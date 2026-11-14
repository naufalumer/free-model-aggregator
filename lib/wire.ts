// The client wires and the three upstream wires.
//
// The trial's client calls this server as an OpenAI service
// (POST /v1/chat/completions) or an Anthropic one (POST /v1/messages), in
// JSON or server-sent events. Each of the ten providers speaks one of three
// wires (openai, anthropic, gemini), so six translations exist:
//
//   client openai     -> upstream openai     : pass through, model replaced
//   client openai     -> upstream anthropic  : rewrite to Messages
//   client openai     -> upstream gemini     : rewrite to generateContent
//   client anthropic  -> upstream openai     : rewrite to Chat Completions
//   client anthropic  -> upstream anthropic  : pass through, model replaced
//   client anthropic  -> upstream gemini     : rewrite to generateContent
//
// The heart of the wire score is echo preservation: every provider answers
// with text `ANSWER ECHO{...}` echoing what it parsed from the upstream
// request, and the client compares that echo with what it sent. So every
// field the client sent — system text, user text, images, tool definitions,
// max_tokens, stop sequences, JSON mode — is passed through exactly, in the
// target wire's shape, and never invented when absent.
//
// Same-wire answers are piped through unchanged. Cross-wire streaming
// answers are buffered whole, parsed, and re-emitted in the client's event
// order; that is deterministic and keeps the usage numbers exact.
//
// Everything here is a pure function of its arguments: no clock, no
// randomness. The two model listings use fixed timestamps for the same
// reason.

import { isRecord, type Json } from './providers.ts';

/** One model this server lists on GET /v1/models. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** A tool call's arguments as an object: the JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function finishReasonOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as OpenAI's finish_reason. */
export function finishReasonOfGemini(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as Anthropic's stop_reason. */
export function stopReasonOfGemini(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'SAFETY') return 'refusal';
  return 'end_turn';
}

// ---------------------------------------------------------------------------
// Shared message translation helpers
// ---------------------------------------------------------------------------

/** The system text of an OpenAI body: its system messages' text, joined with "\n". */
function openAISystemText(body: Json): string {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const parts: string[] = [];
  for (const raw of messages) {
    if (!isRecord(raw) || raw.role !== 'system') continue;
    const text = textOf(raw.content);
    if (text !== '') parts.push(text);
  }
  return parts.join('\n');
}

/** The system text of an Anthropic body: its system field, string or blocks. */
function anthropicSystemText(body: Json): string {
  return textOf(body.system);
}

/** One OpenAI content part as an Anthropic content block: text or image; null for the rest. */
function anthropicBlockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) {
      return { type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** One Anthropic content block as an OpenAI content part: text or image; null for the rest. */
function openAIPartOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** One content part (either wire's image/text spelling) as a Gemini part; null for the rest. */
function geminiPartOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    return { fileData: { fileUri: url } };
  }
  if (part.type === 'image' && isRecord(part.source)) {
    const source = part.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { inlineData: { mimeType: media, data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** OpenAI tool definitions as Anthropic tools, order preserved. */
function anthropicToolsOf(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
}

/** Anthropic tool definitions as OpenAI tools, order preserved. */
function openAIToolsOf(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        type: 'function',
        function: {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      },
    ];
  });
}

/** Tool definitions of either wire as Gemini function declarations, order preserved. */
function geminiDeclarationsOf(tools: Json[]): Json[] {
  return tools.flatMap((raw) => {
    if (!isRecord(raw)) return [];
    if (typeof raw.name === 'string') {
      return [
        {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      ];
    }
    if (isRecord(raw.function) && typeof raw.function.name === 'string') {
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    }
    return [];
  });
}

/** The names of a tool list, either wire's spelling. */
function toolNamesOf(tools: Json[]): string[] {
  return tools.flatMap((raw) => {
    if (!isRecord(raw)) return [];
    if (typeof raw.name === 'string') return [raw.name];
    if (isRecord(raw.function) && typeof raw.function.name === 'string') return [raw.function.name];
    return [];
  });
}

/**
 * A Gemini toolConfig for a tool_choice: omitted for auto (the default),
 * ANY with the allowed names for a forced choice, NONE when tools are off.
 * `tools` is the request's tool list in either wire's spelling.
 */
function geminiToolConfigOf(choice: unknown, tools: Json[]): Json | null {
  if (typeof choice === 'string') {
    if (choice === 'required') return { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: toolNamesOf(tools) } };
    if (choice === 'none') return { functionCallingConfig: { mode: 'NONE' } };
    return null;
  }
  if (isRecord(choice)) {
    if (choice.type === 'any') return { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: toolNamesOf(tools) } };
    if (choice.type === 'tool' && typeof choice.name === 'string') {
      return { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.name] } };
    }
    if (isRecord(choice.function) && typeof choice.function.name === 'string') {
      return { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.function.name] } };
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Client OpenAI -> upstream
// ---------------------------------------------------------------------------

/** An OpenAI Chat Completions request as an Anthropic Messages request naming `model`. */
export function openAIRequestToAnthropic(body: Json, model: string): Json {
  const system = openAISystemText(body);
  const messages: Json[] = [];
  // Tool-role messages are folded into the following user message's
  // tool_result blocks, so the last user message's text echo is unchanged.
  let pendingToolResults: Json[] = [];
  const flushToolResults = (): Json[] => {
    const out = pendingToolResults;
    pendingToolResults = [];
    return out;
  };
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') continue;
    if (raw.role === 'tool') {
      pendingToolResults.push({
        type: 'tool_result',
        tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
        content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content),
      });
      continue;
    }
    if (raw.role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        content.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (content.length > 0) messages.push({ role: 'assistant', content });
      continue;
    }
    // A user message: its text/image parts plus any pending tool results.
    const content: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') content.push({ type: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        const block = anthropicBlockOf(part);
        if (block !== null) content.push(block);
      }
    }
    content.push(...flushToolResults());
    // Always emit the user entry, even when it holds only tool results: a
    // missing entry would leave the provider's last user text at an earlier
    // message's text instead of "".
    messages.push({ role: 'user', content });
  }
  if (pendingToolResults.length > 0) {
    const lastUser = [...messages].reverse().find((m) => m.role === 'user');
    if (lastUser !== undefined && Array.isArray(lastUser.content)) lastUser.content.push(...flushToolResults());
    else messages.push({ role: 'user', content: flushToolResults() });
  }

  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop)) out.stop_sequences = body.stop.filter((s): s is string => typeof s === 'string');
  const tools = anthropicToolsOf(body);
  if (tools.length > 0) out.tools = tools;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice.type === 'any') out.tool_choice = { type: 'any' };
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'tool', name: choice.name };
  } else if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  if (body.stream === true) out.stream = true;
  return out;
}

/** An OpenAI Chat Completions request as a Gemini generateContent request naming `model`. */
export function openAIToGemini(body: Json, model: string): Json {
  const system = openAISystemText(body);
  const contents: Json[] = [];
  let pendingResponses: Json[] = [];
  const flushResponses = (): Json[] => {
    const out = pendingResponses;
    pendingResponses = [];
    return out;
  };
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') continue;
    if (raw.role === 'tool') {
      pendingResponses.push({
        functionResponse: {
          name: typeof raw.tool_call_id === 'string' && raw.tool_call_id !== '' ? raw.tool_call_id : 'tool',
          response: { result: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) },
        },
      });
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          functionCall: {
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: argumentsOf(call.function.arguments),
          },
        });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        const geminiPart = geminiPartOf(part);
        if (geminiPart !== null) parts.push(geminiPart);
      }
    }
    parts.push(...flushResponses());
    contents.push({ role: 'user', parts });
  }
  if (pendingResponses.length > 0) {
    const lastUser = [...contents].reverse().find((c) => c.role === 'user');
    if (lastUser !== undefined && Array.isArray(lastUser.parts)) lastUser.parts.push(...flushResponses());
    else contents.push({ role: 'user', parts: flushResponses() });
  }

  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarations = geminiDeclarationsOf(Array.isArray(body.tools) ? body.tools : []);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (typeof body.stop === 'string') config.stopSequences = [body.stop];
  else if (Array.isArray(body.stop)) config.stopSequences = body.stop.filter((s): s is string => typeof s === 'string');
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  const toolConfig = geminiToolConfigOf(body.tool_choice, Array.isArray(body.tools) ? body.tools : []);
  if (toolConfig !== null) out.toolConfig = toolConfig;
  return out;
}

// ---------------------------------------------------------------------------
// Client Anthropic -> upstream
// ---------------------------------------------------------------------------

/** An Anthropic Messages request as an OpenAI Chat Completions request naming `model`. */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = anthropicSystemText(body);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    // User message: text/image parts first, then its tool results as tool
    // messages. The user entry is always emitted, so a tool-result-only
    // message still leaves the provider's last user text at "".
    const parts: Json[] = [];
    const toolMessages: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        toolMessages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = openAIPartOf(block);
      if (part !== null) parts.push(part);
    }
    messages.push({ role: 'user', content: parts.length > 0 ? parts : '' });
    messages.push(...toolMessages);
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) {
    out.stop = body.stop_sequences.filter((s): s is string => typeof s === 'string');
  }
  const tools = openAIToolsOf(body);
  if (tools.length > 0) out.tools = tools;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** An Anthropic Messages request as a Gemini generateContent request naming `model`. */
export function anthropicToGemini(body: Json, model: string): Json {
  const system = anthropicSystemText(body);
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const parts: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (assistant && block.type === 'tool_use') {
          parts.push({
            functionCall: {
              name: typeof block.name === 'string' ? block.name : '',
              args: isRecord(block.input) ? block.input : {},
            },
          });
          continue;
        }
        if (!assistant && block.type === 'tool_result') {
          parts.push({
            functionResponse: {
              name: typeof block.tool_use_id === 'string' ? block.tool_use_id : 'tool',
              response: { result: toolResultText(block.content) },
            },
          });
          continue;
        }
        const part = geminiPartOf(block);
        if (part !== null) parts.push(part);
      }
    }
    if (parts.length > 0) contents.push({ role: assistant ? 'model' : 'user', parts });
  }

  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarations = geminiDeclarationsOf(Array.isArray(body.tools) ? body.tools : []);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) {
    config.stopSequences = body.stop_sequences.filter((s): s is string => typeof s === 'string');
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  const toolConfig = geminiToolConfigOf(body.tool_choice, Array.isArray(body.tools) ? body.tools : []);
  if (toolConfig !== null) out.toolConfig = toolConfig;
  return out;
}

// ---------------------------------------------------------------------------
// Upstream JSON answers -> client JSON answers
// ---------------------------------------------------------------------------

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** An Anthropic Messages answer as a Chat Completions object. */
export function anthropicToOpenAIAnswer(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const textParts: string[] = [];
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') textParts.push(block.text);
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : 'call_unknown',
        type: 'function',
        function: {
          name: typeof block.name === 'string' ? block.name : '',
          arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
        },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const message: Json = { role: 'assistant', content: textParts.join('') };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  return {
    id: `chatcmpl_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: finishReasonOf(body.stop_reason), logprobs: null }],
    usage: {
      prompt_tokens: numberOrZero(usage.input_tokens),
      completion_tokens: numberOrZero(usage.output_tokens),
      total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens),
    },
  };
}

/** The text, token counts and finish reason of a Gemini answer's first candidate. */
function geminiAnswerParts(body: Json): { text: string; input: number; output: number; finish: unknown } {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    input: numberOrZero(meta.promptTokenCount),
    output: numberOrZero(meta.candidatesTokenCount),
    finish: candidate.finishReason,
  };
}

/** A Gemini generateContent answer as a Chat Completions object. */
export function geminiToOpenAIAnswer(body: Json, model: string): Json {
  const answer = geminiAnswerParts(body);
  return {
    id: 'chatcmpl_gemini',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: finishReasonOfGemini(answer.finish), logprobs: null }],
    usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
export function geminiToAnthropicAnswer(body: Json, model: string): Json {
  const answer = geminiAnswerParts(body);
  return {
    id: 'msg_gemini',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: stopReasonOfGemini(answer.finish),
    stop_sequence: null,
    usage: { input_tokens: answer.input, output_tokens: answer.output },
  };
}

// ---------------------------------------------------------------------------
// SSE: parse an upstream stream, emit a client stream
// ---------------------------------------------------------------------------

/** What a buffered upstream stream holds: the answer text, its usage and its stop. */
export interface StreamAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stop: string;
}

/** Split SSE text into frames: {event, data} pairs, data lines joined with "\n". */
function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseJsonObject(payload: string): Json | null {
  try {
    const value: unknown = JSON.parse(payload);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** Parse a Chat Completions SSE stream into its answer. */
export function parseOpenAISSE(text: string): StreamAnswer {
  let out = '';
  let input = 0;
  let output = 0;
  let stop = 'stop';
  for (const frame of sseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    const chunk = parseJsonObject(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      input = numberOrZero(chunk.usage.prompt_tokens);
      output = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { text: out, inputTokens: input, outputTokens: output, stop };
}

/** Parse an Anthropic Messages SSE stream into its answer. */
export function parseAnthropicSSE(text: string): StreamAnswer {
  let out = '';
  let input = 0;
  let output = 0;
  let stop = 'end_turn';
  for (const frame of sseFrames(text)) {
    const data = parseJsonObject(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') out += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) output = numberOrZero(data.usage.output_tokens);
    }
  }
  return { text: out, inputTokens: input, outputTokens: output, stop };
}

/** Parse a Gemini streamGenerateContent SSE stream into its answer. */
export function parseGeminiSSE(text: string): StreamAnswer {
  let out = '';
  let input = 0;
  let output = 0;
  let stop = 'STOP';
  for (const frame of sseFrames(text)) {
    const data = parseJsonObject(frame.data);
    if (data === null) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      const content = isRecord(candidate.content) ? candidate.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const part of parts) {
        if (isRecord(part) && typeof part.text === 'string') out += part.text;
      }
      if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
    }
    if (isRecord(data.usageMetadata)) {
      input = numberOrZero(data.usageMetadata.promptTokenCount);
      output = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  }
  return { text: out, inputTokens: input, outputTokens: output, stop };
}

/** One SSE frame: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split text into 13-character pieces, the way the providers chunk their deltas. */
function splitPieces(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

/**
 * A buffered answer as a Chat Completions SSE stream: a role chunk, text
 * deltas, one finish chunk, a usage chunk, then [DONE].
 */
export function openAISSEOf(answer: StreamAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl_aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitPieces(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, finishReasonOf(answer.stop))));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl_aggregate',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/**
 * A buffered answer as an Anthropic Messages SSE stream: message_start,
 * one text content block, message_delta with the stop reason and output
 * usage, then message_stop.
 */
export function anthropicSSEOf(answer: StreamAnswer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregate',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitPieces(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopReasonOf(answer.stop), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
