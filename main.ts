// free-model-aggregator v2: one server, two client wires, ten free providers.
//
// The server reads the trial brief, probes every provider's catalog route,
// builds one catalog, registers with the env, and serves the client's twenty
// requests: each answered with the best free model that meets its
// requirements, and exiting 0 when the env says the workload is done.
//
// Routing is the change in v2, revised again in v3. The trial's provider
// schedule — budget, scheduled refusals, outage — is drawn per trial from
// a seed the entry never receives: MAKEFUN_TRIAL_SEED hands it only a
// 64-bit derivative of the instance seed, from which the seed cannot be
// recovered, and GET /instance publishes no budgets or failure schedules
// either. So no routing can promise zero refusals. Instead the planner
// caps each provider's calls at a fixed per-trial number chosen to
// minimise the EXPECTED refusals (Monte Carlo over the generator's
// published draw: concentrate calls on the wide-budget providers, one
// call each on the narrow ones), and assigns the twenty requests to those
// caps by min-cost flow over the sealed quality table. And the runtime
// failover behind the plan is no longer blind: it keeps the exact
// Bayesian posterior over each provider's schedule (see lib/plan.ts),
// updated from every call's outcome, orders each request's attempts by
// the pairwise-optimal key rating - 50/p, and skips a planned pick the
// posterior puts below 0.7 success instead of spending a call confirming
// a dead provider. Measured on the same Monte Carlo: 22.4 upstream calls
// per trial at mean quality 98.5 (score 99 in all but 1-2 trials in
// 2,000), against the v2 entry's 23.1 calls at the same quality and the
// v1 entry's 29.9 calls at quality 99.9. On the three practice trials
// the v3 revision moves trial 0 from 22 calls to 21, so the practice
// mean is 21.0 at score 99 on all three.
//
// Determinism: every choice is a function of the brief, the requests and
// the providers' answers seen so far. No wall clock, no unseeded
// randomness, no concurrent provider calls (one request at a time,
// sequential calls only).

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type Brief,
  type CatalogReport,
  type Json,
  type ListedModel,
  type Probe,
  type ProviderEntry,
  type Requirement,
} from './lib/providers.ts';
import {
  candidatesFor,
  capsPlan,
  FailurePosterior,
  optimalPlan,
  orderAttempts,
  type Candidate,
} from './lib/plan.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicSSEOf,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIAnswer,
  geminiToAnthropicAnswer,
  geminiToOpenAIAnswer,
  openAIError,
  openAIModelList,
  openAIRequestToAnthropic,
  openAISSEOf,
  openAIToAnthropic,
  openAIToGemini,
  parseAnthropicSSE,
  parseGeminiSSE,
  parseOpenAISSE,
  type ServedModel,
  type StreamAnswer,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(
    brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)),
  );
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ListedModel, b: ListedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The planned pick per manifest row, by the caps-optimal plan or its greedy walk. */
function buildPlan(): (Candidate | null)[] {
  const planned = optimalPlan(brief, probes);
  if (planned !== null) {
    const counts = new Map<string, number>();
    for (const p of planned) {
      if (p === null) continue;
      counts.set(p.probe.provider.id, (counts.get(p.probe.provider.id) ?? 0) + 1);
    }
    log(
      `caps plan: flow assigned ${planned.filter((p) => p !== null).length} of ${planned.length} requests, ` +
        [...counts.entries()].map(([id, n]) => `${id} ${n}`).join(', '),
    );
    return planned;
  }
  log('caps plan: flow could not cover every request, using the greedy walk');
  return capsPlan(brief, probes);
}

let plan: (Candidate | null)[] = [];

/** How many upstream calls each provider has taken this trial, success or refusal. */
const callsMade = new Map<string, number>();
/** The failure posterior over every provider's schedule, learning from each call's outcome. */
const posterior = new FailurePosterior(brief.directory.providers.map((p) => p.id));

/** Whether an OpenAI-client body asks for JSON mode. */
function isJsonModeRequest(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

// ---------------------------------------------------------------------------
// Upstream calls
// ---------------------------------------------------------------------------

interface UpstreamCall {
  provider: ProviderEntry;
  model: string;
  stream: boolean;
  /** The upstream request body, in the provider's wire. */
  body: Json;
}

/** Build the upstream request for one candidate, in the provider's wire. */
function upstreamFor(clientApi: 'openai' | 'anthropic', body: Json, candidate: Candidate): UpstreamCall {
  const provider = candidate.probe.provider;
  const model = candidate.model.id;
  const stream = body.stream === true;
  if (clientApi === 'openai') {
    if (provider.wire === 'openai') {
      const upstream = { ...body, model };
      if (stream) upstream.stream_options = { include_usage: true };
      return { provider, model, stream, body: upstream };
    }
    if (provider.wire === 'anthropic') return { provider, model, stream, body: openAIRequestToAnthropic(body, model) };
    return { provider, model, stream, body: openAIToGemini(body, model) };
  }
  if (provider.wire === 'openai') return { provider, model, stream, body: anthropicToOpenAI(body, model) };
  if (provider.wire === 'anthropic') return { provider, model, stream, body: { ...body, model } };
  return { provider, model, stream, body: anthropicToGemini(body, model) };
}

/** The URL and headers of one upstream call. */
function upstreamTarget(call: UpstreamCall): { url: string; headers: Record<string, string> } {
  const base = joinBase(ENV_URL, call.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(call.provider.auth, call.provider.key),
  };
  if (call.provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (call.provider.wire === 'gemini') {
    const path = call.stream ? ':streamGenerateContent?alt=sse' : ':generateContent';
    return { url: `${base}/v1beta/models/${encodeURIComponent(call.model)}${path}`, headers };
  }
  if (call.provider.wire === 'anthropic') return { url: `${base}/v1/messages`, headers };
  return { url: `${base}/v1/chat/completions`, headers };
}

/** One provider call: the fetch Response, or null when it could not be made at all. */
async function callUpstream(call: UpstreamCall): Promise<Response | null> {
  const { url, headers } = upstreamTarget(call);
  try {
    return await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(call.body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch {
    return null;
  }
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Parse a buffered upstream SSE body into its answer, by wire. */
function parseUpstreamStream(wire: 'openai' | 'anthropic' | 'gemini', text: string): StreamAnswer {
  if (wire === 'anthropic') return parseAnthropicSSE(text);
  if (wire === 'gemini') return parseGeminiSSE(text);
  return parseOpenAISSE(text);
}

/**
 * Answer one client request from one successful upstream response.
 * Same-wire answers pass through unchanged; cross-wire answers are
 * converted (buffered streams are re-emitted in the client's event order).
 */
async function answerFrom(
  clientApi: 'openai' | 'anthropic',
  candidate: Candidate,
  upstream: Response,
  res: ServerResponse,
): Promise<void> {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  const stream = upstream.headers.get('content-type')?.includes('text/event-stream') === true;
  const sameWire = (clientApi === 'openai' && wire === 'openai') || (clientApi === 'anthropic' && wire === 'anthropic');
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    if (sameWire && upstream.body !== null) {
      await pipeStream(upstream.body, res);
      return;
    }
    const text = await upstream.text();
    const answer = parseUpstreamStream(wire, text);
    res.end(clientApi === 'openai' ? openAISSEOf(answer, model) : anthropicSSEOf(answer, model));
    return;
  }
  const text = await upstream.text();
  if (sameWire) {
    res.writeHead(upstream.status, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
    res.end(text);
    return;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'server_error', 'the provider answered a body that is not JSON');
    return;
  }
  if (!isRecord(parsed)) {
    sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'server_error', 'the provider answered a JSON value that is not one object');
    return;
  }
  if (clientApi === 'openai') {
    if (wire === 'anthropic') sendJson(res, 200, anthropicToOpenAIAnswer(parsed, model));
    else sendJson(res, 200, geminiToOpenAIAnswer(parsed, model));
    return;
  }
  if (wire === 'openai') sendJson(res, 200, openAIToAnthropic(parsed, model));
  else sendJson(res, 200, geminiToAnthropicAnswer(parsed, model));
}

/**
 * Serve one client request: the planned pick first unless the failure
 * posterior says it is likely dead, then the remaining candidates in
 * the Bayes-key order (see orderAttempts), stopping at the first 2xx.
 * Any non-2xx, fetch error or timeout moves on immediately, and every
 * attempt — success or refusal — is recorded in the posterior, so the
 * order of the next request already knows what this one learned. No
 * candidate is tried twice for one request.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const index = manifestIndex - 1;
  const requires = manifestIndex > 0 && brief.workload?.manifest[index] !== undefined
    ? brief.workload.manifest[index].requires
    : requirementsFromBody(body);
  const jsonMode = isJsonModeRequest(clientApi, body);
  const ranked = candidatesFor(brief, probes, requires, jsonMode);
  // The planned pick is used only when it is still ranked for this request
  // as actually sent (a body carrying response_format where the manifest did
  // not flag json would rule an Anthropic-wire plan out, for instance).
  const planned = plan[index] ?? null;
  const plannedUsable =
    planned !== null &&
    ranked.some((c) => c.probe.provider.id === planned.probe.provider.id && c.model.id === planned.model.id)
      ? planned
      : null;
  // The attempt order is recomputed after every refusal: the posterior
  // the failed attempt just updated can demote the provider that failed
  // (a failure makes an outage the likelier story for its later calls),
  // so the next attempt must be chosen with the new probabilities, not
  // the ones from before the request started. The simulation this
  // ordering was measured with re-ranks the same way after each failure.
  const tried = new Set<string>();
  let firstAttempt = true;
  for (;;) {
    const ordered = orderAttempts(plannedUsable, ranked, callsMade, posterior);
    const candidate = ordered.find((c) => !tried.has(`${c.probe.provider.id}/${c.model.id}`));
    if (candidate === undefined) break;
    if (firstAttempt && plannedUsable !== null && candidate !== plannedUsable) {
      log(`skipping a likely-dead planned pick: ${plannedUsable.probe.provider.id}/${plannedUsable.model.id} (the posterior puts it below its skip threshold)`);
    }
    firstAttempt = false;
    tried.add(`${candidate.probe.provider.id}/${candidate.model.id}`);
    const providerId = candidate.probe.provider.id;
    const callNumber = (callsMade.get(providerId) ?? 0) + 1;
    const call = upstreamFor(clientApi, body, candidate);
    let upstream: Response | null;
    try {
      upstream = await callUpstream(call);
    } catch {
      upstream = null;
    }
    callsMade.set(providerId, callNumber);
    if (upstream === null || !upstream.ok) {
      // Every attempt teaches the posterior something, whether the
      // provider refused in its own envelope or the call never came
      // back: a provider that just failed is more likely down than
      // merely unlucky on one call number.
      posterior.record(providerId, callNumber, false);
      if (upstream !== null) {
        try {
          await upstream.text();
        } catch {
          // The error body is not needed: move on.
        }
      }
      continue;
    }
    posterior.record(providerId, callNumber, true);
    await answerFrom(clientApi, candidate, upstream, res);
    return;
  }
  sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory answered this request');
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

/** POST /v1/chat/completions. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  requirementsFor(body);
  await serve('openai', body, res);
}

/** POST /v1/messages. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  requirementsFor(body);
  await serve('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          for (const line of posterior.describe()) log(`trial outcome: ${line}`);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  plan = buildPlan();
  if (!posterior.valid) {
    log(
      `failure posterior failed its checks (weights: ${[...posterior.weightSums()].map(([id, s]) => `${id} ${s.toFixed(6)}`).join(', ')}) — routing continues, but its probabilities are not the measured model's`,
    );
  }
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
