// The request plan: which (provider, model) answers each manifest row.
//
// There is exactly one plan, and it is a caps-constrained assignment.
// Each provider gets a fixed per-trial call cap, and the twenty requests
// are assigned to (provider, model) pairs by min-cost flow, maximising the
// sealed quality the assignment earns under those caps. The runtime
// failover behind the plan absorbs whatever a provider actually refuses.
//
// Why caps, and why these numbers: the trial's provider schedule —
// budget, scheduled refusals, outage — is drawn per trial from a seed the
// entry never receives. MAKEFUN_TRIAL_SEED hands the entry only the first
// eight bytes of sha256("agent-fun/trial-seed/v1\n" + instance_sha256 +
// "\n" + trial index), rendered as a decimal number: a one-way 64-bit
// derivative from which the 256-bit instance seed cannot be recovered, and
// GET /instance publishes no budgets or failure schedules either. So the
// schedule cannot be derived at runtime, and no routing can promise zero
// refusals. The caps are therefore chosen to minimise the EXPECTED number
// of refusals, by Monte Carlo over the generator's own published draw
// (2,000 random schedules): a provider's second-and-later calls fail with
// probability ~1.5 / (budget - 2), so calls are concentrated on the
// wide-budget providers where that density is lowest — kilol (budget
// 12–16) takes six, openrouter and pollinations three each — and capped to
// one on the narrow providers (zai, sambanova, ovh, cerebras: budget 4–6)
// whose later calls are the most likely to be refused. Measured on the
// same Monte Carlo: this assignment averages 23.0 upstream calls per
// trial at mean quality 98.5 (score 99 in 100% of 500 three-trial finals),
// against the v1 entry's loose caps at 29.9 calls and quality 99.9. The
// quality given up is the price of the calls saved; the score stays 99
// because the score's quality column only needs 98.
//
// The flow runs on fixed node and edge orders with no clock and no
// randomness, so a live pass and its replay plan identically.

// ---------------------------------------------------------------------------
// The failure posterior (the v3 revision of this file).
//
// The caps plan is only half the routing: it picks the first attempt of
// each request, and the runtime failover behind it used to walk the
// ranked candidates in pure rating order, retrying a provider that had
// just refused as if nothing had been learned. That wastes calls — the
// trial's failure schedule is fixed per trial, so every refusal is
// evidence about what the provider will do next.
//
// The posterior below is the exact Bayesian update over the generator's
// own published draw (kit program/generator/draw.mjs): each provider's
// budget is uniform on its published range, it carries one or two
// scheduled fail calls uniform in [2, budget - 1], one provider in ten
// is designated to fail its second call, and one provider in ten goes
// down from an outage call uniform in [2, budget - 1]. The entry never
// sees the drawn schedule — but the DISTRIBUTION is published, so the
// prior is legitimate knowledge, the same distribution the caps were
// chosen from. Every observed call outcome drops the hypotheses it
// contradicts, and the survivors give P(the provider's next call
// answers). Two structural facts fall out of it for free: a provider's
// first call always answers (no fail call or outage can be numbered 1),
// and its second call is the riskiest (the designated failure lives
// there, P ≈ 0.25-0.27 before anything is observed).
//
// The failover order is the pairwise-optimal order for a chain of tries
// (see failoverKey): a candidate's key is rating - K/p, K = 50 milli
// per failed call. And the planned pick itself is skipped when the
// posterior puts its success below 0.7 — a provider that just failed
// is more likely down than merely unlucky on one call number, so
// spending a call confirming the dead pick is worse than going straight
// to the key order. Measured on the Monte Carlo over the published draw
// (2,000 trials, re-checked on a fresh 4,000-trial seed set): 22.4 mean
// upstream calls at mean quality 98.5, score 99 in all but 1-2 trials
// in 2,000, against the rating-ordered failover's 23.1 calls at the
// same quality. On the three practice trials the revision moves trial 0
// from 22 calls to 21 (trials 1-2 stay 21), so the practice mean is
// 21.0 against v2's 21.33, score 99 on all three.

import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type Brief,
  type ListedModel,
  type Probe,
  type Requirement,
} from './providers.ts';

export interface Candidate {
  probe: Probe;
  model: ListedModel;
  rating: number;
}

// ---------------------------------------------------------------------------
// Failure hypotheses: the generator's published draw, as a prior
// ---------------------------------------------------------------------------

/**
 * The generator's published per-provider budget ranges (kit
 * program/generator/draw.mjs, the PROVIDERS table's calls_min /
 * calls_max). The entry never sees a trial's drawn budget, but the
 * range is published, so it is a legitimate prior.
 */
const BUDGET_RANGES: Record<string, readonly [number, number]> = {
  groq: [9, 12],
  cerebras: [5, 6],
  gemini: [6, 8],
  zai: [4, 5],
  openrouter: [10, 12],
  sambanova: [4, 5],
  pollinations: [10, 14],
  cloudflare: [7, 8],
  ovh: [5, 6],
  kilol: [12, 16],
};

/** n choose k, for the small integers the hypothesis count needs. */
function choose(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  let out = 1;
  for (let i = 0; i < k; i++) out = (out * (n - i)) / (i + 1);
  return out;
}

/** Every k-subset of `pool`, in lexicographic order. */
function combinations(pool: number[], k: number): number[][] {
  if (k === 0) return [[]];
  if (pool.length < k) return [];
  const [head, ...tail] = pool;
  return [
    ...combinations(tail, k - 1).map((rest) => [head, ...rest]),
    ...combinations(tail, k),
  ];
}

interface FailureHypotheses {
  /** The drawn budget of each hypothesis. */
  budget: number[];
  /** Bitmask of the scheduled fail calls (bit k set when call k refuses). */
  failMask: number[];
  /** The outage's first refusing call, or 0 when the provider never goes down. */
  outage: number[];
  /** The prior weight of each hypothesis; sums to 1. */
  weight: number[];
}

/**
 * Every (budget, fail set, designated, outage) hypothesis for one
 * provider, with its prior weight. The draw, as published: budget
 * uniform on the provider's range; one or two fail calls drawn uniform
 * without replacement from [2, budget - 1]; one provider in ten
 * designated to fail its second call (its drawn fail set is rewritten
 * to [2, second-smallest drawn]); one provider in ten going down, its
 * outage call uniform in [2, budget - 1]. Unknown providers have no
 * hypotheses and are never penalised for what cannot be modelled.
 */
function hypothesesFor(providerId: string): FailureHypotheses | null {
  const range = BUDGET_RANGES[providerId];
  if (range === undefined) return null;
  const [lo, hi] = range;
  const budget: number[] = [];
  const failMask: number[] = [];
  const outage: number[] = [];
  const weight: number[] = [];
  const nB = hi - lo + 1;
  for (let b = lo; b <= hi; b++) {
    const pool: number[] = [];
    for (let k = 2; k <= b - 1; k++) pool.push(k);
    for (const count of [1, 2]) {
      if (pool.length < count) continue;
      const prior = (1 / nB) * 0.5 * (1 / choose(pool.length, count));
      for (const drawn of combinations(pool, count)) {
        for (const designated of [false, true]) {
          const fails = new Set<number>(drawn);
          if (designated && !fails.has(2)) {
            const rest = [...fails].sort((x, y) => x - y).slice(1);
            fails.clear();
            fails.add(2);
            for (const f of rest) fails.add(f);
          }
          let mask = 0;
          for (const f of fails) mask |= 1 << f;
          const designatedWeight = prior * (designated ? 0.1 : 0.9);
          for (const out of [0, ...pool]) {
            budget.push(b);
            failMask.push(mask);
            outage.push(out);
            weight.push(designatedWeight * (out === 0 ? 0.9 : 0.1 / (b - 2)));
          }
        }
      }
    }
  }
  return { budget, failMask, outage, weight };
}

/** Rank two candidates: rating DESC, provider id ASC, model id ASC. */
export function compareCandidates(a: Candidate, b: Candidate): number {
  if (a.rating !== b.rating) return b.rating - a.rating;
  if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
  return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
}

/**
 * Every (provider, free model) that can serve a request, ranked best first.
 * `jsonMode` is true for an OpenAI-client request carrying response_format
 * json_object/json_schema.
 */
export function candidatesFor(brief: Brief, probes: readonly Probe[], requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort(compareCandidates);
  return out;
}

/**
 * Planned per-provider call caps. The caps sum to exactly the twenty
 * requests, so the plan spends one call per request when nothing is
 * refused and every refusal is paid for by the runtime failover. The
 * spread follows the failure density of the generator's draw: wide-budget
 * providers take the most calls, narrow ones take one each, and the
 * single most-used provider is openrouter, whose top model the heavy
 * requests need. The caps were chosen by hill-climbing over the exact
 * quality-maximal feasible assignment (min-cost flow): this spread is the
 * cheapest one whose planned quality stays at or above 98.4, a 0.4-point
 * margin over the 98.0 floor the score-99 column needs. (Monte Carlo
 * over the published draw, 500 three-trial finals: mean 23.0 calls,
 * mean quality 98.5, score 99 in 100% of finals; v1's loose caps: 29.9
 * calls, quality 99.9. The first caps tried here — kilol-heavy, planned
 * quality 97.7 — scored 98 on two of three practice trials and were
 * rejected on that measurement.)
 */
export const PLANNED_CAPS: Record<string, number> = {
  groq: 1,
  cerebras: 1,
  gemini: 1,
  zai: 1,
  openrouter: 6,
  sambanova: 1,
  pollinations: 3,
  cloudflare: 2,
  ovh: 1,
  kilol: 3,
};

/** The greedy walk: highest-ranked candidate with planned capacity left. */
export function capsPlan(brief: Brief, probes: readonly Probe[]): (Candidate | null)[] {
  const remaining: Record<string, number> = { ...PLANNED_CAPS };
  const manifest = brief.workload?.manifest ?? [];
  return manifest.map((entry) => {
    const jsonMode = entry.api === 'openai' && entry.requires.json === true;
    const ranked = candidatesFor(brief, probes, entry.requires, jsonMode);
    if (ranked.length === 0) return null;
    for (const candidate of ranked) {
      const left = remaining[candidate.probe.provider.id] ?? 0;
      if (left > 0) {
        remaining[candidate.probe.provider.id] = left - 1;
        return candidate;
      }
    }
    return ranked[0];
  });
}

// ---------------------------------------------------------------------------
// Min-cost flow over the planned caps
// ---------------------------------------------------------------------------

interface FlowEdge {
  to: number;
  rev: number;
  cap: number;
  cost: number;
}

/** Successive shortest paths with SPFA; fixed orders, no randomness. */
class MinCostFlow {
  private readonly graph: FlowEdge[][];

  constructor(size: number) {
    this.graph = Array.from({ length: size }, () => []);
  }

  add(from: number, to: number, cap: number, cost: number): void {
    this.graph[from].push({ to, rev: this.graph[to].length, cap, cost });
    this.graph[to].push({ to: from, rev: this.graph[from].length - 1, cap: 0, cost: -cost });
  }

  /** Move as much flow as possible from source to sink, cheapest first. Returns units moved. */
  run(source: number, sink: number, want: number): number {
    const n = this.graph.length;
    let moved = 0;
    while (moved < want) {
      const dist = new Array<number>(n).fill(Number.POSITIVE_INFINITY);
      const prevNode = new Array<number>(n).fill(-1);
      const prevEdge = new Array<number>(n).fill(-1);
      const inQueue = new Array<boolean>(n).fill(false);
      const queue: number[] = [source];
      dist[source] = 0;
      inQueue[source] = true;
      for (let head = 0; head < queue.length; head++) {
        const u = queue[head];
        inQueue[u] = false;
        for (let i = 0; i < this.graph[u].length; i++) {
          const e = this.graph[u][i];
          if (e.cap > 0 && dist[u] + e.cost < dist[e.to]) {
            dist[e.to] = dist[u] + e.cost;
            prevNode[e.to] = u;
            prevEdge[e.to] = i;
            if (!inQueue[e.to]) {
              queue.push(e.to);
              inQueue[e.to] = true;
            }
          }
        }
      }
      if (dist[sink] === Number.POSITIVE_INFINITY) break;
      // One unit: every source edge carries one request.
      let v = sink;
      while (v !== source) {
        const u = prevNode[v];
        const e = this.graph[u][prevEdge[v]];
        e.cap -= 1;
        this.graph[v][e.rev].cap += 1;
        v = u;
      }
      moved += 1;
    }
    return moved;
  }

  /** Whether the edge from `from` to `to` at index `i` was saturated by flow. */
  used(from: number, i: number): boolean {
    return this.graph[from][i].cap === 0;
  }

  edge(from: number, i: number): FlowEdge {
    return this.graph[from][i];
  }

  edgesOf(node: number): FlowEdge[] {
    return this.graph[node];
  }
}

const VALUE_SCALE = 1000000;

/**
 * The optimal plan: the quality-maximal assignment of the manifest to the
 * planned per-provider caps. Returns null when the ratings table or the
 * workload is missing, or when the flow cannot cover every manifest row
 * (the caller then uses the greedy walk).
 */
export function optimalPlan(brief: Brief, probes: readonly Probe[]): (Candidate | null)[] | null {
  const manifest = brief.workload?.manifest ?? [];
  const ratings = brief.ratings;
  if (manifest.length === 0 || ratings === null) return null;
  const providerIds = probes.filter((p) => p.ok).map((p) => p.provider.id);

  // Candidate lists, one per manifest row, in ranking order.
  const ranked: Candidate[][] = manifest.map((entry) => {
    const jsonMode = entry.api === 'openai' && entry.requires.json === true;
    return candidatesFor(brief, probes, entry.requires, jsonMode);
  });
  if (ranked.some((list) => list.length === 0)) return null;

  // Node ids: 0 source, 1..R requests, then option nodes, then provider
  // nodes, then the sink. Option nodes are numbered in first-seen order,
  // walking the manifest and each row's ranked list.
  const requestCount = manifest.length;
  const optionId = new Map<string, number>();
  const optionKey: string[] = [];
  let next = 1 + requestCount;
  for (const list of ranked) {
    for (const c of list) {
      const key = `${c.probe.provider.id}\n${c.model.id}`;
      if (!optionId.has(key)) {
        optionId.set(key, next);
        optionKey.push(key);
        next += 1;
      }
    }
  }
  const providerNode = new Map(providerIds.map((id, i) => [id, next + i]));
  const sink = next + providerIds.length;
  const flow = new MinCostFlow(sink + 1);

  // Source -> request, request -> option (priced by the rating earned),
  // option -> provider, provider -> sink capped at its planned cap.
  const requestEdges: { node: number; edgeIndex: number; candidate: Candidate }[][] = [];
  for (let r = 0; r < requestCount; r++) {
    const node = 1 + r;
    flow.add(0, node, 1, 0);
    const edges: { node: number; edgeIndex: number; candidate: Candidate }[] = [];
    const entry = manifest[r];
    for (const c of ranked[r]) {
      const key = `${c.probe.provider.id}\n${c.model.id}`;
      const option = optionId.get(key);
      if (option === undefined) continue;
      // Price the edge by the quality the pick actually earns — weight
      // times rating — not by rating normalised per request: normalising
      // by each request's best rating distorts the assignment, trading
      // real quality points between requests (measured: a normalised
      // objective lost a full quality point on the practice trials).
      const value = entry.weight * c.rating * VALUE_SCALE;
      const edgeIndex = flow.edgesOf(node).length;
      flow.add(node, option, 1, -value);
      edges.push({ node, edgeIndex, candidate: c });
    }
    requestEdges.push(edges);
  }
  for (const [key, node] of optionId) {
    const providerId = key.split('\n')[0];
    const provider = providerNode.get(providerId);
    if (provider !== undefined) flow.add(node, provider, requestCount, 0);
  }
  for (const id of providerIds) {
    const node = providerNode.get(id);
    if (node !== undefined) flow.add(node, sink, PLANNED_CAPS[id] ?? 0, 0);
  }

  const moved = flow.run(0, sink, requestCount);
  if (moved < requestCount) return null;

  return requestEdges.map((edges) => {
    for (const e of edges) {
      if (flow.used(e.node, e.edgeIndex)) return e.candidate;
    }
    return null;
  });
}

// ---------------------------------------------------------------------------
// Failure posterior: what each provider does next, given what it did
// ---------------------------------------------------------------------------

/**
 * Reference posterior values from the Python model of the same draw
 * (sim_bayes_fast.py, exact enumeration): a fresh provider's call
 * answers with the probability below. The last row is the structural
 * fact that zai's budget tops out at 5, so its sixth call never
 * answers. The constructor checks all five; a mismatch means this
 * port drifted from the model the routing was measured against.
 */
const POSTERIOR_SELF_TEST: [string, number, number][] = [
  ['groq', 2, 0.729542],
  ['kilol', 2, 0.779271],
  ['openrouter', 2, 0.740375],
  ['zai', 2, 0.324375],
  ['zai', 6, 0],
];

/**
 * The exact posterior over one provider's schedule hypotheses, updated
 * from the outcomes this trial has actually observed. A hypothesis that
 * contradicts an observation is dropped (its weight set to 0); the
 * survivors keep their prior weights and are renormalised on read, so
 * the posterior is a pure function of the observation sequence — a live
 * pass and its replay observe the same schedule and compute the same
 * probabilities, in the same arithmetic order, every time.
 */
export class FailurePosterior {
  private readonly hypos = new Map<string, FailureHypotheses>();
  private readonly weights = new Map<string, number[]>();
  private readonly observations = new Map<string, [number, boolean][]>();
  /** How many observations each provider has: the cache key's version. */
  private readonly versions = new Map<string, number>();
  private readonly cache = new Map<string, number>();
  /** Each provider's prior weight sum: one when the enumeration is right. */
  private readonly sums = new Map<string, number>();
  /** False when the hypothesis tables fail their checks (see the constructor). */
  valid = true;

  constructor(providerIds: readonly string[]) {
    for (const id of providerIds) {
      const hypos = hypothesesFor(id);
      if (hypos === null) continue;
      this.hypos.set(id, hypos);
      this.weights.set(id, [...hypos.weight]);
      this.observations.set(id, []);
      this.versions.set(id, 0);
    }
    // The hypothesis weights are a prior: they must sum to one per
    // provider. A sum anywhere else means the enumeration above is
    // wrong, and every routing decision downstream would silently
    // inherit the bias — so the failure is recorded, not swallowed.
    for (const [id, w] of this.weights) {
      let sum = 0;
      for (const x of w) sum += x;
      this.sums.set(id, sum);
      if (Math.abs(sum - 1) > 1e-9) this.valid = false;
    }
    // The enumeration must also reproduce the Python model of the same
    // draw (sim_bayes_fast.py): a fresh provider's second call answers
    // with the probability below. A mismatch means this port drifted
    // from the model the routing was measured against.
    for (const [id, n, want] of POSTERIOR_SELF_TEST) {
      if (Math.abs(this.pSuccess(id, n) - want) > 1e-6) this.valid = false;
    }
  }

  /** Whether call `n` to this provider answers, under hypothesis `i`. */
  private predicts(id: string, n: number, i: number): boolean {
    const h = this.hypos.get(id);
    if (h === undefined) return true;
    return n <= h.budget[i] && (h.failMask[i] & (1 << n)) === 0 && (h.outage[i] === 0 || n < h.outage[i]);
  }

  /** Record that this provider's call `n` answered (true) or refused (false). */
  record(id: string, n: number, ok: boolean): void {
    const h = this.hypos.get(id);
    const w = this.weights.get(id);
    if (h === undefined || w === undefined) return;
    this.observations.get(id)?.push([n, ok]);
    for (let i = 0; i < w.length; i++) {
      if (this.predicts(id, n, i) !== ok) w[i] = 0;
    }
    this.versions.set(id, (this.versions.get(id) ?? 0) + 1);
  }

  /** P(this provider's call `n` answers | everything observed so far). */
  pSuccess(id: string, n: number): number {
    const h = this.hypos.get(id);
    const w = this.weights.get(id);
    // No hypotheses for this provider: nothing is known against it, so
    // do not penalise what cannot be modelled.
    if (h === undefined || w === undefined) return 1;
    const key = `${id}:${this.versions.get(id) ?? 0}:${n}`;
    const hit = this.cache.get(key);
    if (hit !== undefined) return hit;
    let total = 0;
    let good = 0;
    for (let i = 0; i < w.length; i++) {
      total += w[i];
      if (this.predicts(id, n, i)) good += w[i];
    }
    // No surviving hypothesis: the observations contradict the model, so
    // say the honest thing (maximum ignorance) instead of a confident number.
    const p = total > 0 ? good / total : 0.5;
    this.cache.set(key, p);
    return p;
  }

  /** How many calls to this provider this trial has observed. */
  observedCalls(id: string): number {
    return this.observations.get(id)?.length ?? 0;
  }

  /** The prior weight sums, one per provider: each must be 1. */
  weightSums(): ReadonlyMap<string, number> {
    return this.sums;
  }

  /**
   * One line per provider: calls made, refusals seen, and the posterior
   * probability that its next call answers. For the trial-end log: it
   * shows what the trial actually taught the router, provider by
   * provider, in one place instead of scattered through the request logs.
   */
  describe(): string[] {
    const out: string[] = [];
    for (const id of this.hypos.keys()) {
      const obs = this.observations.get(id) ?? [];
      let refusals = 0;
      for (const [, ok] of obs) if (!ok) refusals += 1;
      const p = this.pSuccess(id, obs.length + 1);
      out.push(`${id}: ${obs.length} calls, ${refusals} refusals, P(next)=${p.toFixed(3)}`);
    }
    return out;
  }
}

// Monte Carlo over the generator's published draw (2,000 trials, the
// same seeds for every policy; re-checked on a fresh 4,000-trial seed
// set, where the winner held at 22.38 calls). The shipped caps plan is
// the baseline in every row; only the failover order changes:
//
//   rating-ordered failover (v2):  23.05 calls, quality 98.52, score 99 in all 2,000
//   fresh-provider-first failover: 23.07 calls, quality 98.46, score 99 in 1,998
//   naive-E Bayes failover K=100:  22.69 calls, quality 98.37, score 99 in 1,934
//   Bayes-key failover K=50:       22.72 calls, quality 98.51, score 99 in all 2,000
//   Bayes-key + skip T=0.50:       22.61 calls, quality 98.51, score 99 in all 2,000
//   Bayes-key + skip T=0.70:       22.36 calls, quality 98.52, score 99 in all 2,000
//   Bayes-key + skip T=0.75:       22.62 calls, quality 98.42, score 99 in 1,991
//
// Two things the table settles. First, the naive single-step value
// undervalues a risky high-rated try (a failure still leaves the safe
// fallback next, so its true cost is one extra call, not the request),
// which is why it drops 3% of trials to score 98 while the key above
// holds them. Second, the skip threshold peaks at 0.70: past 0.75 the
// router deviates so often it reverts to the worse key-full behaviour
// and starts leaving requests unserved. The caps were also re-searched
// with this failover in the loop (kilol-heavy, balanced, thin-spread,
// openrouter-4/kilol-5, openrouter-5/kilol-4): the shipped caps stay
// the best set whose planned quality clears 98.40 — the others either
// cost more calls or drop planned quality under the safety margin.

/** The price of one failed upstream call, in rating-milli: the failover's ranking trades rating against it. */
const FAILOVER_PENALTY_MILLI = 50;
/** Skip the planned pick when the posterior puts its success below this. */
const PLANNED_SKIP_BELOW = 0.7;
/** The floor under a success probability used in a ranking key: a certain failure still sorts last instead of dividing by zero. */
const MIN_SUCCESS_PROB = 0.05;

/**
 * The pairwise-optimal order for a chain of tries. Trying A before B is
 * better exactly when rating(A) - K/p(A) > rating(B) - K/p(B), where K
 * is the price of a failed call and p the posterior success
 * probability: a failure still leaves the next candidate to try, so the
 * naive single-step value p*rating - K*(1-p) undervalues a risky
 * high-rated try — on the Monte Carlo it cost 0.4 mean calls against
 * this key at the same quality, with 3% of trials dropping to score 98.
 */
export function failoverKey(candidate: Candidate, callsMade: number, posterior: FailurePosterior): number {
  const p = Math.max(posterior.pSuccess(candidate.probe.provider.id, callsMade + 1), MIN_SUCCESS_PROB);
  return candidate.rating - FAILOVER_PENALTY_MILLI / p;
}

/** Rank two candidates for one request: the failover key, then rating, provider, model — a total order, so live and replay agree. */
export function compareFailover(
  a: Candidate,
  b: Candidate,
  callsMade: ReadonlyMap<string, number>,
  posterior: FailurePosterior,
): number {
  const ka = failoverKey(a, callsMade.get(a.probe.provider.id) ?? 0, posterior);
  const kb = failoverKey(b, callsMade.get(b.probe.provider.id) ?? 0, posterior);
  if (ka !== kb) return kb - ka;
  return compareCandidates(a, b);
}

/**
 * The attempt order for one request. The planned pick goes first while
 * the posterior still gives it a fair chance; when it is likely dead —
 * its provider just failed and an outage is now the likeliest story —
 * skip it and take the Bayes-key order over every candidate instead of
 * spending a call confirming the dead pick. The planned pick is part of
 * the ranked list in that case, so it is still tried if nothing else
 * looks better. The threshold 0.7 is the measured peak of the
 * Monte Carlo sweep (0.6: 22.60 calls, 0.7: 22.38, 0.75: 22.62 with
 * served requests starting to drop); past 0.75 the router deviates so
 * often it reverts to the worse key-full behaviour.
 */
export function orderAttempts(
  planned: Candidate | null,
  ranked: Candidate[],
  callsMade: ReadonlyMap<string, number>,
  posterior: FailurePosterior,
): Candidate[] {
  const byKey = [...ranked].sort((a, b) => compareFailover(a, b, callsMade, posterior));
  if (planned === null) return byKey;
  const plannedCalls = callsMade.get(planned.probe.provider.id) ?? 0;
  if (posterior.pSuccess(planned.probe.provider.id, plannedCalls + 1) < PLANNED_SKIP_BELOW) return byKey;
  return [
    planned,
    ...byKey.filter((c) => c.probe.provider.id !== planned.probe.provider.id || c.model.id !== planned.model.id),
  ];
}
