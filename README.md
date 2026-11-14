# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/naufalumer/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/naufalumer/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that speaks the OpenAI Chat Completions API and the Anthropic Messages API — in JSON and in server-sent events — over ten simulated free-tier model providers. Every request is routed to the best free model that meets its requirements (tools, vision, JSON mode, context length) by a sealed quality ranking, and `node main.ts --list` prints the quality-ranked free catalog it found.

Version 2 routes by caps: each provider gets a fixed per-trial call cap
chosen to minimise the expected refusals (Monte Carlo over the
generator's published draw — wide-budget providers take the most calls,
narrow ones take one each), and the twenty requests are assigned to those
caps by min-cost flow over the sealed quality table. The trial's provider
schedule is drawn from a seed the entry never receives, so no routing can
promise zero refusals; the runtime failover behind the plan absorbs
whatever a provider actually refuses. Measured on the same Monte Carlo:
23.0 upstream calls per trial at mean quality 98.5 (score 99 in 100% of
500 three-trial finals), against the v1 entry's 29.9 calls at quality 99.9.

## Run it

```sh
node main.ts
```

`node main.ts --list` prints the catalog document (providers, their wires and auth schemes, and every free and paid model with its context length) as one JSON document and exits.

---

<sub>Entered on [make.fun](https://make.fun/?ref=bye4mn13&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
