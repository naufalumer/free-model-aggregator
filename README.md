# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/naufalumer/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/naufalumer/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that speaks the OpenAI Chat Completions API and the Anthropic Messages API — in JSON and in server-sent events — over ten simulated free-tier model providers. Every request is routed to the best free model that meets its requirements (tools, vision, JSON mode, context length) by a sealed quality ranking, with budget-aware planning across providers and automatic failover when a provider refuses, and `node main.ts --list` prints the quality-ranked free catalog it found.

## Run it

```sh
node main.ts
```

`node main.ts --list` prints the catalog document (providers, their wires and auth schemes, and every free and paid model with its context length) as one JSON document and exits.

---

<sub>Entered on [make.fun](https://make.fun/?ref=bye4mn13&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
