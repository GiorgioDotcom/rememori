# rememori

[![npm](https://img.shields.io/npm/v/rememori)](https://www.npmjs.com/package/rememori)
[![license](https://img.shields.io/npm/l/rememori)](./LICENSE)
![zero dependencies](https://img.shields.io/badge/dependencies-0-brightgreen)

> Agent memory that runs anywhere JavaScript runs.

Pure TypeScript. **Zero runtime dependencies.** No native bindings, no compiler toolchain, no database server. One file on disk — or IndexedDB in the browser. Works in Node, Bun and browsers today.

**Status: v0.4 — early but moving fast.** Vector recall, entity graph, temporal decay, HNSW index, browser/IndexedDB support and an [MCP server](./mcp) are in. The API below is tested; minor breaking changes possible until v1.0.

## Why

Every AI agent forgets everything when the session ends. Fixing that today means wiring up a vector database, an embedding pipeline, and a retrieval layer — or adopting a memory *platform* with servers, backends, and cloud tiers.

rememori is not a platform. It's a primitive — the SQLite of agent memory:

```bash
npm install rememori
```

```ts
import { Memory } from 'rememori';
import { ollama } from 'rememori/embedders';

const mem = await Memory.open('./agent.rememori', {
  embedder: ollama('nomic-embed-text'),
});

// remember
await mem.remember('User prefers dark mode, hates notifications', {
  tags: ['prefs'],
  importance: 0.8,
});

// recall — cosine similarity × importance × optional temporal decay
const hits = await mem.recall('what UI settings does the user like?', {
  limit: 5,
  halfLifeDays: 90,
});

// forget
await mem.forget(hits[0].id);
```

Three verbs. That's the API.

### Entity graph

Every memory is linked to the named entities it mentions (bipartite memory↔entity graph). Extraction is pluggable: the zero-dependency default spots capitalized names; plug an LLM-backed extractor for quality, or pass `extractor: false` to opt out.

```ts
// recall is hybrid by default: memories sharing entities with the query
// get a score bonus, so graph-linked memories surface even when
// embedding similarity alone would miss them
const hits = await mem.recall('any updates from Giorgio?');
hits[0].sharedEntities; // ['Giorgio']

// explore the graph
mem.entities();        // top entities by linked memories
mem.entity('Giorgio'); // linked memories + co-occurring entities
```

### Fully local chat agent (Ollama)

A complete chat agent with persistent memory across restarts — everything on your machine, no cloud: [`examples/ollama-chat.mjs`](./examples/ollama-chat.mjs) (~50 lines, runnable).

### In the browser — no server at all

Storage works on IndexedDB out of the box (`idb://` paths), and [transformers.js](https://github.com/huggingface/transformers.js) gives you local embeddings. Fully private semantic memory, nothing leaves the machine:

```ts
import { pipeline } from '@huggingface/transformers';
import { Memory } from 'rememori';

const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

const mem = await Memory.open('idb://agent', {
  embedder: {
    async embed(texts) {
      const out = await extractor(texts, { pooling: 'mean', normalize: true });
      return out.tolist().map((v) => new Float32Array(v));
    },
  },
});
```

Runnable single-file demo: [`examples/browser-demo.html`](./examples/browser-demo.html).

## Design

- **Embedder is an interface, never bundled.** Bring Ollama (local, private), any OpenAI-compatible endpoint, or your own `(texts) => Float32Array[]` function.
- **Storage is an adapter.** Append-only JSONL file with tombstones and compaction on Node/Bun, IndexedDB in the browser (`idb://name`), `:memory:` for tests. KV (edge) adapter is on the roadmap.
- **Recall scoring:** `(cosine + entityBonus) × importance × 0.5^(age/halfLife)` where `entityBonus = min(0.3, 0.1 × shared entities)`. Recency, importance and the graph are first-class, not an afterthought.
- **Exact search first, HNSW when it pays.** Below ~1000 memories, an exact scan over contiguous `Float32Array`s wins on every axis. Past that, a pure-TS HNSW graph index kicks in automatically (`index: 'auto'`, the default; force with `'hnsw'` or `'flat'`). Tag/date-filtered recalls always use the exact scan.

## Scale

Synthetic 384-dim embeddings with realistic low-dimensional structure, Apple M-class CPU, `ef=200` (what `recall()` uses). Run it yourself: `node bench/hnsw.bench.mjs`.

| memories | exact scan | HNSW | recall@10 |
|---|---|---|---|
| 1,000 | 0.5 ms/query | (index off — scan wins) | 100% |
| 10,000 | 4.4 ms/query | 2.3 ms/query | 100% |
| 50,000 | 26 ms/query | 2.8 ms/query | 100% |

Index build is one-time, ~2–4 ms per memory, incremental on `remember()`. Honest caveat: uniformly random high-dimensional vectors (the worst case for *any* ANN index — distances concentrate) degrade recall; real embedding models produce structured vectors that behave like the table above.

### MCP server — memory for Claude Code

[`rememori-mcp`](./mcp) wraps the engine as a Model Context Protocol server. One command gives Claude Code (or any MCP client) persistent semantic memory across sessions:

```bash
claude mcp add rememori -- npx -y rememori-mcp
```

## Roadmap

- ~~v0.2 — entity graph (bipartite memory↔entity) + hybrid recall~~ ✅ shipped
- ~~v0.3 — browser support: IndexedDB storage + transformers.js recipe~~ ✅ shipped
- ~~MCP server wrapper~~ ✅ shipped as [`rememori-mcp`](./mcp)
- ~~v0.4 — pure-TS HNSW index~~ ✅ shipped
- v0.5 — consolidation/forgetting policies, LongMemEval harness

## Non-goals

Multi-user servers, auth, cloud sync, multimodal ingestion, LLM-managed memory. If you need a full memory platform with document pipelines and graph databases, use [Cognee](https://github.com/topoteretes/cognee) or [Mem0](https://github.com/mem0ai/mem0) — they're good at that. rememori is for when you want memory *inside* your process, in five minutes.

## License

MIT
