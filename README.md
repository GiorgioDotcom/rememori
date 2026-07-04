# rememori

> Agent memory that runs anywhere JavaScript runs.

Pure TypeScript. **Zero runtime dependencies.** No native bindings, no compiler toolchain, no database server. One file on disk. Works in Node, Bun — and soon browsers and edge runtimes.

**Status: early development (v0.0.x).** The core API below works and is tested, but expect breaking changes until v0.1.0.

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

## Design

- **Embedder is an interface, never bundled.** Bring Ollama (local, private), any OpenAI-compatible endpoint, or your own `(texts) => Float32Array[]` function.
- **Storage is an adapter.** Append-only JSONL file with tombstones and compaction on Node/Bun, `:memory:` for tests. IndexedDB (browser) and KV (edge) adapters are on the roadmap.
- **Recall scoring:** `cosine × importance × 0.5^(age/halfLife)`. Recency and importance are first-class, not an afterthought.
- **Brute-force search over contiguous `Float32Array`s.** Agent memory is thousands of records, not billions — exact search stays fast far beyond that (HNSW planned for when it isn't).

## Roadmap

- v0.2 — entity graph (bipartite memory↔entity) + hybrid recall
- v0.3 — browser support: IndexedDB storage + transformers.js recipe (fully local semantic memory in the browser)
- v0.4 — pure-TS HNSW index
- v0.5 — consolidation/forgetting policies, MCP server wrapper, LongMemEval harness

## Non-goals

Multi-user servers, auth, cloud sync, multimodal ingestion, LLM-managed memory. If you need a full memory platform with document pipelines and graph databases, use [Cognee](https://github.com/topoteretes/cognee) or [Mem0](https://github.com/mem0ai/mem0) — they're good at that. rememori is for when you want memory *inside* your process, in five minutes.

## License

MIT
