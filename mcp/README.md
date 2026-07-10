# rememori-mcp

Persistent semantic memory for Claude Code — and any MCP-capable agent — backed by [rememori](https://github.com/GiorgioDotcom/rememori).

Your agent remembers across sessions: facts, decisions, preferences. Stored in one local file, searched semantically, never leaves your machine (with a local embedder).

## Setup

With [Ollama](https://ollama.com) running locally (default):

```bash
ollama pull nomic-embed-text
claude mcp add rememori -- npx -y rememori-mcp
```

That's it. Claude Code now has the full memory toolset — see [Tools](#tools) below.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `REMEMORI_PATH` | `~/.rememori/memory.mem` | Where memories live |
| `REMEMORI_EMBEDDER` | `ollama` | `ollama` or `openai` |
| `OLLAMA_MODEL` | `nomic-embed-text` | Local embedding model |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `OPENAI_API_KEY` | — | Required for `openai` |
| `OPENAI_MODEL` | `text-embedding-3-small` | |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible endpoint |
| `REMEMORI_MIN_SIMILARITY` | `0.5` | Relevance floor: off-topic recalls return nothing instead of noise (0.5 measured for nomic; lower it if recalls come back empty too often) |

Example with an OpenAI-compatible endpoint:

```bash
claude mcp add rememori -e REMEMORI_EMBEDDER=openai -e OPENAI_API_KEY=sk-... -- npx -y rememori-mcp
```

## Tools

- **remember**(text, tags?, importance?) — store a memory
- **recall**(query, limit?, halfLifeDays?, tags?) — semantic search with entity-graph bonus and optional time decay
- **reinforce**(id) — mark a memory as actually used: resets its decay clock, hardens its ranking (log-damped)
- **reinforce_from_output**(ids, output) — evidence-gated version: only memories verifiably quoted/used in the answer get reinforced (preferred)
- **demote**(id) — negative feedback for used-and-wrong memories or contradiction losers
- **collisions**(id, threshold?, limit?) — near-duplicates of a memory; the agent judges duplicate vs update vs contradiction, then demotes/forgets the loser
- **get**(id) — read one memory with its reinforcement state
- **forget**(id) — delete
- **entities**(name?, limit?) — explore the memory↔entity graph

## License

MIT
