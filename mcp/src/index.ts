#!/usr/bin/env node
/**
 * rememori-mcp — persistent semantic memory over the Model Context Protocol.
 *
 * Environment:
 *   REMEMORI_PATH      memory file (default: ~/.rememori/memory.mem)
 *   REMEMORI_EMBEDDER  "ollama" (default) or "openai"
 *   OLLAMA_MODEL       default: nomic-embed-text
 *   OLLAMA_URL         default: http://localhost:11434
 *   OPENAI_API_KEY     required when REMEMORI_EMBEDDER=openai
 *   OPENAI_MODEL       default: text-embedding-3-small
 *   OPENAI_BASE_URL    any OpenAI-compatible endpoint
 */
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import { Memory } from 'rememori';
import { ollama, openai } from 'rememori/embedders';
import { mkdir } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';

function makeEmbedder() {
  if (process.env.REMEMORI_EMBEDDER === 'openai') {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) throw new Error('rememori-mcp: OPENAI_API_KEY is required with REMEMORI_EMBEDDER=openai');
    const options: { apiKey: string; baseUrl?: string } = { apiKey };
    if (process.env.OPENAI_BASE_URL) options.baseUrl = process.env.OPENAI_BASE_URL;
    return openai(process.env.OPENAI_MODEL ?? 'text-embedding-3-small', options);
  }
  const options: { baseUrl?: string } = {};
  if (process.env.OLLAMA_URL) options.baseUrl = process.env.OLLAMA_URL;
  return ollama(process.env.OLLAMA_MODEL ?? 'nomic-embed-text', options);
}

const path = process.env.REMEMORI_PATH ?? join(homedir(), '.rememori', 'memory.mem');
await mkdir(dirname(path), { recursive: true });
/* relevance floor: off-topic recalls return nothing instead of noise.
   0.5 measured for nomic-embed-text; override via REMEMORI_MIN_SIMILARITY. */
const minSimilarity = Number(process.env.REMEMORI_MIN_SIMILARITY ?? 0.5);
const mem = await Memory.open(path, { embedder: makeEmbedder(), minSimilarity });

const server = new McpServer({ name: 'rememori', version: '0.1.0' });

const text = (data: unknown) => ({
  content: [{ type: 'text' as const, text: JSON.stringify(data, null, 2) }],
});

server.tool(
  'remember',
  'Store a memory. Use for facts, decisions, preferences or events worth recalling in future sessions.',
  {
    text: z.string().min(1).describe('The memory to store, as a self-contained sentence.'),
    tags: z.array(z.string()).optional().describe('Optional tags for filtering.'),
    importance: z.number().min(0).max(1).optional().describe('0..1, weighs into recall ranking. Default 1.'),
  },
  async ({ text: memoryText, tags, importance }) => {
    const options: { tags?: string[]; importance?: number } = {};
    if (tags) options.tags = tags;
    if (importance !== undefined) options.importance = importance;
    const id = await mem.remember(memoryText, options);
    return text({ id, stored: memoryText, total: mem.size });
  },
);

server.tool(
  'recall',
  'Semantic search over stored memories. Ranks by similarity, shared entities, importance and optional time decay.',
  {
    query: z.string().min(1).describe('What to look for, in natural language.'),
    limit: z.number().int().min(1).max(50).optional().describe('Max results. Default 10.'),
    halfLifeDays: z.number().positive().optional().describe('Temporal decay half-life in days. Unset = no decay.'),
    tags: z.array(z.string()).optional().describe('Only memories carrying ALL of these tags.'),
  },
  async ({ query, limit, halfLifeDays, tags }) => {
    const options: { limit?: number; halfLifeDays?: number; tags?: string[] } = {};
    if (limit !== undefined) options.limit = limit;
    if (halfLifeDays !== undefined) options.halfLifeDays = halfLifeDays;
    if (tags) options.tags = tags;
    const hits = await mem.recall(query, options);
    return text(hits.map((h) => ({
      id: h.id,
      text: h.text,
      score: Number(h.score.toFixed(4)),
      sharedEntities: h.sharedEntities,
      tags: h.tags,
      createdAt: new Date(h.createdAt).toISOString(),
    })));
  },
);

server.tool(
  'forget',
  'Delete a memory by id (get ids from recall).',
  { id: z.string().describe('Memory id to delete.') },
  async ({ id }) => text({ forgotten: await mem.forget(id), total: mem.size }),
);

server.tool(
  'entities',
  'List the most-referenced entities in the memory graph, or inspect one entity with its linked memories.',
  {
    name: z.string().optional().describe('Entity name to inspect. Omit to list top entities.'),
    limit: z.number().int().min(1).max(100).optional().describe('Max items. Default 20.'),
  },
  async ({ name, limit }) => {
    if (name) {
      const card = mem.entity(name, limit !== undefined ? { limit } : {});
      return text(card ?? { error: `unknown entity: ${name}` });
    }
    return text(mem.entities(limit ?? 20));
  },
);

await server.connect(new StdioServerTransport());
