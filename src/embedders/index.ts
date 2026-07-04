import type { Embedder } from '../types.js';

export interface OllamaEmbedderOptions {
  /** Default: http://localhost:11434 */
  baseUrl?: string;
}

/** Local embeddings via Ollama (e.g. nomic-embed-text). */
export function ollama(model: string, options: OllamaEmbedderOptions = {}): Embedder {
  const baseUrl = (options.baseUrl ?? 'http://localhost:11434').replace(/\/$/, '');
  return {
    async embed(texts: string[]): Promise<Float32Array[]> {
      const res = await fetch(`${baseUrl}/api/embed`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model, input: texts }),
      });
      if (!res.ok) {
        throw new Error(`rememori: ollama embed failed (${res.status} ${await res.text()})`);
      }
      const data = (await res.json()) as { embeddings: number[][] };
      return data.embeddings.map((e) => new Float32Array(e));
    },
  };
}

export interface OpenAIEmbedderOptions {
  apiKey: string;
  /** Default: https://api.openai.com/v1 — any OpenAI-compatible endpoint works. */
  baseUrl?: string;
}

/** Embeddings via an OpenAI-compatible /embeddings endpoint. */
export function openai(model: string, options: OpenAIEmbedderOptions): Embedder {
  const baseUrl = (options.baseUrl ?? 'https://api.openai.com/v1').replace(/\/$/, '');
  return {
    async embed(texts: string[]): Promise<Float32Array[]> {
      const res = await fetch(`${baseUrl}/embeddings`, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          authorization: `Bearer ${options.apiKey}`,
        },
        body: JSON.stringify({ model, input: texts }),
      });
      if (!res.ok) {
        throw new Error(`rememori: openai embed failed (${res.status} ${await res.text()})`);
      }
      const data = (await res.json()) as { data: { index: number; embedding: number[] }[] };
      return data.data
        .sort((a, b) => a.index - b.index)
        .map((d) => new Float32Array(d.embedding));
    },
  };
}
