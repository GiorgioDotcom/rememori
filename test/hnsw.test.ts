import { describe, expect, it } from 'vitest';
import { HnswIndex, Memory, dot, normalize } from '../src/index.js';
import type { Embedder } from '../src/index.js';

/** deterministic PRNG so failures are reproducible */
function mulberry32(seed: number) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomUnitVectors(n: number, dim: number, seed = 42): Float32Array[] {
  const rand = mulberry32(seed);
  return Array.from({ length: n }, () => {
    const v = new Float32Array(dim);
    for (let i = 0; i < dim; i++) v[i] = rand() * 2 - 1;
    return normalize(v);
  });
}

function bruteTopK(vectors: Float32Array[], q: Float32Array, k: number): number[] {
  return vectors
    .map((v, i) => ({ i, s: dot(q, v) }))
    .sort((a, b) => b.s - a.s)
    .slice(0, k)
    .map((x) => x.i);
}

describe('HnswIndex', () => {
  it('finds near-exact top-10 on 2000 random vectors (recall ≥ 0.9)', () => {
    const vectors = randomUnitVectors(2000, 32);
    const index = new HnswIndex();
    vectors.forEach((v, i) => index.add(String(i), v));

    const queries = randomUnitVectors(30, 32, 7);
    let hits = 0, total = 0;
    for (const q of queries) {
      const exact = new Set(bruteTopK(vectors, q, 10).map(String));
      const approx = index.search(q, 10);
      for (const a of approx) if (exact.has(a.id)) hits++;
      total += 10;
    }
    expect(hits / total).toBeGreaterThanOrEqual(0.9);
  });

  it('never returns removed ids and re-elects the entry point', () => {
    const vectors = randomUnitVectors(300, 16);
    const index = new HnswIndex();
    vectors.forEach((v, i) => index.add(String(i), v));

    for (let i = 0; i < 150; i++) index.remove(String(i));
    expect(index.size).toBe(150);
    expect(index.deletedRatio).toBeCloseTo(0.5);

    for (const q of randomUnitVectors(10, 16, 9)) {
      for (const hit of index.search(q, 20)) {
        expect(Number(hit.id)).toBeGreaterThanOrEqual(150);
      }
    }
  });

  it('returns results sorted by similarity', () => {
    const vectors = randomUnitVectors(500, 16);
    const index = new HnswIndex();
    vectors.forEach((v, i) => index.add(String(i), v));
    const [q] = randomUnitVectors(1, 16, 3);
    const hits = index.search(q!, 10);
    for (let i = 1; i < hits.length; i++) {
      expect(hits[i]!.similarity).toBeLessThanOrEqual(hits[i - 1]!.similarity);
    }
  });
});

/** hash-based embedder: same text → same vector, spread across dims */
const hashEmbedder: Embedder = {
  async embed(texts) {
    return texts.map((text) => {
      const v = new Float32Array(24);
      for (let i = 0; i < text.length; i++) {
        v[(text.charCodeAt(i) * 31 + i) % 24]! += 1;
      }
      return normalize(v);
    });
  },
};

describe('Memory with HNSW index', () => {
  it('index:"hnsw" matches flat results on unfiltered recall', async () => {
    const flat = await Memory.open(':memory:', { embedder: hashEmbedder, index: 'flat' });
    const indexed = await Memory.open(':memory:', { embedder: hashEmbedder, index: 'hnsw' });

    const texts = Array.from({ length: 200 }, (_, i) => `memory number ${i} about topic ${i % 17}`);
    for (const t of texts) { await flat.remember(t); await indexed.remember(t); }

    const a = await flat.recall('topic 5', { limit: 5 });
    const b = await indexed.recall('topic 5', { limit: 5 });
    // approximate index: demand exact top-1 and ≥4/5 overlap (near-ties may swap)
    expect(b[0]!.text).toBe(a[0]!.text);
    const exact = new Set(a.map((h) => h.text));
    const overlap = b.filter((h) => exact.has(h.text)).length;
    expect(overlap).toBeGreaterThanOrEqual(4);
  });

  it('filtered recall stays exact and correct with index on', async () => {
    const mem = await Memory.open(':memory:', { embedder: hashEmbedder, index: 'hnsw' });
    await mem.remember('tagged memory about topic 3', { tags: ['keep'] });
    await mem.remember('untagged memory about topic 3');

    const hits = await mem.recall('topic 3', { tags: ['keep'] });
    expect(hits).toHaveLength(1);
    expect(hits[0]!.tags).toEqual(['keep']);
  });

  it('forget removes from the index too', async () => {
    const mem = await Memory.open(':memory:', { embedder: hashEmbedder, index: 'hnsw' });
    const id = await mem.remember('ephemeral topic 9');
    await mem.remember('lasting topic 9');
    // warm the index, then delete
    await mem.recall('topic 9');
    await mem.forget(id);
    const hits = await mem.recall('topic 9', { limit: 10 });
    expect(hits.every((h) => h.id !== id)).toBe(true);
  });
});
