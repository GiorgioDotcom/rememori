import { afterEach, describe, expect, it } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Memory } from '../src/index.js';
import type { Embedder } from '../src/index.js';

/**
 * Deterministic fake embedder: maps known keywords onto orthogonal axes,
 * so semantic similarity is fully predictable in tests.
 */
const AXES = ['coffee', 'code', 'car', 'music'];

const fakeEmbedder: Embedder = {
  async embed(texts) {
    return texts.map((text) => {
      const v = new Float32Array(AXES.length + 1);
      const lower = text.toLowerCase();
      let hit = false;
      AXES.forEach((axis, i) => {
        if (lower.includes(axis)) {
          v[i] = 1;
          hit = true;
        }
      });
      if (!hit) v[AXES.length] = 1; // unknown texts share a "misc" axis
      return v;
    });
  },
};

const cleanups: string[] = [];
afterEach(async () => {
  while (cleanups.length) await rm(cleanups.pop()!, { recursive: true, force: true });
});

async function tmpFile(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'rememori-'));
  cleanups.push(dir);
  return join(dir, 'test.rememori');
}

describe('Memory', () => {
  it('remembers and recalls by semantic similarity', async () => {
    const mem = await Memory.open(':memory:', { embedder: fakeEmbedder });
    await mem.remember('I love coffee in the morning');
    await mem.remember('TypeScript code should be strict');
    await mem.remember('The car needs new spark plugs');

    const hits = await mem.recall('where do I get coffee?');
    expect(hits[0]!.text).toContain('coffee');
    // unrelated memories (zero similarity) are never returned
    expect(hits).toHaveLength(1);
  });

  it('forgets', async () => {
    const mem = await Memory.open(':memory:', { embedder: fakeEmbedder });
    const id = await mem.remember('coffee is life');
    expect(mem.size).toBe(1);
    expect(await mem.forget(id)).toBe(true);
    expect(await mem.forget(id)).toBe(false);
    expect(mem.size).toBe(0);
    expect(await mem.recall('coffee')).toHaveLength(0);
  });

  it('filters by tags and date', async () => {
    const mem = await Memory.open(':memory:', { embedder: fakeEmbedder });
    await mem.remember('coffee old', { tags: ['drink'], createdAt: Date.parse('2020-01-01') });
    await mem.remember('coffee new', { tags: ['drink', 'fresh'] });
    await mem.remember('code stuff', { tags: ['work'] });

    const tagged = await mem.recall('coffee', { tags: ['fresh'] });
    expect(tagged).toHaveLength(1);
    expect(tagged[0]!.text).toBe('coffee new');

    const recent = await mem.recall('coffee', { after: '2024-01-01' });
    expect(recent.map((h) => h.text)).toEqual(['coffee new']);
  });

  it('weighs importance and temporal decay', async () => {
    const mem = await Memory.open(':memory:', { embedder: fakeEmbedder });
    await mem.remember('coffee weak signal', { importance: 0.2 });
    await mem.remember('coffee strong signal', { importance: 1 });

    const byImportance = await mem.recall('coffee');
    expect(byImportance[0]!.text).toBe('coffee strong signal');

    const yearAgo = Date.now() - 365 * 86_400_000;
    const mem2 = await Memory.open(':memory:', { embedder: fakeEmbedder });
    await mem2.remember('coffee ancient', { createdAt: yearAgo });
    await mem2.remember('coffee today');
    const decayed = await mem2.recall('coffee', { halfLifeDays: 30 });
    expect(decayed[0]!.text).toBe('coffee today');
    expect(decayed[0]!.score).toBeGreaterThan(decayed[1]!.score * 100);
  });

  it('persists across reopen (file storage)', async () => {
    const path = await tmpFile();
    const mem1 = await Memory.open(path, { embedder: fakeEmbedder });
    const keepId = await mem1.remember('coffee persisted', { tags: ['x'], meta: { n: 1 } });
    const dropId = await mem1.remember('car persisted');
    await mem1.forget(dropId);
    await mem1.close();

    const mem2 = await Memory.open(path, { embedder: fakeEmbedder });
    expect(mem2.size).toBe(1);
    const hits = await mem2.recall('coffee');
    expect(hits[0]!.id).toBe(keepId);
    expect(hits[0]!.meta).toEqual({ n: 1 });
    expect(hits[0]!.tags).toEqual(['x']);
  });

  it('prunes by age and importance', async () => {
    const path = await tmpFile();
    const mem = await Memory.open(path, { embedder: fakeEmbedder });
    await mem.remember('coffee ancient', { createdAt: Date.parse('2020-01-01') });
    await mem.remember('music trivial', { importance: 0.1 });
    await mem.remember('code current');

    const removed = await mem.prune({ olderThan: '365d', belowImportance: 0.3 });
    expect(removed).toBe(2);
    expect(mem.size).toBe(1);

    // prune compacted the file — reopen sees the same single record
    const reopened = await Memory.open(path, { embedder: fakeEmbedder });
    expect(reopened.size).toBe(1);
  });
});
