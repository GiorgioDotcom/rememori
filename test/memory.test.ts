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

describe('Reinforcement', () => {
  it('resets the decay anchor: a used old memory outranks an unused one', async () => {
    const yearAgo = Date.now() - 365 * 86_400_000;
    const mem = await Memory.open(':memory:', { embedder: fakeEmbedder });
    const used = await mem.remember('coffee useful fact', { createdAt: yearAgo });
    await mem.remember('coffee stale fact', { createdAt: yearAgo });

    // before reinforcement: both decayed equally
    const before = await mem.recall('coffee', { halfLifeDays: 30 });
    expect(before[0]!.score).toBeCloseTo(before[1]!.score, 3);

    await mem.reinforce(used);
    const after = await mem.recall('coffee', { halfLifeDays: 30 });
    expect(after[0]!.id).toBe(used);
    expect(after[0]!.score).toBeGreaterThan(after[1]!.score * 100);
  });

  it('hardening is log-damped and capped', async () => {
    const mem = await Memory.open(':memory:', { embedder: fakeEmbedder });
    const id = await mem.remember('coffee habit');
    for (let i = 0; i < 100; i++) await mem.reinforce(id);
    const [hit] = await mem.recall('coffee');
    // similarity 1 + hardening capped at 0.15 → score ≤ 1.15
    expect(hit!.score).toBeLessThanOrEqual(1.15 + 1e-9);
    expect(hit!.score).toBeGreaterThan(1.14);
  });

  it('returns false for unknown ids', async () => {
    const mem = await Memory.open(':memory:', { embedder: fakeEmbedder });
    expect(await mem.reinforce('nope')).toBe(false);
  });

  it('persists across reopen (file storage)', async () => {
    const path = await tmpFile();
    const yearAgo = Date.now() - 365 * 86_400_000;
    const mem = await Memory.open(path, { embedder: fakeEmbedder });
    const id = await mem.remember('coffee persisted', { createdAt: yearAgo });
    await mem.reinforce(id);
    await mem.close();

    const reopened = await Memory.open(path, { embedder: fakeEmbedder });
    const [hit] = await reopened.recall('coffee', { halfLifeDays: 30 });
    // anchor survived: no year-long decay applied
    expect(hit!.score).toBeGreaterThan(0.9);
  });
});

describe('Demotion and collision plumbing (v0.7)', () => {
  it('demote lowers ranking below an equal twin, symmetric cap', async () => {
    const mem = await Memory.open(':memory:', { embedder: fakeEmbedder });
    const bad = await mem.remember('coffee fact wrong');
    await mem.remember('coffee fact right');

    await mem.demote(bad);
    const hits = await mem.recall('coffee');
    expect(hits[1]!.id).toBe(bad);

    for (let i = 0; i < 100; i++) await mem.demote(bad);
    const [, worst] = await mem.recall('coffee');
    // similarity ~1, penalty capped at −0.15 → score never below ~0.85
    expect(worst!.score).toBeGreaterThan(0.84);
    expect(await mem.demote('nope')).toBe(false);
  });

  it('negative reinforcements survive reopen', async () => {
    const path = await tmpFile();
    const mem = await Memory.open(path, { embedder: fakeEmbedder });
    const id = await mem.remember('coffee demoted');
    await mem.demote(id);
    await mem.demote(id);
    await mem.close();
    const reopened = await Memory.open(path, { embedder: fakeEmbedder });
    expect(reopened.get(id)!.reinforcements).toBe(-2);
  });

  it('collisions finds near-duplicates, not unrelated memories', async () => {
    const mem = await Memory.open(':memory:', { embedder: fakeEmbedder });
    const a = await mem.remember('coffee machine on floor two');
    await mem.remember('coffee machine moved to floor three'); // same axis → sim 1
    await mem.remember('music playlist for gym');              // different axis → sim 0

    const clashes = mem.collisions(a);
    expect(clashes).toHaveLength(1);
    expect(clashes[0]!.text).toContain('floor three');
    expect(clashes[0]!.similarity).toBeCloseTo(1, 5);
    expect(mem.collisions('nope')).toEqual([]);
  });

  it('a demoted low-similarity memory may vanish entirely — locked-in behavior', async () => {
    const mem = await Memory.open(':memory:', { embedder: fakeEmbedder });
    // entity-rescued memory: query shares "Giorgio", similarity 0
    const id = await mem.remember('Giorgio fixed the car');
    expect(await mem.recall('updates from Giorgio?')).toHaveLength(1);

    // demote past the entity bonus (0.1) → score goes ≤ 0 → gone
    for (let i = 0; i < 10; i++) await mem.demote(id);
    expect(await mem.recall('updates from Giorgio?')).toHaveLength(0);
  });

  it('demote then reinforce round-trips the counter', async () => {
    const mem = await Memory.open(':memory:', { embedder: fakeEmbedder });
    const id = await mem.remember('coffee round trip');
    await mem.demote(id);
    await mem.demote(id);
    await mem.reinforce(id);
    expect(mem.get(id)!.reinforcements).toBe(-1);
    await mem.reinforce(id);
    expect(mem.get(id)!.reinforcements).toBe(0);
  });

  it('collisions works with the HNSW index active', async () => {
    const mem = await Memory.open(':memory:', { embedder: fakeEmbedder, index: 'hnsw' });
    const a = await mem.remember('coffee machine on floor two');
    await mem.remember('coffee machine on floor three');
    await mem.remember('music playlist');
    await mem.recall('coffee'); // force index build
    const clashes = mem.collisions(a);
    expect(clashes).toHaveLength(1);
  });

  it('get() returns defensive copies', async () => {
    const mem = await Memory.open(':memory:', { embedder: fakeEmbedder });
    const id = await mem.remember('coffee copy', { tags: ['keep'], meta: { n: 1 } });
    const snapshot = mem.get(id)!;
    snapshot.tags.push('EVIL');
    (snapshot.meta as Record<string, unknown>).n = 999;
    expect(mem.get(id)!.tags).toEqual(['keep']);
    expect(mem.get(id)!.meta).toEqual({ n: 1 });
  });

  it('reinforceFromOutput dedupes repeated ids', async () => {
    const mem = await Memory.open(':memory:', { embedder: fakeEmbedder });
    const id = await mem.remember('the coffee deploy pipeline breaks when redis cache is cold');
    const hit = { id, text: 'the coffee deploy pipeline breaks when redis cache is cold' };
    const out = 'the deploy pipeline breaks when redis cache is cold';
    const reinforced = await mem.reinforceFromOutput([hit, hit, hit], out);
    expect(reinforced).toEqual([id]);
    expect(mem.get(id)!.reinforcements).toBe(1);
  });

  it('reinforceFromOutput reinforces only verifiably used hits', async () => {
    const mem = await Memory.open(':memory:', { embedder: fakeEmbedder });
    const used = await mem.remember('the coffee deploy pipeline breaks when redis cache is cold');
    const unused = await mem.remember('coffee standup meeting notes for monday');

    const hits = (await mem.recall('coffee')).map((h) => ({ id: h.id, text: h.text }));
    const output =
      'Based on what I know, the deploy pipeline breaks when redis cache is cold, so warm it first.';
    const reinforced = await mem.reinforceFromOutput(hits, output);

    expect(reinforced).toEqual([used]);
    expect(mem.get(used)!.reinforcements).toBe(1);
    expect(mem.get(unused)!.reinforcements).toBe(0);
  });
});

describe('Relevance floor (minSimilarity)', () => {
  it('drops off-topic hits below the floor', async () => {
    const mem = await Memory.open(':memory:', { embedder: fakeEmbedder, minSimilarity: 0.3 });
    await mem.remember('coffee is life');
    await mem.remember('music playlist for gym');

    // "coffee brand cars" hits the coffee axis partially — sim < 1 but > 0.3
    const onTopic = await mem.recall('coffee please');
    expect(onTopic).toHaveLength(1);

    // unknown-word query lands on the misc axis: sim 0 vs both → nothing
    const offTopic = await mem.recall('unrelated gibberish query');
    expect(offTopic).toHaveLength(0);
  });

  it('entity matches bypass the floor (graph rescue)', async () => {
    const mem = await Memory.open(':memory:', { embedder: fakeEmbedder, minSimilarity: 0.9 });
    await mem.remember('Giorgio fixed the car');
    // similarity between query (misc axis) and record (car axis) is 0 < 0.9,
    // but the shared entity keeps it in
    const hits = await mem.recall('news from Giorgio?');
    expect(hits).toHaveLength(1);
    expect(hits[0]!.sharedEntities).toEqual(['Giorgio']);
  });

  it('per-call option overrides the instance default', async () => {
    // instance floor above the max possible similarity → default excludes everything
    const mem = await Memory.open(':memory:', { embedder: fakeEmbedder, minSimilarity: 1.5 });
    await mem.remember('coffee is life');
    expect(await mem.recall('coffee please')).toHaveLength(0);
    expect(await mem.recall('coffee please', { minSimilarity: 0 })).toHaveLength(1);
  });
});

describe('Entity graph', () => {
  it('extracts entities heuristically and builds the graph', async () => {
    const mem = await Memory.open(':memory:', { embedder: fakeEmbedder });
    await mem.remember('Giorgio fixed the car with Mario Rossi');
    await mem.remember('Giorgio writes code in NestJS');

    const top = mem.entities();
    expect(top[0]).toEqual({ name: 'Giorgio', count: 2 });

    const card = mem.entity('giorgio'); // case-insensitive lookup
    expect(card).not.toBeNull();
    expect(card!.name).toBe('Giorgio');
    expect(card!.count).toBe(2);
    expect(card!.coEntities.map((e) => e.name)).toEqual(
      expect.arrayContaining(['Mario Rossi', 'NestJS']),
    );
  });

  it('graph bonus surfaces memories that pure similarity misses', async () => {
    const mem = await Memory.open(':memory:', { embedder: fakeEmbedder });
    // "car" axis record — orthogonal to the query's "misc" axis
    await mem.remember('Giorgio fixed the car');
    await mem.remember('music playlist for gym');

    // query has zero cosine with both, but shares entity "Giorgio" with the first
    const hits = await mem.recall('any updates from Giorgio?');
    expect(hits).toHaveLength(1);
    expect(hits[0]!.text).toBe('Giorgio fixed the car');
    expect(hits[0]!.sharedEntities).toEqual(['Giorgio']);
    expect(hits[0]!.similarity).toBe(0);
    expect(hits[0]!.score).toBeCloseTo(0.1, 5);

    // graph disabled → nothing surfaces
    const noGraph = await mem.recall('any updates from Giorgio?', { graph: false });
    expect(noGraph).toHaveLength(0);
  });

  it('respects explicit entities and extractor: false', async () => {
    const mem = await Memory.open(':memory:', {
      embedder: fakeEmbedder,
      extractor: false,
    });
    await mem.remember('Giorgio fixed the car');
    expect(mem.entities()).toHaveLength(0);

    const mem2 = await Memory.open(':memory:', { embedder: fakeEmbedder });
    await mem2.remember('some coffee note', { entities: ['Lavazza'] });
    expect(mem2.entity('Lavazza')!.count).toBe(1);
  });

  it('persists entities and unindexes on forget', async () => {
    const path = await tmpFile();
    const mem = await Memory.open(path, { embedder: fakeEmbedder });
    const id = await mem.remember('Giorgio fixed the car');
    await mem.remember('Giorgio drinks coffee');
    await mem.close();

    const reopened = await Memory.open(path, { embedder: fakeEmbedder });
    expect(reopened.entity('Giorgio')!.count).toBe(2);

    await reopened.forget(id);
    expect(reopened.entity('Giorgio')!.count).toBe(1);
  });
});
