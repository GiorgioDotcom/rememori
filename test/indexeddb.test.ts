import 'fake-indexeddb/auto';
import { describe, expect, it } from 'vitest';
import { IDBFactory } from 'fake-indexeddb';
import { Memory } from '../src/index.js';
import type { Embedder } from '../src/index.js';

const fakeEmbedder: Embedder = {
  async embed(texts) {
    return texts.map((text) => {
      const v = new Float32Array(2);
      v[0] = text.toLowerCase().includes('coffee') ? 1 : 0;
      v[1] = v[0] === 1 ? 0 : 1;
      return v;
    });
  },
};

function freshIdb(): void {
  // isolate each test: brand-new in-memory IndexedDB
  (globalThis as { indexedDB: IDBFactory }).indexedDB = new IDBFactory();
}

describe('IndexedDBStorage (idb:// path)', () => {
  it('persists across reopen', async () => {
    freshIdb();
    const mem = await Memory.open('idb://agent', { embedder: fakeEmbedder });
    const id = await mem.remember('coffee in the browser', { tags: ['web'], meta: { n: 1 } });
    await mem.close();

    const reopened = await Memory.open('idb://agent', { embedder: fakeEmbedder });
    expect(reopened.size).toBe(1);
    const hits = await reopened.recall('coffee');
    expect(hits[0]!.id).toBe(id);
    expect(hits[0]!.meta).toEqual({ n: 1 });
    expect(hits[0]!.tags).toEqual(['web']);
    // vector round-trips as a real Float32Array
    expect(hits[0]!.similarity).toBeCloseTo(1, 5);
  });

  it('forget and prune survive reopen', async () => {
    freshIdb();
    const mem = await Memory.open('idb://agent', { embedder: fakeEmbedder });
    const a = await mem.remember('coffee one');
    await mem.remember('coffee two', { importance: 0.1 });
    await mem.forget(a);
    await mem.prune({ belowImportance: 0.5 });
    await mem.close();

    const reopened = await Memory.open('idb://agent', { embedder: fakeEmbedder });
    expect(reopened.size).toBe(0);
  });

  it('separate idb:// names are separate stores', async () => {
    freshIdb();
    const memA = await Memory.open('idb://alpha', { embedder: fakeEmbedder });
    await memA.remember('coffee alpha');
    const memB = await Memory.open('idb://beta', { embedder: fakeEmbedder });
    expect(memB.size).toBe(0);
  });

  it('entities persist through IndexedDB', async () => {
    freshIdb();
    const mem = await Memory.open('idb://graph', { embedder: fakeEmbedder });
    await mem.remember('Giorgio drinks coffee');
    await mem.close();

    const reopened = await Memory.open('idb://graph', { embedder: fakeEmbedder });
    expect(reopened.entity('Giorgio')!.count).toBe(1);
  });
});
