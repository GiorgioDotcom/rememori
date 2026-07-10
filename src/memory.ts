import type {
  CollisionHit,
  Embedder,
  EntityCard,
  EntityExtractor,
  MemoryOptions,
  MemoryRecord,
  PruneOptions,
  RecallHit,
  RecallOptions,
  RememberOptions,
  StorageAdapter,
} from './types.js';
import { useEvidence, type EvidenceOptions } from './evidence.js';
import { dot, normalize } from './similarity.js';
import { parseDuration, toEpochMs } from './duration.js';
import { heuristicExtractor } from './extract.js';
import { HnswIndex } from './index/hnsw.js';
import { FileStorage } from './storage/file.js';
import { IndexedDBStorage } from './storage/indexeddb.js';
import { InMemoryStorage } from './storage/memory.js';

const DAY_MS = 86_400_000;
const ENTITY_BONUS = 0.1;
const ENTITY_BONUS_CAP = 0.3;
/* reinforcement hardening: log-damped, capped below the entity bonus */
const HARDENING_STEP = 0.05;
const HARDENING_CAP = 0.15;
/** below this many records, exact scan beats the index overhead */
const HNSW_AUTO_THRESHOLD = 1000;
/** rebuild the index when this fraction of its nodes are tombstones */
const HNSW_REBUILD_RATIO = 0.2;

export class Memory {
  private records: MemoryRecord[] = [];
  private byId = new Map<string, MemoryRecord>();
  /** lowercase entity -> ids of linked memories */
  private entityIndex = new Map<string, Set<string>>();
  /** lowercase entity -> display name (first seen) */
  private entityNames = new Map<string, string>();

  private hnsw: HnswIndex | null = null;

  private constructor(
    private readonly embedder: Embedder,
    private readonly storage: StorageAdapter,
    private readonly extractor: EntityExtractor | null,
    private readonly indexMode: 'auto' | 'hnsw' | 'flat',
    private readonly defaultMinSimilarity: number,
  ) {}

  /**
   * Open a memory store.
   * `path` = file path (Node/Bun), "idb://name" (browser IndexedDB),
   * or ":memory:" for volatile storage.
   */
  static async open(path: string, options: MemoryOptions): Promise<Memory> {
    const storage = options.storage ?? defaultStorage(path);
    const extractor = options.extractor === false ? null : (options.extractor ?? heuristicExtractor);
    const memory = new Memory(
      options.embedder,
      storage,
      extractor,
      options.index ?? 'auto',
      options.minSimilarity ?? 0,
    );
    memory.records = await storage.load();
    memory.byId = new Map(memory.records.map((r) => [r.id, r]));
    for (const record of memory.records) memory.indexEntities(record);
    return memory;
  }

  /** Store a memory. Returns its id. */
  async remember(text: string, options: RememberOptions = {}): Promise<string> {
    if (!text.trim()) throw new Error('rememori: cannot remember empty text');
    const [vector] = await this.embedder.embed([text]);
    if (!vector) throw new Error('rememori: embedder returned no vector');
    const entities =
      options.entities ?? (this.extractor ? await this.extractor.extract(text) : []);
    const record: MemoryRecord = {
      id: randomId(),
      text,
      vector: normalize(vector),
      tags: options.tags ?? [],
      entities,
      importance: clamp01(options.importance ?? 1),
      meta: options.meta ?? {},
      createdAt: options.createdAt ?? Date.now(),
      reinforcements: 0,
    };
    this.records.push(record);
    this.byId.set(record.id, record);
    this.indexEntities(record);
    this.hnsw?.add(record.id, record.vector);
    await this.storage.append(record);
    return record.id;
  }

  /**
   * Hybrid recall: (cosine + entity bonus) × importance × optional decay.
   * The entity bonus lets graph-linked memories surface even when the
   * embedding similarity alone would miss them.
   */
  async recall(query: string, options: RecallOptions = {}): Promise<RecallHit[]> {
    const [queryVec] = await this.embedder.embed([query]);
    if (!queryVec) throw new Error('rememori: embedder returned no vector');
    normalize(queryVec);

    const useGraph = options.graph !== false && this.extractor !== null;
    const queryEntities = useGraph ? await this.extractor!.extract(query) : [];
    const queryEntitySet = new Set(queryEntities.map((e) => e.toLowerCase()));

    const limit = options.limit ?? 10;
    const minScore = options.minScore ?? 0;
    const minSimilarity = options.minSimilarity ?? this.defaultMinSimilarity;
    const after = options.after === undefined ? -Infinity : toEpochMs(options.after);
    const before = options.before === undefined ? Infinity : toEpochMs(options.before);
    const now = Date.now();

    /* Exact scan when filtered or small; HNSW candidates otherwise.
       Entity-matched records are always unioned in, so the graph bonus
       can surface memories the vector index alone would rank out. */
    const filtered =
      options.tags !== undefined || options.after !== undefined || options.before !== undefined;
    let pool: Iterable<MemoryRecord> = this.records;
    if (!filtered && this.ensureIndex()) {
      const candidates = new Map<string, MemoryRecord>();
      const k = Math.max(limit * 8, 64);
      for (const hit of this.hnsw!.search(queryVec, k, 200)) {
        const record = this.byId.get(hit.id);
        if (record) candidates.set(record.id, record);
      }
      for (const lower of queryEntitySet) {
        for (const id of this.entityIndex.get(lower) ?? []) {
          const record = this.byId.get(id);
          if (record) candidates.set(record.id, record);
        }
      }
      pool = candidates.values();
    }

    const hits: RecallHit[] = [];
    for (const record of pool) {
      if (record.createdAt < after || record.createdAt > before) continue;
      if (options.tags && !options.tags.every((t) => record.tags.includes(t))) continue;

      const similarity = dot(queryVec, record.vector);
      const shared = queryEntitySet.size
        ? record.entities.filter((e) => queryEntitySet.has(e.toLowerCase()))
        : [];
      /* relevance floor on raw similarity; entity matches are exempt —
         the graph exists precisely to rescue low-similarity connections */
      if (similarity < minSimilarity && shared.length === 0) continue;
      const bonus = Math.min(ENTITY_BONUS_CAP, ENTITY_BONUS * shared.length);
      /* signed: reinforcements may be negative after demote() */
      const n = record.reinforcements;
      const mag = Math.min(HARDENING_CAP, HARDENING_STEP * Math.log2(1 + Math.abs(n)));
      const hardening = n >= 0 ? mag : -mag;

      let score = (similarity + bonus + hardening) * record.importance;
      if (options.halfLifeDays !== undefined) {
        /* decay is anchored to the last useful recall, not creation:
           memories that keep proving useful stop ageing */
        const anchor = record.reinforcedAt ?? record.createdAt;
        const ageDays = Math.max(0, now - anchor) / DAY_MS;
        score *= 0.5 ** (ageDays / options.halfLifeDays);
      }
      if (score <= 0 || score < minScore) continue;

      hits.push({
        id: record.id,
        text: record.text,
        score,
        similarity,
        sharedEntities: shared,
        tags: record.tags,
        entities: record.entities,
        meta: record.meta,
        createdAt: record.createdAt,
      });
    }

    hits.sort((a, b) => b.score - a.score);
    return hits.slice(0, limit);
  }

  /**
   * Signal that a memory was actually USED — it made it into the answer,
   * not merely into the candidate set. Resets its decay anchor and adds a
   * log-damped scoring bonus. Deliberately explicit: recall() stays a pure
   * read, and reinforcement coming from outside the ranking system is what
   * prevents surfacing-feedback loops. Returns false if the id is unknown.
   */
  async reinforce(id: string): Promise<boolean> {
    const record = this.byId.get(id);
    if (!record) return false;
    record.reinforcements += 1;
    record.reinforcedAt = Date.now();
    await this.storage.append(record); // append is an upsert: last write for an id wins
    return true;
  }

  /**
   * Negative feedback: this memory was used and the outcome was bad, or it
   * lost a contradiction to a newer memory. Symmetric with reinforce() —
   * log-damped penalty capped at −0.15. Does not touch the decay anchor.
   * A demoted memory ranks lower immediately instead of waiting for decay,
   * so "recalled and harmful" stops sharing a path with "never recalled".
   */
  async demote(id: string): Promise<boolean> {
    const record = this.byId.get(id);
    if (!record) return false;
    record.reinforcements -= 1;
    /* note: a heavily demoted low-similarity memory can drop out of recall
       entirely via the score<=0 cutoff, even when entity-rescued. That is
       intended — "recalled and harmful" is allowed to disappear — and
       covered by a test that locks the behavior in. */
    await this.storage.append(record);
    return true;
  }

  /**
   * Evidence-gated reinforcement: reinforce only the hits whose text
   * verifiably appears in `output` (a quoted token run, or high distinct-token
   * containment). Text evidence only, no embeddings — embedding similarity
   * between memory and answer would rebuild the self-report loop softly.
   * Returns the ids that were reinforced.
   */
  async reinforceFromOutput(
    hits: readonly { id: string; text: string }[],
    output: string,
    options: EvidenceOptions = {},
  ): Promise<string[]> {
    const reinforced: string[] = [];
    const seen = new Set<string>();
    for (const hit of hits) {
      if (seen.has(hit.id) || !this.byId.has(hit.id)) continue;
      seen.add(hit.id);
      if (useEvidence(hit.text, output, options).used) {
        await this.reinforce(hit.id);
        reinforced.push(hit.id);
      }
    }
    return reinforced;
  }

  /**
   * Memories suspiciously close to the given one — near-duplicates, updates
   * or contradictions. The engine only detects proximity; judging which of
   * the three it is needs semantics the caller has (an LLM) and embeddings
   * don't: contradictory statements embed almost identically. Typical flow:
   * remember() → collisions() → caller adjudicates → demote()/forget().
   */
  collisions(id: string, options: { threshold?: number; limit?: number } = {}): CollisionHit[] {
    const record = this.byId.get(id);
    if (!record) return [];
    const threshold = options.threshold ?? 0.8;
    const limit = options.limit ?? 5;

    const out: CollisionHit[] = [];
    for (const other of this.records) {
      if (other.id === id) continue;
      const similarity = dot(record.vector, other.vector);
      if (similarity >= threshold) {
        out.push({ id: other.id, text: other.text, similarity, createdAt: other.createdAt });
      }
    }
    return out.sort((a, b) => b.similarity - a.similarity).slice(0, limit);
  }

  /** Read one memory (without its vector). Defensive copy — mutating the
   *  result never touches engine state. Null if unknown. */
  get(id: string): Omit<MemoryRecord, 'vector'> | null {
    const record = this.byId.get(id);
    if (!record) return null;
    const { vector: _vector, ...rest } = record;
    return { ...rest, tags: [...record.tags], entities: [...record.entities], meta: { ...record.meta } };
  }

  /** Delete a memory. Returns false if the id was unknown. */
  async forget(id: string): Promise<boolean> {
    const record = this.byId.get(id);
    if (!record) return false;
    this.byId.delete(id);
    this.records = this.records.filter((r) => r.id !== id);
    this.unindexEntities(record);
    this.hnsw?.remove(id);
    await this.storage.tombstone(id);
    return true;
  }

  /** Bulk-remove old or unimportant memories. Returns number removed. */
  async prune(options: PruneOptions): Promise<number> {
    const cutoff =
      options.olderThan === undefined
        ? -Infinity
        : typeof options.olderThan === 'number'
          ? options.olderThan
          : Date.now() - parseDuration(options.olderThan);
    const belowImportance = options.belowImportance ?? -Infinity;

    const keep: MemoryRecord[] = [];
    let removed = 0;
    for (const record of this.records) {
      const tooOld = record.createdAt < cutoff;
      const tooWeak = record.importance < belowImportance;
      if (tooOld || tooWeak) {
        this.byId.delete(record.id);
        this.unindexEntities(record);
        removed++;
      } else {
        keep.push(record);
      }
    }
    if (removed > 0) {
      this.records = keep;
      this.hnsw = null; // rebuilt lazily on the next indexed recall
      await this.storage.compact(keep);
    }
    return removed;
  }

  /** Build/rebuild the HNSW index when the mode and size call for it. */
  private ensureIndex(): boolean {
    if (this.indexMode === 'flat') return false;
    if (this.indexMode === 'auto' && this.records.length < HNSW_AUTO_THRESHOLD) return false;
    if (this.hnsw && this.hnsw.deletedRatio > HNSW_REBUILD_RATIO) this.hnsw = null;
    if (!this.hnsw) {
      this.hnsw = new HnswIndex();
      for (const record of this.records) this.hnsw.add(record.id, record.vector);
    }
    return true;
  }

  /** Graph card for one entity, or null if unknown. */
  entity(name: string, options: { limit?: number } = {}): EntityCard | null {
    const lower = name.toLowerCase();
    const ids = this.entityIndex.get(lower);
    if (!ids || ids.size === 0) return null;
    const limit = options.limit ?? 20;

    const linked = [...ids]
      .map((id) => this.byId.get(id))
      .filter((r): r is MemoryRecord => r !== undefined)
      .sort((a, b) => b.createdAt - a.createdAt);

    const coCounts = new Map<string, number>();
    for (const record of linked) {
      for (const e of record.entities) {
        const el = e.toLowerCase();
        if (el === lower) continue;
        coCounts.set(el, (coCounts.get(el) ?? 0) + 1);
      }
    }

    return {
      name: this.entityNames.get(lower) ?? name,
      count: linked.length,
      memories: linked.slice(0, limit).map((r) => ({
        id: r.id,
        text: r.text,
        createdAt: r.createdAt,
        tags: r.tags,
      })),
      coEntities: [...coCounts.entries()]
        .sort((a, b) => b[1] - a[1])
        .map(([el, count]) => ({ name: this.entityNames.get(el) ?? el, count })),
    };
  }

  /** Most-linked entities in the graph. */
  entities(limit = 20): { name: string; count: number }[] {
    return [...this.entityIndex.entries()]
      .map(([lower, ids]) => ({ name: this.entityNames.get(lower) ?? lower, count: ids.size }))
      .filter((e) => e.count > 0)
      .sort((a, b) => b.count - a.count)
      .slice(0, limit);
  }

  /** Number of stored memories. */
  get size(): number {
    return this.records.length;
  }

  /** Rewrite storage without tombstones/duplicates. */
  async compact(): Promise<void> {
    await this.storage.compact(this.records);
  }

  async close(): Promise<void> {
    await this.storage.close();
  }

  private indexEntities(record: MemoryRecord): void {
    for (const e of record.entities) {
      const lower = e.toLowerCase();
      let set = this.entityIndex.get(lower);
      if (!set) {
        set = new Set();
        this.entityIndex.set(lower, set);
        this.entityNames.set(lower, e);
      }
      set.add(record.id);
    }
  }

  private unindexEntities(record: MemoryRecord): void {
    for (const e of record.entities) {
      const lower = e.toLowerCase();
      const set = this.entityIndex.get(lower);
      if (!set) continue;
      set.delete(record.id);
      if (set.size === 0) {
        this.entityIndex.delete(lower);
        this.entityNames.delete(lower);
      }
    }
  }
}

function defaultStorage(path: string) {
  if (path === ':memory:') return new InMemoryStorage();
  if (path.startsWith('idb://')) return new IndexedDBStorage(path.slice('idb://'.length));
  return new FileStorage(path);
}

function clamp01(n: number): number {
  return Math.min(1, Math.max(0, n));
}

/** UUID v4 via WebCrypto when available (browser, edge, Node 19+), else Math.random (Node 18 without the global). IDs are not security tokens. */
function randomId(): string {
  const c = (globalThis as { crypto?: { randomUUID?: () => string } }).crypto;
  if (c?.randomUUID) return c.randomUUID();
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (ch) => {
    const r = (Math.random() * 16) | 0;
    return (ch === 'x' ? r : (r & 0x3) | 0x8).toString(16);
  });
}
