import type {
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
import { dot, normalize } from './similarity.js';
import { parseDuration, toEpochMs } from './duration.js';
import { heuristicExtractor } from './extract.js';
import { FileStorage } from './storage/file.js';
import { InMemoryStorage } from './storage/memory.js';

const DAY_MS = 86_400_000;
const ENTITY_BONUS = 0.1;
const ENTITY_BONUS_CAP = 0.3;

export class Memory {
  private records: MemoryRecord[] = [];
  private byId = new Map<string, MemoryRecord>();
  /** lowercase entity -> ids of linked memories */
  private entityIndex = new Map<string, Set<string>>();
  /** lowercase entity -> display name (first seen) */
  private entityNames = new Map<string, string>();

  private constructor(
    private readonly embedder: Embedder,
    private readonly storage: StorageAdapter,
    private readonly extractor: EntityExtractor | null,
  ) {}

  /**
   * Open a memory store.
   * `path` = file path (Node/Bun) or ":memory:" for volatile storage.
   */
  static async open(path: string, options: MemoryOptions): Promise<Memory> {
    const storage =
      options.storage ?? (path === ':memory:' ? new InMemoryStorage() : new FileStorage(path));
    const extractor = options.extractor === false ? null : (options.extractor ?? heuristicExtractor);
    const memory = new Memory(options.embedder, storage, extractor);
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
      id: crypto.randomUUID(),
      text,
      vector: normalize(vector),
      tags: options.tags ?? [],
      entities,
      importance: clamp01(options.importance ?? 1),
      meta: options.meta ?? {},
      createdAt: options.createdAt ?? Date.now(),
    };
    this.records.push(record);
    this.byId.set(record.id, record);
    this.indexEntities(record);
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
    const after = options.after === undefined ? -Infinity : toEpochMs(options.after);
    const before = options.before === undefined ? Infinity : toEpochMs(options.before);
    const now = Date.now();

    const hits: RecallHit[] = [];
    for (const record of this.records) {
      if (record.createdAt < after || record.createdAt > before) continue;
      if (options.tags && !options.tags.every((t) => record.tags.includes(t))) continue;

      const similarity = dot(queryVec, record.vector);
      const shared = queryEntitySet.size
        ? record.entities.filter((e) => queryEntitySet.has(e.toLowerCase()))
        : [];
      const bonus = Math.min(ENTITY_BONUS_CAP, ENTITY_BONUS * shared.length);

      let score = (similarity + bonus) * record.importance;
      if (options.halfLifeDays !== undefined) {
        const ageDays = Math.max(0, now - record.createdAt) / DAY_MS;
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

  /** Delete a memory. Returns false if the id was unknown. */
  async forget(id: string): Promise<boolean> {
    const record = this.byId.get(id);
    if (!record) return false;
    this.byId.delete(id);
    this.records = this.records.filter((r) => r.id !== id);
    this.unindexEntities(record);
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
      await this.storage.compact(keep);
    }
    return removed;
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

function clamp01(n: number): number {
  return Math.min(1, Math.max(0, n));
}
