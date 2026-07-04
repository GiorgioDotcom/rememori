import type {
  Embedder,
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
import { FileStorage } from './storage/file.js';
import { InMemoryStorage } from './storage/memory.js';

const DAY_MS = 86_400_000;

export class Memory {
  private records: MemoryRecord[] = [];
  private byId = new Map<string, MemoryRecord>();

  private constructor(
    private readonly embedder: Embedder,
    private readonly storage: StorageAdapter,
  ) {}

  /**
   * Open a memory store.
   * `path` = file path (Node/Bun) or ":memory:" for volatile storage.
   */
  static async open(path: string, options: MemoryOptions): Promise<Memory> {
    const storage =
      options.storage ?? (path === ':memory:' ? new InMemoryStorage() : new FileStorage(path));
    const memory = new Memory(options.embedder, storage);
    memory.records = await storage.load();
    memory.byId = new Map(memory.records.map((r) => [r.id, r]));
    return memory;
  }

  /** Store a memory. Returns its id. */
  async remember(text: string, options: RememberOptions = {}): Promise<string> {
    if (!text.trim()) throw new Error('rememori: cannot remember empty text');
    const [vector] = await this.embedder.embed([text]);
    if (!vector) throw new Error('rememori: embedder returned no vector');
    const record: MemoryRecord = {
      id: crypto.randomUUID(),
      text,
      vector: normalize(vector),
      tags: options.tags ?? [],
      importance: clamp01(options.importance ?? 1),
      meta: options.meta ?? {},
      createdAt: options.createdAt ?? Date.now(),
    };
    this.records.push(record);
    this.byId.set(record.id, record);
    await this.storage.append(record);
    return record.id;
  }

  /** Semantic recall: cosine similarity × importance × optional temporal decay. */
  async recall(query: string, options: RecallOptions = {}): Promise<RecallHit[]> {
    const [queryVec] = await this.embedder.embed([query]);
    if (!queryVec) throw new Error('rememori: embedder returned no vector');
    normalize(queryVec);

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
      let score = similarity * record.importance;
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
        tags: record.tags,
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
}

function clamp01(n: number): number {
  return Math.min(1, Math.max(0, n));
}
