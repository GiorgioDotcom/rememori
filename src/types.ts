/** A single stored memory. */
export interface MemoryRecord {
  id: string;
  text: string;
  /** L2-normalized embedding. */
  vector: Float32Array;
  tags: string[];
  /** 0..1, weighs into recall score. Default 1. */
  importance: number;
  /** Arbitrary user metadata. */
  meta: Record<string, unknown>;
  /** Epoch milliseconds. */
  createdAt: number;
}

export interface RememberOptions {
  tags?: string[];
  importance?: number;
  meta?: Record<string, unknown>;
  /** Override creation time (epoch ms). Mainly for imports/tests. */
  createdAt?: number;
}

export interface RecallOptions {
  /** Max results. Default 10. */
  limit?: number;
  /** Only records carrying ALL of these tags. */
  tags?: string[];
  /** Only records created at/after this time (epoch ms or ISO string). */
  after?: number | string;
  /** Only records created at/before this time (epoch ms or ISO string). */
  before?: number | string;
  /**
   * Temporal decay half-life in days. When set, a memory's score is
   * multiplied by 0.5 ** (ageDays / halfLifeDays). Unset = no decay.
   */
  halfLifeDays?: number;
  /** Minimum final score (0..1) to include. Default 0. */
  minScore?: number;
}

export interface RecallHit {
  id: string;
  text: string;
  /** Final score: cosine × importance × decay. */
  score: number;
  /** Raw cosine similarity, before importance/decay weighting. */
  similarity: number;
  tags: string[];
  meta: Record<string, unknown>;
  createdAt: number;
}

export interface PruneOptions {
  /** Remove records older than e.g. "90d", "12h", or epoch ms. */
  olderThan?: string | number;
  /** Remove records with importance strictly below this value. */
  belowImportance?: number;
}

/** Turns texts into embedding vectors. The only external capability rememori needs. */
export interface Embedder {
  embed(texts: string[]): Promise<Float32Array[]>;
}

/** Persistence backend. Append-oriented; compaction folds tombstones. */
export interface StorageAdapter {
  /** Load all live records (tombstones already applied). */
  load(): Promise<MemoryRecord[]>;
  append(record: MemoryRecord): Promise<void>;
  /** Record a deletion tombstone. */
  tombstone(id: string): Promise<void>;
  /** Rewrite storage with exactly these records. */
  compact(records: MemoryRecord[]): Promise<void>;
  close(): Promise<void>;
}

export interface MemoryOptions {
  embedder: Embedder;
  /** Custom storage. Defaults: file storage for a path, in-memory for ":memory:". */
  storage?: StorageAdapter;
}
