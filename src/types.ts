/** A single stored memory. */
export interface MemoryRecord {
  id: string;
  text: string;
  /** L2-normalized embedding. */
  vector: Float32Array;
  tags: string[];
  /** Named entities linked to this memory (bipartite memory↔entity graph). */
  entities: string[];
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
  /** Explicit entities. When omitted, the configured extractor runs on the text. */
  entities?: string[];
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
  /** Minimum final score to include. Default 0. */
  minScore?: number;
  /**
   * Relevance floor on RAW cosine similarity, applied before importance,
   * decay and the graph bonus. Hits below it are dropped unless they share
   * an entity with the query. Model-dependent — measured good defaults:
   * ~0.5 for nomic-embed-text, ~0.3 for all-MiniLM-L6-v2.
   * Overrides the instance-level default from MemoryOptions.
   */
  minSimilarity?: number;
  /**
   * Hybrid graph recall: entities are extracted from the query and memories
   * sharing them get a score bonus of min(0.3, 0.1 × shared). Default true.
   */
  graph?: boolean;
}

export interface RecallHit {
  id: string;
  text: string;
  /** Final score: (cosine + entity bonus) × importance × decay. */
  score: number;
  /** Raw cosine similarity, before graph/importance/decay weighting. */
  similarity: number;
  /** Entities shared with the query (empty when graph recall is off). */
  sharedEntities: string[];
  tags: string[];
  entities: string[];
  meta: Record<string, unknown>;
  createdAt: number;
}

export interface PruneOptions {
  /** Remove records older than e.g. "90d", "12h", or epoch ms. */
  olderThan?: string | number;
  /** Remove records with importance strictly below this value. */
  belowImportance?: number;
}

/** Summary of one entity in the memory↔entity graph. */
export interface EntityCard {
  /** Display name (first-seen casing). */
  name: string;
  /** Number of memories linked to this entity. */
  count: number;
  /** Linked memories, newest first. */
  memories: { id: string; text: string; createdAt: number; tags: string[] }[];
  /** Entities co-occurring in the same memories, most frequent first. */
  coEntities: { name: string; count: number }[];
}

/** Turns texts into embedding vectors. The only external capability rememori needs. */
export interface Embedder {
  embed(texts: string[]): Promise<Float32Array[]>;
}

/** Extracts named entities from text. Default: zero-dep capitalization heuristic. */
export interface EntityExtractor {
  extract(text: string): Promise<string[]>;
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
  /**
   * Entity extractor for the graph layer. Defaults to the built-in
   * capitalization heuristic; pass `false` to disable the graph entirely.
   */
  extractor?: EntityExtractor | false;
  /**
   * Vector index strategy. "auto" (default) scans exhaustively until the
   * store passes ~1000 memories, then switches to an HNSW graph index.
   * "hnsw" forces the index at any size; "flat" always scans.
   * Tag/date-filtered recalls always use the exact scan.
   */
  index?: 'auto' | 'hnsw' | 'flat';
  /**
   * Default relevance floor for every recall (see RecallOptions.minSimilarity).
   * Without it, embedding models never score unrelated texts at zero, so an
   * off-topic query returns the least-irrelevant memories instead of nothing.
   */
  minSimilarity?: number;
}
