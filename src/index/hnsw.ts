import { dot } from '../similarity.js';

/**
 * Pure-TypeScript HNSW (Hierarchical Navigable Small World) index.
 * Approximate nearest-neighbor search over unit vectors, cosine metric
 * (vectors must be L2-normalized; similarity = dot product).
 *
 * Deletions are tombstones: deleted nodes keep routing but never appear
 * in results. Check `deletedRatio` and rebuild when it grows.
 */

export interface HnswOptions {
  /** Max links per node on upper layers. Layer 0 uses 2×M. Default 16. */
  M?: number;
  /** Beam width during construction. Default 200. */
  efConstruction?: number;
}

export interface HnswHit {
  id: string;
  similarity: number;
}

interface HeapEntry {
  idx: number;
  /** negative dot product — smaller is closer */
  dist: number;
}

/** Binary heap over HeapEntry; `max` decides which extreme pops first. */
class Heap {
  private a: HeapEntry[] = [];
  constructor(private readonly max: boolean) {}

  get size(): number { return this.a.length; }
  peek(): HeapEntry | undefined { return this.a[0]; }

  private worse(x: HeapEntry, y: HeapEntry): boolean {
    return this.max ? x.dist < y.dist : x.dist > y.dist;
  }

  push(e: HeapEntry): void {
    const a = this.a;
    a.push(e);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.worse(a[p]!, a[i]!)) { [a[p], a[i]] = [a[i]!, a[p]!]; i = p; }
      else break;
    }
  }

  pop(): HeapEntry | undefined {
    const a = this.a;
    if (a.length === 0) return undefined;
    const top = a[0]!;
    const last = a.pop()!;
    if (a.length > 0) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let best = i;
        if (l < a.length && this.worse(a[best]!, a[l]!)) best = l;
        if (r < a.length && this.worse(a[best]!, a[r]!)) best = r;
        if (best === i) break;
        [a[i], a[best]] = [a[best]!, a[i]!];
        i = best;
      }
    }
    return top;
  }
}

export class HnswIndex {
  private readonly M: number;
  private readonly M0: number;
  private readonly efC: number;
  private readonly mL: number;

  private vectors: Float32Array[] = [];
  private extIds: string[] = [];
  private byExtId = new Map<string, number>();
  /** links[idx][level] = neighbor idx array */
  private links: number[][][] = [];
  private nodeLevel: number[] = [];
  private deleted = new Set<number>();
  private entry = -1;
  private maxLevel = -1;

  constructor(options: HnswOptions = {}) {
    this.M = options.M ?? 16;
    this.M0 = this.M * 2;
    this.efC = options.efConstruction ?? 200;
    this.mL = 1 / Math.log(this.M);
  }

  get size(): number { return this.extIds.length - this.deleted.size; }

  get deletedRatio(): number {
    return this.extIds.length === 0 ? 0 : this.deleted.size / this.extIds.length;
  }

  has(id: string): boolean {
    const idx = this.byExtId.get(id);
    return idx !== undefined && !this.deleted.has(idx);
  }

  add(id: string, vector: Float32Array): void {
    if (this.byExtId.has(id)) return;
    const idx = this.vectors.length;
    const level = Math.floor(-Math.log(Math.random()) * this.mL);

    this.vectors.push(vector);
    this.extIds.push(id);
    this.byExtId.set(id, idx);
    this.nodeLevel.push(level);
    const nodeLinks: number[][] = [];
    for (let l = 0; l <= level; l++) nodeLinks.push([]);
    this.links.push(nodeLinks);

    if (this.entry === -1) {
      this.entry = idx;
      this.maxLevel = level;
      return;
    }

    const q = vector;
    let ep = this.entry;

    /* greedy descent through layers above the new node's level */
    for (let lev = this.maxLevel; lev > level; lev--) {
      ep = this.greedyClosest(q, ep, lev);
    }

    /* insert with beam search on each layer from min(level, maxLevel) to 0 */
    let eps = [ep];
    for (let lev = Math.min(level, this.maxLevel); lev >= 0; lev--) {
      const found = this.searchLayer(q, eps, this.efC, lev);
      const maxLinks = lev === 0 ? this.M0 : this.M;
      const neighbors = this.selectNeighbors(q, found, this.M);
      this.links[idx]![lev] = neighbors.map((n) => n.idx);
      for (const n of neighbors) {
        const theirs = this.links[n.idx]![lev]!;
        theirs.push(idx);
        if (theirs.length > maxLinks) {
          const pruned = this.selectNeighbors(
            this.vectors[n.idx]!,
            theirs.map((t) => ({ idx: t, dist: -dot(this.vectors[n.idx]!, this.vectors[t]!) })),
            maxLinks,
          );
          this.links[n.idx]![lev] = pruned.map((p) => p.idx);
        }
      }
      eps = found.map((f) => f.idx);
    }

    if (level > this.maxLevel) {
      this.entry = idx;
      this.maxLevel = level;
    }
  }

  /** Tombstone-delete. The node keeps routing links but never returns as a result. */
  remove(id: string): boolean {
    const idx = this.byExtId.get(id);
    if (idx === undefined || this.deleted.has(idx)) return false;
    this.deleted.add(idx);
    if (idx === this.entry) {
      /* pick the live node with the highest level as the new entry */
      let best = -1, bestLevel = -1;
      for (let i = 0; i < this.extIds.length; i++) {
        if (this.deleted.has(i)) continue;
        if (this.nodeLevel[i]! > bestLevel) { bestLevel = this.nodeLevel[i]!; best = i; }
      }
      this.entry = best;
      this.maxLevel = bestLevel;
    }
    return true;
  }

  search(q: Float32Array, k: number, ef = 64): HnswHit[] {
    if (this.entry === -1) return [];
    let ep = this.entry;
    for (let lev = this.maxLevel; lev > 0; lev--) {
      ep = this.greedyClosest(q, ep, lev);
    }
    const found = this.searchLayer(q, [ep], Math.max(ef, k), 0);
    return found
      .filter((f) => !this.deleted.has(f.idx))
      .sort((a, b) => a.dist - b.dist)
      .slice(0, k)
      .map((f) => ({ id: this.extIds[f.idx]!, similarity: -f.dist }));
  }

  private greedyClosest(q: Float32Array, start: number, level: number): number {
    let cur = start;
    let curDist = -dot(q, this.vectors[cur]!);
    for (;;) {
      let improved = false;
      for (const n of this.links[cur]![level] ?? []) {
        const d = -dot(q, this.vectors[n]!);
        if (d < curDist) { cur = n; curDist = d; improved = true; }
      }
      if (!improved) return cur;
    }
  }

  /** Beam search on one layer. Returns up to `ef` closest entries (may include tombstones — callers filter). */
  private searchLayer(q: Float32Array, eps: number[], ef: number, level: number): HeapEntry[] {
    const visited = new Set<number>(eps);
    const candidates = new Heap(false); // min-heap: closest first
    const results = new Heap(true);     // max-heap: farthest on top

    for (const ep of eps) {
      const e = { idx: ep, dist: -dot(q, this.vectors[ep]!) };
      candidates.push(e);
      results.push(e);
    }

    while (candidates.size > 0) {
      const c = candidates.pop()!;
      const worst = results.peek();
      if (results.size >= ef && worst && c.dist > worst.dist) break;
      for (const n of this.links[c.idx]![level] ?? []) {
        if (visited.has(n)) continue;
        visited.add(n);
        const d = -dot(q, this.vectors[n]!);
        const w = results.peek();
        if (results.size < ef || (w && d < w.dist)) {
          const e = { idx: n, dist: d };
          candidates.push(e);
          results.push(e);
          if (results.size > ef) results.pop();
        }
      }
    }

    const out: HeapEntry[] = [];
    for (;;) {
      const e = results.pop();
      if (!e) break;
      out.push(e);
    }
    return out;
  }

  /**
   * Neighbor selection heuristic from the HNSW paper: keep a candidate only
   * if it is closer to the query than to every already-selected neighbor.
   * Produces diverse links, which is what keeps the graph navigable.
   */
  private selectNeighbors(q: Float32Array, candidates: HeapEntry[], m: number): HeapEntry[] {
    const sorted = [...candidates].sort((a, b) => a.dist - b.dist);
    const selected: HeapEntry[] = [];
    for (const c of sorted) {
      if (selected.length >= m) break;
      let ok = true;
      for (const s of selected) {
        if (-dot(this.vectors[c.idx]!, this.vectors[s.idx]!) < c.dist) { ok = false; break; }
      }
      if (ok) selected.push(c);
    }
    /* fall back to plain nearest if the heuristic was too strict */
    if (selected.length < m) {
      for (const c of sorted) {
        if (selected.length >= m) break;
        if (!selected.includes(c)) selected.push(c);
      }
    }
    return selected;
  }
}
