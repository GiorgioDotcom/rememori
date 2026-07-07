// Honest numbers for the README. Run: node bench/hnsw.bench.mjs
// Clustered vectors (mixture around random centers) model real embedding
// distributions; queries are perturbations of stored points, like a
// paraphrased question against a stored memory.
import { HnswIndex, dot, normalize } from '../dist/index.js';

const DIM = 384; // MiniLM / nomic-embed class
const QUERIES = 50;
const EF = 200;  // what Memory.recall uses

function randVec() {
  const v = new Float32Array(DIM);
  for (let i = 0; i < DIM; i++) v[i] = Math.random() * 2 - 1;
  return normalize(v);
}

/* Real embeddings live on a low intrinsic-dimension manifold. Model that:
   sample a latent z in 24 dims, project to DIM with a fixed random matrix.
   (Uniform-random 384-dim vectors are the known pathological case for ANY
   ANN index — distances concentrate — and don't resemble real data.) */
const LATENT = 24;
const PROJ = Array.from({ length: LATENT }, randVec);

function fromLatent(z) {
  const v = new Float32Array(DIM);
  for (let l = 0; l < LATENT; l++) {
    const p = PROJ[l];
    for (let i = 0; i < DIM; i++) v[i] += z[l] * p[i];
  }
  return normalize(v);
}

function makeRealistic(n) {
  return Array.from({ length: n }, () => {
    const z = Array.from({ length: LATENT }, () => Math.random() * 2 - 1);
    return { z, v: fromLatent(z) };
  });
}

function perturbLatent(z) {
  return fromLatent(z.map((x) => x + (Math.random() * 2 - 1) * 0.15));
}

function bruteTopK(vectors, q, k) {
  return vectors.map((v, i) => ({ i, s: dot(q, v) })).sort((a, b) => b.s - a.s).slice(0, k).map((x) => x.i);
}

for (const N of [1_000, 10_000, 50_000]) {
  const data = makeRealistic(N);
  const vectors = data.map((d) => d.v);
  const queries = Array.from({ length: QUERIES }, (_, i) => perturbLatent(data[(i * 37) % N].z));

  let t0 = performance.now();
  const exactSets = queries.map((q) => new Set(bruteTopK(vectors, q, 10).map(String)));
  const bruteMs = (performance.now() - t0) / QUERIES;

  t0 = performance.now();
  const index = new HnswIndex();
  vectors.forEach((v, i) => index.add(String(i), v));
  const buildMs = performance.now() - t0;

  t0 = performance.now();
  const results = queries.map((q) => index.search(q, 10, EF));
  const hnswMs = (performance.now() - t0) / QUERIES;

  let hits = 0;
  results.forEach((rs, qi) => { for (const r of rs) if (exactSets[qi].has(r.id)) hits++; });

  console.log(
    `N=${N.toLocaleString('en')}  brute=${bruteMs.toFixed(2)}ms/q  hnsw=${hnswMs.toFixed(2)}ms/q  ` +
    `speedup=${(bruteMs / hnswMs).toFixed(1)}x  recall@10=${((hits / (QUERIES * 10)) * 100).toFixed(1)}%  ` +
    `build=${(buildMs / 1000).toFixed(1)}s (${(buildMs / N).toFixed(1)}ms/insert)`,
  );
}
