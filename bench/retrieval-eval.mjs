// Retrieval calibration (needs Ollama running + `npm i @huggingface/transformers` for the MiniLM leg): measure similarity distributions for relevant vs
// irrelevant queries across embedder configs, to pick a relevance threshold
// and validate nomic task prefixes.
import { dot, normalize } from '../dist/index.js';

const MEMORIES = [
  // infra / deploys
  'Staging deploys fail when the Redis cache is cold',
  'The production database runs Postgres 15 with pgvector',
  'CI pipeline needs Node 20, the tests break on Node 18',
  'Rollbacks must go through the blue-green switch, never kubectl directly',
  'The S3 bucket for assets is eu-south-1, renaming it breaks the CDN',
  // personal prefs
  'I prefer dark theme in every editor',
  'I hate notification sounds during meetings',
  'I love candy but I am cutting sugar this month',
  'My working hours are 9 to 17 CET, no calls after 18',
  'I write commit messages in English, always imperative mood',
  // people / entities
  'Marco owns the Kubernetes cluster and approves infra changes',
  'Sara from design reviews every UI change before merge',
  'Giorgio fixed the flaky deploy pipeline last Tuesday',
  'The client contact at Acme Corp is Laura, she answers fast on Slack',
  'Davide is on parental leave until September',
  // project decisions
  'We chose NestJS over Express for the billing service',
  'GraphQL was rejected, the public API stays REST',
  'Feature flags go through Unleash, never env variables',
  'The mobile app deadline moved to October 15',
  'Authentication uses JWT with 15 minute access tokens',
  // misc
  'The office coffee machine needs decalcifying every month',
  'Standup is at 9:30 on Mondays and Thursdays',
  'The wifi password for guests rotates every Friday',
  'Lunch break is usually at 13 with the backend team',
];

const RELEVANT = [
  ['why do staging deploys fail?', 0],
  ['what database do we use in production?', 1],
  ['which Node version does CI need?', 2],
  ['how do we do rollbacks?', 3],
  ['what UI theme do I like?', 5],
  ['do I want sounds in meetings?', 6],
  ['when can people schedule calls with me?', 8],
  ['who manages the Kubernetes cluster?', 10],
  ['who is the contact at Acme?', 13],
  ['which framework did we pick for billing?', 15],
  ['is the public API GraphQL?', 16],
  ['when is the mobile deadline?', 18],
];

const IRRELEVANT = [
  'what do I know about cars?',
  'best pizza in Naples?',
  'how tall is mount Everest?',
  'latest football results',
  'how to grow tomatoes on a balcony',
  'what is the capital of Australia?',
  'chords for Wonderwall',
  'is it going to rain tomorrow?',
];

async function ollamaEmbed(texts, prefix = '') {
  const res = await fetch('http://localhost:11434/api/embed', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'nomic-embed-text', input: texts.map((t) => prefix + t) }),
  });
  const data = await res.json();
  return data.embeddings.map((e) => normalize(new Float32Array(e)));
}

let miniLmPipe = null;
async function miniLmEmbed(texts) {
  if (!miniLmPipe) {
    const { pipeline } = await import('@huggingface/transformers');
    miniLmPipe = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  }
  const out = await miniLmPipe(texts, { pooling: 'mean', normalize: true });
  return out.tolist().map((v) => normalize(new Float32Array(v)));
}

function stats(name, memVecs, qRelVecs, qIrrVecs) {
  const relSims = [];   // similarity of the CORRECT memory for each relevant query
  const relTops = [];   // top-1 correct?
  RELEVANT.forEach(([, target], i) => {
    const q = qRelVecs[i];
    const sims = memVecs.map((m) => dot(q, m));
    relSims.push(sims[target]);
    relTops.push(sims.indexOf(Math.max(...sims)) === target);
  });
  const irrMax = qIrrVecs.map((q) => Math.max(...memVecs.map((m) => dot(q, m))));

  const fmt = (a) => `min=${Math.min(...a).toFixed(3)} avg=${(a.reduce((x, y) => x + y) / a.length).toFixed(3)} max=${Math.max(...a).toFixed(3)}`;
  console.log(`\n=== ${name} ===`);
  console.log(`top-1 accuracy: ${relTops.filter(Boolean).length}/${relTops.length}`);
  console.log(`correct-hit sim:      ${fmt(relSims)}`);
  console.log(`irrelevant TOP sim:   ${fmt(irrMax)}`);
  const gap = Math.min(...relSims) - Math.max(...irrMax);
  console.log(`separation gap: ${gap.toFixed(3)} ${gap > 0 ? '→ clean threshold exists' : '→ OVERLAP'}`);
  const threshold = (Math.min(...relSims) + Math.max(...irrMax)) / 2;
  console.log(`suggested minSimilarity: ${threshold.toFixed(2)}`);
}

const relQ = RELEVANT.map(([q]) => q);

// A) nomic, current behavior (no prefixes)
{
  const m = await ollamaEmbed(MEMORIES);
  const qr = await ollamaEmbed(relQ);
  const qi = await ollamaEmbed(IRRELEVANT);
  stats('nomic-embed-text — NO prefixes (current)', m, qr, qi);
}
// B) nomic with task prefixes (as the model card mandates)
{
  const m = await ollamaEmbed(MEMORIES, 'search_document: ');
  const qr = await ollamaEmbed(relQ, 'search_query: ');
  const qi = await ollamaEmbed(IRRELEVANT, 'search_query: ');
  stats('nomic-embed-text — WITH task prefixes', m, qr, qi);
}
// C) MiniLM (what the site demo uses)
{
  const m = await miniLmEmbed(MEMORIES);
  const qr = await miniLmEmbed(relQ);
  const qi = await miniLmEmbed(IRRELEVANT);
  stats('Xenova/all-MiniLM-L6-v2 (site demo)', m, qr, qi);
}
