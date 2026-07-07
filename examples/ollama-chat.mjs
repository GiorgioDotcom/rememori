// Persistent memory for a local Ollama agent — 100% on your machine.
//
//   ollama pull nomic-embed-text
//   ollama pull llama3.2
//   npm install rememori
//   node examples/ollama-chat.mjs
//
// Tell it things ("my server runs Debian on a Pi 4"), quit, restart —
// it still knows. Memory lives in ./agent.rememori, a single JSONL file.

import readline from 'node:readline/promises';
import { Memory } from 'rememori';
import { ollama } from 'rememori/embedders';

const CHAT_MODEL = process.env.CHAT_MODEL ?? 'llama3.2';

const mem = await Memory.open('./agent.rememori', {
  embedder: ollama('nomic-embed-text'),
});

async function chat(system, user) {
  const res = await fetch('http://localhost:11434/api/chat', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: CHAT_MODEL,
      stream: false,
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: user },
      ],
    }),
  });
  if (!res.ok) throw new Error(`ollama chat failed: ${res.status}`);
  return (await res.json()).message.content;
}

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
console.log('Local agent with persistent memory. Ctrl+C to quit.\n');

while (true) {
  const user = (await rl.question('you > ')).trim();
  if (!user) continue;

  // 1. Recall what's relevant — recent memories score higher (30-day half-life).
  const hits = await mem.recall(user, { limit: 5, halfLifeDays: 30 });
  const context = hits.map((h) => `- ${h.text}`).join('\n');

  // 2. Answer with memories injected into the system prompt.
  const system =
    'You are a helpful local assistant.' +
    (context ? `\n\nThings you remember about this user:\n${context}` : '');
  const answer = await chat(system, user);
  console.log(`\nagent > ${answer}\n`);

  // 3. Remember what the user said. That's it — embedded, indexed, on disk.
  await mem.remember(user, { tags: ['chat'] });
}
