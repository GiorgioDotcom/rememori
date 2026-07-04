import type { EntityExtractor } from './types.js';

/**
 * Single capitalized words that are almost always sentence starters or
 * function words, not entities (English + Italian). Multi-word matches
 * are never filtered by this list.
 */
const STOP_SINGLE = new Set([
  'the', 'a', 'an', 'i', 'it', 'this', 'that', 'these', 'those',
  'what', 'who', 'where', 'when', 'why', 'how', 'which',
  'my', 'your', 'his', 'her', 'our', 'their', 'if', 'and', 'but', 'or', 'so',
  'il', 'la', 'lo', 'le', 'gli', 'un', 'una', 'uno',
  'che', 'cosa', 'chi', 'dove', 'quando', 'perche', 'perché', 'come', 'quale',
  'questo', 'questa', 'quello', 'quella', 'e', 'ma', 'se', 'non', 'per', 'con',
]);

const CAPITALIZED_RUN = /\p{Lu}[\p{L}\d'’.-]*(?:[ \t]+\p{Lu}[\p{L}\d'’.-]*)*/gu;

/**
 * Zero-dependency heuristic entity extractor: runs of Capitalized Words.
 * Good enough as a default; plug an LLM-backed extractor for quality.
 */
export const heuristicExtractor: EntityExtractor = {
  async extract(text: string): Promise<string[]> {
    const seen = new Map<string, string>(); // lowercase -> original
    for (const match of text.matchAll(CAPITALIZED_RUN)) {
      const name = match[0].replace(/[.,;:]+$/, '').trim();
      if (name.length < 2) continue;
      const lower = name.toLowerCase();
      if (!name.includes(' ') && STOP_SINGLE.has(lower)) continue;
      if (!seen.has(lower)) seen.set(lower, name);
    }
    return [...seen.values()];
  },
};

/** Case-insensitive intersection size of two entity lists. */
export function sharedEntityCount(a: string[], b: string[]): number {
  if (a.length === 0 || b.length === 0) return 0;
  const set = new Set(a.map((e) => e.toLowerCase()));
  let count = 0;
  for (const e of b) if (set.has(e.toLowerCase())) count++;
  return count;
}
