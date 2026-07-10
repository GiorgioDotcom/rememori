/**
 * Evidence-gated reinforcement helpers: decide whether a memory was
 * VERIFIABLY used in an output, from text alone. Deliberately dumb and
 * transparent — no embeddings here. Embedding similarity between memory
 * and answer would be the soft version of the self-report loop.
 */

/* word runs of 3+, but standalone numbers at ANY length: "Node 20" vs
   "Node 18" differ only in a 2-digit token, and numbers are exactly the
   kind of high-information detail evidence must not be blind to */
const WORD = /[\p{L}\d]{3,}|\d{1,2}/gu;

function tokens(text: string): string[] {
  return (text.toLowerCase().match(WORD) ?? []);
}

/** Longest run of consecutive memory tokens appearing consecutively in the output. */
function longestSharedRun(memoryTokens: string[], outputTokens: string[]): number {
  if (memoryTokens.length === 0 || outputTokens.length === 0) return 0;
  /* classic DP over token sequences; inputs are short (memories are sentences) */
  let best = 0;
  let prev = new Int32Array(outputTokens.length + 1);
  for (let i = 1; i <= memoryTokens.length; i++) {
    const cur = new Int32Array(outputTokens.length + 1);
    for (let j = 1; j <= outputTokens.length; j++) {
      if (memoryTokens[i - 1] === outputTokens[j - 1]) {
        cur[j] = prev[j - 1]! + 1;
        if (cur[j]! > best) best = cur[j]!;
      }
    }
    prev = cur;
  }
  return best;
}

export interface UseEvidence {
  /** True when the output verifiably used the memory. */
  used: boolean;
  /** Longest consecutive shared token run (quoted-span signal). */
  run: number;
  /** Fraction of the memory's distinct tokens present anywhere in the output. */
  containment: number;
}

export interface EvidenceOptions {
  /** Consecutive shared tokens that count as a quoted span. Default 4. */
  minRun?: number;
  /** Distinct-token containment that counts as use. Default 0.6. */
  minContainment?: number;
}

/**
 * Textual evidence that `memoryText` was used in `output`.
 *
 * The containment path has two guards beyond the threshold:
 * - it needs at least 3 distinct tokens (one-word memories would match any
 *   output containing that word);
 * - every NUMERIC token of the memory must appear in the output — numbers
 *   carry the discriminating detail, and an output that says "Node 18"
 *   must never count as evidence for a memory that says "Node 20".
 *
 * Known limitation: unsegmented CJK text tokenizes as one long token, so
 * only exact substring reuse matches; paraphrases score zero.
 */
export function useEvidence(
  memoryText: string,
  output: string,
  options: EvidenceOptions = {},
): UseEvidence {
  const minRun = options.minRun ?? 4;
  const minContainment = options.minContainment ?? 0.6;

  const memTokens = tokens(memoryText);
  const outTokens = tokens(output);
  const outSet = new Set(outTokens);

  const distinct = [...new Set(memTokens)];
  const contained = distinct.filter((t) => outSet.has(t)).length;
  const containment = distinct.length === 0 ? 0 : contained / distinct.length;
  const run = longestSharedRun(memTokens, outTokens);

  /* the numeric veto gates BOTH paths: a "quote" that changes the number
     is not a quote — the number is the payload */
  const numbersIntact = distinct
    .filter((t) => /^\d+$/.test(t))
    .every((t) => outSet.has(t));
  const runUsed = run >= minRun;
  const containmentUsed = distinct.length >= 3 && containment >= minContainment;

  return { used: numbersIntact && (runUsed || containmentUsed), run, containment };
}
