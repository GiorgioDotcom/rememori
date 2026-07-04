const UNIT_MS: Record<string, number> = {
  ms: 1,
  s: 1000,
  m: 60_000,
  h: 3_600_000,
  d: 86_400_000,
  w: 604_800_000,
};

/** Parse "90d" | "12h" | "30m" into milliseconds. */
export function parseDuration(input: string): number {
  const match = /^(\d+(?:\.\d+)?)(ms|s|m|h|d|w)$/.exec(input.trim());
  if (!match) throw new Error(`memoro: invalid duration "${input}" (use e.g. "90d", "12h")`);
  return Number(match[1]) * UNIT_MS[match[2]!]!;
}

/** Accepts epoch ms or ISO date string, returns epoch ms. */
export function toEpochMs(input: number | string): number {
  if (typeof input === 'number') return input;
  const ms = Date.parse(input);
  if (Number.isNaN(ms)) throw new Error(`memoro: invalid date "${input}"`);
  return ms;
}
