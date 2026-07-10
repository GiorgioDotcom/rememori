import { describe, expect, it } from 'vitest';
import { useEvidence } from '../src/index.js';

describe('useEvidence', () => {
  it('a contradicting number is NOT evidence (the Node 20 vs 18 case)', () => {
    const e = useEvidence('the project needs Node 20', 'yes — the project needs Node 18');
    expect(e.used).toBe(false);
  });

  it('matching numbers with high containment ARE evidence', () => {
    const e = useEvidence('the project needs Node 20', 'upgrade first: the project needs Node 20');
    expect(e.used).toBe(true);
  });

  it('a quoted run of 4+ tokens is evidence', () => {
    const e = useEvidence(
      'staging deploys fail when the redis cache is cold',
      'as noted, deploys fail when the redis cache is cold, so warm it',
    );
    expect(e.used).toBe(true);
    expect(e.run).toBeGreaterThanOrEqual(4);
  });

  it('one-word memories never pass via containment', () => {
    const e = useEvidence('important!', 'this is important for everyone');
    expect(e.containment).toBe(1);
    expect(e.used).toBe(false);
  });

  it('handles accented unicode text', () => {
    const e = useEvidence(
      'la città preferita di Aurélie è Zurigo',
      'ricorda che la città preferita di Aurélie è Zurigo, prenota lì',
    );
    expect(e.used).toBe(true);
  });

  it('unrelated text is not evidence', () => {
    const e = useEvidence('the wifi password rotates every friday', 'tomatoes grow best in full sun');
    expect(e.used).toBe(false);
    expect(e.run).toBe(0);
  });
});
