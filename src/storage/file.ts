import type { MemoryRecord, StorageAdapter } from '../types.js';

interface AppendLine {
  op: 'add';
  id: string;
  text: string;
  /** base64-encoded Float32Array */
  vector: string;
  tags: string[];
  importance: number;
  meta: Record<string, unknown>;
  createdAt: number;
}

interface TombstoneLine {
  op: 'del';
  id: string;
}

type LogLine = AppendLine | TombstoneLine;

function encodeVector(v: Float32Array): string {
  return Buffer.from(v.buffer, v.byteOffset, v.byteLength).toString('base64');
}

function decodeVector(s: string): Float32Array {
  const buf = Buffer.from(s, 'base64');
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

/**
 * Append-only JSONL log with tombstones. `compact()` rewrites the file.
 * Node/Bun only — browser and edge runtimes use their own adapters.
 */
export class FileStorage implements StorageAdapter {
  private fs: typeof import('node:fs/promises') | null = null;

  constructor(private readonly path: string) {}

  private async nodeFs() {
    if (!this.fs) this.fs = await import('node:fs/promises');
    return this.fs;
  }

  async load(): Promise<MemoryRecord[]> {
    const fs = await this.nodeFs();
    let raw: string;
    try {
      raw = await fs.readFile(this.path, 'utf8');
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') return [];
      throw err;
    }
    const live = new Map<string, MemoryRecord>();
    for (const line of raw.split('\n')) {
      if (!line.trim()) continue;
      const parsed = JSON.parse(line) as LogLine;
      if (parsed.op === 'del') {
        live.delete(parsed.id);
      } else {
        live.set(parsed.id, {
          id: parsed.id,
          text: parsed.text,
          vector: decodeVector(parsed.vector),
          tags: parsed.tags,
          importance: parsed.importance,
          meta: parsed.meta,
          createdAt: parsed.createdAt,
        });
      }
    }
    return [...live.values()];
  }

  async append(record: MemoryRecord): Promise<void> {
    const line: AppendLine = {
      op: 'add',
      id: record.id,
      text: record.text,
      vector: encodeVector(record.vector),
      tags: record.tags,
      importance: record.importance,
      meta: record.meta,
      createdAt: record.createdAt,
    };
    const fs = await this.nodeFs();
    await fs.appendFile(this.path, JSON.stringify(line) + '\n', 'utf8');
  }

  async tombstone(id: string): Promise<void> {
    const fs = await this.nodeFs();
    const line: TombstoneLine = { op: 'del', id };
    await fs.appendFile(this.path, JSON.stringify(line) + '\n', 'utf8');
  }

  async compact(records: MemoryRecord[]): Promise<void> {
    const fs = await this.nodeFs();
    const lines = records.map((record) => {
      const line: AppendLine = {
        op: 'add',
        id: record.id,
        text: record.text,
        vector: encodeVector(record.vector),
        tags: record.tags,
        importance: record.importance,
        meta: record.meta,
        createdAt: record.createdAt,
      };
      return JSON.stringify(line);
    });
    const tmp = `${this.path}.tmp`;
    await fs.writeFile(tmp, lines.join('\n') + (lines.length ? '\n' : ''), 'utf8');
    await fs.rename(tmp, this.path);
  }

  async close(): Promise<void> {
    // Nothing held open; appendFile opens/closes per call.
  }
}
