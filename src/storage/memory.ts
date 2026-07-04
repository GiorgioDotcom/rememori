import type { MemoryRecord, StorageAdapter } from '../types.js';

/** Volatile storage. Everything is lost on close. */
export class InMemoryStorage implements StorageAdapter {
  private records = new Map<string, MemoryRecord>();

  async load(): Promise<MemoryRecord[]> {
    return [...this.records.values()];
  }

  async append(record: MemoryRecord): Promise<void> {
    this.records.set(record.id, record);
  }

  async tombstone(id: string): Promise<void> {
    this.records.delete(id);
  }

  async compact(records: MemoryRecord[]): Promise<void> {
    this.records = new Map(records.map((r) => [r.id, r]));
  }

  async close(): Promise<void> {
    this.records.clear();
  }
}
