import type { MemoryRecord, StorageAdapter } from '../types.js';

/**
 * Browser storage on IndexedDB. Vectors are stored as Float32Array
 * directly — structured clone handles them, no encoding needed.
 * Open with `Memory.open('idb://name', ...)`.
 */
export class IndexedDBStorage implements StorageAdapter {
  private db: IDBDatabase | null = null;

  constructor(private readonly dbName: string) {}

  private async openDb(): Promise<IDBDatabase> {
    if (this.db) return this.db;
    const idb = (globalThis as { indexedDB?: IDBFactory }).indexedDB;
    if (!idb) throw new Error('rememori: indexedDB is not available in this runtime');
    this.db = await new Promise<IDBDatabase>((resolve, reject) => {
      const req = idb.open(this.dbName, 1);
      req.onupgradeneeded = () => {
        req.result.createObjectStore('memories', { keyPath: 'id' });
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error ?? new Error('rememori: failed to open IndexedDB'));
    });
    return this.db;
  }

  private async tx<T>(
    mode: IDBTransactionMode,
    run: (store: IDBObjectStore) => IDBRequest<T> | void,
  ): Promise<T> {
    const db = await this.openDb();
    return new Promise<T>((resolve, reject) => {
      const transaction = db.transaction('memories', mode);
      let result: T;
      const request = run(transaction.objectStore('memories'));
      if (request) request.onsuccess = () => (result = request.result);
      transaction.oncomplete = () => resolve(result);
      transaction.onerror = () => reject(transaction.error);
      transaction.onabort = () => reject(transaction.error ?? new Error('rememori: IndexedDB transaction aborted'));
    });
  }

  async load(): Promise<MemoryRecord[]> {
    const rows = await this.tx<MemoryRecord[]>('readonly', (store) => store.getAll());
    // Defensive copy into real Float32Array (some polyfills return plain objects)
    return rows.map((r) => ({ ...r, vector: new Float32Array(r.vector) }));
  }

  async append(record: MemoryRecord): Promise<void> {
    await this.tx('readwrite', (store) => void store.put(record));
  }

  async tombstone(id: string): Promise<void> {
    await this.tx('readwrite', (store) => void store.delete(id));
  }

  async compact(records: MemoryRecord[]): Promise<void> {
    await this.tx('readwrite', (store) => {
      store.clear();
      for (const record of records) store.put(record);
    });
  }

  async close(): Promise<void> {
    this.db?.close();
    this.db = null;
  }
}
