/**
 * vector-store-adapter.ts
 *
 * Canonical definition of the pluggable {@link VectorStoreAdapter} interface.
 *
 * Implement this interface to connect any vector database — Pinecone,
 * Weaviate, Qdrant, pgvector, Redis Stack, an in-memory store, etc. — to
 * next-semantic-cache. The package ships with `RedisVectorAdapter` as the
 * reference implementation.
 *
 * The interface is split into two layers:
 *
 *   1. Standard vector-store methods — `upsert`, `query`, `delete`.
 *      These map 1:1 onto the primitive operations every vector database
 *      exposes and give callers direct, id-addressable access to entries.
 *
 *   2. Cache-facing convenience methods — `search`, `save`.
 *      These are what `SemanticCacheMiddleware` calls internally. They can
 *      usually be implemented as thin wrappers over `query` / `upsert`.
 */

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

/**
 * Arbitrary JSON-serialisable metadata stored alongside a vector.
 * Adapters must persist this map verbatim and return it from {@link
 * VectorStoreAdapter.query} results.
 */
export type VectorMetadata = Record<string, unknown>;

/**
 * A single result returned by {@link VectorStoreAdapter.query}.
 */
export interface VectorQueryMatch {
  /** The unique id the entry was upserted under. */
  id: string;

  /**
   * Cosine *similarity* score in [0, 1] between the query vector and this
   * entry — higher is more similar. Adapters whose backend reports cosine
   * *distance* must convert it (`similarity = 1 − distance`).
   */
  score: number;

  /** The metadata map supplied at upsert time. */
  metadata: VectorMetadata;
}

// ---------------------------------------------------------------------------
// VectorStoreAdapter
// ---------------------------------------------------------------------------

/**
 * Strict contract for a pluggable vector-store backend.
 *
 * All methods are asynchronous and must never throw for routine backend
 * unavailability — adapters should degrade gracefully (log and return an
 * empty / null result) so a cache outage never takes down the primary
 * LLM call path.
 */
export interface VectorStoreAdapter {
  // ─── Standard vector-store methods ────────────────────────────────────────

  /**
   * Inserts a vector under `id`, or replaces the existing entry when an
   * entry with the same `id` already exists (idempotent write).
   *
   * @param id       - Unique identifier for the entry.
   * @param vector   - The embedding as a float array.
   * @param metadata - Arbitrary metadata persisted alongside the vector and
   *                   returned verbatim in {@link query} matches.
   */
  upsert(id: string, vector: number[], metadata: VectorMetadata): Promise<void>;

  /**
   * Finds the `topK` entries whose vectors are most similar to `vector`,
   * ordered by descending cosine similarity.
   *
   * @param vector - The query embedding.
   * @param topK   - Maximum number of matches to return.
   * @returns Up to `topK` matches; an empty array when the store is empty
   *          or the backend is unreachable.
   */
  query(vector: number[], topK: number): Promise<VectorQueryMatch[]>;

  /**
   * Removes the entry stored under `id`. A no-op when no such entry exists.
   *
   * @param id - The identifier the entry was upserted under.
   */
  delete(id: string): Promise<void>;

  // ─── Cache-facing convenience methods (used by SemanticCacheMiddleware) ──

  /**
   * Search for a cached response whose embedding is within `threshold`
   * cosine similarity of `vector`.
   *
   * @param vector    - The embedding of the current prompt (float32 array).
   * @param threshold - Similarity threshold in [0, 1].  A value of 0.92 is a
   *                    reasonable starting point for sentence-level deduplication.
   * @returns The cached LLM response string, or `null` on a miss.
   */
  search(vector: number[], threshold: number): Promise<string | null>;

  /**
   * Persist a prompt embedding alongside the LLM response it produced.
   *
   * @param promptVector - The embedding that was used for the cache miss.
   * @param response     - The raw text returned by the LLM.
   */
  save(promptVector: number[], response: string): Promise<void>;
}
