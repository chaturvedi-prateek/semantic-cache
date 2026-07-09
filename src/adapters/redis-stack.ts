/**
 * redis-stack-adapter.ts
 *
 * Concrete implementation of {@link VectorStoreAdapter} for a self-hosted
 * Redis Stack instance (or any Redis server with the RediSearch module
 * enabled, including Redis Cloud with Search).
 *
 * This adapter is designed for the `redis` npm package (node-redis v4,
 * https://github.com/redis/node-redis).  Its `sendCommand()` method is used
 * to dispatch raw FT.* and HSET commands so no RediSearch-specific SDK is
 * required.
 *
 * ioredis compatibility
 * ─────────────────────
 * If you prefer `ioredis`, wrap the client to satisfy {@link RedisClientLike}:
 *
 *   import IORedis from "ioredis";
 *   const ioredis = new IORedis(process.env.REDIS_URL);
 *
 *   const adapter = new RedisStackVectorAdapter({
 *     client: {
 *       sendCommand: (args) =>
 *         ioredis.call(args[0] as string, ...(args.slice(1) as string[])) as any,
 *     },
 *   });
 *
 * Storage layout
 * ─────────────
 *  Key prefix : sc:<uuid>          (Redis HASH)
 *  Fields     : embedding  — Float32 vector encoded as a binary blob
 *               _metadata  — full metadata JSON for round-trip retrieval
 *               response   — LLM response text (also in _metadata)
 *               createdAt  — ISO-8601 timestamp (also in _metadata)
 *
 * Required index
 * ──────────────
 * The adapter creates the RediSearch index automatically on the first
 * `save()` or `query()` call.  If you prefer to create it manually
 * (e.g. via a migration step), run the following command against your
 * Redis instance once:
 *
 *   FT.CREATE semantic_cache_idx
 *     ON HASH PREFIX 1 sc:
 *     SCHEMA
 *       embedding VECTOR HNSW 6
 *         TYPE FLOAT32
 *         DIM 384
 *         DISTANCE_METRIC COSINE
 *       response TEXT NOSTEM
 *       createdAt TAG
 *
 * Replace DIM 384 with your actual embedding dimensionality if different
 * (e.g. 1536 for text-embedding-ada-002, 3072 for text-embedding-3-large).
 *
 * DIALECT 2 is required for the KNN vector query syntax used by this
 * adapter.  This requires Redis Stack 2.x or Redis 7.x with the
 * RediSearch module at version >= 2.4.
 *
 * Failure policy
 * ──────────────
 *  • Standard vector-store methods (`upsert`, `query`, `delete`) throw a
 *    descriptive error when a Redis command fails.
 *  • Cache-facing methods (`search`, `save`) catch every error and degrade
 *    gracefully (search → null, save → no-op) so a cache outage never takes
 *    down the primary LLM call path.
 *
 * Dependencies:
 *   npm i redis          # node-redis v4
 *   npm i -D @types/node
 */

import type {
  VectorMetadataFilter,
  VectorStoreAdapter,
  VectorMetadata,
  VectorQueryMatch,
} from "../vector-store-adapter";
import { buildRedisFilterPrefix } from "../redis-metadata-filter";

// ---------------------------------------------------------------------------
// RedisClientLike — minimal interface compatible with node-redis v4
// ---------------------------------------------------------------------------

/**
 * Minimal Redis client interface required by {@link RedisStackVectorAdapter}.
 *
 * A `redis` (node-redis v4) client satisfies this shape directly:
 *
 * ```ts
 * import { createClient } from "redis";
 * const client = createClient({ url: process.env.REDIS_URL });
 * await client.connect();
 * const adapter = new RedisStackVectorAdapter({ client });
 * ```
 */
export interface RedisClientLike {
  sendCommand(args: (string | Uint8Array)[]): Promise<unknown>;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Prefix applied to every Redis HASH key used for cache entries. */
const KEY_PREFIX = "sc:" as const;

/**
 * Name of the RediSearch index created over the HASH keys with KEY_PREFIX.
 * Must match the index name used in the manual FT.CREATE command shown in
 * the file header.
 */
const INDEX_NAME = "semantic_cache_idx" as const;

/** Metadata key under which `save()` stores the cached LLM response. */
const RESPONSE_METADATA_KEY = "response" as const;
const USER_ID_METADATA_KEY = "userId" as const;
const TENANT_ID_METADATA_KEY = "tenantId" as const;

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

export interface RedisStackVectorAdapterOptions {
  /**
   * A connected node-redis v4 client (or any object implementing
   * {@link RedisClientLike}).
   *
   * @example
   * ```ts
   * import { createClient } from "redis";
   * const client = createClient({ url: process.env.REDIS_URL });
   * await client.connect();
   * const adapter = new RedisStackVectorAdapter({ client });
   * ```
   */
  client: RedisClientLike;

  /**
   * Dimensionality of the embedding vectors.
   * Must match your embedding model and the DIM value in the FT.CREATE
   * command (384 for all-MiniLM-L6-v2).
   * @default 384
   */
  dimensions?: number;

  /**
   * Optional TTL in seconds.  When set, each cache entry will automatically
   * expire after this many seconds via the Redis EXPIRE command.
   * Omit (or set to 0) to keep entries indefinitely.
   */
  ttlSeconds?: number;
}

// ---------------------------------------------------------------------------
// Helpers — vector encoding
// ---------------------------------------------------------------------------

/**
 * Encodes a JS `number[]` into a `Uint8Array` of raw Float32 bytes.
 * RediSearch VECTOR fields require the embedding as a packed binary blob
 * with TYPE FLOAT32.
 */
function encodeVector(vector: number[]): Uint8Array {
  const float32 = new Float32Array(vector);
  return new Uint8Array(float32.buffer);
}

// ---------------------------------------------------------------------------
// RedisStackVectorAdapter
// ---------------------------------------------------------------------------

/**
 * A {@link VectorStoreAdapter} backed by a self-hosted Redis Stack (or any
 * Redis server with the RediSearch module) using the FT.SEARCH KNN vector
 * similarity syntax.
 *
 * Pass a pre-connected node-redis v4 client.  The adapter creates the
 * RediSearch index automatically on first use.
 *
 * @example
 * ```ts
 * import { createClient } from "redis";
 * import { RedisStackVectorAdapter } from "next-semantic-cache/redis-stack-adapter";
 *
 * const client = createClient({ url: process.env.REDIS_URL });
 * await client.connect();
 *
 * const adapter = new RedisStackVectorAdapter({
 *   client,
 *   ttlSeconds: 86_400, // 24-hour cache expiry
 * });
 * ```
 */
export class RedisStackVectorAdapter implements VectorStoreAdapter {
  private readonly client: RedisClientLike;
  private readonly dimensions: number;
  private readonly ttlSeconds: number;

  /**
   * Memoised promise for the one-time FT.CREATE index setup.
   * Ensures the index is created at most once per adapter instance,
   * even under concurrent first calls.
   */
  private ensureIndexPromise: Promise<void> | null = null;

  constructor(options: RedisStackVectorAdapterOptions) {
    if (!options.client) {
      throw new Error("RedisStackVectorAdapter: `client` is required.");
    }
    this.client = options.client;
    this.dimensions = options.dimensions ?? 384;
    this.ttlSeconds = options.ttlSeconds ?? 0;
  }

  // =========================================================================
  // Index management
  // =========================================================================

  /**
   * Creates the RediSearch HNSW vector index if it does not already exist.
   * The call is memoised — the FT.CREATE command is sent at most once per
   * adapter instance regardless of how many concurrent callers reach this
   * method simultaneously.
   *
   * The command issued is equivalent to:
   *
   *   FT.CREATE semantic_cache_idx
   *     ON HASH PREFIX 1 sc:
   *     SCHEMA
   *       embedding VECTOR HNSW 6
   *         TYPE FLOAT32 DIM <dimensions> DISTANCE_METRIC COSINE
   *       response TEXT NOSTEM
   *       createdAt TAG
   *
   * Redis returns an "Index already exists" error when the index is already
   * present; this method silently ignores that error so the adapter is safe
   * to reinitialise across warm lambda invocations or hot-reloads.
   */
  private async ensureIndex(): Promise<void> {
    if (this.ensureIndexPromise) return this.ensureIndexPromise;

    this.ensureIndexPromise = (async () => {
      try {
        // FT.CREATE — HNSW index with COSINE distance over HASH keys
        // prefixed with KEY_PREFIX ("sc:").
        // The 6 after HNSW is the number of HNSW attribute key-value pairs
        // that follow (TYPE, DIM, DISTANCE_METRIC = 3 pairs × 2 tokens = 6).
        await this.client.sendCommand([
          "FT.CREATE",
          INDEX_NAME,
          "ON", "HASH",
          "PREFIX", "1", KEY_PREFIX,
          "SCHEMA",
          // Vector field: packed Float32 bytes, HNSW graph, cosine similarity.
          "embedding", "VECTOR", "HNSW", "6",
          "TYPE", "FLOAT32",
          "DIM", String(this.dimensions),
          "DISTANCE_METRIC", "COSINE",
          // Stored LLM response — indexed as a full-text field so future
          // callers can layer text-filter predicates onto KNN queries.
          "response", "TEXT", "NOSTEM",
          // ISO-8601 creation timestamp — indexed as TAG for equality filters.
          "createdAt", "TAG",
          USER_ID_METADATA_KEY, "TAG",
          TENANT_ID_METADATA_KEY, "TAG",
        ]);
      } catch (err: unknown) {
        // "Index already exists" (BUSYKEY) is expected on warm starts.
        const msg = err instanceof Error ? err.message : String(err);
        if (!msg.includes("Index already exists")) {
          throw err;
        }
      }
    })();

    return this.ensureIndexPromise;
  }

  // =========================================================================
  // VectorStoreAdapter — standard methods (upsert / query / delete)
  // =========================================================================

  /**
   * Stores a vector and its metadata as a Redis HASH under `sc:<id>`.
   *
   * Uses HSET, which creates the key if absent or updates fields in-place
   * if it already exists (idempotent write, matching the upsert semantics
   * required by {@link VectorStoreAdapter}).
   *
   * The embedding is stored as a packed Float32 binary blob so RediSearch
   * can read it directly via the VECTOR schema field.  The full metadata
   * object is additionally stored as a JSON string in `_metadata` for
   * lossless round-trip retrieval in {@link query}.
   */
  async upsert(
    id: string,
    vector: number[],
    metadata: VectorMetadata
  ): Promise<void> {
    const key = `${KEY_PREFIX}${id}`;
    // Encode the float32 array as raw bytes for the VECTOR index field.
    const embeddingBytes = encodeVector(vector);

    // Build the HSET argument list.  The (string | Uint8Array)[] type is required
    // because the embedding bytes must be sent as a raw Uint8Array, while all
    // other values are plain strings.
    const args: (string | Uint8Array)[] = [
      "HSET", key,
      // VECTOR field consumed by RediSearch for KNN queries.
      "embedding", embeddingBytes,
      // Full metadata stored as JSON for lossless round-trip in query().
      "_metadata", JSON.stringify(metadata ?? {}),
    ];

    // Duplicate known metadata fields as individual hash fields so that
    // RediSearch can index them (TEXT / TAG) for combined text+vector queries.
    const response = (metadata as Record<string, unknown>)[RESPONSE_METADATA_KEY];
    if (typeof response === "string") {
      args.push("response", response);
    }
    const createdAt = (metadata as Record<string, unknown>).createdAt;
    if (typeof createdAt === "string") {
      args.push("createdAt", createdAt);
    }
    const userId = (metadata as Record<string, unknown>)[USER_ID_METADATA_KEY];
    if (typeof userId === "string") {
      args.push(USER_ID_METADATA_KEY, userId);
    }
    const tenantId = (metadata as Record<string, unknown>)[TENANT_ID_METADATA_KEY];
    if (typeof tenantId === "string") {
      args.push(TENANT_ID_METADATA_KEY, tenantId);
    }

    await this.client.sendCommand(args);

    if (this.ttlSeconds > 0) {
      // EXPIRE sets the key to auto-delete after ttlSeconds seconds.
      await this.client.sendCommand([
        "EXPIRE", key, String(this.ttlSeconds),
      ]);
    }
  }

  /**
   * Performs a KNN vector similarity search using FT.SEARCH.
   *
   * The query issued is:
   *
   *   FT.SEARCH semantic_cache_idx
   *     "*=>[KNN <topK> @embedding $vec AS __score]"
   *     PARAMS 2 vec <float32-bytes>
   *     RETURN 2 _metadata __score
   *     SORTBY __score ASC
   *     DIALECT 2
   *
   * `__score` is the cosine *distance* returned by RediSearch, in [0, 2]
   * (0 = identical vectors, 2 = opposite vectors).  This method converts it
   * to cosine *similarity* via `score = 1 − distance`, clamped to [0, 1],
   * as required by the {@link VectorStoreAdapter} contract.
   */
  async query(
    vector: number[],
    topK: number,
    filter?: VectorMetadataFilter
  ): Promise<VectorQueryMatch[]> {
    await this.ensureIndex();

    const k = Math.max(1, Math.floor(topK));
    const vecBytes = encodeVector(vector);
    const filterPrefix = buildRedisFilterPrefix(filter);

    // FT.SEARCH RESP2 response format:
    //   [totalCount, key1, [field, value, ...], key2, [field, value, ...], ...]
    const raw = await this.client.sendCommand([
      "FT.SEARCH",
      INDEX_NAME,
      // KNN filter: find the k nearest neighbours of $vec in @embedding field.
      // The $vec placeholder is resolved via the PARAMS clause below.
      `${filterPrefix}=>[KNN ${k} @embedding $vec AS __score]`,
      // PARAMS passes named binary parameters; nargs=2 means one name+value pair.
      "PARAMS", "2", "vec", vecBytes,
      // RETURN limits the fields included in each result to save bandwidth.
      "RETURN", "2", "_metadata", "__score",
      // Sort ascending by cosine distance (closest first).
      "SORTBY", "__score", "ASC",
      // DIALECT 2 is required for the KNN vector search syntax.
      "DIALECT", "2",
    ]) as unknown[];

    if (!Array.isArray(raw) || raw.length < 1) return [];

    const totalCount = Number(raw[0]);
    if (totalCount === 0) return [];

    const matches: VectorQueryMatch[] = [];

    // Results occupy pairs starting at index 1: [key, fieldsArray] per result.
    for (let i = 1; i < raw.length; i += 2) {
      const key = raw[i] as string;
      const id = key.startsWith(KEY_PREFIX)
        ? key.slice(KEY_PREFIX.length)
        : key;

      const fields = raw[i + 1] as string[];
      if (!Array.isArray(fields)) continue;

      // Parse the flat [field, value, field, value, ...] array into a map.
      const fieldMap: Record<string, string> = {};
      for (let f = 0; f < fields.length; f += 2) {
        fieldMap[fields[f]] = fields[f + 1];
      }

      // __score is the cosine *distance* in [0, 2]; convert to similarity.
      const distance = parseFloat(fieldMap["__score"] ?? "2");
      const score = Math.min(1, Math.max(0, 1 - distance));

      // Reconstruct the full metadata from the stored JSON blob.
      let metadata: VectorMetadata = {};
      try {
        if (fieldMap["_metadata"]) {
          metadata = JSON.parse(fieldMap["_metadata"]) as VectorMetadata;
        }
      } catch {
        // Malformed JSON — return an empty metadata map rather than crashing.
      }

      matches.push({ id, score, metadata });
    }

    return matches;
  }

  /** Removes the entry stored under `id`. A no-op when no such entry exists. */
  async delete(id: string): Promise<void> {
    await this.client.sendCommand(["DEL", `${KEY_PREFIX}${id}`]);
  }

  // =========================================================================
  // VectorStoreAdapter — cache-facing convenience methods
  // =========================================================================

  /**
   * Searches for a cached response whose embedding is within `threshold`
   * cosine similarity of `vector`. Returns `null` on a miss or any backend
   * failure so the middleware falls back to the live LLM call.
   */
  async search(
    vector: number[],
    threshold: number,
    filter?: VectorMetadataFilter
  ): Promise<string | null> {
    try {
      const [best] = await this.query(vector, 1, filter);
      if (!best || best.score < threshold) return null;

      const response = best.metadata[RESPONSE_METADATA_KEY];
      return typeof response === "string" ? response : null;
    } catch (err: unknown) {
      console.error(
        "[RedisStackVectorAdapter] search() failed (falling back to LLM):",
        err instanceof Error ? err.message : err
      );
      return null;
    }
  }

  /**
   * Persists a prompt embedding alongside the LLM response it produced.
   * Failures are logged but never re-thrown — caching is best-effort.
   */
  async save(
    promptVector: number[],
    response: string,
    metadata: VectorMetadata = {}
  ): Promise<void> {
    try {
      await this.ensureIndex();
      await this.upsert(crypto.randomUUID(), promptVector, {
        ...metadata,
        [RESPONSE_METADATA_KEY]: response,
        createdAt: new Date().toISOString(),
      });
    } catch (err: unknown) {
      console.error(
        "[RedisStackVectorAdapter] save() failed (cache write skipped):",
        err instanceof Error ? err.message : err
      );
    }
  }
}
