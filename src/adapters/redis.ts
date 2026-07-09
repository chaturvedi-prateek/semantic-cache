/**
 * redis-vector-adapter.ts
 *
 * Concrete implementation of VectorStoreAdapter using Upstash Redis
 * (compatible with any Redis instance that has the RediSearch / Redis Stack
 * module enabled — the same FT.* command surface is used in both cases).
 *
 * Storage layout
 * ─────────────
 *  Key prefix : sc:<uuid>          (Redis HASH)
 *  Fields     : response  — raw LLM response text
 *               vector    — Float32 embedding encoded as a binary string
 *               createdAt — ISO-8601 timestamp for observability
 *
 * Index       : FT.CREATE on the HASH prefix using an HNSW VECTOR field
 *               with COSINE distance.  Created lazily on first save/search.
 *
 * Failure policy (defensive-first)
 * ──────────────────────────────────
 *  Every Redis operation is wrapped in try/catch.  If Redis is unreachable,
 *  times out, or returns an unexpected payload:
 *    • search() returns null  → middleware falls back to the real LLM.
 *    • save()   is a no-op    → the miss is simply not cached; no crash.
 *
 * Dependencies:
 *   npm i @upstash/redis
 *   npm i -D @types/node
 *
 * Edge Runtime: this adapter uses only Web-standard APIs (Uint8Array,
 * Web Crypto `crypto.randomUUID`) and the REST-based @upstash/redis client
 * (which uses `fetch`), so it runs unchanged on Vercel Edge functions.
 */

import { Redis } from "@upstash/redis";
import type {
  VectorStoreAdapter,
  VectorMetadataFilter,
  VectorMetadata,
  VectorQueryMatch,
} from "../vector-store-adapter";
import { buildRedisFilterPrefix } from "../redis-metadata-filter";
import {
  buildCacheTags,
  revalidateNextTag,
  withNextCache,
  SEMANTIC_CACHE_TAG,
} from "../next/cache";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Prefix for every cache entry stored as a Redis HASH. */
const KEY_PREFIX = "sc:" as const;

/** Name of the RediSearch index created over the cache entries. */
const INDEX_NAME = "semantic_cache_idx" as const;

/**
 * Dimensionality of the all-MiniLM-L6-v2 embedding space.
 * Must match the model used in the middleware (384 for MiniLM-L6).
 */
const VECTOR_DIM = 384 as const;

/**
 * Number of HNSW graph neighbours to explore at query time.
 * Higher = more accurate recall, higher latency.  32 is a safe default.
 */
const HNSW_EF_RUNTIME = 32 as const;
const RESPONSE_METADATA_KEY = "response" as const;
const USER_ID_METADATA_KEY = "userId" as const;
const TENANT_ID_METADATA_KEY = "tenantId" as const;

// ---------------------------------------------------------------------------
// Helpers — raw command dispatch
// ---------------------------------------------------------------------------

/**
 * Dispatches a raw Redis command through the @upstash/redis client.
 *
 * The @upstash/redis SDK does not expose a public `sendCommand` /
 * `executeCommand` method for RediSearch (FT.*) or other low-level commands.
 * Internally every client forwards a command array via its private request
 * pipeline. We funnel ALL raw commands through this single helper so the
 * cast to the internal shape is isolated to one place rather than scattered
 * across the class.
 *
 * @param redis - The @upstash/redis client instance.
 * @param args  - The command name followed by its arguments, as a flat array.
 * @returns The decoded command result, typed by the caller via `T`.
 */
async function rawCommand<T = unknown>(
  redis: Redis,
  args: (string | number)[]
): Promise<T> {
  // The internal `request` method accepts `{ command: [...] }` and returns
  // `{ result }`. It is not part of the public type surface, so we cast here
  // deliberately and keep the cast confined to this helper.
  const client = redis as unknown as {
    request: (body: { command: (string | number)[] }) => Promise<{ result: T }>;
  };

  const { result } = await client.request({ command: args });
  return result;
}

// ---------------------------------------------------------------------------
// Helpers — vector encoding / decoding
// ---------------------------------------------------------------------------

/**
 * Encodes a JS number[] into a Uint8Array of raw Float32 bytes.
 * Redis vector fields require the embedding as a binary blob.
 *
 * Uses the Web-standard `Uint8Array` rather than Node's `Buffer` so the
 * adapter runs unchanged on the Next.js Edge Runtime (Vercel Edge functions),
 * where the Node `Buffer` global is unavailable.
 */
function encodeVector(vector: number[]): Uint8Array {
  const float32 = new Float32Array(vector);
  return new Uint8Array(float32.buffer);
}

/**
 * Converts a byte array into a latin1/"binary" string — one character per
 * byte (code points 0–255).
 *
 * This is the Edge-compatible equivalent of Node's `buffer.toString("binary")`.
 * The RediSearch REST layer expects the raw vector bytes as such a string.
 * Chunking avoids call-stack overflows from spreading very large arrays into
 * `String.fromCharCode`.
 */
function toBinaryString(bytes: Uint8Array): string {
  const CHUNK_SIZE = 0x8000; // 32768 bytes per chunk
  let binary = "";
  for (let i = 0; i < bytes.length; i += CHUNK_SIZE) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK_SIZE));
  }
  return binary;
}

/**
 * Wraps a promise with a timeout.
 * Rejects with a timeout error if the promise does not settle within
 * `ms` milliseconds.
 */
function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error(`Redis operation timed out after ${ms}ms`)),
      ms
    );
    promise.then(
      (value) => { clearTimeout(timer); resolve(value); },
      (err)   => { clearTimeout(timer); reject(err); }
    );
  });
}

// ---------------------------------------------------------------------------
// RedisVectorAdapter
// ---------------------------------------------------------------------------

export interface RedisVectorAdapterOptions {
  /**
   * Pre-configured @upstash/redis client instance.
   * Pass this if you want to share a client across your application.
   * Mutually exclusive with `redisUrl` + `redisToken`.
   */
  client?: Redis;

  /**
   * Upstash Redis REST endpoint URL.
   * Required when `client` is not provided.
   * Example: "https://your-db.upstash.io"
   */
  redisUrl?: string;

  /**
   * Upstash Redis REST API token.
   * Required when `client` is not provided.
   */
  redisToken?: string;

  /**
   * Optional TTL in seconds.  When set, each cache entry will automatically
   * expire after this many seconds via Redis EXPIRE.
   * Omit (or set to 0) to keep entries indefinitely.
   */
  ttlSeconds?: number;

  /**
   * Maximum number of milliseconds to wait for a Redis command before
   * treating the operation as a failure and falling back to the LLM.
   * @default 2000
   */
  connectionTimeoutMs?: number;

  /**
   * Optional logical namespace for this cache instance (e.g. a tenant slug,
   * a feature area, or a specific user ID).
   *
   * When set, the namespace is woven into the Next.js cache tags produced for
   * each lookup — `['semantic-cache', namespace]` — so that {@link
   * RedisVectorAdapter.invalidate} can purge only the queries belonging to
   * this namespace rather than the entire semantic cache.
   *
   * Has no effect when the package runs outside of Next.js.
   */
  namespace?: string;
}

/**
 * A {@link VectorStoreAdapter} backed by Redis with the RediSearch (Redis
 * Stack) vector similarity extension.
 *
 * Compatible with:
 *  • Upstash Redis (recommended — serverless, REST-based)
 *  • Redis Stack (self-hosted or Redis Cloud with Search enabled)
 *
 * @example
 * ```ts
 * const adapter = new RedisVectorAdapter({
 *   redisUrl:   process.env.UPSTASH_REDIS_REST_URL!,
 *   redisToken: process.env.UPSTASH_REDIS_REST_TOKEN!,
 *   ttlSeconds: 60 * 60 * 24, // 24-hour cache expiry
 * });
 *
 * const model = wrapLanguageModel({
 *   model: openai("gpt-4o"),
 *   middleware: SemanticCacheMiddleware({
 *     vectorStore: adapter,
 *     similarityThreshold: 0.92,
 *   }),
 * });
 * ```
 */
export class RedisVectorAdapter implements VectorStoreAdapter {
  private readonly redis: Redis;
  private readonly ttlSeconds: number;
  private readonly timeoutMs: number;

  /**
   * Logical namespace used to scope Next.js cache tags for this instance.
   * `undefined` when the caller did not supply one.
   */
  private readonly namespace?: string;

  /**
   * Memoised, Next.js-cache-wrapped view of {@link performSearch}.
   *
   * Built lazily on first use. When Next.js is available this is the
   * `unstable_cache`-wrapped fetch; otherwise it is `performSearch` itself,
   * so behaviour is identical outside of Next.js.
   */
  private _cachedSearch?: (
    vector: number[],
    threshold: number,
    filter?: VectorMetadataFilter
  ) => Promise<string | null>;

  /**
   * Whether the RediSearch HNSW index has been confirmed to exist this
   * process lifetime.  Avoids issuing FT.CREATE on every request.
   */
  private indexReady = false;

  constructor(options: RedisVectorAdapterOptions) {
    const { client, redisUrl, redisToken, ttlSeconds = 0, connectionTimeoutMs = 2000, namespace } = options;

    if (client) {
      this.redis = client;
    } else if (redisUrl && redisToken) {
      this.redis = new Redis({
        url:   redisUrl,
        token: redisToken,
      });
    } else {
      throw new Error(
        "RedisVectorAdapter: supply either `client` or both `redisUrl` and `redisToken`."
      );
    }

    this.ttlSeconds = ttlSeconds;
    this.timeoutMs  = connectionTimeoutMs;
    this.namespace  = namespace;
  }

  // =========================================================================
  // Index bootstrap
  // =========================================================================

  /**
   * Lazily creates the RediSearch HNSW index on first use.
   * The operation is idempotent — if the index already exists Redis returns
   * an error that we swallow silently.
   */
  private async ensureIndex(): Promise<void> {
    if (this.indexReady) return;

    try {
      await withTimeout(
        rawCommand(this.redis, [
          "FT.CREATE", INDEX_NAME,
          "ON",     "HASH",
          "PREFIX", "1", KEY_PREFIX,
          "SCHEMA",
          // ── Full-text field (for human inspection / future hybrid search) ──
          "response",  "TEXT",
          "createdAt", "TEXT",
          USER_ID_METADATA_KEY, "TAG",
          TENANT_ID_METADATA_KEY, "TAG",
          // ── Vector field — HNSW with Cosine distance ──────────────────────
          "vector", "VECTOR", "HNSW",
          "6",                         // number of attribute/value pairs below
          "TYPE",            "FLOAT32",
          "DIM",             String(VECTOR_DIM),
          "DISTANCE_METRIC", "COSINE",
        ]),
        this.timeoutMs
      );
    } catch (err: unknown) {
      // Redis returns "Index already exists" as an error string; swallow it.
      // Any other error is also swallowed here — if the index genuinely fails
      // to create, FT.SEARCH will error too and be caught in search().
      const message = err instanceof Error ? err.message : String(err);
      if (!message.includes("already exists") && !message.includes("Index already exists")) {
        console.warn("[RedisVectorAdapter] FT.CREATE warning (non-fatal):", message);
      }
    }

    // Mark as ready regardless — subsequent failures will be caught per-call.
    this.indexReady = true;
  }

  // =========================================================================
  // VectorStoreAdapter — standard methods (upsert / query / delete)
  // =========================================================================

  /**
   * Inserts or replaces the entry stored under `id`.
   *
   * The entry is written as a Redis HASH under `sc:<id>` with the vector as a
   * binary blob and the metadata serialised to JSON, so it participates in
   * the same HNSW index used by {@link search}.
   *
   * Failures are logged but never re-thrown — writes are best-effort.
   */
  async upsert(
    id: string,
    vector: number[],
    metadata: VectorMetadata
  ): Promise<void> {
    try {
      await this.ensureIndex();

      const key       = `${KEY_PREFIX}${id}`;
      const vectorBuf = encodeVector(vector);

      await withTimeout(
        rawCommand(this.redis, [
          "HSET", key,
          "vector",    toBinaryString(vectorBuf),
          "metadata",  JSON.stringify(metadata ?? {}),
          "createdAt", new Date().toISOString(),
          ...(typeof metadata?.[USER_ID_METADATA_KEY] === "string"
           ? [USER_ID_METADATA_KEY, metadata[USER_ID_METADATA_KEY] as string]
           : []),
          ...(typeof metadata?.[TENANT_ID_METADATA_KEY] === "string"
           ? [TENANT_ID_METADATA_KEY, metadata[TENANT_ID_METADATA_KEY] as string]
           : []),
        ]),
        this.timeoutMs
      );

      if (this.ttlSeconds > 0) {
        await withTimeout(
          rawCommand(this.redis, ["EXPIRE", key, String(this.ttlSeconds)]),
          this.timeoutMs
        );
      }
    } catch (err: unknown) {
      console.error(
        "[RedisVectorAdapter] upsert() failed (write skipped):",
        err instanceof Error ? err.message : err
      );
    }
  }

  /**
   * Returns the `topK` nearest entries to `vector`, ordered by descending
   * cosine similarity. Redis reports cosine *distance*, so scores are
   * converted via `similarity = 1 − distance`.
   *
   * Returns an empty array on any backend failure.
   */
  async query(
    vector: number[],
    topK: number,
    filter?: VectorMetadataFilter
  ): Promise<VectorQueryMatch[]> {
    try {
      await this.ensureIndex();

      const k = Math.max(1, Math.floor(topK));
      const queryBlob = encodeVector(vector);
      const filterPrefix = buildRedisFilterPrefix(filter);

      const rawResult = await withTimeout(
        rawCommand<[number, ...Array<string | string[]>]>(this.redis, [
          "FT.SEARCH", INDEX_NAME,
          `${filterPrefix}=>[KNN ${k} @vector $vec AS __score]`,
          "PARAMS", "2",
          "vec", toBinaryString(queryBlob),
          "RETURN", "2", "__score", "metadata",
          "SORTBY", "__score", "ASC",
          "LIMIT", "0", String(k),
          "DIALECT", "2",
        ]),
        this.timeoutMs
      );

      // rawResult layout (RESP2): [ totalCount, key1, [field, value, ...], ... ]
      const matches: VectorQueryMatch[] = [];
      for (let i = 1; i + 1 < rawResult.length; i += 2) {
        const key    = rawResult[i] as string;
        const fields = rawResult[i + 1] as string[];

        const fieldMap: Record<string, string> = {};
        for (let j = 0; j < fields.length; j += 2) {
          fieldMap[fields[j]] = fields[j + 1];
        }

        let metadata: VectorMetadata = {};
        if (fieldMap["metadata"]) {
          try {
            metadata = JSON.parse(fieldMap["metadata"]) as VectorMetadata;
          } catch {
            /* malformed metadata — return the empty map */
          }
        }

        const distance = parseFloat(fieldMap["__score"] ?? "1");

        matches.push({
          id: key.startsWith(KEY_PREFIX) ? key.slice(KEY_PREFIX.length) : key,
          score: 1 - distance,
          metadata,
        });
      }

      return matches;
    } catch (err: unknown) {
      console.error(
        "[RedisVectorAdapter] query() failed (returning no matches):",
        err instanceof Error ? err.message : err
      );
      return [];
    }
  }

  /**
   * Deletes the entry stored under `id`. A no-op when the key does not exist
   * or the backend is unreachable.
   */
  async delete(id: string): Promise<void> {
    try {
      await withTimeout(
        rawCommand(this.redis, ["DEL", `${KEY_PREFIX}${id}`]),
        this.timeoutMs
      );
    } catch (err: unknown) {
      console.error(
        "[RedisVectorAdapter] delete() failed (skipped):",
        err instanceof Error ? err.message : err
      );
    }
  }

  // =========================================================================
  // VectorStoreAdapter — search
  // =========================================================================

  /**
   * Performs a KNN-1 cosine similarity search against the HNSW index.
   *
   * When running inside a Next.js app, the underlying vector-database fetch is
   * transparently wrapped in Next.js's `unstable_cache`, tagged with
   * `['semantic-cache', namespace]`, so repeated identical lookups are served
   * from Next.js's data cache and can be purged via {@link invalidate}.
   *
   * Outside of Next.js, `unstable_cache` is unavailable and this method falls
   * straight through to the live vector-store fetch — no behavioural change.
   *
   * @param vector    - The float32 embedding of the incoming prompt.
   * @param threshold - Minimum cosine similarity score in [0, 1].
   * @returns The cached LLM response string, or `null` on a miss / error.
   */
  async search(
    vector: number[],
    threshold: number,
    filter?: VectorMetadataFilter
  ): Promise<string | null> {
    return this.getCachedSearch()(vector, threshold, filter);
  }

  /**
   * Lazily builds the Next.js-cache-wrapped view of {@link performSearch}.
   *
   * The wrapper is scoped by `keyParts` (which include the namespace) and
   * tagged via {@link buildCacheTags}. When Next.js is not present,
   * {@link withNextCache} returns `performSearch` unchanged.
   */
  private getCachedSearch(): (
    vector: number[],
    threshold: number,
    filter?: VectorMetadataFilter
  ) => Promise<string | null> {
    if (!this._cachedSearch) {
      const keyParts = [
        SEMANTIC_CACHE_TAG,
        "search",
        this.namespace ?? "default",
      ];
      const tags = buildCacheTags(this.namespace);

      // Align the Next.js data-cache lifetime with the Redis TTL when one is
      // configured, so both layers expire in lockstep. A ttl of 0 leaves the
      // revalidation window unset (Next.js default).
      const revalidate = this.ttlSeconds > 0 ? this.ttlSeconds : undefined;

      this._cachedSearch = withNextCache(
        (
          vector: number[],
          threshold: number,
          filter?: VectorMetadataFilter
        ) => this.performSearch(vector, threshold, filter),
        keyParts,
        tags,
        revalidate
      );
    }

    return this._cachedSearch;
  }

  /**
   * The raw vector-database fetch that backs {@link search}.
   *
   * Cosine *distance* (returned by Redis) = 1 − cosine *similarity*.
   * Therefore the `threshold` parameter (expressed as similarity, e.g. 0.92)
   * is converted: `maxDistance = 1 − threshold` (e.g. 0.08).
   *
   * If Redis is unavailable or returns an error, `null` is returned and the
   * middleware gracefully falls through to a live LLM call.
   */
  private async performSearch(
    vector: number[],
    threshold: number,
    filter?: VectorMetadataFilter
  ): Promise<string | null> {
    try {
      await this.ensureIndex();

      // Convert similarity threshold → cosine distance upper bound.
      const maxDistance = 1 - threshold;

      // Encode the query vector as a binary buffer for the KNN parameter.
      const queryBlob = encodeVector(vector);
      const filterPrefix = buildRedisFilterPrefix(filter);

      // ── FT.SEARCH KNN query ──────────────────────────────────────────────
      const rawResult = await withTimeout(
        rawCommand<[number, ...Array<string | string[]>]>(this.redis, [
          "FT.SEARCH", INDEX_NAME,
          `${filterPrefix}=>[KNN 1 @vector $vec AS __score]`,
          "PARAMS", "2",
          "vec", toBinaryString(queryBlob),  // binary string for the REST layer
          "RETURN", "2", "__score", "response",
          "SORTBY", "__score", "ASC",
          "DIALECT", "2",
        ]),
        this.timeoutMs
      );

      // rawResult layout (RESP2): [ totalCount, key1, [field, value, ...], ... ]
      const totalCount = rawResult[0] as number;
      if (totalCount === 0) {
        return null; // no entries in the index yet
      }

      // The first result's field-value array is at index 2.
      const fields = rawResult[2] as string[];

      // Parse the flat field-value array into a map.
      const fieldMap: Record<string, string> = {};
      for (let i = 0; i < fields.length; i += 2) {
        fieldMap[fields[i]] = fields[i + 1];
      }

      const score    = parseFloat(fieldMap["__score"] ?? "1");
      const response = fieldMap["response"] ?? null;

      // `score` is cosine distance; reject if it exceeds the max allowed distance.
      if (score > maxDistance || response === null) {
        return null;
      }

      return response;
    } catch (err: unknown) {
      // ── Defensive fallback ────────────────────────────────────────────────
      console.error(
        "[RedisVectorAdapter] search() failed (falling back to LLM):",
        err instanceof Error ? err.message : err
      );
      return null;
    }
  }

  // =========================================================================
  // VectorStoreAdapter — save
  // =========================================================================

  /**
   * Stores a prompt embedding and its LLM response as a Redis HASH.
   *
   * Failures are logged but never re-thrown — caching is best-effort.
   *
   * @param promptVector - The float32 embedding of the prompt.
   * @param response     - The LLM-generated text to cache.
   */
  async save(
    promptVector: number[],
    response: string,
    metadata: VectorMetadata = {}
  ): Promise<void> {
    try {
      await this.ensureIndex();

      const key       = `${KEY_PREFIX}${crypto.randomUUID()}`;
      const vectorBuf = encodeVector(promptVector);
      const entryMetadata: VectorMetadata & { createdAt: string } = {
        ...metadata,
        [RESPONSE_METADATA_KEY]: response,
        createdAt: new Date().toISOString(),
      };

      // Store the vector as a binary string field so RediSearch can index it.
      await withTimeout(
        rawCommand(this.redis, [
          "HSET", key,
          "vector",    toBinaryString(vectorBuf),
          "metadata",  JSON.stringify(entryMetadata),
          "response",  response,
          "createdAt", entryMetadata.createdAt,
          ...(typeof entryMetadata[USER_ID_METADATA_KEY] === "string"
            ? [USER_ID_METADATA_KEY, entryMetadata[USER_ID_METADATA_KEY] as string]
            : []),
          ...(typeof entryMetadata[TENANT_ID_METADATA_KEY] === "string"
            ? [TENANT_ID_METADATA_KEY, entryMetadata[TENANT_ID_METADATA_KEY] as string]
            : []),
        ]),
        this.timeoutMs
      );

      // Apply optional TTL — Redis will remove the key after `ttlSeconds`.
      if (this.ttlSeconds > 0) {
        await withTimeout(
          rawCommand(this.redis, ["EXPIRE", key, String(this.ttlSeconds)]),
          this.timeoutMs
        );
      }
    } catch (err: unknown) {
      // ── Defensive fallback ────────────────────────────────────────────────
      // A failed cache write must never degrade the primary LLM response
      // that was already returned to the user.
      console.error(
        "[RedisVectorAdapter] save() failed (cache write skipped):",
        err instanceof Error ? err.message : err
      );
    }
  }

  // =========================================================================
  // Next.js native cache invalidation
  // =========================================================================

  /**
   * Purges the Next.js native cache for semantic queries via `revalidateTag`.
   *
   * Because every lookup wrapped by {@link search} is tagged with
   * `['semantic-cache', namespace]`, callers can invalidate at two levels:
   *
   *   • `invalidate("user_42")` — purge only the `user_42` namespace's queries.
   *   • `invalidate()`          — purge this adapter's configured namespace, or,
   *                               if none was configured, every semantic-cache
   *                               entry (the base `'semantic-cache'` tag).
   *
   * This only affects Next.js's data cache — it does **not** delete the
   * underlying Redis entries (use {@link flush} for that). When the package
   * runs outside of Next.js, `revalidateTag` is unavailable and this method is
   * a safe no-op.
   *
   * @param namespace - Optional namespace tag to purge. Falls back to the
   *                    adapter's configured namespace when omitted.
   */
  async invalidate(namespace?: string): Promise<void> {
    const ns = namespace ?? this.namespace;
    const tag = ns ?? SEMANTIC_CACHE_TAG;
    revalidateNextTag(tag);
  }

  // =========================================================================
  // Utility — explicit teardown (optional, for graceful shutdown)
  // =========================================================================

  /**
   * Deletes ALL semantic cache entries from Redis.
   * Useful for testing or manual cache invalidation.
   *
   *     This scans the keyspace — avoid on large production databases.
   *     Prefer TTL-based expiry for automatic eviction in production.
   */
  async flush(): Promise<number> {
    try {
      let cursor = "0";
      let deletedCount = 0;

      do {
        const [nextCursor, keys] = await withTimeout(
          rawCommand<[string, string[]]>(this.redis, [
            "SCAN", cursor, "MATCH", `${KEY_PREFIX}*`, "COUNT", "100"
          ]),
          this.timeoutMs
        );

        cursor = nextCursor;

        if (keys.length > 0) {
          await withTimeout(
            rawCommand(this.redis, ["DEL", ...keys]),
            this.timeoutMs
          );
          deletedCount += keys.length;
        }
      } while (cursor !== "0");

      return deletedCount;
    } catch (err: unknown) {
      console.error("[RedisVectorAdapter] flush() failed:", err instanceof Error ? err.message : err);
      return 0;
    }
  }
}