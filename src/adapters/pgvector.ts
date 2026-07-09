/**
 * pgvector-adapter.ts
 *
 * Concrete implementation of {@link VectorStoreAdapter} backed by a
 * PostgreSQL database with the pgvector extension
 * (https://github.com/pgvector/pgvector).
 *
 * The adapter accepts any pool/client whose `.query()` method is compatible
 * with the `pg` package — this includes:
 *  • `pg.Pool` / `pg.Client`      (https://node-postgres.com)
 *  • `@neondatabase/serverless`   (https://github.com/neondatabase/serverless)
 *
 * Required schema
 * ───────────────
 * Before using this adapter, run the following SQL once against your database:
 *
 *   -- 1. Enable the pgvector extension (requires pgvector >= 0.5.0):
 *   CREATE EXTENSION IF NOT EXISTS vector;
 *
 *   -- 2. Create the cache table. Adjust VECTOR(384) to match your embedding
 *   --    model's dimensionality (384 for all-MiniLM-L6-v2, 1536 for
 *   --    text-embedding-ada-002, etc.):
 *   CREATE TABLE IF NOT EXISTS semantic_cache (
 *     id          TEXT PRIMARY KEY,
 *     embedding   VECTOR(384)  NOT NULL,
 *     metadata    JSONB        NOT NULL DEFAULT '{}'
 *   );
 *
 *   -- 3. Create an HNSW index for fast approximate nearest-neighbour search
 *   --    using cosine distance (`vector_cosine_ops`).  Replace 384 with your
 *   --    actual vector dimension if different:
 *   CREATE INDEX IF NOT EXISTS semantic_cache_embedding_idx
 *     ON semantic_cache
 *     USING hnsw (embedding vector_cosine_ops);
 *
 * Failure policy
 * ──────────────
 *  • Standard vector-store methods (`upsert`, `query`, `delete`) throw a
 *    descriptive error when the SQL query fails, so callers can react to
 *    backend outages.
 *  • Cache-facing methods (`search`, `save`) catch every error and degrade
 *    gracefully (search → null, save → no-op) so a cache outage never takes
 *    down the primary LLM call path.
 *
 * Dependencies:
 *   npm i pg              # node-postgres
 *   npm i -D @types/pg
 *   -- OR --
 *   npm i @neondatabase/serverless
 */

import type {
  VectorMetadataFilter,
  VectorStoreAdapter,
  VectorMetadata,
  VectorQueryMatch,
} from "./vector-store-adapter";
import { getAllowedMetadataFilterEntries } from "./metadata-filter";

// ---------------------------------------------------------------------------
// PgPoolLike — minimal interface compatible with pg.Pool / pg.Client and
// @neondatabase/serverless Pool.
// ---------------------------------------------------------------------------

/**
 * Minimal Postgres pool/client interface required by {@link PgVectorAdapter}.
 *
 * Both `pg.Pool` and `@neondatabase/serverless` satisfy this shape out of
 * the box — no wrapper is needed.
 *
 * @example — pg
 * ```ts
 * import { Pool } from "pg";
 * const pool = new Pool({ connectionString: process.env.DATABASE_URL });
 * const adapter = new PgVectorAdapter({ pool });
 * ```
 *
 * @example — @neondatabase/serverless
 * ```ts
 * import { Pool } from "@neondatabase/serverless";
 * const pool = new Pool({ connectionString: process.env.DATABASE_URL });
 * const adapter = new PgVectorAdapter({ pool });
 * ```
 */
export interface PgPoolLike {
  query<R extends Record<string, unknown> = Record<string, unknown>>(
    text: string,
    values?: unknown[]
  ): Promise<{ rows: R[] }>;
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

export interface PgVectorAdapterOptions {
  /**
   * A configured `pg.Pool`, `pg.Client`, or `@neondatabase/serverless` Pool.
   * The pool is used for every SQL operation; connection management (pooling,
   * retries) is the caller's responsibility.
   */
  pool: PgPoolLike;

  /**
   * Name of the table used to store cache entries.
   * Must match the table created by the required schema migration.
   * @default "semantic_cache"
   */
  tableName?: string;
}

/** Metadata key under which `save()` stores the cached LLM response. */
const RESPONSE_METADATA_KEY = "response" as const;
function buildMetadataFilterWhereClause(
  filter: VectorMetadataFilter | undefined,
  startingParameterIndex: number
): { clause: string; values: string[] } {
  if (!filter) {
    return { clause: "", values: [] };
  }

  const entries = getAllowedMetadataFilterEntries(filter);

  if (entries.length === 0) {
    return { clause: "", values: [] };
  }

  return {
    clause:
      "WHERE " +
      entries
        .map(
          ([key], index) =>
            key === "userId"
              ? `metadata ->> 'userId' = $${startingParameterIndex + index}`
              : `metadata ->> 'tenantId' = $${startingParameterIndex + index}`
        )
        .join(" AND "),
    values: entries.map(([, value]) => value),
  };
}

// ---------------------------------------------------------------------------
// PgVectorAdapter
// ---------------------------------------------------------------------------

/**
 * A {@link VectorStoreAdapter} backed by PostgreSQL with the pgvector
 * extension, using the cosine distance operator (`<=>`) for similarity
 * search.
 *
 * Cosine *distance* values returned by pgvector are in [0, 2] (0 = identical,
 * 2 = opposite).  This adapter converts them to cosine *similarity* in [0, 1]
 * via `similarity = 1 − distance`, clamped to [0, 1], as required by the
 * {@link VectorStoreAdapter} contract.
 *
 * @example
 * ```ts
 * import { Pool } from "pg";
 * import { PgVectorAdapter } from "next-semantic-cache/pgvector-adapter";
 *
 * const pool = new Pool({ connectionString: process.env.DATABASE_URL });
 * const adapter = new PgVectorAdapter({ pool });
 * ```
 */
export class PgVectorAdapter implements VectorStoreAdapter {
  private readonly pool: PgPoolLike;
  private readonly tableName: string;

  constructor(options: PgVectorAdapterOptions) {
    if (!options.pool) {
      throw new Error("PgVectorAdapter: `pool` is required.");
    }
    this.pool = options.pool;
    this.tableName = options.tableName ?? "semantic_cache";
  }

  // =========================================================================
  // VectorStoreAdapter — standard methods (upsert / query / delete)
  // =========================================================================

  /**
   * Inserts a vector under `id`, replacing any existing entry (idempotent).
   *
   * The vector is supplied as a pgvector array-literal string (e.g.
   * `"[0.1,0.2,0.3]"`) and cast to `VECTOR` via the `::vector` syntax.
   * Metadata is stored as `JSONB`.
   */
  async upsert(
    id: string,
    vector: number[],
    metadata: VectorMetadata
  ): Promise<void> {
    // pgvector expects vectors as '[x,y,z,...]' array-literal strings.
    // The ::vector cast converts this to the internal VECTOR representation.
    await this.pool.query(
      `INSERT INTO ${this.tableName} (id, embedding, metadata)
       VALUES ($1, $2::vector, $3::jsonb)
       ON CONFLICT (id) DO UPDATE
         SET embedding = EXCLUDED.embedding,
             metadata  = EXCLUDED.metadata`,
      [id, `[${vector.join(",")}]`, JSON.stringify(metadata ?? {})]
    );
  }

  /**
   * Returns the `topK` nearest entries to `vector`, ordered by ascending
   * cosine distance (i.e. descending cosine similarity).
   *
   * The `<=>` operator is pgvector's cosine distance operator, producing
   * distances in [0, 2].  Scores are reported as `1 − distance`, clamped
   * to [0, 1].
   */
  async query(
    vector: number[],
    topK: number,
    filter?: VectorMetadataFilter
  ): Promise<VectorQueryMatch[]> {
    const metadataFilter = buildMetadataFilterWhereClause(filter, 2);
    // `embedding <=> $1::vector` computes the cosine distance between the
    // stored embedding and the query vector.  Lower distance = more similar.
    // `1 - (embedding <=> ...)` converts distance to similarity in [0, 1].
    const { rows } = await this.pool.query<{
      id: string;
      score: string | number;
      metadata: VectorMetadata;
    }>(
      `SELECT id,
              1 - (embedding <=> $1::vector) AS score,
              metadata
       FROM ${this.tableName}
      ${metadataFilter.clause}
       ORDER BY embedding <=> $1::vector
       LIMIT $${metadataFilter.values.length + 2}`,
      [
        `[${vector.join(",")}]`,
        ...metadataFilter.values,
        Math.max(1, Math.floor(topK)),
      ]
    );

    return rows.map((row) => ({
      id: row.id,
      // pg returns numeric columns as strings; coerce and clamp to [0, 1].
      score: Math.min(1, Math.max(0, Number(row.score))),
      metadata: (row.metadata as VectorMetadata) ?? {},
    }));
  }

  /** Removes the entry stored under `id`. A no-op when no such entry exists. */
  async delete(id: string): Promise<void> {
    await this.pool.query(
      `DELETE FROM ${this.tableName} WHERE id = $1`,
      [id]
    );
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
        "[PgVectorAdapter] search() failed (falling back to LLM):",
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
      await this.upsert(crypto.randomUUID(), promptVector, {
        ...metadata,
        [RESPONSE_METADATA_KEY]: response,
        createdAt: new Date().toISOString(),
      });
    } catch (err: unknown) {
      console.error(
        "[PgVectorAdapter] save() failed (cache write skipped):",
        err instanceof Error ? err.message : err
      );
    }
  }
}
