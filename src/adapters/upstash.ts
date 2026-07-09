/**
 * upstash-vector-adapter.ts
 *
 * Concrete implementation of {@link VectorStoreAdapter} backed by
 * Upstash Vector (https://upstash.com/docs/vector).
 *
 * 100% Edge-compatible by design
 * ──────────────────────────────
 * This adapter deliberately does NOT use the official `@upstash/vector`
 * Node.js SDK. All operations are issued directly against the Upstash
 * Vector REST API using the Web-standard `fetch` API, so the adapter runs
 * unchanged on the Next.js Edge Runtime (Vercel Edge functions), Node.js,
 * and any other JavaScript runtime with `fetch` support.
 *
 * REST endpoints used
 * ───────────────────
 *   POST {url}/upsert[/{namespace}]  — insert or replace a vector
 *   POST {url}/query[/{namespace}]   — KNN similarity search
 *   POST {url}/delete[/{namespace}]  — delete vectors by id
 *
 * Failure policy
 * ──────────────
 *  • Standard vector-store methods (`upsert`, `query`, `delete`) throw a
 *    descriptive error when the HTTP request fails, so id-addressable
 *    callers can react to backend outages.
 *  • Cache-facing methods (`search`, `save`) — used internally by
 *    `SemanticCacheMiddleware` — catch every error and degrade gracefully
 *    (search → null, save → no-op) so a cache outage never takes down the
 *    primary LLM call path.
 */

import type {
  VectorMetadataFilter,
  VectorStoreAdapter,
  VectorMetadata,
  VectorQueryMatch,
} from "./vector-store-adapter";
import { getAllowedMetadataFilterEntries } from "./metadata-filter";

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

export interface UpstashVectorAdapterOptions {
  /**
   * Upstash Vector REST endpoint URL.
   * Example: "https://your-index-region-vector.upstash.io"
   */
  url: string;

  /**
   * Upstash Vector REST API token
   * (`UPSTASH_VECTOR_REST_TOKEN` in the Upstash console).
   */
  token: string;

  /**
   * Optional Upstash Vector namespace to scope all operations to.
   * When omitted, the default namespace is used.
   */
  namespace?: string;

  /**
   * Maximum number of milliseconds to wait for an HTTP request before
   * aborting it and treating the operation as failed.
   * @default 2000
   */
  requestTimeoutMs?: number;
}

// ---------------------------------------------------------------------------
// Internal response shapes (subset of the Upstash Vector REST API)
// ---------------------------------------------------------------------------

interface UpstashQueryMatch {
  id: string | number;
  score: number;
  metadata?: VectorMetadata;
}

interface UpstashQueryResponse {
  result?: UpstashQueryMatch[];
}

/** Metadata key under which `save()` stores the cached LLM response. */
const RESPONSE_METADATA_KEY = "response" as const;

function escapeUpstashFilterValue(value: string): string {
  return value.replace(/\\/g, "\\\\").replace(/'/g, "\\'");
}

function toUpstashMetadataFilter(
  filter?: VectorMetadataFilter
): string | undefined {
  if (!filter) return undefined;

  const clauses = getAllowedMetadataFilterEntries(filter)
    .map(([key, value]) => `${key} = '${escapeUpstashFilterValue(value)}'`);

  return clauses.length > 0 ? clauses.join(" AND ") : undefined;
}

// ---------------------------------------------------------------------------
// UpstashVectorAdapter
// ---------------------------------------------------------------------------

/**
 * A {@link VectorStoreAdapter} backed by Upstash Vector over its REST API.
 *
 * @example
 * ```ts
 * const adapter = new UpstashVectorAdapter({
 *   url:       process.env.UPSTASH_VECTOR_REST_URL!,
 *   token:     process.env.UPSTASH_VECTOR_REST_TOKEN!,
 *   namespace: "my-app",
 * });
 * ```
 */
export class UpstashVectorAdapter implements VectorStoreAdapter {
  private readonly url: string;
  private readonly token: string;
  private readonly namespace?: string;
  private readonly timeoutMs: number;

  constructor(options: UpstashVectorAdapterOptions) {
    const { url, token, namespace, requestTimeoutMs = 2000 } = options;

    if (!url || !token) {
      throw new Error(
        "UpstashVectorAdapter: both `url` and `token` are required."
      );
    }

    // Strip trailing slashes (loop avoids regex backtracking concerns).
    let normalizedUrl = url;
    while (normalizedUrl.endsWith("/")) {
      normalizedUrl = normalizedUrl.slice(0, -1);
    }
    this.url = normalizedUrl;
    this.token = token;
    this.namespace = namespace;
    this.timeoutMs = requestTimeoutMs;
  }

  // =========================================================================
  // HTTP plumbing
  // =========================================================================

  /**
   * Issues a POST request to the Upstash Vector REST API and returns the
   * parsed JSON response.
   *
   * Throws a descriptive error when the request times out, the network
   * fails, or the API responds with a non-2xx status.
   */
  private async request<T>(path: string, body: unknown): Promise<T> {
    const namespaceSuffix = this.namespace
      ? `/${encodeURIComponent(this.namespace)}`
      : "";
    const endpoint = `${this.url}${path}${namespaceSuffix}`;

    let response: Response;
    try {
      response = await fetch(endpoint, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${this.token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(this.timeoutMs),
      });
    } catch (err: unknown) {
      const reason = err instanceof Error ? err.message : String(err);
      throw new Error(
        `UpstashVectorAdapter: request to ${path} failed before receiving a response: ${reason}`
      );
    }

    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(
        `UpstashVectorAdapter: request to ${path} failed with HTTP ${response.status} ${response.statusText}` +
          (detail ? ` — ${detail}` : "")
      );
    }

    return (await response.json()) as T;
  }

  /**
   * Converts an Upstash similarity score to raw cosine similarity in [0, 1].
   *
   * Upstash normalizes cosine scores to `(1 + cosine) / 2`, so the raw
   * cosine similarity is recovered via `2 * score − 1` and clamped to [0, 1]
   * as required by the {@link VectorStoreAdapter} contract.
   */
  private static toCosineSimilarity(score: number): number {
    return Math.min(1, Math.max(0, 2 * score - 1));
  }

  // =========================================================================
  // VectorStoreAdapter — standard methods (upsert / query / delete)
  // =========================================================================

  /** Inserts a vector under `id`, replacing any existing entry (idempotent). */
  async upsert(
    id: string,
    vector: number[],
    metadata: VectorMetadata
  ): Promise<void> {
    await this.request("/upsert", {
      id,
      vector,
      metadata: metadata ?? {},
    });
  }

  /**
   * Returns the `topK` nearest entries to `vector`, ordered by descending
   * cosine similarity.
   */
  async query(
    vector: number[],
    topK: number,
    filter?: VectorMetadataFilter
  ): Promise<VectorQueryMatch[]> {
    const metadataFilter = toUpstashMetadataFilter(filter);
    const data = await this.request<UpstashQueryResponse>("/query", {
      vector,
      topK: Math.max(1, Math.floor(topK)),
      includeMetadata: true,
      ...(metadataFilter !== undefined ? { filter: metadataFilter } : {}),
    });

    return (data.result ?? []).map((match) => ({
      id: String(match.id),
      score: UpstashVectorAdapter.toCosineSimilarity(match.score),
      metadata: match.metadata ?? {},
    }));
  }

  /** Removes the entry stored under `id`. A no-op when no such entry exists. */
  async delete(id: string): Promise<void> {
    await this.request("/delete", { ids: [id] });
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
        "[UpstashVectorAdapter] search() failed (falling back to LLM):",
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
        "[UpstashVectorAdapter] save() failed (cache write skipped):",
        err instanceof Error ? err.message : err
      );
    }
  }
}
