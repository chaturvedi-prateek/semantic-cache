/**
 * pinecone-vector-adapter.ts
 *
 * Concrete implementation of {@link VectorStoreAdapter} backed by
 * Pinecone (https://docs.pinecone.io).
 *
 * 100% Edge-compatible by design
 * ──────────────────────────────
 * This adapter deliberately does NOT use the official `@pinecone-database/
 * pinecone` Node.js SDK. All operations are issued directly against the
 * Pinecone data-plane REST API using the Web-standard `fetch` API, so the
 * adapter runs unchanged on the Next.js Edge Runtime (Vercel Edge
 * functions), Node.js, and any other JavaScript runtime with `fetch`
 * support.
 *
 * REST endpoints used (data plane, addressed via the index host)
 * ──────────────────────────────────────────────────────────────
 *   POST https://{indexHost}/vectors/upsert  — insert or replace vectors
 *   POST https://{indexHost}/query           — KNN similarity search
 *   POST https://{indexHost}/vectors/delete  — delete vectors by id
 *
 * The index host is shown in the Pinecone console (or returned by the
 * `describe_index` control-plane call), e.g.
 * `my-index-abc1234.svc.us-east-1-aws.pinecone.io`.
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

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

export interface PineconeVectorAdapterOptions {
  /**
   * Pinecone API key (`PINECONE_API_KEY`).
   */
  apiKey: string;

  /**
   * The index host shown in the Pinecone console for your index.
   * Accepted with or without the `https://` scheme.
   * Example: "my-index-abc1234.svc.us-east-1-aws.pinecone.io"
   */
  indexHost: string;

  /**
   * Optional Pinecone namespace to scope all operations to.
   * When omitted, the default ("") namespace is used.
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
// Internal response shapes (subset of the Pinecone data-plane API)
// ---------------------------------------------------------------------------

interface PineconeQueryMatch {
  id: string;
  score: number;
  metadata?: VectorMetadata;
}

interface PineconeQueryResponse {
  matches?: PineconeQueryMatch[];
}

/** Metadata key under which `save()` stores the cached LLM response. */
const RESPONSE_METADATA_KEY = "response" as const;

function toPineconeMetadataFilter(
  filter?: VectorMetadataFilter
): Record<string, { $eq: string }> | undefined {
  if (!filter) return undefined;

  const entries = Object.entries(filter).filter(
    (entry): entry is [string, string] => typeof entry[1] === "string"
  );

  if (entries.length === 0) return undefined;

  return Object.fromEntries(
    entries.map(([key, value]) => [key, { $eq: value }])
  );
}

// ---------------------------------------------------------------------------
// PineconeVectorAdapter
// ---------------------------------------------------------------------------

/**
 * A {@link VectorStoreAdapter} backed by Pinecone over its data-plane
 * REST API.
 *
 * The Pinecone index should be created with the `cosine` metric so that
 * match scores are cosine similarities, as required by the
 * {@link VectorStoreAdapter} contract.
 *
 * @example
 * ```ts
 * const adapter = new PineconeVectorAdapter({
 *   apiKey:    process.env.PINECONE_API_KEY!,
 *   indexHost: process.env.PINECONE_INDEX_HOST!,
 *   namespace: "my-app",
 * });
 * ```
 */
export class PineconeVectorAdapter implements VectorStoreAdapter {
  private readonly apiKey: string;
  private readonly baseUrl: string;
  private readonly namespace?: string;
  private readonly timeoutMs: number;

  constructor(options: PineconeVectorAdapterOptions) {
    const { apiKey, indexHost, namespace, requestTimeoutMs = 2000 } = options;

    if (!apiKey || !indexHost) {
      throw new Error(
        "PineconeVectorAdapter: both `apiKey` and `indexHost` are required."
      );
    }

    // Normalize the host into a base URL: strip any scheme the caller may
    // have included plus trailing slashes, then force https.
    let host = indexHost.replace(/^https?:\/\//, "");
    while (host.endsWith("/")) {
      host = host.slice(0, -1);
    }
    this.baseUrl = `https://${host}`;
    this.apiKey = apiKey;
    this.namespace = namespace;
    this.timeoutMs = requestTimeoutMs;
  }

  // =========================================================================
  // HTTP plumbing
  // =========================================================================

  /**
   * Issues a POST request to the Pinecone data-plane REST API and returns
   * the parsed JSON response.
   *
   * Throws a descriptive error when the request times out, the network
   * fails, or the API responds with a non-2xx status.
   */
  private async request<T>(path: string, body: unknown): Promise<T> {
    const endpoint = `${this.baseUrl}${path}`;

    let response: Response;
    try {
      response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Api-Key": this.apiKey,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(this.timeoutMs),
      });
    } catch (err: unknown) {
      const reason = err instanceof Error ? err.message : String(err);
      throw new Error(
        `PineconeVectorAdapter: request to ${path} failed before receiving a response: ${reason}`
      );
    }

    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(
        `PineconeVectorAdapter: request to ${path} failed with HTTP ${response.status} ${response.statusText}` +
          (detail ? ` — ${detail}` : "")
      );
    }

    return (await response.json()) as T;
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
    await this.request("/vectors/upsert", {
      vectors: [{ id, values: vector, metadata: metadata ?? {} }],
      ...(this.namespace ? { namespace: this.namespace } : {}),
    });
  }

  /**
   * Returns the `topK` nearest entries to `vector`, ordered by descending
   * cosine similarity. Scores are clamped to [0, 1] as required by the
   * {@link VectorStoreAdapter} contract (cosine scores can be negative for
   * opposing vectors).
   */
  async query(
    vector: number[],
    topK: number,
    filter?: VectorMetadataFilter
  ): Promise<VectorQueryMatch[]> {
    const data = await this.request<PineconeQueryResponse>("/query", {
      vector,
      topK: Math.max(1, Math.floor(topK)),
      includeMetadata: true,
      ...(toPineconeMetadataFilter(filter)
        ? { filter: toPineconeMetadataFilter(filter) }
        : {}),
      ...(this.namespace ? { namespace: this.namespace } : {}),
    });

    return (data.matches ?? []).map((match) => ({
      id: match.id,
      score: Math.min(1, Math.max(0, match.score)),
      metadata: match.metadata ?? {},
    }));
  }

  /** Removes the entry stored under `id`. A no-op when no such entry exists. */
  async delete(id: string): Promise<void> {
    await this.request("/vectors/delete", {
      ids: [id],
      ...(this.namespace ? { namespace: this.namespace } : {}),
    });
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
        "[PineconeVectorAdapter] search() failed (falling back to LLM):",
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
        "[PineconeVectorAdapter] save() failed (cache write skipped):",
        err instanceof Error ? err.message : err
      );
    }
  }
}
