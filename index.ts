/**
 * next-semantic-cache — index.ts
 *
 * Public API surface for the package.
 *
 * The single exported helper, `withSemanticCache`, is a thin convenience
 * wrapper around `wrapLanguageModel` from the Vercel AI SDK.  It handles
 * all middleware wiring so callers can go from zero to cached in one line.
 *
 * Usage:
 *   import { withSemanticCache } from "next-semantic-cache";
 *   import { RedisVectorAdapter }  from "next-semantic-cache/adapters/redis";
 *
 *   const model = withSemanticCache(openai("gpt-4o"), {
 *     adapter:   new RedisVectorAdapter({ ... }),
 *     threshold: 0.92,
 *   });
 */

import { wrapLanguageModel, type LanguageModel } from "ai";
import {
  SemanticCacheMiddleware,
  type VectorStoreAdapter,
  type SemanticCacheMiddlewareOptions,
} from "./semantic-cache-middleware";

// ---------------------------------------------------------------------------
// Re-exports — consumers import everything from one place
// ---------------------------------------------------------------------------

export type { VectorStoreAdapter, SemanticCacheMiddlewareOptions };
export type { VectorMetadata, VectorQueryMatch } from "./vector-store-adapter";
export { SemanticCacheMiddleware };
export { RedisVectorAdapter } from "./redis-vector-adapter";
export { UpstashVectorAdapter } from "./upstash-vector-adapter";
export type { UpstashVectorAdapterOptions } from "./upstash-vector-adapter";
export { PineconeVectorAdapter } from "./pinecone-vector-adapter";
export type { PineconeVectorAdapterOptions } from "./pinecone-vector-adapter";

// Next.js native-cache integration helpers (safe to import outside Next.js).
export {
  SEMANTIC_CACHE_TAG,
  buildCacheTags,
  isNextCacheAvailable,
  withNextCache,
  revalidateNextTag,
} from "./next-cache";

// ---------------------------------------------------------------------------
// withSemanticCache options
// ---------------------------------------------------------------------------

/**
 * Configuration for {@link withSemanticCache}.
 */
export interface SemanticCacheOptions {
  /**
   * A concrete {@link VectorStoreAdapter} implementation.
   *
   * The package ships with {@link RedisVectorAdapter} for Upstash / Redis
   * Stack out of the box.  You can also supply your own implementation —
   * any class that satisfies the interface will work.
   */
  adapter: VectorStoreAdapter;

  /**
   * Cosine-similarity threshold in [0, 1].
   *
   * Prompts whose nearest cached neighbour exceeds this similarity are
   * served from cache.  Tune this for your use-case:
   *   • 0.95 — very strict; only near-identical phrasings hit the cache.
   *   • 0.90 — balanced; catches paraphrases and minor rewording.
   *   • 0.85 — aggressive; may serve slightly off-topic cached answers.
   *
   * @default 0.92
   */
  threshold?: number;

  /**
   * Emit cache hit/miss debug logs to `console.debug`.
   * Safe to enable in development; disable in production.
   * @default false
   */
  debug?: boolean;
}

// ---------------------------------------------------------------------------
// withSemanticCache
// ---------------------------------------------------------------------------

/**
 * Wraps any Vercel AI SDK {@link LanguageModelV1} with local semantic caching.
 *
 * Embeddings are computed **locally** using `all-MiniLM-L6-v2` via
 * `@huggingface/transformers` — no third-party embedding API is called.
 * Cache lookups return in ~50 ms; LLM calls are made only on a true miss.
 *
 * @param model   - Any Vercel AI SDK model (OpenAI, Anthropic, Gemini, …).
 * @param options - Adapter, similarity threshold, and debug flag.
 * @returns       The same model type, now augmented with semantic caching.
 *
 * @example — Next.js Edge route
 * ```ts
 * import { withSemanticCache, RedisVectorAdapter } from "next-semantic-cache";
 * import { openai }    from "@ai-sdk/openai";
 * import { streamText } from "ai";
 *
 * const adapter = new RedisVectorAdapter({
 *   redisUrl:   process.env.UPSTASH_REDIS_REST_URL!,
 *   redisToken: process.env.UPSTASH_REDIS_REST_TOKEN!,
 *   ttlSeconds: 86_400,
 * });
 *
 * // Build the cached model once — reused across requests on warm lambdas.
 * const cachedModel = withSemanticCache(openai("gpt-4o"), {
 *   adapter,
 *   threshold: 0.92,
 *   debug: process.env.NODE_ENV === "development",
 * });
 *
 * export const runtime = "edge";
 *
 * export async function POST(req: Request) {
 *   const { prompt } = await req.json();
 *   const result = streamText({ model: cachedModel, prompt });
 *   return result.toDataStreamResponse();
 * }
 * ```
 */
export function withSemanticCache<M extends LanguageModel>(
  model: M,
  options: SemanticCacheOptions
): M {
  const { adapter, threshold = 0.92, debug = false } = options;

  // Build the middleware with the caller-supplied adapter and threshold.
  const middleware = SemanticCacheMiddleware({
    vectorStore: adapter,
    similarityThreshold: threshold,
    debug,
  });

  // wrapLanguageModel injects the middleware into the model's call chain.
  // The returned value is structurally identical to the original model and
  // can be passed to any Vercel AI SDK function (streamText, generateText …).
  return wrapLanguageModel({ model: model as any, middleware }) as M;
}
