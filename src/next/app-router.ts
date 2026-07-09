/**
 * next-app-router.ts
 *
 * A thin, strictly-typed helper for the **Next.js App Router**.
 *
 * `withSemanticCache` wraps a Route Handler (the functions you export from
 * `app/api/.../route.ts`, e.g. `POST`) so that semantically similar requests
 * are served from a vector cache instead of hitting your LLM provider on every
 * call.
 *
 * Unlike the model-level `withSemanticCache` exported from the package root —
 * which wraps a Vercel AI SDK `LanguageModel` — this helper operates at the
 * HTTP boundary. It takes a standard web `Request`, decides hit vs. miss, and
 * returns a `NextResponse`. That makes it framework-native and provider
 * agnostic: your fallback can call OpenAI, Anthropic, a local model, or any
 * async function that ultimately produces text.
 *
 * Pipeline:
 *   1. Extract the prompt text from the incoming `Request`.
 *   2. Embed the prompt (locally, via the same MiniLM pipeline the middleware
 *      uses) and probe the vector store.
 *   3. Cache HIT  → return a `NextResponse` immediately; the LLM is never called.
 *   4. Cache MISS → run the caller's `fallback`, persist the result, and return
 *      a `NextResponse`.
 *
 * Usage (app/api/chat/route.ts):
 * ```ts
 * import { withSemanticCache } from "next-semantic-cache/next";
 * import { RedisVectorAdapter } from "next-semantic-cache/adapters/redis";
 * import { generateText } from "ai";
 * import { openai } from "@ai-sdk/openai";
 *
 * const adapter = new RedisVectorAdapter({
 *   redisUrl:   process.env.UPSTASH_REDIS_REST_URL!,
 *   redisToken: process.env.UPSTASH_REDIS_REST_TOKEN!,
 * });
 *
 * export const POST = withSemanticCache({
 *   adapter,
 *   threshold: 0.92,
 *   namespace: "chat",
 *   extractPrompt: async (req) => (await req.json()).prompt,
 *   fallback: async (prompt) => {
 *     const { text } = await generateText({ model: openai("gpt-4o"), prompt });
 *     return text;
 *   },
 * });
 * ```
 */

import { NextResponse } from "next/server";
import {
  generateEmbedding,
  type VectorStoreAdapter,
} from "../semantic-cache-middleware";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/**
 * Extracts the prompt text to embed from the incoming request.
 *
 * The default implementation reads a JSON body of the shape `{ prompt: string }`
 * (falling back to a `messages` array's last user turn). Supply your own to
 * support different payload shapes, query parameters, headers, etc.
 *
 * Note: reading a request body consumes it. The default extractor clones the
 * request before reading so your `fallback` still receives an unconsumed body.
 * If you provide a custom extractor that reads the body, clone first via
 * `request.clone()` if `fallback` also needs to read it.
 */
export type PromptExtractor = (request: Request) => string | Promise<string>;

/**
 * The LLM (or any async producer) invoked on a cache miss.
 *
 * @param prompt  - The prompt text extracted from the request.
 * @param request - The original, unmodified web `Request`.
 * @returns The text to cache and return to the client.
 */
export type LlmFallback = (
  prompt: string,
  request: Request
) => string | Promise<string>;

/**
 * Configuration for {@link withSemanticCache}.
 */
export interface WithSemanticCacheConfig {
  /**
   * A concrete {@link VectorStoreAdapter} implementation (e.g.
   * `RedisVectorAdapter`, or your own). Used for both the cache lookup and
   * the write-back on a miss.
   */
  adapter: VectorStoreAdapter;

  /**
   * The LLM fallback executed when nothing similar is cached.
   * Its returned text is both cached and returned to the client.
   */
  fallback: LlmFallback;

  /**
   * Optional custom prompt-extraction logic.
   * @default reads `{ prompt }` (or the last user message of `{ messages }`)
   *          from the JSON body of a clone of the request.
   */
  extractPrompt?: PromptExtractor;

  /**
   * Optional logical partition for this handler's cache entries.
   *
   * The adapter interface is intentionally namespace-agnostic, so the
   * namespace is folded into the embedded text as a stable prefix. Different
   * namespaces therefore produce different embeddings and never collide,
   * effectively partitioning the cache per handler/tenant.
   */
  namespace?: string;

  /**
   * Cosine-similarity threshold in [0, 1]. Prompts whose nearest cached
   * neighbour exceeds this similarity are served from cache.
   * @default 0.92
   */
  threshold?: number;

  /**
   * Emit cache hit/miss debug logs to `console.debug`.
   * @default false
   */
  debug?: boolean;
}

/**
 * The shape of the JSON body returned by the wrapped handler.
 */
export interface SemanticCacheResponseBody {
  /** The LLM (or cached) response text. */
  response: string;
  /** `true` when served from the semantic cache, `false` on a live LLM call. */
  cached: boolean;
}

/**
 * A Next.js App Router Route Handler: takes a web `Request`, returns a
 * `NextResponse`. This is the exact signature `withSemanticCache` produces,
 * so its result can be exported directly as `GET`/`POST`/etc.
 */
export type RouteHandler = (request: Request) => Promise<NextResponse>;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Default prompt extractor. Reads a JSON body of `{ prompt }`, or the last
 * user message from a `{ messages: [{ role, content }] }` array. Operates on a
 * clone so the original request body remains readable by `fallback`.
 */
async function defaultExtractPrompt(request: Request): Promise<string> {
  let body: unknown;
  try {
    body = await request.clone().json();
  } catch {
    // Non-JSON or empty body — nothing to embed.
    return "";
  }

  if (!body || typeof body !== "object") {
    return "";
  }

  const record = body as Record<string, unknown>;

  if (typeof record.prompt === "string") {
    return record.prompt;
  }

  if (Array.isArray(record.messages)) {
    const lastUser = [...record.messages]
      .reverse()
      .find(
        (m): m is { role: string; content: string } =>
          !!m &&
          typeof m === "object" &&
          (m as { role?: unknown }).role === "user" &&
          typeof (m as { content?: unknown }).content === "string"
      );
    if (lastUser) {
      return lastUser.content;
    }
  }

  return "";
}

/**
 * Folds an optional namespace into the text used for embedding so that entries
 * from different namespaces never collide in a shared vector store.
 */
function applyNamespace(promptText: string, namespace?: string): string {
  return namespace ? `[${namespace}] ${promptText}` : promptText;
}

// ---------------------------------------------------------------------------
// withSemanticCache
// ---------------------------------------------------------------------------

/**
 * Wraps a Next.js App Router Route Handler with semantic caching.
 *
 * Checks the semantic cache first; on a sufficiently similar hit it returns a
 * {@link NextResponse} immediately. On a miss it runs the provided `fallback`,
 * persists the result, and returns a {@link NextResponse}.
 *
 * @param config - Adapter, LLM fallback, and optional prompt extractor,
 *                 namespace, similarity threshold, and debug flag.
 * @returns A {@link RouteHandler} suitable for `export const POST = ...`.
 */
export function withSemanticCache(config: WithSemanticCacheConfig): RouteHandler {
  const {
    adapter,
    fallback,
    extractPrompt = defaultExtractPrompt,
    namespace,
    threshold = 0.92,
    debug = false,
  } = config;

  const log = (...args: unknown[]): void => {
    if (debug) console.debug("[SemanticCache:route]", ...args);
  };

  return async function handler(request: Request): Promise<NextResponse> {
    const promptText = await extractPrompt(request);

    // No prompt to embed → skip the cache entirely and run the fallback.
    if (!promptText) {
      log("Empty prompt — bypassing cache.");
      const response = await fallback(promptText, request);
      return NextResponse.json<SemanticCacheResponseBody>({
        response,
        cached: false,
      });
    }

    const keyText = applyNamespace(promptText, namespace);
    const promptVector = await generateEmbedding(keyText);

    // ── Cache lookup ──────────────────────────────────────────────────────
    const cached = await adapter.search(promptVector, threshold);
    if (cached !== null) {
      log(`Cache HIT  — "${promptText.slice(0, 72)}…"`);
      return NextResponse.json<SemanticCacheResponseBody>({
        response: cached,
        cached: true,
      });
    }

    // ── Cache MISS → run the LLM fallback ─────────────────────────────────
    log(`Cache MISS — "${promptText.slice(0, 72)}…"`);
    const response = await fallback(promptText, request);

    // Persist the fresh result. Caching is best-effort: a write failure must
    // never fail the response that was successfully produced.
    if (response) {
      try {
        await adapter.save(promptVector, response);
        log(`Saved embedding — "${promptText.slice(0, 72)}…"`);
      } catch (err: unknown) {
        console.error(
          "[SemanticCache:route] Failed to persist to vector store:",
          err instanceof Error ? err.message : err
        );
      }
    }

    return NextResponse.json<SemanticCacheResponseBody>({
      response,
      cached: false,
    });
  };
}
