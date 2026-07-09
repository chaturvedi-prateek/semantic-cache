/**
 * semantic-cache-middleware.ts
 *
 * A Vercel AI SDK LanguageModelV4Middleware that short-circuits LLM calls
 * by performing semantic similarity lookups against a pluggable vector store.
 *
 * Pipeline:
 *   1. transformParams  — embed the incoming prompt.
 *   2. Cache hit        — return a synthetic LanguageModelV4GenerateResult immediately.
 *   3. Cache miss       — fall through to wrapGenerate; store result asynchronously.
 *
 * Dependencies:
 *   npm i @huggingface/transformers ai
 *   npm i -D @types/node
 */

import {
  type LanguageModelMiddleware,
  type LanguageModelCallOptions,
} from "ai";

// LanguageModelV4GenerateResult is not directly exported; we define it locally
type LanguageModelV4GenerateResult = {
  rawCall: { rawPrompt: unknown; rawSettings: Record<string, unknown> };
  finishReason: string;
  usage: {
    promptTokens?: number;
    completionTokens?: number;
    inputTokens?: {
      total: number;
      noCache?: number;
      cacheRead?: number;
      cacheWrite?: number;
    };
    outputTokens?: {
      total: number;
      text?: number;
      reasoning?: number;
    };
  };
  text: string;
  content: Array<{ type: "text"; text: string }>;
  warnings?: unknown[];
};

// ---------------------------------------------------------------------------
// Re-export alias so callers can reference the versioned name.
// The Vercel AI SDK surface currently exports LanguageModelMiddleware;
// swap to LanguageModelV4Middleware once the SDK stabilises that name.
// ---------------------------------------------------------------------------
export type LanguageModelV4Middleware = LanguageModelMiddleware;

// ---------------------------------------------------------------------------
// 1. VectorStoreAdapter — pluggable persistence layer
// ---------------------------------------------------------------------------

// The canonical interface now lives in ./vector-store-adapter — re-exported
// here so existing `import { VectorStoreAdapter } from "./semantic-cache-middleware"`
// call-sites keep working unchanged.
export type {
  VectorStoreAdapter,
  VectorMetadata,
  VectorMetadataFilter,
  VectorQueryMatch,
} from "./vector-store-adapter";
import type {
  VectorMetadata,
  VectorMetadataFilter,
  VectorStoreAdapter,
} from "./vector-store-adapter";

// ---------------------------------------------------------------------------
// 2. Singleton embedding model
// ---------------------------------------------------------------------------

// Internal type alias for the HuggingFace pipeline instance.
type EmbeddingPipeline = (
  text: string,
  opts: Record<string, unknown>
) => Promise<{ data: Float32Array; dims: number[] }>;

/**
 * Internal singleton state — one pipeline instance shared across all
 * middleware invocations in the same Node.js process / serverless warm start.
 */
let _embeddingPipeline: EmbeddingPipeline | null = null;

/** Name of the SBERT-compatible model used for prompt embeddings. */
const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2" as const;

/**
 * Returns (and lazily initialises) the singleton feature-extraction pipeline.
 * Subsequent calls are O(1) — the model is loaded only once per cold start.
 *
 * Thread-safety note: in a single-threaded Node.js runtime this is safe.
 * In environments with true concurrency, wrap the init block with a mutex.
 */
async function getEmbeddingPipeline(): Promise<EmbeddingPipeline> {
  if (_embeddingPipeline !== null) {
    return _embeddingPipeline;
  }

  // Dynamic import keeps the heavy transformer runtime tree-shakeable when
  // the middleware is used in environments that bundle conditionally.
  const { pipeline } = await import("@huggingface/transformers");

  const rawPipeline = await pipeline("feature-extraction", EMBEDDING_MODEL, {
    // Quantised ONNX weights — smaller download, same retrieval quality.
    dtype: "q8" as never,
    // Suppress verbose download progress in production logs.
    progress_callback: undefined,
  });

  // Cast to our internal type; the raw callable matches the signature.
  _embeddingPipeline = rawPipeline as unknown as EmbeddingPipeline;

  return _embeddingPipeline;
}

// ---------------------------------------------------------------------------
// 3. Prompt serialisation helper
// ---------------------------------------------------------------------------

/**
 * Collapses a LanguageModelV1 prompt (an array of messages) into a single
 * plain-text string suitable for embedding.
 *
 * Includes the latest user turn together with surrounding prompt context so
 * different system prompts or prior conversation history do not collide.
 */
function extractPromptText(prompt: unknown): string {
  if (!Array.isArray(prompt)) {
    return "";
  }

  const messages = prompt
    .flatMap((message) => {
      if (!message || typeof message !== "object") {
        return [];
      }

      const role = typeof message.role === "string" ? message.role : "unknown";
      const content = message.content;
      const text =
        typeof content === "string"
          ? content.trim()
          : Array.isArray(content)
            ? content
                .filter(
                  (part): part is { type: "text"; text: string } =>
                    part &&
                    typeof part === "object" &&
                    part.type === "text" &&
                    typeof part.text === "string"
                )
                .map((part) => part.text.trim())
                .filter(Boolean)
                .join("\n")
            : "";

      if (!text) {
        return [];
      }

      return [`${role}: ${text}`];
    })
    .filter(Boolean);

  if (messages.length === 0) {
    return "";
  }

  return messages.join("\n\n");
}

// ---------------------------------------------------------------------------
// 4. Embedding generator
// ---------------------------------------------------------------------------

/**
 * Generates a normalised float32 embedding vector for `text`.
 * Uses mean pooling over all token embeddings (standard for MiniLM).
 *
 * Exported so higher-level helpers (e.g. the Next.js App Router
 * `withSemanticCache` wrapper) can embed prompts through the exact same
 * pipeline the middleware uses, guaranteeing cache-key compatibility.
 */
export async function generateEmbedding(text: string): Promise<number[]> {
  const pipe = await getEmbeddingPipeline();

  // `output` has shape [1, seq_len, hidden_dim] as a Tensor;
  // the `pooling: "mean"` + `normalize: true` options collapse it to [1, 384].
  const output = await pipe(text, {
    pooling: "mean",
    normalize: true,
  });

  // Flatten to a plain JS array for portability across vector store clients.
  return Array.from(output.data);
}

// ---------------------------------------------------------------------------
// 5. Synthetic response builder
// ---------------------------------------------------------------------------

/**
 * Constructs a minimal LanguageModelV1GenerateResult-shaped object that the
 * Vercel AI SDK accepts as a short-circuited response.
 *
 * The SDK's LanguageModelV4GenerateResult is structurally compatible with V1
 * at the fields we populate here.  Update field names as the SDK evolves.
 */
function buildCachedGenerateResult(
  cachedText: string
): LanguageModelV4GenerateResult {
  return {
    /** Passthrough fields required by the SDK result contract. */
    rawCall: {
      rawPrompt: null,
      rawSettings: {},
    },
    /** Signal to the caller that generation ended normally. */
    finishReason: "stop",
    /** Zero usage — no tokens were consumed from the provider. */
    usage: {
      promptTokens: 0,
      completionTokens: 0,
      inputTokens: {
        total: 0,
        noCache: 0,
        cacheRead: 0,
        cacheWrite: 0,
      },
      outputTokens: {
        total: 0,
        text: 0,
        reasoning: 0,
      },
    },
    /** The cached LLM response returned verbatim. */
    text: cachedText,
    /** Content field for LanguageModelV4GenerateResult compatibility. */
    content: [
      {
        type: "text" as const,
        text: cachedText,
      },
    ],
    /** Tool-call fields are undefined for pure-text cached responses. */
    toolCalls: undefined,
    toolResults: undefined,
    /** Warnings field for LanguageModelV4GenerateResult compatibility. */
    warnings: undefined,
    /** Custom metadata so downstream consumers can detect cache hits. */
    providerMetadata: {
      semanticCache: {
        hit: true,
        model: EMBEDDING_MODEL,
      },
    },
  } as LanguageModelV4GenerateResult;
}

// ---------------------------------------------------------------------------
// 6. Middleware options
// ---------------------------------------------------------------------------

export interface SemanticCacheMiddlewareOptions {
  /**
   * Your VectorStoreAdapter implementation.
   * The middleware is entirely storage-agnostic — bring your own backend.
   */
  vectorStore: VectorStoreAdapter;

  /**
   * Cosine-similarity threshold in [0, 1].
   * Prompts whose nearest neighbour exceeds this value are considered cache hits.
   * @default 0.92
   */
  similarityThreshold?: number;

  /**
   * Optional user identifier written into cache metadata and applied as a
   * lookup filter.
   */
  userId?: string;

  /**
   * Optional tenant identifier written into cache metadata and applied as a
   * lookup filter.
   */
  tenantId?: string;

  /**
   * When `true`, logs cache hits/misses to `console.debug`.
   * Disable in production or pipe to your own logger.
   * @default false
   */
  debug?: boolean;
}

// ---------------------------------------------------------------------------
// 7. Internal sentinel key
// ---------------------------------------------------------------------------

/**
 * The key used to smuggle a cache-hit value through `providerMetadata`
 * from `transformParams` into `wrapGenerate`.
 * Prefixed with `__` to minimise collision risk with real provider keys.
 */
const CACHE_HIT_SENTINEL = "__semanticCacheHit" as const;
const PROMPT_VECTOR_SENTINEL = "__semanticCachePromptVector" as const;

// ---------------------------------------------------------------------------
// 8. SemanticCacheMiddleware factory
// ---------------------------------------------------------------------------

/**
 * Creates a LanguageModelV4Middleware instance that intercepts LLM calls and
 * serves semantically equivalent responses from a vector cache.
 *
 * @example
 * ```ts
 * import { wrapLanguageModel } from "ai";
 * import { SemanticCacheMiddleware } from "./semantic-cache-middleware";
 * import { MyPgVectorStore } from "./stores/pg-vector";
 *
 * const model = wrapLanguageModel({
 *   model: openai("gpt-4o"),
 *   middleware: SemanticCacheMiddleware({
 *     vectorStore: new MyPgVectorStore(),
 *     similarityThreshold: 0.92,
 *     debug: process.env.NODE_ENV !== "production",
 *   }),
 * });
 * ```
 */
export function SemanticCacheMiddleware(
  options: SemanticCacheMiddlewareOptions
): LanguageModelV4Middleware {
  const {
    vectorStore,
    similarityThreshold = 0.92,
    userId,
    tenantId,
    debug = false,
  } = options;
  const metadataFilter: VectorMetadataFilter | undefined =
    userId || tenantId ? { userId, tenantId } : undefined;

  /** Scoped logger — no-ops unless `debug` is true. */
  const log = (...args: unknown[]): void => {
    if (debug) console.debug("[SemanticCache]", ...args);
  };

  const buildCacheMetadata = (): VectorMetadata => ({
    ...(metadataFilter ?? {}),
  });

  // =========================================================================
  // transformParams
  //
  // Runs BEFORE the LLM call.  Generates a prompt embedding and probes the
  // vector store.  On a hit, attaches a sentinel to providerMetadata so that
  // wrapGenerate can short-circuit without calling the model.
  // =========================================================================
  const transformParams: LanguageModelV4Middleware["transformParams"] = async ({
    params,
  }) => {
    const promptText = extractPromptText(params.prompt);

    if (!promptText) {
      // Nothing to embed; pass params through unchanged.
      log("Empty prompt text — skipping semantic cache lookup.");
      return params;
    }

    // ------------------------------------------------------------------
    // Generate the embedding for the incoming prompt.
    // This is the only I/O-bound step in the hot path (beyond the cache
    // search itself); keep `similarityThreshold` high to minimise false
    // positives and unnecessary LLM bypasses.
    // ------------------------------------------------------------------
    const promptVector = await generateEmbedding(promptText);

    // Probe the vector store.
    const cachedResponse = await vectorStore.search(
      promptVector,
      similarityThreshold,
      metadataFilter
    );

    if (cachedResponse !== null) {
      // ── Cache HIT ─────────────────────────────────────────────────
      log(`Cache HIT  — "${promptText.slice(0, 72)}…"`);

      // Attach the cached value to providerMetadata.  wrapGenerate reads
      // this sentinel and returns the synthetic result without calling doGenerate.
      return {
        ...params,
        providerMetadata: {
          ...((params as any).providerMetadata ?? {}),
          [CACHE_HIT_SENTINEL]: cachedResponse,
        },
      } as any;
    }

    // ── Cache MISS ────────────────────────────────────────────────────
    log(`Cache MISS — "${promptText.slice(0, 72)}…"`);
    // Thread the vector through providerMetadata so wrapGenerate can save it
    // without relying on any process-scoped shared state.
    return {
      ...params,
      providerMetadata: {
        ...((params as any).providerMetadata ?? {}),
        [PROMPT_VECTOR_SENTINEL]: promptVector,
      },
    } as any;
  };

  // =========================================================================
  // wrapGenerate
  //
  // Wraps model.doGenerate.
  //   • Cache HIT  → returns a synthetic result; doGenerate is NEVER called.
  //   • Cache MISS → calls doGenerate, then fire-and-forgets the save.
  // =========================================================================
  const wrapGenerate: LanguageModelV4Middleware["wrapGenerate"] = async ({
    doGenerate,
    params,
  }) => {
    // ── Cache HIT path ────────────────────────────────────────────────
    const cacheHitValue = ((params as any).providerMetadata ?? {})[
      CACHE_HIT_SENTINEL
    ];

    if (typeof cacheHitValue === "string") {
      log("Short-circuiting LLM call — serving cached response.");
      // Return the fabricated result; the real model is never invoked.
      return buildCachedGenerateResult(cacheHitValue) as any;
    }

    // ── Cache MISS path ───────────────────────────────────────────────
    // Delegate to the real model.
    const result = await doGenerate();

    // Persist asynchronously — we must NOT block the response path.
    const promptText = extractPromptText(params.prompt);
    const promptVector = ((params as any).providerMetadata ?? {})[
      PROMPT_VECTOR_SENTINEL
    ] as number[] | undefined;

    // SDK v4 uses `text` directly; fall back to content array for compatibility.
    const responseText: string =
      (result as any).text ??
      ((result as any).content?.find((c: any) => c.type === "text")?.text ??
        "");

    if (promptVector && responseText) {
      vectorStore
        .save(promptVector, responseText, buildCacheMetadata())
        .then(() => {
          log(`Saved embedding — "${promptText.slice(0, 72)}…"`);
        })
        .catch((err: unknown) => {
          console.error(
            "[SemanticCache] Failed to persist to vector store:",
            err instanceof Error ? err.message : err
          );
        });
    }

    return result;
  };

  return { transformParams, wrapGenerate };
}
