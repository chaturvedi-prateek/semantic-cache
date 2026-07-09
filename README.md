# next-semantic-cache

> Semantic caching middleware for the [Vercel AI SDK](https://sdk.vercel.ai) — short-circuit LLM calls with vector similarity lookups.

[![npm version](https://img.shields.io/npm/v/next-semantic-cache.svg)](https://www.npmjs.com/package/next-semantic-cache)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

`next-semantic-cache` is a drop-in `LanguageModelV4Middleware` that intercepts prompts, embeds them locally, and serves cached responses when a semantically similar prompt has been seen before. It ships with a pluggable `VectorStoreAdapter` interface plus built-in adapters for Redis (RediSearch / Upstash), PostgreSQL (`pgvector`), and lightweight in-memory development workflows.

> ⚠️ **Edge runtime note:** Edge runtime deployment is supported only when using a hosted embedding provider (for example, OpenAI or Upstash). The default local `@huggingface/transformers` embedding path (ONNX/WASM) is not supported in standard Vercel Edge functions.

---

## Why?

LLM calls are **slow** and **expensive**. Traditional caching only matches *exact* strings — but users phrase the same question in countless ways. `next-semantic-cache` matches on **meaning**, not exact text:

| Prompt A | Prompt B | Exact cache | Semantic cache |
| --- | --- | --- | --- |
| "How do I reset my password?" | "What's the steps to change my password?" | ❌ Miss | ✅ Hit |

---

## Features

- 🧠 **Semantic matching** — embeds prompts with `all-MiniLM-L6-v2` and matches on cosine similarity.
- 🔌 **Pluggable storage** — implement `VectorStoreAdapter` for any vector DB (Pinecone, pgvector, Weaviate, in-memory, …).
- 🚀 **Redis adapter included** — works with **Upstash Redis** (serverless) or self-hosted **Redis Stack**.
- 🛡️ **Defensive by design** — every cache operation is wrapped in try/catch. If the store is down, you transparently fall back to the live LLM. **A cache failure never breaks your app.**
- ⏱️ **Configurable TTL** — auto-expire entries via Redis `EXPIRE`.
- 📦 **Zero-config embeddings** — the embedding model runs locally; no extra API keys required.

---

## Installation

```bash
npm install next-semantic-cache ai @huggingface/transformers
```

For the Redis adapter, also install:

```bash
npm install @upstash/redis
```

---

## Quick Start

```ts
import { openai } from "@ai-sdk/openai";
import { wrapLanguageModel, generateText } from "ai";
import { SemanticCacheMiddleware } from "next-semantic-cache";
import { RedisVectorAdapter } from "next-semantic-cache/adapters/redis";

// 1. Create a vector store adapter.
const vectorStore = new RedisVectorAdapter({
  redisUrl:   process.env.UPSTASH_REDIS_REST_URL!,
  redisToken: process.env.UPSTASH_REDIS_REST_TOKEN!,
  ttlSeconds: 60 * 60 * 24, // 24-hour cache expiry
});

// 2. Wrap your model with the middleware.
const model = wrapLanguageModel({
  model: openai("gpt-4o"),
  middleware: SemanticCacheMiddleware({
    vectorStore,
    similarityThreshold: 0.92, // 0–1; higher = stricter match
    tenantId: "tenant-123",
    userId: "user-456",
  }),
});

// 3. Use it exactly like any AI SDK model.
const { text } = await generateText({
  model,
  prompt: "Explain quantum entanglement in one sentence.",
});

console.log(text);
// First call  → hits the LLM, caches the result.
// Later calls with a similar prompt → served instantly from cache.
```

---

## App Router Route Handlers

Prefer to cache at the HTTP boundary instead of at the model? Wrap a Next.js
App Router Route Handler with `withSemanticCache` from `next-semantic-cache/next`.
It takes a standard web `Request`, checks the semantic cache first, and returns
a `NextResponse` — running your LLM fallback only on a miss and caching the
result automatically.

```ts
// app/api/chat/route.ts
import { withSemanticCache } from "next-semantic-cache/next";
import { RedisVectorAdapter } from "next-semantic-cache/adapters/redis";
import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";

const adapter = new RedisVectorAdapter({
  redisUrl:   process.env.UPSTASH_REDIS_REST_URL!,
  redisToken: process.env.UPSTASH_REDIS_REST_TOKEN!,
});

export const POST = withSemanticCache({
  adapter,
  threshold: 0.92,
  namespace: "chat", // optional: partition this handler's cache entries
  // Optional: customise how the prompt is read from the request.
  extractPrompt: async (req) => (await req.json()).prompt,
  // Runs only on a cache miss; its return value is cached and returned.
  fallback: async (prompt) => {
    const { text } = await generateText({ model: openai("gpt-4o"), prompt });
    return text;
  },
});
```

The handler responds with JSON of the shape
`{ response: string, cached: boolean }` — `cached` is `true` when the answer
came from the vector store and `false` on a live LLM call.

### `withSemanticCache(config)` — App Router

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `adapter` | `VectorStoreAdapter` | **required** | The persistence backend for embeddings and responses. |
| `fallback` | `(prompt, request) => string \| Promise<string>` | **required** | LLM (or any producer) invoked on a cache miss. Its text is cached and returned. |
| `extractPrompt` | `(request) => string \| Promise<string>` | reads `{ prompt }` or the last user message of `{ messages }` | How to extract the prompt text to embed. |
| `namespace` | `string` | — | Logical partition; folded into the embedding so entries never collide across namespaces. |
| `threshold` | `number` | `0.92` | Minimum cosine similarity (0–1) for a cache hit. |
| `debug` | `boolean` | `false` | Log cache hits/misses to `console.debug`. |

> Requires `next` (>= 13) as a peer dependency — already present in any App
> Router project.

---

## Configuration

### `SemanticCacheMiddleware(options)`

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `vectorStore` | `VectorStoreAdapter` | **required** | The persistence backend for embeddings and responses. |
| `similarityThreshold` | `number` | `0.92` | Minimum cosine similarity (0–1) for a cache hit. Higher = fewer, more accurate hits. |
| `tenantId` | `string` | — | Optional tenant-level metadata filter applied to lookups and persisted on writes. |
| `userId` | `string` | — | Optional user-level metadata filter applied to lookups and persisted on writes. |

### `RedisVectorAdapter(options)`

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `client` | `Redis` | — | A pre-configured `@upstash/redis` client. Mutually exclusive with `redisUrl` + `redisToken`. |
| `redisUrl` | `string` | — | Upstash REST endpoint. Required if `client` is omitted. |
| `redisToken` | `string` | — | Upstash REST token. Required if `client` is omitted. |
| `ttlSeconds` | `number` | `0` | Auto-expire entries after N seconds. `0` = keep indefinitely. |
| `connectionTimeoutMs` | `number` | `2000` | Timeout per Redis command before falling back to the LLM. |
| `namespace` | `string` | — | Logical namespace (tenant, feature, or user ID) woven into the Next.js cache tags. Enables scoped invalidation via `invalidate()`. No effect outside Next.js. |

### `MemoryVectorAdapter()`

In-memory adapter for local development and tests.

- Stores vectors in a process-local `Map`.
- Uses pure JavaScript cosine similarity for `query()`/`search()`.
- Requires no external database or additional dependencies.

### `PgVectorAdapter(options)`

PostgreSQL + `pgvector` adapter for production workloads where vectors are
stored in your relational database.

### `RedisStackVectorAdapter(options)`

Redis Stack adapter for self-hosted Redis deployments using RediSearch vector
indexing.

---

## Next.js Native Cache Integration

`RedisVectorAdapter` integrates **optionally** with Next.js's native data cache. When it detects that it is running inside a Next.js app, it transparently wraps the underlying vector-database fetch performed by `search()` in [`unstable_cache`](https://nextjs.org/docs/app/api-reference/functions/unstable_cache), tagging every entry with `['semantic-cache', namespace]`.

You can then purge those cached lookups with [`revalidateTag`](https://nextjs.org/docs/app/api-reference/functions/revalidateTag) via the `invalidate()` method:

```ts
const adapter = new RedisVectorAdapter({
  redisUrl:   process.env.UPSTASH_REDIS_REST_URL!,
  redisToken: process.env.UPSTASH_REDIS_REST_TOKEN!,
  namespace:  "user_42", // → tags: ['semantic-cache', 'user_42']
});

// In a Server Action or Route Handler:
await adapter.invalidate("user_42"); // purge only this namespace's queries
await adapter.invalidate();          // purge the adapter's namespace (or all, if none)
```

### `invalidate(namespace?)`

| Call | Effect |
| --- | --- |
| `invalidate("user_42")` | Revalidates the `user_42` tag — purges only that namespace's queries. |
| `invalidate()` (namespace configured) | Revalidates the adapter's configured namespace tag. |
| `invalidate()` (no namespace) | Revalidates the base `semantic-cache` tag — purges every entry. |

> `invalidate()` only affects Next.js's **data cache** — it does not delete the underlying Redis entries. Use `flush()` to remove Redis keys.

### Graceful fallback outside Next.js

`next/cache` is treated as an **optional** module. When the package is used outside of Next.js (plain Node, tests, edge workers without the App Router, …), the module cannot be resolved and every integration point degrades safely:

- `search()` falls straight through to the live vector-store fetch (no caching layer, unchanged behaviour).
- `invalidate()` becomes a no-op.

**Nothing throws, and Next.js is never a required dependency.**

---

## Writing a Custom Adapter

Implement the `VectorStoreAdapter` interface to plug in any backend:

```ts
import type { VectorStoreAdapter } from "next-semantic-cache";

export class MyVectorAdapter implements VectorStoreAdapter {
  /**
   * Return the cached response if a stored vector is within `threshold`
   * cosine similarity of `vector`, otherwise `null`.
   */
  async search(vector: number[], threshold: number, filter?: { userId?: string; tenantId?: string }): Promise<string | null> {
    // ... query your vector DB ...
    return null;
  }

  /**
   * Persist a prompt embedding and its LLM response.
   * Failures should be swallowed — caching is best-effort.
   */
  async save(promptVector: number[], response: string, metadata?: Record<string, unknown>): Promise<void> {
    // ... upsert into your vector DB ...
  }
}
```

---

## How It Works

```
                      ┌──────────────────────────────┐
   prompt ──────────► │  transformParams             │
                      │  • embed prompt (MiniLM-L6)   │
                      │  • vectorStore.search(...)    │
                      └───────────┬──────────────────┘
                                  │
                 ┌────────────────┴─────────────────┐
                 ▼                                    ▼
          Cache HIT                             Cache MISS
   ┌──────────────────────┐            ┌──────────────────────────┐
   │ return synthetic      │            │ wrapGenerate:            │
   │ result — LLM is       │            │ • call real model        │
   │ NEVER invoked         │            │ • vectorStore.save(...)  │
   └──────────────────────┘            │   (fire-and-forget)      │
                                        └──────────────────────────┘
```

1. **`transformParams`** embeds the incoming prompt and queries the vector store.
2. **Cache hit** → a synthetic `LanguageModelV4GenerateResult` is returned immediately; the real model is short-circuited.
3. **Cache miss** → the request falls through to the real model, and the response is saved to the store asynchronously (never blocking the response path).

---

## Redis Storage Layout

The `RedisVectorAdapter` stores each entry as a Redis `HASH` and creates an HNSW vector index over them:

| Component | Value |
| --- | --- |
| Key prefix | `sc:<uuid>` |
| Fields | `response`, `vector` (Float32 binary), `createdAt` (ISO-8601) |
| Index | `FT.CREATE` — HNSW, `FLOAT32`, `DIM 384`, `COSINE` distance |

The index is created **lazily** on first `save()`/`search()` and is idempotent.

> ⚠️ **Note on Upstash + binary vectors:** `@upstash/redis` transports commands as JSON over HTTP. Binary vector payloads may not round-trip reliably through the REST layer. If you encounter KNN query errors, use a **Redis Stack** instance with a native `ioredis` client for full binary support.

---

## Requirements

- **Node.js** ≥ 18
- **Vercel AI SDK** (`ai`) v4+
- A vector store (Redis Stack / Upstash, or your own adapter)
- Embedding model: `all-MiniLM-L6-v2` (384-dim) via `@huggingface/transformers`

---

## License

MIT © [Prateek Chaturvedi](https://github.com/chaturvedi-prateek)