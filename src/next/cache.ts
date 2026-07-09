/**
 * next-cache.ts
 *
 * Thin, dependency-free bridge to Next.js's native caching primitives
 * (`unstable_cache` and `revalidateTag` from `next/cache`).
 *
 * Design goals
 * ────────────
 *  • **Optional** — `next/cache` is treated as an *optional* runtime module.
 *    When the package runs outside a Next.js app (plain Node, a test harness,
 *    an edge worker without the App Router, …) the module simply isn't there.
 *  • **Graceful** — if `next/cache` cannot be resolved, every helper degrades
 *    to a no-op / pass-through instead of throwing. Callers keep working with
 *    zero Next.js coupling.
 *  • **Zero static coupling** — the module specifier is resolved *indirectly*
 *    so that bundlers do not attempt to statically include `next/cache`, and
 *    TypeScript does not require `next` to be installed to type-check.
 */

/**
 * Signature of Next.js's `unstable_cache`.
 * See: https://nextjs.org/docs/app/api-reference/functions/unstable_cache
 */
type UnstableCache = <T extends (...args: any[]) => Promise<any>>(
  fn: T,
  keyParts?: string[],
  options?: { tags?: string[]; revalidate?: number | false }
) => T;

/**
 * Signature of Next.js's `revalidateTag`.
 * See: https://nextjs.org/docs/app/api-reference/functions/revalidateTag
 */
type RevalidateTag = (tag: string) => void;

/** The subset of the `next/cache` module surface this package relies on. */
interface NextCacheModule {
  unstable_cache?: UnstableCache;
  revalidateTag?: RevalidateTag;
}

// ---------------------------------------------------------------------------
// Lazy, cached module resolution
// ---------------------------------------------------------------------------

/** Memoised resolution state — we probe `next/cache` at most once per process. */
let _resolved = false;
let _module: NextCacheModule | undefined;

/**
 * Attempts to resolve the `next/cache` module exactly once.
 *
 * The specifier is stored in a variable (rather than passed as a string
 * literal to `require`) so that:
 *   • TypeScript does not try to resolve the module's types at compile time
 *     (which would fail when `next` is not a dependency), and
 *   • bundlers do not statically pull `next/cache` into non-Next builds.
 *
 * Any failure — module missing, throwing on load, or `require` being
 * unavailable (e.g. a pure-ESM/edge runtime) — resolves to `undefined`.
 */
function loadNextCache(): NextCacheModule | undefined {
  if (_resolved) return _module;
  _resolved = true;

  try {
    const specifier = "next/cache";
    const req: NodeRequire | undefined =
      typeof require === "function" ? require : undefined;

    _module = req ? (req(specifier) as NextCacheModule) : undefined;
  } catch {
    // next/cache is unavailable — fall back to a Next.js-free code path.
    _module = undefined;
  }

  return _module;
}

// ---------------------------------------------------------------------------
// Public helpers
// ---------------------------------------------------------------------------

/**
 * Base cache tag applied to every semantic-cache entry.
 * Combined with a namespace to build the per-entry tag set.
 */
export const SEMANTIC_CACHE_TAG = "semantic-cache" as const;

/**
 * Builds the Next.js cache tags for a given namespace.
 *
 * @example buildCacheTags("user_42") → ["semantic-cache", "user_42"]
 * @example buildCacheTags()          → ["semantic-cache"]
 */
export function buildCacheTags(namespace?: string): string[] {
  return namespace ? [SEMANTIC_CACHE_TAG, namespace] : [SEMANTIC_CACHE_TAG];
}

/**
 * Returns `true` when Next.js's `unstable_cache` is available in the current
 * runtime. Useful for conditional logging / diagnostics.
 */
export function isNextCacheAvailable(): boolean {
  return typeof loadNextCache()?.unstable_cache === "function";
}

/**
 * Wraps an async function with Next.js's `unstable_cache` when it is available.
 *
 * When `next/cache` cannot be resolved (i.e. the package is used outside of
 * Next.js), the original function is returned **unchanged** so the caller runs
 * exactly as it would without any caching layer.
 *
 * @param fn       - The async function to memoise (e.g. a vector DB fetch).
 * @param keyParts - Stable key segments that scope the cache entry.
 * @param tags     - Cache tags used for targeted `revalidateTag` invalidation.
 * @param revalidate - Optional `unstable_cache` revalidation window in seconds
 *                     (or `false` to cache indefinitely). Omit to use Next.js's
 *                     default behaviour.
 * @returns Either the `unstable_cache`-wrapped function, or `fn` untouched.
 */
export function withNextCache<T extends (...args: any[]) => Promise<any>>(
  fn: T,
  keyParts: string[],
  tags: string[],
  revalidate?: number | false
): T {
  const mod = loadNextCache();

  if (mod && typeof mod.unstable_cache === "function") {
    try {
      const options: { tags: string[]; revalidate?: number | false } = { tags };
      if (revalidate !== undefined) {
        options.revalidate = revalidate;
      }
      const cachedFn = mod.unstable_cache(fn, keyParts, options);
      return (async (...args: any[]) => {
        try {
          return await cachedFn(...args);
        } catch (err: unknown) {
          // If execution of cachedFn fails (e.g. Invariant error under non-Next.js or test env),
          // fall back to the raw function.
          return await fn(...args);
        }
      }) as unknown as T;
    } catch {
      // If wrapping fails for any reason, fall back to the raw function.
      return fn;
    }
  }

  return fn;
}

/**
 * Invalidates a Next.js cache tag via `revalidateTag` when available.
 *
 * Outside of Next.js (or if the call throws — e.g. invoked outside a
 * request/action scope), this is a safe no-op that returns `false`.
 *
 * @returns `true` if `revalidateTag` was invoked successfully, else `false`.
 */
export function revalidateNextTag(tag: string): boolean {
  const mod = loadNextCache();

  if (mod && typeof mod.revalidateTag === "function") {
    try {
      mod.revalidateTag(tag);
      return true;
    } catch {
      return false;
    }
  }

  return false;
}
