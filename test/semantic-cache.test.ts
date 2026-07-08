import { describe, it, expect, vi, beforeEach } from "vitest";
import { generateEmbedding, SemanticCacheMiddleware } from "../semantic-cache-middleware";
import { withSemanticCache } from "../index";
import type { VectorStoreAdapter } from "../semantic-cache-middleware";
import type { LanguageModel, LanguageModelV4 } from "ai";

describe("generateEmbedding", () => {
  it("should generate 384-dimensional embedding using all-MiniLM-L6-v2", async () => {
    // This actually downloads the quantized MiniLM model on first run and embeds text
    const text = "Hello world";
    const embedding = await generateEmbedding(text);
    
    expect(embedding).toBeInstanceOf(Array);
    expect(embedding.length).toBe(384); // all-MiniLM-L6-v2 size is 384
    expect(embedding.every(n => typeof n === "number")).toBe(true);
    // Verify it is normalized (magnitude close to 1)
    const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    expect(magnitude).toBeCloseTo(1, 4);
  }, 60000); // 1 minute timeout since downloading model may take time on first run
});

describe("SemanticCacheMiddleware", () => {
  let mockVectorStore: VectorStoreAdapter;
  let savedData: { vector: number[]; response: string }[] = [];

  beforeEach(() => {
    savedData = [];
    mockVectorStore = {
      search: vi.fn(async (vector: number[], threshold: number) => {
        // Mock simple exact match based on vector similarity or presence in savedData
        if (savedData.length > 0) {
          return savedData[0].response;
        }
        return null;
      }),
      save: vi.fn(async (vector: number[], response: string) => {
        savedData.push({ vector, response });
      }),
    };
  });

  it("should correctly handle a cache miss and then a cache hit", async () => {
    const middleware = SemanticCacheMiddleware({
      vectorStore: mockVectorStore,
      similarityThreshold: 0.92,
      debug: true,
    });

    const mockDoGenerate = vi.fn().mockResolvedValue({
      text: "This is a prompt response from the LLM.",
      finishReason: "stop",
      usage: { promptTokens: 10, completionTokens: 10 },
      content: [{ type: "text", text: "This is a prompt response from the LLM." }],
    });

    // 1. First request -> Cache Miss
    const params = {
      prompt: [{ role: "user", content: [{ type: "text", text: "What is quantum entanglement?" }] }],
      model: {} as any,
      mode: "regular" as const,
    };

    // Transform params
    const transformedParams = await middleware.transformParams!({
      params: params as any,
    });

    expect(transformedParams).toBeDefined();
    expect(transformedParams.providerMetadata).toBeUndefined(); // No hit sentinel

    // Wrap generate
    const result = await middleware.wrapGenerate!({
      doGenerate: mockDoGenerate,
      params: transformedParams as any,
    });

    expect(result.text).toBe("This is a prompt response from the LLM.");
    expect(mockDoGenerate).toHaveBeenCalledTimes(1);

    // Give asynchronous save a moment to run
    await new Promise((resolve) => setTimeout(resolve, 50));

    expect(mockVectorStore.save).toHaveBeenCalledTimes(1);
    expect(savedData.length).toBe(1);
    expect(savedData[0].response).toBe("This is a prompt response from the LLM.");

    // 2. Second request -> Cache Hit
    mockDoGenerate.mockClear();

    const transformedParamsHit = await middleware.transformParams!({
      params: params as any,
    });

    // It should have the cache hit sentinel in providerMetadata
    expect(transformedParamsHit.providerMetadata).toBeDefined();
    expect((transformedParamsHit.providerMetadata as any).__semanticCacheHit).toBe("This is a prompt response from the LLM.");

    const resultHit = await middleware.wrapGenerate!({
      doGenerate: mockDoGenerate,
      params: transformedParamsHit as any,
    });

    expect(resultHit.text).toBe("This is a prompt response from the LLM.");
    expect(mockDoGenerate).not.toHaveBeenCalled(); // Short-circuited!
  });
});
