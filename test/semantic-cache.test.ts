import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("@huggingface/transformers", () => ({
  pipeline: vi.fn(async () => async (text: string) => {
    const vector = new Float32Array(384);
    for (let i = 0; i < vector.length; i++) {
      const code = text.charCodeAt(i % text.length) || i + 1;
      vector[i] = ((code % 97) + 1) / 100;
    }

    const magnitude = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
    for (let i = 0; i < vector.length; i++) {
      vector[i] = vector[i] / magnitude;
    }

    return { data: vector, dims: [1, 384] };
  }),
}));

import {
  generateEmbedding,
  SemanticCacheMiddleware,
} from "../semantic-cache-middleware";
import type {
  VectorMetadata,
  VectorMetadataFilter,
  VectorStoreAdapter,
} from "../semantic-cache-middleware";

describe("generateEmbedding", () => {
  it("should generate a normalized 384-dimensional embedding", async () => {
    const embedding = await generateEmbedding("Hello world");

    expect(embedding).toBeInstanceOf(Array);
    expect(embedding.length).toBe(384);
    expect(embedding.every((n) => typeof n === "number")).toBe(true);

    const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    expect(magnitude).toBeCloseTo(1, 4);
  });
});

describe("SemanticCacheMiddleware", () => {
  let mockVectorStore: VectorStoreAdapter;
  let savedData: Array<{
    vector: number[];
    response: string;
    metadata: VectorMetadata;
  }>;

  beforeEach(() => {
    savedData = [];
    mockVectorStore = {
      search: vi.fn(
        async (
          vector: number[],
          _threshold: number,
          filter?: VectorMetadataFilter
        ) => {
          const match = savedData.find(
            (entry) =>
              JSON.stringify(entry.vector) === JSON.stringify(vector) &&
              entry.metadata.userId === filter?.userId &&
              entry.metadata.tenantId === filter?.tenantId
          );
          return match?.response ?? null;
        }
      ),
      save: vi.fn(
        async (
          vector: number[],
          response: string,
          metadata: VectorMetadata = {}
        ) => {
          savedData.push({ vector, response, metadata });
        }
      ),
      upsert: vi.fn(async () => {}),
      query: vi.fn(async () => []),
      delete: vi.fn(async () => {}),
    };
  });

  it("should correctly handle a cache miss and then a cache hit", async () => {
    const middleware = SemanticCacheMiddleware({
      vectorStore: mockVectorStore,
      similarityThreshold: 0.92,
      userId: "user-1",
      tenantId: "tenant-1",
      debug: true,
    });

    const mockDoGenerate = vi.fn().mockResolvedValue({
      text: "This is a prompt response from the LLM.",
      finishReason: "stop",
      usage: { promptTokens: 10, completionTokens: 10 },
      content: [{ type: "text", text: "This is a prompt response from the LLM." }],
    });

    const params = {
      prompt: [
        { role: "system", content: [{ type: "text", text: "Answer as a tutor." }] },
        { role: "user", content: [{ type: "text", text: "What is quantum entanglement?" }] },
      ],
      model: {} as any,
      mode: "regular" as const,
    };

    const transformedParams = await middleware.transformParams!({
      type: "generate",
      params: params as any,
      model: {} as any,
    });

    expect((transformedParams as any).__semanticCacheHit).toBeUndefined();
    expect(mockVectorStore.search).toHaveBeenCalledWith(
      expect.any(Array),
      0.92,
      { userId: "user-1", tenantId: "tenant-1" }
    );

    const result = await middleware.wrapGenerate!({
      doGenerate: mockDoGenerate,
      doStream: vi.fn(),
      params: transformedParams as any,
      model: {} as any,
    });

    expect((result.content[0] as { type: "text"; text: string }).text).toBe(
      "This is a prompt response from the LLM."
    );
    expect(mockDoGenerate).toHaveBeenCalledTimes(1);

    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(mockVectorStore.save).toHaveBeenCalledTimes(1);
    expect(savedData).toHaveLength(1);
    expect(savedData[0].metadata).toMatchObject({
      userId: "user-1",
      tenantId: "tenant-1",
    });

    mockDoGenerate.mockClear();

    const transformedParamsHit = await middleware.transformParams!({
      type: "generate",
      params: params as any,
      model: {} as any,
    });

    expect((transformedParamsHit as any).providerMetadata?.__semanticCacheHit).toBe(
      "This is a prompt response from the LLM."
    );

    const resultHit = await middleware.wrapGenerate!({
      doGenerate: mockDoGenerate,
      doStream: vi.fn(),
      params: transformedParamsHit as any,
      model: {} as any,
    });

    expect((resultHit.content[0] as { type: "text"; text: string }).text).toBe(
      "This is a prompt response from the LLM."
    );
    expect(mockDoGenerate).not.toHaveBeenCalled();
  });

  it("uses conversation context when generating the cache vector", async () => {
    const middleware = SemanticCacheMiddleware({
      vectorStore: mockVectorStore,
      similarityThreshold: 0.92,
      userId: "user-1",
      tenantId: "tenant-1",
    });

    const promptA = [
      { role: "system", content: [{ type: "text", text: "You are a math tutor." }] },
      { role: "user", content: [{ type: "text", text: "Summarize it." }] },
    ];
    const promptB = [
      { role: "system", content: [{ type: "text", text: "You are a legal analyst." }] },
      { role: "user", content: [{ type: "text", text: "Summarize it." }] },
    ];

    await middleware.transformParams!({
      type: "generate",
      params: { prompt: promptA } as any,
      model: {} as any,
    });
    await middleware.transformParams!({
      type: "generate",
      params: { prompt: promptB } as any,
      model: {} as any,
    });

    const firstVector = (mockVectorStore.search as ReturnType<typeof vi.fn>).mock.calls[0][0];
    const secondVector = (mockVectorStore.search as ReturnType<typeof vi.fn>).mock.calls[1][0];

    expect(firstVector).not.toEqual(secondVector);
  });
});
