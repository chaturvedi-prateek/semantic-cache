import { describe, it, expect, vi, beforeEach } from "vitest";
import "./mock-transformers";

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
  let resolveSave!: () => void;
  let saveCompleted: Promise<void>;

  beforeEach(() => {
    savedData = [];
    saveCompleted = new Promise<void>((resolve) => {
      resolveSave = resolve;
    });
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
          resolveSave();
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

    await saveCompleted;
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

  it("keeps concurrent identical prompts isolated until each save is scheduled", async () => {
    let releaseFirstSave!: () => void;
    const firstSaveReleased = new Promise<void>((resolve) => {
      releaseFirstSave = resolve;
    });
    let resolveGenerateA!: (value: any) => void;

    mockVectorStore.search = vi.fn(async () => null);
    mockVectorStore.save = vi
      .fn()
      .mockImplementationOnce(
        async (vector: number[], response: string, metadata: VectorMetadata = {}) => {
          savedData.push({ vector, response, metadata });
          await firstSaveReleased;
        }
      )
      .mockImplementation(
        async (vector: number[], response: string, metadata: VectorMetadata = {}) => {
          savedData.push({ vector, response, metadata });
        }
      );

    const middleware = SemanticCacheMiddleware({
      vectorStore: mockVectorStore,
      similarityThreshold: 0.92,
    });

    const params = {
      prompt: [{ role: "user", content: [{ type: "text", text: "Same prompt" }] }],
      model: {} as any,
      mode: "regular" as const,
    };

    const transformedA = await middleware.transformParams!({
      type: "generate",
      params: params as any,
      model: {} as any,
    });
    const transformedB = await middleware.transformParams!({
      type: "generate",
      params: params as any,
      model: {} as any,
    });

    const wrapA = middleware.wrapGenerate!({
      doGenerate: vi.fn(
        () =>
          new Promise((resolve) => {
            resolveGenerateA = resolve;
          })
      ),
      doStream: vi.fn(),
      params: transformedA as any,
      model: {} as any,
    });

    const wrapB = await middleware.wrapGenerate!({
      doGenerate: vi.fn().mockResolvedValue({
        text: "response B",
        finishReason: "stop",
        usage: { promptTokens: 1, completionTokens: 1 },
        content: [{ type: "text", text: "response B" }],
      }),
      doStream: vi.fn(),
      params: transformedB as any,
      model: {} as any,
    });

    expect((wrapB.content[0] as { type: "text"; text: string }).text).toBe("response B");
    expect(mockVectorStore.save).toHaveBeenCalledTimes(1);

    releaseFirstSave();
    await Promise.resolve();

    resolveGenerateA({
      text: "response A",
      finishReason: "stop",
      usage: { promptTokens: 1, completionTokens: 1 },
      content: [{ type: "text", text: "response A" }],
    });

    const resultA = await wrapA;
    expect((resultA.content[0] as { type: "text"; text: string }).text).toBe("response A");

    await Promise.resolve();

    expect(mockVectorStore.save).toHaveBeenCalledTimes(2);
    expect(savedData.map((entry) => entry.response)).toEqual(["response B", "response A"]);
  });

  it("short-circuits wrapStream on cache hits and saves streamed text on misses", async () => {
    const middleware = SemanticCacheMiddleware({
      vectorStore: mockVectorStore,
      similarityThreshold: 0.92,
      userId: "user-1",
      tenantId: "tenant-1",
    });

    const params = {
      prompt: [{ role: "user", content: [{ type: "text", text: "Stream this response" }] }],
      model: {} as any,
      mode: "regular" as const,
    };

    const transformedMissParams = await middleware.transformParams!({
      type: "stream",
      params: params as any,
      model: {} as any,
    });

    const missChunks = [
      { type: "stream-start", warnings: [] },
      { type: "text-start", id: "text-1" },
      { type: "text-delta", id: "text-1", delta: "Hello " },
      { type: "text-delta", id: "text-1", delta: "streaming world" },
      { type: "text-end", id: "text-1" },
      {
        type: "finish",
        finishReason: "stop",
        usage: {
          promptTokens: 1,
          completionTokens: 3,
          inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
          outputTokens: { total: 3, text: 3, reasoning: 0 },
        },
      },
    ];

    const missDoStream = vi.fn().mockResolvedValue({
      stream: new ReadableStream({
        start(controller) {
          for (const chunk of missChunks) {
            controller.enqueue(chunk);
          }
          controller.close();
        },
      }),
    });

    const missResult = await middleware.wrapStream!({
      doGenerate: vi.fn(),
      doStream: missDoStream,
      params: transformedMissParams as any,
      model: {} as any,
    });

    let streamedText = "";
    const streamedTypes: string[] = [];
    for await (const part of missResult.stream as any) {
      streamedTypes.push(part.type);
      if (part.type === "text-delta") {
        streamedText += part.delta;
      }
    }

    expect(streamedText).toBe("Hello streaming world");
    expect(streamedTypes).toContain("text-delta");
    expect(missDoStream).toHaveBeenCalledTimes(1);

    await saveCompleted;
    expect(mockVectorStore.save).toHaveBeenCalledTimes(1);
    expect(savedData).toHaveLength(1);
    expect(savedData[0].response).toBe("Hello streaming world");

    const transformedHitParams = await middleware.transformParams!({
      type: "stream",
      params: params as any,
      model: {} as any,
    });

    const hitDoStream = vi.fn();
    const hitResult = await middleware.wrapStream!({
      doGenerate: vi.fn(),
      doStream: hitDoStream,
      params: transformedHitParams as any,
      model: {} as any,
    });

    let hitText = "";
    const hitTypes: string[] = [];
    for await (const part of hitResult.stream as any) {
      hitTypes.push(part.type);
      if (part.type === "text-delta") {
        hitText += part.delta;
      }
    }

    expect(hitText).toBe("Hello streaming world");
    expect(hitTypes).toEqual([
      "stream-start",
      "text-start",
      "text-delta",
      "text-end",
      "finish",
    ]);
    expect(hitDoStream).not.toHaveBeenCalled();
  });
});
