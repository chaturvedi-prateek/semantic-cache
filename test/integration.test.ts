import { describe, it, expect, vi } from "vitest";
import "./mock-transformers";
import { generateText } from "ai";
import { withSemanticCache } from "../src/index";
import type { VectorStoreAdapter } from "../src/semantic-cache-middleware";
import type { LanguageModelV4 } from "ai";

describe("E2E Integration with Vercel AI SDK", () => {
  it("should integrate with Vercel AI SDK and perform end-to-end caching", async () => {
    // 1. Setup a standard mock vector store adapter
    const db = new Map<string, string>();
    const savedVectors: { vector: number[]; response: string }[] = [];
    
    const mockAdapter: VectorStoreAdapter = {
      search: vi.fn(async (vector: number[], threshold: number) => {
        // Since we are mocking, we can treat embeddings within 0.95 cosine similarity as a hit.
        // Let's implement cosine similarity search over our saved vectors!
        for (const item of savedVectors) {
          const sim = cosineSimilarity(vector, item.vector);
          console.log(`Calculated Similarity: ${sim} against threshold: ${threshold}`);
          if (sim >= threshold) {
            return item.response;
          }
        }
        return null;
      }),
      save: vi.fn(async (vector: number[], response: string) => {
        savedVectors.push({ vector, response });
      }),
      upsert: vi.fn(async () => {}),
      query: vi.fn(async () => []),
      delete: vi.fn(async () => {}),
    };

    // Helper to calculate cosine similarity
    function cosineSimilarity(a: number[], b: number[]): number {
      let dotProduct = 0;
      let normA = 0;
      let normB = 0;
      for (let i = 0; i < a.length; i++) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
      }
      return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    // 2. Setup a dummy Vercel AI SDK Language Model
    let designModelCalls = 0;
    const dummyModel: LanguageModelV4 = {
      specificationVersion: "v1",
      provider: "mock-provider",
      modelId: "mock-model",
      defaultObjectGenerationMode: "json",
      doGenerate: vi.fn().mockImplementation(async (options) => {
        designModelCalls++;
        const userMsg = options.prompt.find((m: any) => m.role === "user");
        let promptText = "empty";
        if (userMsg && Array.isArray(userMsg.content)) {
          const textPart = userMsg.content.find((p: any) => p.type === "text");
          if (textPart && typeof textPart === "object" && "text" in textPart) {
            promptText = textPart.text;
          }
        }
        const text = `Real response to: ${promptText}. (Call #${designModelCalls})`;
        return {
          text,
          finishReason: "stop",
          content: [{ type: "text", text }],
          usage: {
            inputTokens: { total: 15, noCache: 15, cacheRead: 0, cacheWrite: 0 },
            outputTokens: { total: 15, text: 15, reasoning: 0 },
          },
          rawCall: { rawPrompt: options.prompt, rawSettings: {} },
        };
      }) as any,
      doStream: vi.fn() as any,
    };

    // 3. Wrap with semantic cache
    const cachedModel = withSemanticCache(dummyModel, {
      adapter: mockAdapter,
      threshold: 0.80, // reasonable threshold for miniLM embeddings
      debug: true,
    });

    // 4. First call: "What's the process to reset a password?"
    const result1 = await generateText({
      model: cachedModel,
      prompt: "What's the process to reset a password?",
    });

    expect(result1.text).toBe("Real response to: What's the process to reset a password?. (Call #1)");
    expect(dummyModel.doGenerate).toHaveBeenCalledTimes(1);
    expect(mockAdapter.save).toHaveBeenCalledTimes(1);

    // Give transient async save operation a moment
    await new Promise((resolve) => setTimeout(resolve, 100));

    // 5. Second call: repeat the same prompt to verify cache reuse
    const result2 = await generateText({
      model: cachedModel,
      prompt: "What's the process to reset a password?",
    });

    expect(result2.text).toBe("Real response to: What's the process to reset a password?. (Call #1)");
    
    // The underlying dummy model doGenerate should NOT have been called again (still 1 call)
    expect(dummyModel.doGenerate).toHaveBeenCalledTimes(1);
    expect(result2.providerMetadata?.semanticCache?.hit).toBe(true);

    // 6. Third call: "Explain thermodynamics" (completely unrelated prompt)
    const result3 = await generateText({
      model: cachedModel,
      prompt: "Explain thermodynamics",
    });

    // This should result in a cache miss
    expect(result3.text).toBe("Real response to: Explain thermodynamics. (Call #2)");
    expect(dummyModel.doGenerate).toHaveBeenCalledTimes(2);
  }, 15000);
});
