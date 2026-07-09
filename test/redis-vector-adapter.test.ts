import { describe, it, expect, vi, beforeEach } from "vitest";
import { RedisVectorAdapter } from "../redis-vector-adapter";
import { Redis } from "@upstash/redis";

describe("RedisVectorAdapter", () => {
  let mockRequest: any;
  let mockRedis: Redis;

  beforeEach(() => {
    mockRequest = vi.fn();
    mockRedis = {
      request: mockRequest,
    } as unknown as Redis;
  });

  it("should initialize successfully with client", () => {
    const adapter = new RedisVectorAdapter({ client: mockRedis });
    expect(adapter).toBeDefined();
  });

  it("should throw if neither client nor url/token are provided", () => {
    expect(() => new RedisVectorAdapter({})).toThrow();
  });

  it("should correctly handle save", async () => {
    const adapter = new RedisVectorAdapter({ client: mockRedis, ttlSeconds: 3600 });
    
    // First call to save will trigger FT.CREATE (ensureIndex) and then HSET + EXPIRE
    mockRequest.mockResolvedValue({ result: "OK" });

    const promptVector = new Array(384).fill(0.1);
    await adapter.save(promptVector, "cached_response", {
      userId: "user-1",
      tenantId: "tenant-1",
    });

    // We expect multiple raw command requests:
    // 1st request is FT.CREATE
    // 2nd request is HSET
    // 3rd request is EXPIRE
    expect(mockRequest).toHaveBeenCalledTimes(3);

    // Let's verify the HSET command structure
    const hsetCall = mockRequest.mock.calls[1][0].command;
    expect(hsetCall[0]).toBe("HSET");
    expect(hsetCall[1]).toContain("sc:");
    expect(hsetCall[2]).toBe("vector");
    expect(typeof hsetCall[3]).toBe("string"); // encoded binary string
    expect(hsetCall[4]).toBe("metadata");
    expect(JSON.parse(hsetCall[5])).toMatchObject({
      response: "cached_response",
      userId: "user-1",
      tenantId: "tenant-1",
    });
    expect(hsetCall[6]).toBe("response");
    expect(hsetCall[7]).toBe("cached_response");
    expect(hsetCall[8]).toBe("createdAt");
    expect(hsetCall).toContain("userId");
    expect(hsetCall).toContain("tenantId");

    // Let's verify the EXPIRE command
    const expireCall = mockRequest.mock.calls[2][0].command;
    expect(expireCall).toEqual(["EXPIRE", hsetCall[1], "3600"]);
  });

  it("should correctly perform search on hit within threshold", async () => {
    const adapter = new RedisVectorAdapter({ client: mockRedis });

    // Mock search responses
    // 1st call is save/search ensureIndex -> FT.CREATE
    // 2nd call is FT.SEARCH
    mockRequest
      .mockResolvedValueOnce({ result: "OK" }) // FT.CREATE
      .mockResolvedValueOnce({
        result: [
          1, // totalCount
          "sc:test-uuid", // key
          ["__score", "0.05", "response", "matched_response"] // fields
        ]
      });

    const promptVector = new Array(384).fill(0.1);
    const result = await adapter.search(promptVector, 0.92); // similarity threshold 0.92 -> max distance 0.08

    expect(result).toBe("matched_response");
    expect(mockRequest).toHaveBeenCalledTimes(2);

    const searchCall = mockRequest.mock.calls[1][0].command;
    expect(searchCall[0]).toBe("FT.SEARCH");
    expect(searchCall[1]).toBe("semantic_cache_idx");
    expect(searchCall[2]).toBe("*=>[KNN 1 @vector $vec AS __score]");
  });

  it("should scope search queries with metadata filters", async () => {
    const adapter = new RedisVectorAdapter({ client: mockRedis });

    mockRequest
      .mockResolvedValueOnce({ result: "OK" })
      .mockResolvedValueOnce({ result: [0] });

    const promptVector = new Array(384).fill(0.1);
    await adapter.search(promptVector, 0.92, {
      userId: "user-1",
      tenantId: "tenant-1",
    });

    const searchCall = mockRequest.mock.calls[1][0].command;
    expect(searchCall[2]).toContain("@userId:{user-1}");
    expect(searchCall[2]).toContain("@tenantId:{tenant-1}");
  });

  it("should return null on search score exceeding maxDistance (similarity lower than threshold)", async () => {
    const adapter = new RedisVectorAdapter({ client: mockRedis });

    mockRequest
      .mockResolvedValueOnce({ result: "OK" }) // FT.CREATE
      .mockResolvedValueOnce({
        result: [
          1, // totalCount
          "sc:test-uuid",
          ["__score", "0.15", "response", "matched_response"] // distance 0.15 is greater than maxDistance (0.08)
        ]
      });

    const promptVector = new Array(384).fill(0.1);
    const result = await adapter.search(promptVector, 0.92); // max distance = 1 - 0.92 = 0.08

    expect(result).toBeNull();
  });

  it("should return null on search when totalCount is 0", async () => {
    const adapter = new RedisVectorAdapter({ client: mockRedis });

    mockRequest
      .mockResolvedValueOnce({ result: "OK" }) // FT.CREATE
      .mockResolvedValueOnce({
        result: [0] // totalCount = 0
      });

    const promptVector = new Array(384).fill(0.1);
    const result = await adapter.search(promptVector, 0.92);

    expect(result).toBeNull();
  });

  it("should handle error in search gracefully and return null (non-blocking fallback design)", async () => {
    const adapter = new RedisVectorAdapter({ client: mockRedis });

    mockRequest
      .mockResolvedValueOnce({ result: "OK" }) // FT.CREATE
      .mockRejectedValueOnce(new Error("Redis disconnect!"));

    const promptVector = new Array(384).fill(0.1);
    const result = await adapter.search(promptVector, 0.92);

    expect(result).toBeNull(); // should fall back gracefully
  });
});
