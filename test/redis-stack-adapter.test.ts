import { describe, it, expect, vi, beforeEach } from "vitest";
import { RedisStackVectorAdapter } from "../redis-stack-adapter";

describe("RedisStackVectorAdapter", () => {
  let sendCommand: ReturnType<typeof vi.fn>;
  let mockClient: { sendCommand: typeof sendCommand };

  beforeEach(() => {
    sendCommand = vi.fn();
    mockClient = { sendCommand };
  });

  const makeAdapter = (ttlSeconds?: number, dimensions?: number) =>
    new RedisStackVectorAdapter({ client: mockClient, ttlSeconds, dimensions });

  // ── constructor ───────────────────────────────────────────────────────────

  it("throws when client is not provided", () => {
    expect(
      () => new RedisStackVectorAdapter({ client: undefined as any })
    ).toThrow("RedisStackVectorAdapter: `client` is required.");
  });

  it("initialises successfully with a valid client", () => {
    expect(makeAdapter()).toBeDefined();
  });

  // ── ensureIndex (FT.CREATE) ───────────────────────────────────────────────

  it("creates the RediSearch index via FT.CREATE on first query()", async () => {
    // FT.CREATE → "OK"; FT.SEARCH → totalCount=0
    sendCommand
      .mockResolvedValueOnce("OK")       // FT.CREATE
      .mockResolvedValueOnce([0]);       // FT.SEARCH total=0

    await makeAdapter().query([0.1], 1);

    const createArgs = sendCommand.mock.calls[0];
    expect(createArgs[0][0]).toBe("FT.CREATE");
    expect(createArgs[0][1]).toBe("semantic_cache_idx");
    expect(createArgs[0]).toContain("COSINE");
    expect(createArgs[0]).toContain("HNSW");
  });

  it("creates the index only once across multiple query() calls", async () => {
    sendCommand.mockResolvedValue([0]); // FT.SEARCH total=0

    const adapter = makeAdapter();
    // First call → FT.CREATE + FT.SEARCH
    sendCommand.mockResolvedValueOnce("OK").mockResolvedValueOnce([0]);
    await adapter.query([0.1], 1);
    // Second call → only FT.SEARCH (index already memoised)
    sendCommand.mockResolvedValueOnce([0]);
    await adapter.query([0.1], 1);

    const createCalls = sendCommand.mock.calls.filter(
      (c) => c[0][0] === "FT.CREATE"
    );
    expect(createCalls).toHaveLength(1);
  });

  it("silently ignores the 'Index already exists' error from FT.CREATE", async () => {
    sendCommand
      .mockRejectedValueOnce(new Error("Index already exists"))
      .mockResolvedValueOnce([0]); // FT.SEARCH total=0

    await expect(makeAdapter().query([0.1], 1)).resolves.toEqual([]);
  });

  it("uses the configured dimensions in the FT.CREATE schema", async () => {
    sendCommand
      .mockResolvedValueOnce("OK")
      .mockResolvedValueOnce([0]);

    await makeAdapter(undefined, 1536).query([0.1], 1);

    const createArgs: string[] = sendCommand.mock.calls[0][0];
    const dimIndex = createArgs.indexOf("DIM");
    expect(createArgs[dimIndex + 1]).toBe("1536");
  });

  // ── upsert ────────────────────────────────────────────────────────────────

  it("upserts via HSET with key prefix, binary embedding, and metadata fields", async () => {
    sendCommand.mockResolvedValue("OK");

    const adapter = makeAdapter();
    await adapter.upsert("id-1", [0.1, 0.2], {
      response: "cached!",
      createdAt: "2024-01-01T00:00:00.000Z",
    });

    expect(sendCommand).toHaveBeenCalledTimes(1);
    const args: (string | Buffer)[] = sendCommand.mock.calls[0][0];
    expect(args[0]).toBe("HSET");
    expect(args[1]).toBe("sc:id-1");

    // embedding field must be a Uint8Array of Float32 bytes
    const embIdx = args.indexOf("embedding");
    expect(embIdx).toBeGreaterThan(1);
    expect(args[embIdx + 1]).toBeInstanceOf(Uint8Array);

    // _metadata JSON round-trips the full metadata object
    const metaIdx = args.indexOf("_metadata");
    expect(metaIdx).toBeGreaterThan(1);
    const meta = JSON.parse(args[metaIdx + 1] as string);
    expect(meta.response).toBe("cached!");

    // response and createdAt are duplicated as individual fields
    const respIdx = args.indexOf("response");
    expect(respIdx).toBeGreaterThan(1);
    expect(args[respIdx + 1]).toBe("cached!");
  });

  it("stores user and tenant metadata as indexed tag fields", async () => {
    sendCommand.mockResolvedValue("OK");

    await makeAdapter().upsert("id-1", [0.1, 0.2], {
      response: "cached!",
      userId: "user-1",
      tenantId: "tenant-1",
    });

    const args: (string | Buffer)[] = sendCommand.mock.calls[0][0];
    expect(args).toContain("userId");
    expect(args).toContain("tenantId");
  });

  it("issues an EXPIRE command when ttlSeconds > 0", async () => {
    sendCommand.mockResolvedValue("OK");

    await makeAdapter(3600).upsert("id-2", [0.1], { response: "x" });

    const calls = sendCommand.mock.calls.map((c) => c[0]);
    const expireCall = calls.find((c) => c[0] === "EXPIRE");
    expect(expireCall).toBeDefined();
    expect(expireCall![1]).toBe("sc:id-2");
    expect(expireCall![2]).toBe("3600");
  });

  it("does not issue EXPIRE when ttlSeconds is 0 (default)", async () => {
    sendCommand.mockResolvedValue("OK");

    await makeAdapter().upsert("id-3", [0.1], {});

    const hasTtl = sendCommand.mock.calls.some((c) => c[0][0] === "EXPIRE");
    expect(hasTtl).toBe(false);
  });

  // ── query ─────────────────────────────────────────────────────────────────

  it("queries via FT.SEARCH with KNN syntax and DIALECT 2", async () => {
    sendCommand
      .mockResolvedValueOnce("OK") // FT.CREATE
      .mockResolvedValueOnce([
        1,
        "sc:abc",
        [
          "_metadata", JSON.stringify({ response: "hit!" }),
          "__score", "0.05",
        ],
      ]);

    const matches = await makeAdapter().query([0.1, 0.2], 1);

    const searchArgs: string[] = sendCommand.mock.calls[1][0];
    expect(searchArgs[0]).toBe("FT.SEARCH");
    expect(searchArgs[1]).toBe("semantic_cache_idx");
    expect(searchArgs[2]).toContain("KNN");
    expect(searchArgs[2]).toContain("@embedding");
    expect(searchArgs).toContain("DIALECT");
    expect(searchArgs).toContain("2");

    expect(matches).toHaveLength(1);
    expect(matches[0].id).toBe("abc");
    // cosine distance 0.05 → similarity 0.95
    expect(matches[0].score).toBeCloseTo(0.95);
    expect(matches[0].metadata).toEqual({ response: "hit!" });
  });

  it("returns an empty array when totalCount is 0", async () => {
    sendCommand
      .mockResolvedValueOnce("OK")
      .mockResolvedValueOnce([0]);

    const matches = await makeAdapter().query([0.1], 1);
    expect(matches).toEqual([]);
  });

  it("clamps similarity scores below 0 and above 1", async () => {
    sendCommand
      .mockResolvedValueOnce("OK")
      .mockResolvedValueOnce([
        2,
        "sc:x",
        ["_metadata", "{}", "__score", "2.5"], // distance > 2 → clamped to 0
        "sc:y",
        ["_metadata", "{}", "__score", "-0.1"], // negative → clamped to 1
      ]);

    const matches = await makeAdapter().query([0.1], 2);
    expect(matches[0].score).toBe(0);
    expect(matches[1].score).toBe(1);
  });

  it("enforces topK >= 1", async () => {
    sendCommand.mockResolvedValueOnce("OK").mockResolvedValueOnce([0]);

    await makeAdapter().query([0.1], 0);

    const searchArgs: string[] = sendCommand.mock.calls[1][0];
    expect(searchArgs[2]).toContain("KNN 1");
  });

  it("applies metadata filters to FT.SEARCH queries", async () => {
    sendCommand.mockResolvedValueOnce("OK").mockResolvedValueOnce([0]);

    await makeAdapter().query([0.1], 1, {
      userId: "user-1",
      tenantId: "tenant-1",
    });

    const searchArgs: string[] = sendCommand.mock.calls[1][0];
    expect(searchArgs[2]).toContain("@userId:{user-1}");
    expect(searchArgs[2]).toContain("@tenantId:{tenant-1}");
  });

  it("strips the 'sc:' key prefix from returned ids", async () => {
    sendCommand
      .mockResolvedValueOnce("OK")
      .mockResolvedValueOnce([
        1,
        "sc:my-uuid-123",
        ["_metadata", "{}", "__score", "0.1"],
      ]);

    const [match] = await makeAdapter().query([0.1], 1);
    expect(match.id).toBe("my-uuid-123");
  });

  it("handles malformed _metadata JSON gracefully (returns empty metadata)", async () => {
    sendCommand
      .mockResolvedValueOnce("OK")
      .mockResolvedValueOnce([
        1,
        "sc:id",
        ["_metadata", "NOT_JSON", "__score", "0.0"],
      ]);

    const [match] = await makeAdapter().query([0.1], 1);
    expect(match.metadata).toEqual({});
  });

  // ── delete ────────────────────────────────────────────────────────────────

  it("deletes via DEL with the sc: key prefix", async () => {
    sendCommand.mockResolvedValue(1);

    await makeAdapter().delete("my-id");

    const args: string[] = sendCommand.mock.calls[0][0];
    expect(args[0]).toBe("DEL");
    expect(args[1]).toBe("sc:my-id");
  });

  // ── search ────────────────────────────────────────────────────────────────

  it("search returns the cached response on a hit above threshold", async () => {
    sendCommand
      .mockResolvedValueOnce("OK") // FT.CREATE
      .mockResolvedValueOnce([
        1,
        "sc:abc",
        [
          "_metadata", JSON.stringify({ response: "cached!" }),
          "__score", "0.04",
        ],
      ]);

    await expect(makeAdapter().search([0.1], 0.92)).resolves.toBe("cached!");
  });

  it("search returns null when distance exceeds (1 - threshold)", async () => {
    // threshold 0.92 → max distance 0.08; score 0.15 > 0.08 → miss
    sendCommand
      .mockResolvedValueOnce("OK")
      .mockResolvedValueOnce([
        1,
        "sc:abc",
        [
          "_metadata", JSON.stringify({ response: "stale" }),
          "__score", "0.15",
        ],
      ]);

    await expect(makeAdapter().search([0.1], 0.92)).resolves.toBeNull();
  });

  it("search returns null when result set is empty", async () => {
    sendCommand.mockResolvedValueOnce("OK").mockResolvedValueOnce([0]);
    await expect(makeAdapter().search([0.1], 0.92)).resolves.toBeNull();
  });

  it("search degrades gracefully (returns null) on backend failure", async () => {
    sendCommand
      .mockResolvedValueOnce("OK")
      .mockRejectedValueOnce(new Error("Redis timeout"));
    const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    await expect(makeAdapter().search([0.1], 0.92)).resolves.toBeNull();
    expect(errSpy).toHaveBeenCalled();
    errSpy.mockRestore();
  });

  // ── save ──────────────────────────────────────────────────────────────────

  it("save creates the index and stores the response under the metadata 'response' key", async () => {
    sendCommand.mockResolvedValue("OK");

    await makeAdapter().save([0.1, 0.2], "the answer", {
      userId: "user-1",
      tenantId: "tenant-1",
    });

    // First call is FT.CREATE, second is HSET
    const createArgs: string[] = sendCommand.mock.calls[0][0];
    expect(createArgs[0]).toBe("FT.CREATE");

    const hsetArgs: (string | Buffer)[] = sendCommand.mock.calls[1][0];
    expect(hsetArgs[0]).toBe("HSET");
    const metaIdx = (hsetArgs as string[]).indexOf("_metadata");
    const meta = JSON.parse(hsetArgs[metaIdx + 1] as string);
    expect(meta.response).toBe("the answer");
    expect(typeof meta.createdAt).toBe("string");
    expect(meta.userId).toBe("user-1");
    expect(meta.tenantId).toBe("tenant-1");
  });

  it("save never throws even when the client rejects", async () => {
    sendCommand.mockRejectedValue(new Error("connection lost"));
    const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    await expect(makeAdapter().save([0.1], "x")).resolves.toBeUndefined();
    expect(errSpy).toHaveBeenCalled();
    errSpy.mockRestore();
  });
});
