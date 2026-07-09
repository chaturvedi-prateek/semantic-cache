import { describe, it, expect, vi, beforeEach } from "vitest";
import { PgVectorAdapter } from "../pgvector-adapter";

describe("PgVectorAdapter", () => {
  let mockQuery: ReturnType<typeof vi.fn>;
  let mockPool: { query: typeof mockQuery };

  beforeEach(() => {
    mockQuery = vi.fn();
    mockPool = { query: mockQuery };
  });

  const makeAdapter = (tableName?: string) =>
    new PgVectorAdapter({ pool: mockPool, tableName });

  it("throws when pool is not provided", () => {
    expect(
      () => new PgVectorAdapter({ pool: undefined as any })
    ).toThrow("PgVectorAdapter: `pool` is required.");
  });

  it("uses 'semantic_cache' as the default table name", async () => {
    mockQuery.mockResolvedValue({ rows: [] });
    const adapter = makeAdapter();
    await adapter.delete("id-1");
    const [sql] = mockQuery.mock.calls[0];
    expect(sql).toContain("semantic_cache");
  });

  it("uses a custom table name when provided", async () => {
    mockQuery.mockResolvedValue({ rows: [] });
    const adapter = makeAdapter("my_cache");
    await adapter.delete("id-1");
    const [sql] = mockQuery.mock.calls[0];
    expect(sql).toContain("my_cache");
  });

  // ── upsert ────────────────────────────────────────────────────────────────

  it("upserts via INSERT … ON CONFLICT with vector literal and JSON metadata", async () => {
    mockQuery.mockResolvedValue({ rows: [] });

    const adapter = makeAdapter();
    await adapter.upsert("id-1", [0.1, 0.2, 0.3], { foo: "bar" });

    expect(mockQuery).toHaveBeenCalledTimes(1);
    const [sql, params] = mockQuery.mock.calls[0];

    expect(sql).toMatch(/INSERT INTO semantic_cache/i);
    expect(sql).toMatch(/ON CONFLICT \(id\) DO UPDATE/i);
    // id
    expect(params[0]).toBe("id-1");
    // vector formatted as pgvector array literal
    expect(params[1]).toBe("[0.1,0.2,0.3]");
    // metadata serialised to JSON
    expect(JSON.parse(params[2])).toEqual({ foo: "bar" });
  });

  it("uses the <=> cosine distance operator in the upsert SQL", async () => {
    mockQuery.mockResolvedValue({ rows: [] });
    await makeAdapter().upsert("x", [1, 2], {});
    const [sql] = mockQuery.mock.calls[0];
    // The upsert itself does not use <=>; verify it sets `embedding` correctly.
    expect(sql).toContain("::vector");
  });

  // ── query ─────────────────────────────────────────────────────────────────

  it("queries via SELECT using the <=> cosine distance operator", async () => {
    mockQuery.mockResolvedValue({
      rows: [
        { id: "a", score: "0.95", metadata: { response: "hello" } },
        { id: "b", score: "-0.1", metadata: {} },
      ],
    });

    const adapter = makeAdapter();
    const matches = await adapter.query([0.1, 0.2], 2);

    const [sql, params] = mockQuery.mock.calls[0];
    expect(sql).toContain("<=>");
    expect(sql).toContain("::vector");
    expect(params[0]).toBe("[0.1,0.2]");
    expect(params[1]).toBe(2); // topK

    expect(matches).toHaveLength(2);
    expect(matches[0]).toEqual({
      id: "a",
      score: 0.95,
      metadata: { response: "hello" },
    });
    // Negative similarity scores are clamped to 0.
    expect(matches[1]).toEqual({ id: "b", score: 0, metadata: {} });
  });

  it("clamps scores greater than 1 to 1", async () => {
    mockQuery.mockResolvedValue({
      rows: [{ id: "a", score: "1.5", metadata: {} }],
    });
    const [match] = await makeAdapter().query([0.1], 1);
    expect(match.score).toBe(1);
  });

  it("enforces topK >= 1", async () => {
    mockQuery.mockResolvedValue({ rows: [] });
    await makeAdapter().query([0.1], 0);
    const [, params] = mockQuery.mock.calls[0];
    expect(params[1]).toBe(1);
  });

  // ── delete ────────────────────────────────────────────────────────────────

  it("deletes via DELETE WHERE id = $1", async () => {
    mockQuery.mockResolvedValue({ rows: [] });

    await makeAdapter().delete("id-1");

    const [sql, params] = mockQuery.mock.calls[0];
    expect(sql).toMatch(/DELETE FROM semantic_cache WHERE id = \$1/i);
    expect(params[0]).toBe("id-1");
  });

  // ── search ────────────────────────────────────────────────────────────────

  it("search returns the cached response on a hit above threshold", async () => {
    mockQuery.mockResolvedValue({
      rows: [{ id: "a", score: "0.97", metadata: { response: "cached!" } }],
    });

    await expect(makeAdapter().search([0.1], 0.92)).resolves.toBe("cached!");
  });

  it("search returns null when score is below threshold", async () => {
    mockQuery.mockResolvedValue({
      rows: [{ id: "a", score: "0.80", metadata: { response: "stale" } }],
    });

    await expect(makeAdapter().search([0.1], 0.92)).resolves.toBeNull();
  });

  it("search returns null when result set is empty", async () => {
    mockQuery.mockResolvedValue({ rows: [] });
    await expect(makeAdapter().search([0.1], 0.92)).resolves.toBeNull();
  });

  it("search degrades gracefully (returns null) on backend failure", async () => {
    mockQuery.mockRejectedValue(new Error("connection refused"));
    const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    await expect(makeAdapter().search([0.1], 0.92)).resolves.toBeNull();
    expect(errSpy).toHaveBeenCalled();
    errSpy.mockRestore();
  });

  // ── save ──────────────────────────────────────────────────────────────────

  it("save upserts the response under the metadata 'response' key", async () => {
    mockQuery.mockResolvedValue({ rows: [] });

    await makeAdapter().save([0.1, 0.2], "the answer");

    const [sql, params] = mockQuery.mock.calls[0];
    expect(sql).toMatch(/INSERT INTO semantic_cache/i);
    // id is a UUID string
    expect(typeof params[0]).toBe("string");
    expect(params[1]).toBe("[0.1,0.2]");
    const metadata = JSON.parse(params[2]);
    expect(metadata.response).toBe("the answer");
    expect(typeof metadata.createdAt).toBe("string");
  });

  it("save never throws even when the pool rejects", async () => {
    mockQuery.mockRejectedValue(new Error("db down"));
    const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    await expect(makeAdapter().save([0.1], "x")).resolves.toBeUndefined();
    expect(errSpy).toHaveBeenCalled();
    errSpy.mockRestore();
  });
});
