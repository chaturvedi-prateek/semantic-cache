import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { UpstashVectorAdapter } from "../upstash-vector-adapter";

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

describe("UpstashVectorAdapter", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  const makeAdapter = (namespace?: string) =>
    new UpstashVectorAdapter({
      url: "https://example-vector.upstash.io/",
      token: "test-token",
      namespace,
    });

  it("throws when url or token is missing", () => {
    expect(() => new UpstashVectorAdapter({ url: "", token: "t" })).toThrow();
    expect(() => new UpstashVectorAdapter({ url: "https://x", token: "" })).toThrow();
  });

  it("upserts via POST /upsert with auth header", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ result: "Success" }));

    const adapter = makeAdapter();
    await adapter.upsert("id-1", [0.1, 0.2], { foo: "bar" });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("https://example-vector.upstash.io/upsert");
    expect(init.method).toBe("POST");
    expect(init.headers.Authorization).toBe("Bearer test-token");
    expect(JSON.parse(init.body)).toEqual({
      id: "id-1",
      vector: [0.1, 0.2],
      metadata: { foo: "bar" },
    });
  });

  it("scopes requests to the configured namespace", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ result: "Success" }));

    const adapter = makeAdapter("my-ns");
    await adapter.delete("id-1");

    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe("https://example-vector.upstash.io/delete/my-ns");
  });

  it("queries and converts Upstash scores to cosine similarity", async () => {
    // Upstash cosine score = (1 + cosine) / 2 → score 0.96 ⇒ cosine 0.92
    fetchMock.mockResolvedValue(
      jsonResponse({
        result: [{ id: "a", score: 0.96, metadata: { response: "hi" } }],
      })
    );

    const adapter = makeAdapter();
    const matches = await adapter.query([0.1, 0.2], 3);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("https://example-vector.upstash.io/query");
    expect(JSON.parse(init.body)).toEqual({
      vector: [0.1, 0.2],
      topK: 3,
      includeMetadata: true,
    });

    expect(matches).toHaveLength(1);
    expect(matches[0].id).toBe("a");
    expect(matches[0].score).toBeCloseTo(0.92);
    expect(matches[0].metadata).toEqual({ response: "hi" });
  });

  it("deletes via POST /delete with ids array", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ result: { deleted: 1 } }));

    const adapter = makeAdapter();
    await adapter.delete("id-1");

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("https://example-vector.upstash.io/delete");
    expect(JSON.parse(init.body)).toEqual({ ids: ["id-1"] });
  });

  it("throws a descriptive error on non-2xx responses", async () => {
    fetchMock.mockResolvedValue(
      new Response("unauthorized", { status: 401, statusText: "Unauthorized" })
    );

    const adapter = makeAdapter();
    await expect(adapter.upsert("id", [0.1], {})).rejects.toThrow(
      /UpstashVectorAdapter.*\/upsert.*401/
    );
  });

  it("search returns the cached response on a hit above threshold", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse({
        result: [{ id: "a", score: 0.98, metadata: { response: "cached!" } }],
      })
    );

    const adapter = makeAdapter();
    await expect(adapter.search([0.1], 0.92)).resolves.toBe("cached!");
  });

  it("search returns null below threshold", async () => {
    // score 0.9 ⇒ cosine 0.8, below 0.92 threshold
    fetchMock.mockResolvedValue(
      jsonResponse({
        result: [{ id: "a", score: 0.9, metadata: { response: "cached!" } }],
      })
    );

    const adapter = makeAdapter();
    await expect(adapter.search([0.1], 0.92)).resolves.toBeNull();
  });

  it("search degrades gracefully (null) on backend failure", async () => {
    fetchMock.mockRejectedValue(new Error("network down"));
    const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    const adapter = makeAdapter();
    await expect(adapter.search([0.1], 0.92)).resolves.toBeNull();
    errSpy.mockRestore();
  });

  it("save upserts the response under metadata and never throws", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ result: "Success" }));

    const adapter = makeAdapter();
    await adapter.save([0.1, 0.2], "the answer");

    const [, init] = fetchMock.mock.calls[0];
    const body = JSON.parse(init.body);
    expect(typeof body.id).toBe("string");
    expect(body.vector).toEqual([0.1, 0.2]);
    expect(body.metadata.response).toBe("the answer");

    // failure path is swallowed
    fetchMock.mockRejectedValue(new Error("boom"));
    const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    await expect(adapter.save([0.1], "x")).resolves.toBeUndefined();
    errSpy.mockRestore();
  });
});
