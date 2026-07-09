import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { PineconeVectorAdapter } from "../src/adapters/pinecone";

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

describe("PineconeVectorAdapter", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  const makeAdapter = (namespace?: string) =>
    new PineconeVectorAdapter({
      apiKey: "test-key",
      indexHost: "my-index-abc.svc.us-east-1-aws.pinecone.io",
      namespace,
    });

  it("throws when apiKey or indexHost is missing", () => {
    expect(
      () => new PineconeVectorAdapter({ apiKey: "", indexHost: "h" })
    ).toThrow();
    expect(
      () => new PineconeVectorAdapter({ apiKey: "k", indexHost: "" })
    ).toThrow();
  });

  it("accepts an indexHost with an https:// scheme", async () => {
    fetchMock.mockResolvedValue(jsonResponse({}));

    const adapter = new PineconeVectorAdapter({
      apiKey: "test-key",
      indexHost: "https://my-index-abc.svc.us-east-1-aws.pinecone.io/",
    });
    await adapter.delete("id-1");

    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe(
      "https://my-index-abc.svc.us-east-1-aws.pinecone.io/vectors/delete"
    );
  });

  it("upserts via POST /vectors/upsert with Api-Key header", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ upsertedCount: 1 }));

    const adapter = makeAdapter("my-ns");
    await adapter.upsert("id-1", [0.1, 0.2], { foo: "bar" });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(
      "https://my-index-abc.svc.us-east-1-aws.pinecone.io/vectors/upsert"
    );
    expect(init.method).toBe("POST");
    expect(init.headers["Api-Key"]).toBe("test-key");
    expect(JSON.parse(init.body)).toEqual({
      vectors: [{ id: "id-1", values: [0.1, 0.2], metadata: { foo: "bar" } }],
      namespace: "my-ns",
    });
  });

  it("queries via POST /query and maps matches", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse({
        matches: [
          { id: "a", score: 0.95, metadata: { response: "hi" } },
          { id: "b", score: -0.2 },
        ],
      })
    );

    const adapter = makeAdapter();
    const matches = await adapter.query([0.1, 0.2], 2);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(
      "https://my-index-abc.svc.us-east-1-aws.pinecone.io/query"
    );
    expect(JSON.parse(init.body)).toEqual({
      vector: [0.1, 0.2],
      topK: 2,
      includeMetadata: true,
    });

    expect(matches).toHaveLength(2);
    expect(matches[0]).toEqual({
      id: "a",
      score: 0.95,
      metadata: { response: "hi" },
    });
    // negative cosine scores are clamped to 0
    expect(matches[1]).toEqual({ id: "b", score: 0, metadata: {} });
  });

  it("passes metadata filters through query requests", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ matches: [] }));

    const adapter = makeAdapter();
    await adapter.query([0.1, 0.2], 2, { userId: "user-1", tenantId: "tenant-1" });

    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body)).toMatchObject({
      filter: {
        userId: { $eq: "user-1" },
        tenantId: { $eq: "tenant-1" },
      },
    });
  });

  it("deletes via POST /vectors/delete with ids and namespace", async () => {
    fetchMock.mockResolvedValue(jsonResponse({}));

    const adapter = makeAdapter("my-ns");
    await adapter.delete("id-1");

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(
      "https://my-index-abc.svc.us-east-1-aws.pinecone.io/vectors/delete"
    );
    expect(JSON.parse(init.body)).toEqual({
      ids: ["id-1"],
      namespace: "my-ns",
    });
  });

  it("throws a descriptive error on non-2xx responses", async () => {
    fetchMock.mockResolvedValue(
      new Response("forbidden", { status: 403, statusText: "Forbidden" })
    );

    const adapter = makeAdapter();
    await expect(adapter.query([0.1], 1)).rejects.toThrow(
      /PineconeVectorAdapter.*\/query.*403/
    );
  });

  it("search returns the cached response on a hit above threshold", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse({
        matches: [{ id: "a", score: 0.97, metadata: { response: "cached!" } }],
      })
    );

    const adapter = makeAdapter();
    await expect(adapter.search([0.1], 0.92)).resolves.toBe("cached!");
  });

  it("search returns null below threshold or on empty results", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse({
        matches: [{ id: "a", score: 0.5, metadata: { response: "cached!" } }],
      })
    );

    const adapter = makeAdapter();
    await expect(adapter.search([0.1], 0.92)).resolves.toBeNull();

    fetchMock.mockResolvedValue(jsonResponse({ matches: [] }));
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
    fetchMock.mockResolvedValue(jsonResponse({ upsertedCount: 1 }));

    const adapter = makeAdapter();
    await adapter.save([0.1, 0.2], "the answer", {
      userId: "user-1",
      tenantId: "tenant-1",
    });

    const [, init] = fetchMock.mock.calls[0];
    const body = JSON.parse(init.body);
    expect(body.vectors).toHaveLength(1);
    expect(typeof body.vectors[0].id).toBe("string");
    expect(body.vectors[0].values).toEqual([0.1, 0.2]);
    expect(body.vectors[0].metadata.response).toBe("the answer");
    expect(body.vectors[0].metadata.userId).toBe("user-1");
    expect(body.vectors[0].metadata.tenantId).toBe("tenant-1");

    // failure path is swallowed
    fetchMock.mockRejectedValue(new Error("boom"));
    const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    await expect(adapter.save([0.1], "x")).resolves.toBeUndefined();
    errSpy.mockRestore();
  });
});
