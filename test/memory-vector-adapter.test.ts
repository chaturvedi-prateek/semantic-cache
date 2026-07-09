import { describe, it, expect } from "vitest";
import { MemoryVectorAdapter } from "../memory-vector-adapter";

describe("MemoryVectorAdapter", () => {
  it("upserts and queries by descending cosine similarity", async () => {
    const adapter = new MemoryVectorAdapter();

    await adapter.upsert("a", [1, 0], { response: "A" });
    await adapter.upsert("b", [0.8, 0.2], { response: "B" });
    await adapter.upsert("c", [0, 1], { response: "C" });

    const matches = await adapter.query([1, 0], 2);
    expect(matches).toHaveLength(2);
    expect(matches[0].id).toBe("a");
    expect(matches[1].id).toBe("b");
    expect(matches[0].score).toBeGreaterThan(matches[1].score);
  });

  it("respects topK and handles non-positive topK", async () => {
    const adapter = new MemoryVectorAdapter();

    await adapter.upsert("a", [1, 0], {});
    await adapter.upsert("b", [0, 1], {});

    await expect(adapter.query([1, 0], 1)).resolves.toHaveLength(1);
    await expect(adapter.query([1, 0], 0)).resolves.toEqual([]);
  });

  it("search returns response when score is above threshold", async () => {
    const adapter = new MemoryVectorAdapter();
    await adapter.upsert("a", [1, 0], { response: "cached!" });

    await expect(adapter.search([0.9, 0.1], 0.9)).resolves.toBe("cached!");
    await expect(adapter.search([0.9, 0.1], 0.999)).resolves.toBeNull();
  });

  it("search returns null when response metadata is missing", async () => {
    const adapter = new MemoryVectorAdapter();
    await adapter.upsert("a", [1, 0], { foo: "bar" });

    await expect(adapter.search([1, 0], 0.5)).resolves.toBeNull();
  });

  it("save stores response and delete removes entries", async () => {
    const adapter = new MemoryVectorAdapter();

    await adapter.save([1, 0], "from-save");
    await expect(adapter.search([1, 0], 0.99)).resolves.toBe("from-save");

    await adapter.upsert("delete-me", [1, 0], { response: "gone" });
    await adapter.delete("delete-me");
    const matches = await adapter.query([1, 0], 10);
    expect(matches.find((m) => m.id === "delete-me")).toBeUndefined();
  });

  it("returns score 0 for zero vectors or mismatched dimensions", async () => {
    const adapter = new MemoryVectorAdapter();

    await adapter.upsert("zero", [0, 0], { response: "x" });
    await adapter.upsert("dim", [1, 0, 0], { response: "y" });

    const zeroMatches = await adapter.query([1, 0], 10);
    const zero = zeroMatches.find((m) => m.id === "zero");
    expect(zero?.score).toBe(0);

    const mismatchedMatches = await adapter.query([1, 0], 10);
    const dim = mismatchedMatches.find((m) => m.id === "dim");
    expect(dim?.score).toBe(0);
  });
});
