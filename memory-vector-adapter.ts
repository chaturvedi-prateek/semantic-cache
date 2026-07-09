/**
 * memory-vector-adapter.ts
 *
 * Lightweight in-memory {@link VectorStoreAdapter} implementation intended
 * for local development and tests.
 */

import type {
  VectorMetadata,
  VectorQueryMatch,
  VectorStoreAdapter,
} from "./vector-store-adapter";

/** Metadata key under which `save()` stores the cached LLM response. */
const RESPONSE_METADATA_KEY = "response" as const;

interface MemoryVectorEntry {
  vector: number[];
  metadata: VectorMetadata;
}

/**
 * A simple in-memory vector store backed by a JavaScript `Map`.
 *
 * Similarity is calculated with pure JavaScript cosine similarity, no external
 * dependencies required.
 */
export class MemoryVectorAdapter implements VectorStoreAdapter {
  private readonly entries = new Map<string, MemoryVectorEntry>();

  async upsert(
    id: string,
    vector: number[],
    metadata: VectorMetadata
  ): Promise<void> {
    this.entries.set(id, {
      vector: [...vector],
      metadata: { ...(metadata ?? {}) },
    });
  }

  async query(vector: number[], topK: number): Promise<VectorQueryMatch[]> {
    if (topK <= 0) return [];
    return this.searchMatches(vector, topK, 0);
  }

  async delete(id: string): Promise<void> {
    this.entries.delete(id);
  }

  async search(vector: number[], threshold: number): Promise<string | null> {
    const [best] = await this.searchMatches(vector, 1, threshold);
    if (!best) return null;

    const response = best.metadata[RESPONSE_METADATA_KEY];
    return typeof response === "string" ? response : null;
  }

  async save(promptVector: number[], response: string): Promise<void> {
    await this.upsert(crypto.randomUUID(), promptVector, {
      [RESPONSE_METADATA_KEY]: response,
      createdAt: new Date().toISOString(),
    });
  }

  private async searchMatches(
    vector: number[],
    topK: number,
    minThreshold: number
  ): Promise<VectorQueryMatch[]> {
    const threshold = Math.min(1, Math.max(0, minThreshold));
    const limit = Math.max(1, Math.floor(topK));

    const matches = [...this.entries.entries()]
      .map(([id, entry]) => ({
        id,
        score: this.cosineSimilarity(vector, entry.vector),
        metadata: entry.metadata,
      }))
      .filter((match) => match.score >= threshold)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);

    return matches.map((match) => ({
      id: match.id,
      score: match.score,
      metadata: { ...match.metadata },
    }));
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length === 0 || b.length === 0 || a.length !== b.length) return 0;

    let dot = 0;
    let aNormSq = 0;
    let bNormSq = 0;

    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      aNormSq += a[i] * a[i];
      bNormSq += b[i] * b[i];
    }

    const denominator = Math.sqrt(aNormSq) * Math.sqrt(bNormSq);
    if (denominator === 0) return 0;

    const cosine = dot / denominator;
    return Math.min(1, Math.max(0, cosine));
  }
}
