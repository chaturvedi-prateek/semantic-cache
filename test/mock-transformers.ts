import { vi } from "vitest";

vi.mock("@huggingface/transformers", () => ({
  pipeline: vi.fn(async () => async (text: string) => {
    const vector = new Float32Array(384);
    for (let i = 0; i < vector.length; i++) {
      const code = text.charCodeAt(i % text.length) || i + 1;
      vector[i] = ((code % 97) + 1) / 100;
    }

    const magnitude = Math.sqrt(
      vector.reduce((sum, value) => sum + value * value, 0)
    );
    for (let i = 0; i < vector.length; i++) {
      vector[i] = vector[i] / magnitude;
    }

    return { data: vector, dims: [1, 384] };
  }),
}));
