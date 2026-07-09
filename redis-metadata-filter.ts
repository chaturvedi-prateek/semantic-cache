import type { VectorMetadataFilter } from "./vector-store-adapter";
import { getAllowedMetadataFilterEntries } from "./metadata-filter";

export function escapeRedisTagValue(value: string): string {
  let escaped = "";
  for (const char of value) {
    escaped += /[A-Za-z0-9_-]/.test(char) ? char : `\\${char}`;
  }
  return escaped;
}

export function buildRedisFilterPrefix(
  filter?: VectorMetadataFilter
): string {
  if (!filter) return "*";

  const clauses = getAllowedMetadataFilterEntries(filter)
    .map(([key, value]) => `@${key}:{${escapeRedisTagValue(value)}}`);

  return clauses.length > 0 ? clauses.join(" ") : "*";
}
