import type { VectorMetadataFilter } from "./vector-store-adapter";

export const ALLOWED_METADATA_FILTER_KEYS = ["userId", "tenantId"] as const;

export type AllowedMetadataFilterKey =
  (typeof ALLOWED_METADATA_FILTER_KEYS)[number];

export function getAllowedMetadataFilterEntries(
  filter?: VectorMetadataFilter
): Array<[AllowedMetadataFilterKey, string]> {
  if (!filter) return [];

  const entries: Array<[AllowedMetadataFilterKey, string]> = [];
  for (const key of ALLOWED_METADATA_FILTER_KEYS) {
    const value = filter[key];
    if (typeof value === "string") {
      entries.push([key, value]);
    }
  }

  return entries;
}
