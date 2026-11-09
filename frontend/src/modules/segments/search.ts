export type Section = "segments" | "unusual";
export type SegmentsSearch = { section?: Section; run?: string; unusual?: string };

const ID = /^[0-9a-f]{12}$/;

/** The page's URL: which section, which segmentation and which unusual-rows run are shown. */
export function validateSegmentsSearch(search: Record<string, unknown>): SegmentsSearch {
  const id = (value: unknown) => (typeof value === "string" && ID.test(value) ? value : undefined);
  return {
    section: search.section === "unusual" ? "unusual" : undefined,
    run: id(search.run),
    unusual: id(search.unusual),
  };
}
