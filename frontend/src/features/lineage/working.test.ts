import { describe, expect, it } from "vitest";

import type { Family } from "./api";
import { latestReady, samePageOn } from "./working";

const version = (seq: number, status: string) => ({ dataset_id: `v${seq}`, seq, status }) as Family["versions"][number];

describe("useWorkingDataset helpers (C-62)", () => {
  it("picks the latest ready cleaned version", () => {
    const family = { dataset_id: "o", original_id: "o", versions: [version(1, "ready"), version(3, "importing"), version(2, "ready")] } as Family;
    expect(latestReady(family)?.seq).toBe(2);
    expect(latestReady({ ...family, versions: [] })).toBeNull();
    expect(latestReady(null)).toBeNull();
  });

  it("keeps the page when it changes the dataset", () => {
    expect(samePageOn("/d/o/kpi/x", "o", "v2")).toBe("/d/v2/kpi/x");
    expect(samePageOn("/recipes", "o", "v2")).toBe("/d/v2");
  });
});
