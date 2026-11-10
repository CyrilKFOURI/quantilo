import type { Dataset, VersionInfo } from "../../api/client";
import type { Family, FamilyVersion } from "../../features/lineage/api";
import { t } from "../../lib/i18n";

/* 095: what the Changes page compares with. A target is another dataset's id, or `@n` for an
   earlier import n of this dataset (sent as `?version=n`), or null for the server's default (the
   previous import). */

export type Target = string | null;
export type VersusOption = { value: string; label: string };

export const importTarget = (version: number) => `@${version}`;
export const importOf = (target: Target) => (target?.startsWith("@") ? Number(target.slice(1)) : null);

const day = (iso: string) => iso.slice(0, 10);
export const versionName = (seq: number) => t("clean.versions.name", { seq: String(seq) });

/** The cleaned version `id` of the family, if it is one. */
export function cleanedVersion(family: Family | null, id: string): FamilyVersion | undefined {
  return family?.versions.find((version) => version.dataset_id === id);
}

/** R-095-01: a cleaned version compares with its original; anything else with its previous
 *  import (the server's default), or, never reimported, with its latest cleaned version. An
 *  explicit `version` or `vs` wins. */
export function targetOf(
  datasetId: string,
  family: Family | null,
  imports: VersionInfo[],
  vs: string | undefined,
  version: number | undefined,
): Target {
  if (version) return importTarget(version);
  if (vs && vs !== datasetId) return vs;
  if (cleanedVersion(family, datasetId)) return family?.original_id ?? null;
  if (imports.some((info) => !info.current)) return null;
  const latest = (family?.versions ?? []).filter((one) => one.status === "ready").sort((a, b) => b.seq - a.seq)[0];
  return latest?.dataset_id ?? null;
}

/** R-095-04: "This file's versions" (cleaned versions, the original, then earlier imports; newest
 *  first, the current one left out) and "Other datasets". */
export function versusGroups(
  datasetId: string,
  family: Family | null,
  imports: VersionInfo[],
  datasets: Dataset[],
): { family: VersusOption[]; others: VersusOption[] } {
  const versions = [...(family?.versions ?? [])].filter((version) => version.status === "ready").sort((a, b) => b.seq - a.seq);
  const relatives = new Set([family?.original_id, ...versions.map((version) => version.dataset_id)]);
  const own: VersusOption[] = versions
    .filter((version) => version.dataset_id !== datasetId)
    .map((version) => ({ value: version.dataset_id, label: versionName(version.seq) }));
  if (family && versions.length > 0 && family.original_id !== datasetId) {
    own.push({ value: family.original_id, label: t("changes.versus.original") });
  }
  const earlier = imports.filter((info) => !info.current);
  earlier.forEach((info, index) =>
    own.push({
      value: importTarget(info.version),
      label: t(index === 0 ? "changes.versus.previousImport" : "changes.versus.import", { date: day(info.imported_at) }),
    }),
  );
  const others = datasets
    .filter((other) => other.id !== datasetId && other.status === "ready" && !relatives.has(other.id))
    .map((other) => ({ value: other.id, label: other.name }));
  return { family: own, others };
}

/** The select's value: the target, or the previous import it stands for by default. */
export function pickedValue(target: Target, imports: VersionInfo[]): string | null {
  if (target) return target;
  const previous = imports.find((info) => !info.current);
  return previous ? importTarget(previous.version) : null;
}

/** R-095-02/03: the sentence that names both sides, and the cleaning steps that explain them. */
export function relationOf(
  datasetId: string,
  target: Target,
  family: Family | null,
  imports: VersionInfo[],
  datasets: Dataset[],
): { sentence: string; version: FamilyVersion | null } | null {
  const version = importOf(target);
  if (target === null || version !== null) {
    const chosen = version === null ? imports.find((info) => !info.current) : imports.find((info) => info.version === version);
    if (!chosen) return null;
    const key = version === null ? "changes.versus.sincePrevious" : "changes.versus.sinceImport";
    return { sentence: t(key, { date: day(chosen.imported_at) }), version: null };
  }
  const current = cleanedVersion(family, datasetId);
  const other = cleanedVersion(family, target);
  if (current && target === family?.original_id) {
    return { sentence: t("changes.versus.byCleaning", { version: versionName(current.seq) }), version: current };
  }
  if (other && datasetId === family?.original_id) {
    return { sentence: t("changes.versus.originalVs", { version: versionName(other.seq) }), version: other };
  }
  if (current && other) {
    return { sentence: t("changes.versus.twoCleanings", { after: versionName(current.seq), before: versionName(other.seq) }), version: null };
  }
  const name = datasets.find((dataset) => dataset.id === target)?.name ?? target;
  return { sentence: t("changes.versus.otherDataset", { name }), version: null };
}
