import { useNavigate, useRouterState } from "@tanstack/react-router";

import { useDataset } from "../../api/hooks";
import { t } from "../../lib/i18n";
import { Select } from "../../ui/fields";
import { useLineage } from "./api";
import { familyOf, samePageOf } from "./lineage";

/** The dataset's name in the top bar; with cleaned versions (065), a native list of the family
 *  (Original · Cleaned v1 · Cleaned v2): picking one keeps the page and changes the dataset. */
export function DatasetName({ datasetId, name }: { datasetId: string; name: string }) {
  const lineage = useLineage();
  const family = familyOf(datasetId, lineage);
  const navigate = useNavigate();
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const original = family[0];
  // a version's name repeats its original's ("sales · Cleaned v1"): show the original's, and the list
  const originalName = useDataset(original && original.id !== datasetId ? original.id : undefined).data?.name;
  if (!original) return <span className="truncate font-medium text-ink-1">{name}</span>;
  return (
    <span className="flex min-w-0 items-center gap-2">
      <span className="truncate font-medium text-ink-1">{original.id === datasetId ? name : (originalName ?? name)}</span>
      <Select
        aria-label={t("clean.family.label")}
        value={datasetId}
        onChange={(event) => void navigate({ to: samePageOf(pathname, event.target.value) })}
        className="h-7 max-w-44 shrink-0 text-xs"
      >
        {family.map((member) => (
          <option key={member.id} value={member.id}>
            {member.id === original.id ? t("clean.family.original") : t("clean.family.version", { seq: String(member.seq) })}
          </option>
        ))}
      </Select>
    </span>
  );
}
