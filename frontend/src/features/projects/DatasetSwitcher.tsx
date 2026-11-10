import { Link, useNavigate, useRouterState } from "@tanstack/react-router";

import { useDataset, useDatasets } from "../../api/hooks";
import { t } from "../../lib/i18n";
import { useLineage } from "../../modules/clean/api";
import { DatasetName } from "../../modules/clean/FamilySelect";
import { familyOf, samePageOf } from "../../modules/clean/lineage";
import { Select } from "../../ui/fields";
import { projectOf, useProjects } from "./api";

/** The open dataset in the top bar (087): its project first ("Churn / telco", the project opens
 *  the home page on it), then its name and one native list, *Versions* (065) then *In <project>*
 *  (the other datasets of the project). Picking one keeps the page. On a phone the project is in
 *  the list only (no room for the crumb). Without a project: the 065 name and list, unchanged. */
export function DatasetSwitcher({ datasetId, name }: { datasetId: string; name: string }) {
  const projects = useProjects();
  const project = projectOf(datasetId, projects);
  const lineage = useLineage();
  const datasets = useDatasets().data ?? [];
  const navigate = useNavigate();
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const family = familyOf(datasetId, lineage);
  const original = family[0];
  const originalName = useDataset(original && original.id !== datasetId ? original.id : undefined).data?.name;
  if (!project) return <DatasetName datasetId={datasetId} name={name} />;

  const familyIds = new Set(family.map((member) => member.id));
  const versions = new Set(lineage.map((item) => item.dataset_id));
  const siblings = datasets.filter(
    (dataset) =>
      project.dataset_ids.includes(dataset.id) &&
      dataset.id !== datasetId &&
      !familyIds.has(dataset.id) &&
      !versions.has(dataset.id) &&
      dataset.status === "ready",
  );
  const shownName = original && original.id !== datasetId ? (originalName ?? name) : name;
  return (
    <span className="flex min-w-0 items-center gap-1.5">
      <Link
        to="/"
        search={{ project: project.id }}
        className="max-w-48 shrink truncate text-ink-3 hover:text-accent max-sm:hidden"
        title={t("projects.openTitle", { name: project.name })}
      >
        {project.name}
      </Link>
      <span aria-hidden className="shrink-0 text-ink-3 max-sm:hidden">
        /
      </span>
      <span className="min-w-10 truncate font-medium text-ink-1">{shownName}</span>
      {(family.length > 0 || siblings.length > 0) && (
        <Select
          aria-label={t("projects.switch.label")}
          value={datasetId}
          onChange={(event) => void navigate({ to: samePageOf(pathname, event.target.value) })}
          className="ml-0.5 h-7 max-w-44 shrink-0 text-xs max-sm:max-w-18"
        >
          {family.length > 0 ? (
            <optgroup label={t("projects.switch.versions")}>
              {family.map((member) => (
                <option key={member.id} value={member.id}>
                  {member.id === original?.id
                    ? t("clean.family.original")
                    : t("clean.family.version", { seq: String(member.seq) })}
                </option>
              ))}
            </optgroup>
          ) : (
            <option value={datasetId}>{t("projects.switch.thisOne", { name })}</option>
          )}
          {siblings.length > 0 && (
            <optgroup label={t("projects.switch.inProject", { name: project.name })}>
              {siblings.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.name}
                </option>
              ))}
            </optgroup>
          )}
        </Select>
      )}
    </span>
  );
}
