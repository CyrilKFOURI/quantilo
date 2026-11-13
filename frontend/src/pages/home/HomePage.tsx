import { Navigate, useSearch } from "@tanstack/react-router";

import { useDatasets } from "../../api/hooks";
import { DatasetList } from "../../features/datasets/DatasetList";
import { HealthIndicator } from "../../features/health/HealthIndicator";
import { ImportPanel } from "../../features/import/ImportPanel";
import { LegalLinks } from "../../features/legal/LegalLinks";
import { useProjects } from "../../features/projects/api";
import { t } from "../../lib/i18n";
import { ErrorNote } from "../../ui/fields";
import { PageHeader } from "../../ui/PageHeader";

/** 087: `?project=<id>` opens one project (its datasets only; an import goes in it, R-PRJ-05). */
export type HomeSearch = { project?: string };

const ID = /^[\w-]{1,64}$/;

export function validateHomeSearch(search: Record<string, unknown>): HomeSearch {
  return typeof search.project === "string" && ID.test(search.project) ? { project: search.project } : {};
}

export function HomePage() {
  const datasets = useDatasets();
  const search = useSearch({ strict: false }) as HomeSearch;
  const projects = useProjects();
  // A project deleted elsewhere (or a stale link): the home page of every dataset
  const open = projects.find((project) => project.id === search.project) ?? null;
  const hasDatasets = (datasets.data?.length ?? 0) > 0;
  // C1: a project has its own page; the old `/?project=` links lead there
  if (open) return <Navigate to="/projects/$projectId" params={{ projectId: open.id }} replace />;

  return (
    <div className="mx-auto flex min-h-full max-w-5xl flex-col gap-8 px-(--gutter) py-8">
      <PageHeader title={t("home.title")} description={t("home.body")} />
      <ImportPanel compact={hasDatasets} project={null} />
      {datasets.isError && <ErrorNote>{datasets.error.message}</ErrorNote>}
      {hasDatasets && datasets.data && <DatasetList datasets={datasets.data} />}
      <div className="mt-auto space-y-3">
        <HealthIndicator />
        <LegalLinks statement />
      </div>
    </div>
  );
}
