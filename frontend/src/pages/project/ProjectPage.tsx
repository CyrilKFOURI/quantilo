import { useQueryClient } from "@tanstack/react-query";
import { Link, useParams } from "@tanstack/react-router";
import { Images, Network } from "lucide-react";

import type { Dataset } from "../../api/client";
import { useDatasets, useUpdateColumns } from "../../api/hooks";
import { DatasetList } from "../../features/datasets/DatasetList";
import { ImportPanel } from "../../features/import/ImportPanel";
import { projectKeys, useProjectImages, useProjects, type ImageRef, type Project } from "../../features/projects/api";
import { countLabel } from "../../features/projects/ProjectGroup";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Button, buttonClass } from "../../ui/Button";
import { ErrorNote } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { PageHeader } from "../../ui/PageHeader";

/** C1 (087, "page projet séparée"): `/projects/$projectId`. Its tables (the home list of this
 *  project: Move to…, drag, versions), its image sets, and the columns of its tables that name its
 *  images (R-PRJ-11), each with "Mark as image reference". Nothing is computed for the images
 *  part without both a table and an image set. */
export function ProjectPage() {
  const { projectId } = useParams({ strict: false }) as { projectId: string };
  const projects = useProjects();
  const datasets = useDatasets();
  const project = projects.find((item) => item.id === projectId) ?? null;
  if (!datasets.data) return <Loading label={t("browse.loading")} className="m-(--gutter) h-40 rounded-lg" />;
  if (!project) {
    return (
      <div className="mx-auto max-w-5xl px-(--gutter) py-8">
        <ErrorNote>{t("projects.page.notFound")}</ErrorNote>
      </div>
    );
  }
  const members = datasets.data.filter((dataset) => project.dataset_ids.includes(dataset.id));
  const tables = members.filter((dataset) => dataset.format !== "images");
  const images = members.filter((dataset) => dataset.format === "images");
  return (
    <div className="mx-auto flex min-h-full max-w-5xl flex-col gap-8 px-(--gutter) py-8">
      <div className="space-y-1.5">
        <nav aria-label={t("projects.breadcrumb")} className="text-[13px] text-ink-3">
          <Link to="/" className="hover:text-accent">
            {t("projects.all")}
          </Link>
          <span aria-hidden className="px-1.5">
            /
          </span>
        </nav>
        <div className="flex flex-wrap items-end justify-between gap-3">
          <PageHeader title={project.name} description={countLabel(project.dataset_ids.length)} />
          {tables.length >= 2 && (
            <Link to="/projects/$projectId/model" params={{ projectId: project.id }} className={buttonClass("secondary")}>
              <Network size={14} strokeWidth={1.5} aria-hidden />
              {t("model.open")}
            </Link>
          )}
        </div>
      </div>
      <ImportPanel compact={members.length > 0} project={project} />
      <section aria-label={t("projects.page.tables")} className="space-y-2">
        <h2 className="text-sm font-medium text-ink-1">{t("projects.page.tables")}</h2>
        <DatasetList datasets={tables} openProject={project} />
      </section>
      <ImageSets images={images} />
      {images.length > 0 && tables.length > 0 && <ImageReferences project={project} tables={tables} images={images} />}
    </div>
  );
}

function ImageSets({ images }: { images: Dataset[] }) {
  return (
    <section aria-label={t("projects.page.images")} className="space-y-2">
      <h2 className="text-sm font-medium text-ink-1">{t("projects.page.images")}</h2>
      {images.length === 0 ? (
        <p className="text-[13px] text-ink-3">{t("projects.page.noImages")}</p>
      ) : (
        <>
          <p className="text-xs text-ink-3">{t("projects.page.imagesHint")}</p>
          <ul className="divide-y divide-line overflow-hidden rounded-lg border border-line bg-surface-1">
            {images.map((dataset) => (
              <li key={dataset.id} className="flex min-w-0 flex-wrap items-center justify-between gap-2 px-3 py-2 text-[13px]">
                <span className="flex min-w-0 items-center gap-2">
                  <Images size={15} strokeWidth={1.5} className="shrink-0 text-ink-3" aria-hidden />
                  <span className="min-w-0 truncate text-ink-1">{dataset.name}</span>
                  <span className="shrink-0 text-xs text-ink-3 tabular-nums">{formatCount(dataset.row_count ?? 0)}</span>
                </span>
                <Link to={"/d/$datasetId/vision" as never} params={{ datasetId: dataset.id } as never} className="text-xs text-accent hover:underline">
                  {t("projects.page.openVision")}
                </Link>
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}

function ImageReferences({ project, tables, images }: { project: Project; tables: Dataset[]; images: Dataset[] }) {
  const found = useProjectImages(project.id, true);
  const names = new Map([...tables, ...images].map((dataset) => [dataset.id, dataset.name]));
  return (
    <section aria-label={t("projects.page.refs")} className="space-y-2">
      <h2 className="text-sm font-medium text-ink-1">{t("projects.page.refs")}</h2>
      <p className="max-w-prose text-xs text-ink-3">{t("projects.page.refsHint")}</p>
      {found.isError && <ErrorNote>{found.error.message}</ErrorNote>}
      {found.isPending && <Loading label={t("projects.page.refs")} className="h-16 rounded-md" compact />}
      {found.data && found.data.references.length === 0 && <p className="text-[13px] text-ink-3">{t("projects.page.refsNone")}</p>}
      {found.data && found.data.references.length > 0 && (
        <ul className="space-y-1.5">
          {found.data.references.map((ref) => (
            <Reference key={`${ref.dataset_id}:${ref.column}:${ref.image_set_id}`} reference={ref} table={names.get(ref.dataset_id) ?? ref.dataset_id} projectId={project.id} />
          ))}
        </ul>
      )}
    </section>
  );
}

function Reference({ reference, table, projectId }: { reference: ImageRef; table: string; projectId: string }) {
  const update = useUpdateColumns(reference.dataset_id);
  const client = useQueryClient();
  return (
    <li className="flex flex-wrap items-start justify-between gap-2 rounded-md border border-line bg-surface-1 px-3 py-2">
      <p className="min-w-0 flex-1 basis-72 text-xs text-ink-2">
        <span className="font-medium text-ink-1">
          {table} · <span className="font-mono">{reference.column}</span>
        </span>{" "}
        {reference.reason}
        {reference.examples.length > 0 && <span className="block truncate font-mono text-[11px] text-ink-3">{reference.examples.join(", ")}</span>}
      </p>
      {reference.applied ? (
        <span className="text-xs text-ink-3">{t("projects.page.applied")}</span>
      ) : (
        <Button
          disabled={update.isPending}
          onClick={() =>
            update.mutate([{ name: reference.column, role: "image_ref" }], {
              onSuccess: () => void client.invalidateQueries({ queryKey: [...projectKeys.all, projectId, "images"] }),
            })
          }
        >
          {t("projects.page.apply")}
        </Button>
      )}
      {update.isError && <ErrorNote>{update.error.message}</ErrorNote>}
    </li>
  );
}
