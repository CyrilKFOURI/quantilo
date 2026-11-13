import { useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "@tanstack/react-router";
import { FolderPlus, GitCompareArrows, Pencil, RefreshCw, Trash2, X } from "lucide-react";
import { useState, type DragEvent, type ReactNode } from "react";

import type { Dataset } from "../../api/client";
import { useDeleteDataset, useReimportDataset, useRenameDataset } from "../../api/hooks";
import { formatBytes, formatCount, formatRelativeTime } from "../../lib/format";
import { useStoredState } from "../../lib/useStoredState";
import {
  projectKeys,
  useCreateProject,
  useDeleteProject,
  useAddToProject,
  useMoveDatasets,
  useProjects,
  useRenameProject,
  type Project,
} from "../projects/api";
import { familyIds, groupByProject, groupKey } from "../projects/grouping";
import { DRAG_TYPE, EmptyGroup, MoveToMenu, NewProjectGroup, ProjectGroup } from "../projects/ProjectGroup";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { cleanKeys, useLineage } from "../../modules/clean/api";
import { groupFamilies, type FamilyRow } from "../../modules/clean/lineage";
import { Dialog } from "../../ui/Dialog";
import { Checkbox, ErrorNote, TextInput } from "../../ui/fields";
import { StatusBadge } from "./StatusBadge";

/** The "New project…" of the ticked datasets' Move to… (they go in once it is created). */
const SELECTION = "__selection__";

/** Of two datasets, the one imported last is the new version; the other is the reference. */
export function versionsOrder(first: Dataset, second: Dataset): [Dataset, Dataset] {
  const stamp = (d: Dataset) => d.imported_at ?? d.created_at;
  return stamp(first) >= stamp(second) ? [first, second] : [second, first];
}

export function DatasetList({ datasets, openProject = null }: { datasets: Dataset[]; openProject?: Project | null }) {
  const [deleting, setDeleting] = useState<FamilyRow | null>(null);
  const [withVersions, setWithVersions] = useState(false);
  // 065: each original followed by its cleaned versions, indented
  const rows = groupFamilies(datasets, useLineage());
  const client = useQueryClient();
  const [selected, setSelected] = useState<string[]>([]);
  const navigate = useNavigate();
  const chosen = datasets.filter((dataset) => selected.includes(dataset.id) && dataset.status === "ready");
  // C1: any number can be ticked (to move them together); exactly two are compared.
  const toggle = (id: string) =>
    setSelected((current) => (current.includes(id) ? current.filter((item) => item !== id) : [...current, id]));
  const compare = () => {
    const [first, second] = chosen;
    if (!first || !second) return;
    const [after, before] = versionsOrder(first, second);
    void navigate({ to: "/d/$datasetId/changes", params: { datasetId: after.id }, search: { vs: before.id } });
  };
  const remove = useDeleteDataset();
  const reimport = useReimportDataset();
  // Deleting a dataset that is still importing cancels its import.
  const cancelling = deleting?.dataset.status === "importing";
  const versions = deleting?.children ?? [];
  const confirmDelete = (row: FamilyRow) => {
    const doomed = [row.dataset, ...(withVersions ? row.children : [])];
    void Promise.all(doomed.map((dataset) => remove.mutateAsync(dataset.id))).then(() => {
      setDeleting(null);
      void client.invalidateQueries({ queryKey: cleanKeys.lineage });
      void client.invalidateQueries({ queryKey: projectKeys.all });
    });
  };

  // 087: projects, groups, drag and drop, Move to…
  const projects = useProjects();
  const createProject = useCreateProject();
  const renameProject = useRenameProject();
  const deleteProject = useDeleteProject();
  const move = useMoveDatasets();
  const addTo = useAddToProject();
  const [collapsed, setCollapsed] = useStoredState<string[]>("de.projects.collapsed", []);
  // `pendingMove`: "New project…" from a dataset's Move to… menu: the dataset goes in once created
  const [creating, setCreating] = useState<{ pendingMove: string | null } | null>(null);
  const [deletingProject, setDeletingProject] = useState<Project | null>(null);
  // the row dragged and the group it is dragged from (C1: a dataset may be in several projects)
  const [dragging, setDraggingState] = useState<{ row: FamilyRow; from: string | null } | null>(null);
  const setDragging = (row: FamilyRow | null, from: string | null = null) => setDraggingState(row ? { row, from } : null);
  const [over, setOver] = useState<string | null>(null);
  const error = remove.error ?? reimport.error ?? move.error ?? addTo.error ?? renameProject.error ?? deleteProject.error;
  // R-PRJ-10: no project, no group (the list of before)
  const grouped = projects.length > 0 || creating !== null || openProject !== null;
  const groups = groupByProject(rows, projects).filter((group) => !openProject || group.project?.id === openProject.id);
  const memberOf = (row: FamilyRow) =>
    projects.filter((project) => project.dataset_ids.includes(row.version?.parentId ?? row.dataset.id)).map((project) => project.id);

  /** Out of the group it is shown in (`from`), into `projectId` (C1: its other projects kept). */
  const moveRow = (row: FamilyRow, projectId: string | null, from: string | null) => {
    if (from === projectId) return;
    move.mutate({ datasetIds: familyIds(row, rows), projectId, fromProjectId: from });
  };
  const addRow = (row: FamilyRow, projectId: string) => addTo.mutate({ datasetIds: familyIds(row, rows), projectId });
  // C1: "Move to…" on the ticked datasets (their families), out of every other project
  const selectedRows = rows.filter((row) => selected.includes(row.dataset.id));
  const selectionIds = [...new Set(selectedRows.flatMap((row) => familyIds(row, rows)))];
  const startCreate = (pendingMove: string | null = null) => {
    createProject.reset();
    setCreating({ pendingMove });
  };
  const create = (name: string) =>
    createProject.mutate(name, {
      onSuccess: (project) => {
        const pending = rows.find((row) => row.dataset.id === creating?.pendingMove);
        setCreating(null);
        if (pending) move.mutate({ datasetIds: familyIds(pending, rows), projectId: project.id });
        else if (creating?.pendingMove === SELECTION) move.mutate({ datasetIds: selectionIds, projectId: project.id });
      },
    });
  const toggleGroup = (key: string) =>
    setCollapsed(collapsed.includes(key) ? collapsed.filter((item) => item !== key) : [...collapsed, key]);

  const dropHandlers = (project: Project | null) => {
    const key = groupKey(project);
    const target = project?.id ?? null;
    const accepts = (event: DragEvent) => dragging !== null && event.dataTransfer.types.includes(DRAG_TYPE);
    return {
      dropActive: over === key && dragging !== null && dragging.from !== target,
      onDragOver: (event: DragEvent) => {
        if (!accepts(event)) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = "move";
        if (over !== key) setOver(key);
      },
      onDragLeave: (event: DragEvent) => {
        // only when leaving the group, not when passing from one of its rows to the next
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setOver(null);
      },
      onDrop: (event: DragEvent) => {
        if (!accepts(event) || !dragging) return;
        event.preventDefault();
        moveRow(dragging.row, target, dragging.from);
        setOver(null);
        setDragging(null);
      },
    };
  };

  const renderRows = (list: FamilyRow[], from: string | null = null) =>
    list.map((row) => (
      <DatasetRow
        key={row.dataset.id}
        dataset={row.dataset}
        version={row.version}
        checked={selected.includes(row.dataset.id)}
        onToggle={() => toggle(row.dataset.id)}
        onDelete={() => {
          setWithVersions(false);
          setDeleting(row);
        }}
        onReimport={() => reimport.mutate(row.dataset.id)}
        dragging={dragging !== null && familyIds(dragging.row, rows).includes(row.dataset.id)}
        onDragStart={grouped ? () => setDragging(row, from) : undefined}
        onDragEnd={() => {
          setDragging(null);
          setOver(null);
        }}
        moveMenu={
          projects.length > 0 ? (
            <MoveToMenu
              name={row.dataset.name}
              current={from}
              memberOf={memberOf(row)}
              projects={projects}
              onMove={(projectId) => moveRow(row, projectId, from)}
              onAdd={(projectId) => addRow(row, projectId)}
              onNew={() => startCreate(row.dataset.id)}
            />
          ) : null
        }
      />
    ));

  return (
    <section>
      <div className="mb-2 flex min-h-(--control-h) flex-wrap items-center justify-between gap-x-3 gap-y-2">
        <h2 className="text-xs text-ink-3">{t(openProject ? "projects.datasets" : "datasets.recent")}</h2>
        <div className="flex flex-wrap items-center gap-3">
          {chosen.length === 2 ? (
            <Button icon={GitCompareArrows} onClick={compare}>
              {t("changes.compareVersions")}
            </Button>
          ) : (
            datasets.length > 1 && (
              <span className="text-xs text-ink-3">{t(chosen.length === 1 ? "changes.pickOneMore" : chosen.length > 2 ? "projects.pickTwoToCompare" : "changes.pickTwo")}</span>
            )
          )}
          {/* C1: the ticked datasets moved together (keyboard, phone, tablet: no dragging) */}
          {projects.length > 0 && selectedRows.length > 0 && (
            <MoveToMenu
              name={t("projects.selection", { count: String(selectedRows.length) })}
              current={null}
              memberOf={[]}
              projects={projects}
              label={t("projects.moveSelected", { count: String(selectedRows.length) })}
              onMove={(projectId) => move.mutate({ datasetIds: selectionIds, projectId }, { onSuccess: () => setSelected([]) })}
              onAdd={(projectId) => addTo.mutate({ datasetIds: selectionIds, projectId }, { onSuccess: () => setSelected([]) })}
              onNew={() => startCreate(SELECTION)}
            />
          )}
          {!openProject && (
            <Button icon={FolderPlus} onClick={() => startCreate()} disabled={creating !== null}>
              {t("projects.new")}
            </Button>
          )}
        </div>
      </div>
      {error && (
        <div className="mb-2">
          <ErrorNote>{error.message}</ErrorNote>
        </div>
      )}
      <div className="overflow-x-auto rounded-lg border border-line bg-surface-1">
        {/* Under 768 px the name and the actions stay; shape, size, last opened and status go
            on a line under the name. Short values never break letter by letter (`data-keep-line`). */}
        <table className="w-full border-collapse text-[13px] max-md:[&_tr>:nth-child(n+3):nth-child(-n+6)]:hidden">
          <thead>
            <tr className="border-b border-line text-left text-xs text-ink-3">
              <th scope="col" className="w-0 py-2 pl-3"><span className="sr-only">{t("changes.selectToCompare")}</span></th>
              <th scope="col" className="w-full px-3 py-2 font-normal">{t("datasets.col.name")}</th>
              <th scope="col" data-keep-line className="px-3 py-2 text-right font-normal whitespace-nowrap">{t("datasets.col.shape")}</th>
              <th scope="col" data-keep-line className="px-3 py-2 text-right font-normal whitespace-nowrap">{t("datasets.col.size")}</th>
              <th scope="col" data-keep-line className="px-3 py-2 font-normal whitespace-nowrap">{t("datasets.col.opened")}</th>
              <th scope="col" data-keep-line className="px-3 py-2 font-normal whitespace-nowrap">{t("datasets.col.status")}</th>
              <th scope="col" className="w-0 px-2 py-2"><span className="sr-only">{t("datasets.actions")}</span></th>
            </tr>
          </thead>
          {creating && (
            <NewProjectGroup onSubmit={create} onCancel={() => setCreating(null)} error={createProject.error?.message} />
          )}
          {grouped ? (
            groups
              // Not in a project shows when it has datasets, or while dragging (the way out of a project)
              .filter((group) => group.project || group.rows.length > 0 || dragging !== null)
              .map((group) => {
                const key = groupKey(group.project);
                const isOpen = openProject !== null && group.project?.id === openProject.id;
                const project = group.project;
                return (
                  <ProjectGroup
                    key={key}
                    project={project}
                    count={group.rows.length}
                    collapsed={collapsed.includes(key)}
                    onToggle={() => toggleGroup(key)}
                    open={isOpen}
                    {...dropHandlers(project)}
                    onRename={project ? (name) => renameProject.mutate({ id: project.id, name }) : undefined}
                    onDelete={project ? () => setDeletingProject(project) : undefined}
                  >
                    {group.rows.length > 0 ? renderRows(group.rows, project?.id ?? null) : <EmptyGroup open={isOpen} />}
                  </ProjectGroup>
                );
              })
          ) : (
            <tbody>{renderRows(rows)}</tbody>
          )}
        </table>
      </div>

      {deletingProject && (
        <DeleteProjectDialog
          project={deletingProject}
          count={groups.find((group) => group.project?.id === deletingProject.id)?.rows.length ?? 0}
          pending={deleteProject.isPending}
          onCancel={() => setDeletingProject(null)}
          onConfirm={() =>
            deleteProject.mutate(deletingProject.id, {
              onSuccess: () => {
                setDeletingProject(null);
                if (openProject) void navigate({ to: "/" });
              },
            })
          }
        />
      )}

      {deleting && (
        <Dialog
          open
          onOpenChange={(open) => !open && setDeleting(null)}
          title={t(cancelling ? "datasets.cancelImportTitle" : "datasets.deleteTitle", { name: deleting.dataset.name })}
          size="sm"
          footer={
            <>
              <Button variant="ghost" onClick={() => setDeleting(null)}>
                {t("import.cancel")}
              </Button>
              <Button
                variant="danger"
                disabled={remove.isPending}
                onClick={() => confirmDelete(deleting)}
              >
                {t(cancelling ? "datasets.cancelImport" : "datasets.delete")}
              </Button>
            </>
          }
        >
          <p className="text-ink-2">{t(cancelling ? "datasets.cancelImportBody" : "datasets.deleteBody")}</p>
          {versions.length > 0 && (
            <Checkbox
              className="mt-3 text-[13px] text-ink-1"
              checked={withVersions}
              onChange={(event) => setWithVersions(event.target.checked)}
              label={versions.length === 1 ? t("clean.home.alsoDeleteOne") : t("clean.home.alsoDelete", { count: String(versions.length) })}
            />
          )}
        </Dialog>
      )}
    </section>
  );
}

/** R-PRJ-03: says what stays (every dataset), not only what goes. */
function DeleteProjectDialog({
  project,
  count,
  pending,
  onCancel,
  onConfirm,
}: {
  project: Project;
  count: number;
  pending: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const body =
    count === 0
      ? t("projects.deleteEmpty")
      : count === 1
        ? t("projects.deleteBodyOne")
        : t("projects.deleteBody", { count: String(count) });
  return (
    <Dialog
      open
      onOpenChange={(open) => !open && onCancel()}
      title={t("projects.deleteTitle", { name: project.name })}
      size="sm"
      footer={
        <>
          <Button variant="ghost" onClick={onCancel}>
            {t("import.cancel")}
          </Button>
          <Button variant="danger" disabled={pending} onClick={onConfirm}>
            {t("projects.delete")}
          </Button>
        </>
      }
    >
      <p className="text-ink-2">{body}</p>
    </Dialog>
  );
}

function DatasetRow({
  dataset,
  version,
  checked,
  onToggle,
  onDelete,
  onReimport,
  dragging = false,
  onDragStart,
  onDragEnd,
  moveMenu = null,
}: {
  dataset: Dataset;
  /** 065: a cleaned version, shown indented under its original. */
  version?: FamilyRow["version"];
  checked: boolean;
  onToggle: () => void;
  onDelete: () => void;
  onReimport: () => void;
  /** 087: this row's family is being dragged. */
  dragging?: boolean;
  /** 087: set when projects exist: the row can be dragged onto another project. */
  onDragStart?: () => void;
  onDragEnd?: () => void;
  /** 087: the Move to… menu (keyboard, phone, tablet). */
  moveMenu?: ReactNode;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(dataset.name);
  const rename = useRenameDataset();

  const save = () => {
    const name = draft.trim();
    if (name && name !== dataset.name) rename.mutate({ id: dataset.id, name });
    setEditing(false);
  };

  return (
    <tr
      className={`group border-b border-line last:border-b-0 hover:bg-surface-0 ${onDragStart && !editing ? "cursor-grab active:cursor-grabbing" : ""} ${dragging ? "opacity-50" : ""}`}
      draggable={Boolean(onDragStart) && !editing}
      onDragStart={(event) => {
        if (!onDragStart) return;
        event.dataTransfer.setData(DRAG_TYPE, dataset.id);
        event.dataTransfer.effectAllowed = "move";
        onDragStart();
      }}
      onDragEnd={onDragEnd}
    >
      <td className="py-2 pl-3">
        <input
          type="checkbox"
          className="size-3.5 accent-(--accent)"
          checked={checked}
          disabled={dataset.status !== "ready"}
          onChange={onToggle}
          aria-label={t("changes.selectDataset", { name: dataset.name })}
        />
      </td>
      <td className={`max-w-0 py-2 pr-3 ${version ? "pl-8" : "pl-3"}`}>
        {editing ? (
          <form
            onSubmit={(event) => {
              event.preventDefault();
              save();
            }}
          >
            <TextInput
              autoFocus
              aria-label={t("datasets.rename")}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onBlur={save}
              onKeyDown={(event) => event.key === "Escape" && setEditing(false)}
              className="w-full"
            />
          </form>
        ) : (
          <div className="min-w-0">
            {dataset.status === "ready" ? (
              <Link
                to="/d/$datasetId"
                params={{ datasetId: dataset.id }}
                draggable={false}
                className="block truncate font-medium text-ink-1 hover:text-accent"
              >
                {dataset.name}
              </Link>
            ) : (
              <span className="block truncate font-medium text-ink-2">{dataset.name}</span>
            )}
            {version ? (
              <span className="block truncate text-[11px] text-ink-3">
                {[
                  t("clean.home.version", { seq: String(version.seq) }),
                  version.origin === "auto" && t("clean.versions.auto"),
                  version.latest && t("clean.versions.latest"),
                  version.outOfDate && t("clean.versions.outOfDate"),
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </span>
            ) : (
              <span className="block truncate font-mono text-[11px] text-ink-3" title={dataset.source_path}>
                {dataset.source_path}
              </span>
            )}
            <span className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs text-ink-2 tabular-nums md:hidden">
              <span>
                {formatCount(dataset.row_count)} × {formatCount(dataset.col_count)}
              </span>
              <span className="text-ink-3">{formatBytes(dataset.size_bytes)}</span>
              <span className="text-ink-3">{formatRelativeTime(dataset.last_opened_at ?? dataset.created_at)}</span>
              <StatusBadge dataset={dataset} />
            </span>
          </div>
        )}
      </td>
      <td data-keep-line className="px-3 py-2 text-right whitespace-nowrap text-ink-2 tabular-nums">
        {formatCount(dataset.row_count)} × {formatCount(dataset.col_count)}
      </td>
      <td data-keep-line className="px-3 py-2 text-right whitespace-nowrap text-ink-2 tabular-nums">
        {formatBytes(dataset.size_bytes)}
      </td>
      <td data-keep-line className="px-3 py-2 whitespace-nowrap text-ink-2">
        {formatRelativeTime(dataset.last_opened_at ?? dataset.created_at)}
      </td>
      <td data-keep-line className="px-3 py-2 whitespace-nowrap">
        <StatusBadge dataset={dataset} />
      </td>
      <td data-keep-line className="px-2 py-2">
        <div className="flex justify-end gap-0.5 opacity-60 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100">
          {moveMenu}
          <Button
            variant="ghost"
            icon={Pencil}
            iconOnly
            aria-label={t("datasets.rename")}
            title={t("datasets.rename")}
            onClick={() => {
              setDraft(dataset.name);
              setEditing(true);
            }}
          />
          <Button
            variant="ghost"
            icon={RefreshCw}
            iconOnly
            aria-label={t("datasets.reimport")}
            title={t("datasets.reimport")}
            disabled={dataset.status === "importing" || dataset.source_state === "missing"}
            onClick={onReimport}
          />
          <Button
            variant="ghost"
            icon={dataset.status === "importing" ? X : Trash2}
            iconOnly
            aria-label={t(dataset.status === "importing" ? "datasets.cancelImport" : "datasets.delete")}
            title={t(dataset.status === "importing" ? "datasets.cancelImport" : "datasets.delete")}
            onClick={onDelete}
          />
        </div>
      </td>
    </tr>
  );
}
