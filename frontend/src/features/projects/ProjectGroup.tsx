import * as Menu from "@radix-ui/react-dropdown-menu";
import { Link } from "@tanstack/react-router";
import { Check, ChevronRight, FolderInput, FolderOpen, MoreHorizontal, Pencil, Plus, Trash2 } from "lucide-react";
import { useState, type DragEvent, type ReactNode } from "react";

import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { TextInput } from "../../ui/fields";
import type { Project } from "./api";

/** The type of a dataset row being dragged (087): the import zone only takes files, and only the
 *  project groups take this. */
export const DRAG_TYPE = "application/x-quantilo-dataset";
/** Columns of the home table: a group's header and empty state span them all. */
export const COLUMNS = 7;

const menuClass =
  "z-50 min-w-52 max-w-[calc(100vw-32px)] rounded-lg border border-line bg-surface-1 p-1 shadow-[0_12px_32px_-12px_rgb(0_0_0/0.35)]";
const itemClass =
  "flex cursor-default items-center gap-2.5 rounded-md px-2.5 py-1.5 text-[13px] text-ink-1 outline-none select-none data-[disabled]:cursor-not-allowed data-[disabled]:text-ink-3 data-[highlighted]:bg-surface-2 pointer-coarse:min-h-11";

export const countLabel = (count: number) =>
  count === 1 ? t("projects.countOne") : t("projects.count", { count: String(count) });

type GroupProps = {
  project: Project | null;
  /** Datasets shown in the group (cleaned versions included). */
  count: number;
  collapsed: boolean;
  onToggle: () => void;
  /** The project open on its own (`/?project=`): no folding, no link to itself. */
  open?: boolean;
  /** A dataset row is over this group and would move here. */
  dropActive: boolean;
  onDragOver: (event: DragEvent) => void;
  onDragLeave: (event: DragEvent) => void;
  onDrop: (event: DragEvent) => void;
  onRename?: (name: string) => void;
  onDelete?: () => void;
  children: ReactNode;
};

/** A project (or Not in a project) on the home list: a `<tbody>` whose first row is the header.
 *  The whole group, header included, takes a dragged dataset. */
export function ProjectGroup({
  project,
  count,
  collapsed,
  onToggle,
  open = false,
  dropActive,
  onDragOver,
  onDragLeave,
  onDrop,
  onRename,
  onDelete,
  children,
}: GroupProps) {
  const [editing, setEditing] = useState(false);
  const name = project?.name ?? t("projects.none");
  const bodyId = `group-${project?.id ?? "none"}`;
  const folded = collapsed && !open;
  return (
    <tbody
      id={bodyId}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      data-drop={dropActive || undefined}
      className="border-b border-line last:border-b-0 data-drop:[&>tr]:bg-accent-soft"
    >
      <tr className="bg-surface-0">
        <th scope="rowgroup" colSpan={COLUMNS} className="py-1.5 pr-2 pl-1 text-left font-normal">
          {editing && project && onRename ? (
            <NameInput
              initial={project.name}
              label={t("projects.rename")}
              onSubmit={(value) => {
                setEditing(false);
                if (value !== project.name) onRename(value);
              }}
              onCancel={() => setEditing(false)}
            />
          ) : (
            <div className="flex min-h-(--control-h) items-center gap-1.5">
              {open ? (
                <FolderOpen size={15} strokeWidth={1.5} className="mx-1.5 shrink-0 text-ink-3" aria-hidden />
              ) : (
                <button
                  type="button"
                  onClick={onToggle}
                  aria-expanded={!folded}
                  aria-controls={bodyId}
                  aria-label={t(folded ? "projects.expand" : "projects.collapse", { name })}
                  className="inline-flex size-7 shrink-0 items-center justify-center rounded-md text-ink-3 hover:bg-surface-2 hover:text-ink-1"
                >
                  <ChevronRight
                    size={15}
                    strokeWidth={1.5}
                    aria-hidden
                    className={`transition-transform duration-150 motion-reduce:transition-none ${folded ? "" : "rotate-90"}`}
                  />
                </button>
              )}
              {project && !open ? (
                <Link
                  to="/"
                  search={{ project: project.id }}
                  className="min-w-0 truncate text-[13px] font-medium text-ink-1 hover:text-accent"
                  title={t("projects.openTitle", { name })}
                >
                  {name}
                </Link>
              ) : (
                <span className={`min-w-0 truncate text-[13px] font-medium ${project ? "text-ink-1" : "text-ink-2"}`}>{name}</span>
              )}
              <span className="shrink-0 text-xs text-ink-3 tabular-nums">
                {dropActive ? t("projects.dropHere") : countLabel(count)}
              </span>
              {project && onRename && onDelete && (
                <span className="ml-auto">
                  <ProjectMenu project={project} open={open} onRename={() => setEditing(true)} onDelete={onDelete} />
                </span>
              )}
            </div>
          )}
        </th>
      </tr>
      {!folded && children}
    </tbody>
  );
}

/** The empty state of a project: how to fill it, by mouse or by the Move to… menu. */
export function EmptyGroup({ open }: { open: boolean }) {
  return (
    <tr>
      <td colSpan={COLUMNS} className="px-3 py-3">
        <p className="rounded-md border border-dashed border-line px-3 py-4 text-center text-xs text-ink-2">
          {open ? (
            <>
              {t("projects.emptyOpen")}{" "}
              <Link to="/" className="text-ink-1 underline decoration-line underline-offset-4 hover:text-accent hover:decoration-accent">
                {t("projects.all")}
              </Link>
            </>
          ) : (
            t("projects.empty")
          )}
        </p>
      </td>
    </tr>
  );
}

/** The header of a project being created (inline, no dialog: Enter creates, Escape cancels). */
export function NewProjectGroup({
  onSubmit,
  onCancel,
  error,
}: {
  onSubmit: (name: string) => void;
  onCancel: () => void;
  error?: string | null;
}) {
  return (
    <tbody className="border-b border-line">
      <tr className="bg-surface-0">
        <th scope="rowgroup" colSpan={COLUMNS} className="py-1.5 pr-2 pl-1 text-left font-normal">
          <NameInput initial="" label={t("projects.newName")} onSubmit={onSubmit} onCancel={onCancel} placeholder={t("projects.newPlaceholder")} />
          {error && <p className="mt-1 pl-9 text-xs text-bad">{error}</p>}
        </th>
      </tr>
    </tbody>
  );
}

function NameInput({
  initial,
  label,
  placeholder,
  onSubmit,
  onCancel,
}: {
  initial: string;
  label: string;
  placeholder?: string;
  onSubmit: (name: string) => void;
  onCancel: () => void;
}) {
  const [draft, setDraft] = useState(initial);
  const submit = () => {
    const name = draft.trim();
    if (name) onSubmit(name);
    else onCancel();
  };
  return (
    <form
      className="flex items-center gap-1.5"
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
    >
      <FolderOpen size={15} strokeWidth={1.5} className="mx-1.5 shrink-0 text-ink-3" aria-hidden />
      <TextInput
        autoFocus
        aria-label={label}
        placeholder={placeholder}
        maxLength={80}
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={(event) => event.key === "Escape" && onCancel()}
        className="w-full max-w-80 min-w-0"
      />
      <Button type="submit" variant="secondary">
        {t("datasets.save")}
      </Button>
      <Button variant="ghost" onClick={onCancel}>
        {t("import.cancel")}
      </Button>
    </form>
  );
}

function ProjectMenu({
  project,
  open,
  onRename,
  onDelete,
}: {
  project: Project;
  open: boolean;
  onRename: () => void;
  onDelete: () => void;
}) {
  return (
    <Menu.Root>
      <Menu.Trigger asChild>
        <Button variant="ghost" icon={MoreHorizontal} iconOnly aria-label={t("projects.menu", { name: project.name })} title={t("projects.actions")} />
      </Menu.Trigger>
      <Menu.Portal>
        <Menu.Content align="end" sideOffset={4} className={menuClass}>
          {!open && (
            <Menu.Item asChild className={itemClass}>
              <Link to="/" search={{ project: project.id }}>
                <FolderOpen size={14} strokeWidth={1.5} className="text-ink-3" aria-hidden />
                {t("projects.open")}
              </Link>
            </Menu.Item>
          )}
          <Menu.Item className={itemClass} onSelect={onRename}>
            <Pencil size={14} strokeWidth={1.5} className="text-ink-3" aria-hidden />
            {t("projects.rename")}
          </Menu.Item>
          <Menu.Item className={itemClass} onSelect={onDelete}>
            <Trash2 size={14} strokeWidth={1.5} className="text-ink-3" aria-hidden />
            {t("projects.delete")}
          </Menu.Item>
        </Menu.Content>
      </Menu.Portal>
    </Menu.Root>
  );
}

/** "Move to…" of a dataset: the way to move without dragging (keyboard, phone, tablet). */
export function MoveToMenu({
  name,
  current,
  projects,
  onMove,
  onNew,
}: {
  name: string;
  current: string | null;
  projects: Project[];
  onMove: (projectId: string | null) => void;
  onNew: () => void;
}) {
  const choices: { id: string | null; label: string }[] = [
    ...projects.map((project) => ({ id: project.id, label: project.name })),
    { id: null, label: t("projects.none") },
  ];
  return (
    <Menu.Root>
      <Menu.Trigger asChild>
        <Button variant="ghost" icon={FolderInput} iconOnly aria-label={t("projects.moveDataset", { name })} title={t("projects.moveTo")} />
      </Menu.Trigger>
      <Menu.Portal>
        <Menu.Content align="end" sideOffset={4} className={menuClass}>
          <Menu.Label className="px-2.5 pt-1.5 pb-1 text-xs text-ink-3">{t("projects.moveTo")}</Menu.Label>
          {choices.map((choice) => {
            const here = choice.id === current;
            return (
              <Menu.Item
                key={choice.id ?? "none"}
                disabled={here}
                className={itemClass}
                onSelect={() => onMove(choice.id)}
              >
                <span className="inline-flex size-3.5 shrink-0 items-center justify-center">
                  {here && <Check size={13} strokeWidth={2} className="text-accent" aria-hidden />}
                </span>
                <span className={`truncate ${choice.id ? "" : "text-ink-2"}`}>{choice.label}</span>
                {here && <span className="sr-only">{t("projects.current")}</span>}
              </Menu.Item>
            );
          })}
          <Menu.Separator className="my-1 h-px bg-line" />
          <Menu.Item className={itemClass} onSelect={onNew}>
            <Plus size={14} strokeWidth={1.5} className="text-ink-3" aria-hidden />
            {t("projects.newEllipsis")}
          </Menu.Item>
        </Menu.Content>
      </Menu.Portal>
    </Menu.Root>
  );
}
