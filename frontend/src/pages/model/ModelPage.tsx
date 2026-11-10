import { Link, useParams } from "@tanstack/react-router";
import { Check, LayoutGrid, Maximize, Table2, ZoomIn, ZoomOut } from "lucide-react";
import { useCallback, useMemo, useState } from "react";

import { BuildDialog } from "../../features/model/BuildDialog";
import { Builds } from "../../features/model/Builds";
import { useAcceptAll, useModel, useSaveLayout, type Model } from "../../features/model/api";
import { roleLabel, summary } from "../../features/model/labels";
import { planeSize, starLayout, type Point } from "../../features/model/layout";
import { fitScale, ModelCanvas, type Selection } from "../../features/model/ModelCanvas";
import { ModelPanel, RelationList } from "../../features/model/ModelPanel";
import { useProjects } from "../../features/projects/api";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { PHONE, useMedia } from "../../lib/useMedia";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { ErrorNote } from "../../ui/fields";
import { Loading } from "../../ui/Loading";

/** The model of a project (088, docs/pages/18-modele.md): `/projects/$projectId/model`. */
export function ModelPage() {
  const { projectId } = useParams({ strict: false }) as { projectId: string };
  const projects = useProjects();
  const project = projects.find((item) => item.id === projectId);
  const model = useModel(projectId);
  const phone = useMedia(PHONE);

  return (
    <div className="flex min-h-full flex-col gap-5 px-(--gutter) py-6">
      <nav aria-label={t("model.breadcrumb")} className="text-[13px] text-ink-3">
        <Link to="/" className="hover:text-accent">
          {t("projects.all")}
        </Link>
        <span aria-hidden className="px-1.5">
          /
        </span>
        {project ? (
          <Link to="/" search={{ project: project.id }} className="hover:text-accent">
            {project.name}
          </Link>
        ) : (
          <span>…</span>
        )}
      </nav>
      {model.isPending && <Loading label={t("model.loading")} className="h-[420px] rounded-lg" />}
      {model.isError && <ErrorNote>{model.error.message}</ErrorNote>}
      {model.data && <ModelView projectId={projectId} model={model.data} phone={phone} />}
    </div>
  );
}

function ModelView({ projectId, model, phone }: { projectId: string; model: Model; phone: boolean }) {
  const [selected, setSelected] = useState<Selection>(null);
  const [building, setBuilding] = useState(false);
  const [moved, setMoved] = useState<Record<string, Point>>({});
  const [room, setRoom] = useState<number | undefined>(undefined);
  // null: fit the diagram's width (not below 70 %); a number: the user's zoom
  const [zoom, setZoom] = useState<number | null>(null);
  // the diagram's width, followed as the window changes (a ref callback, not an effect)
  const measure = useCallback((node: HTMLDivElement | null) => {
    if (!node || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) => setRoom(entry?.contentRect.width));
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  const acceptAll = useAcceptAll(projectId);
  const layout = useSaveLayout(projectId);
  const auto = useMemo(
    () => starLayout(model.tables, model.relations.filter((relation) => relation.cardinality !== "n:m"), model.facts),
    [model.tables, model.relations, model.facts],
  );
  const positions: Record<string, Point> = { ...auto, ...model.positions, ...moved };
  const scale = zoom ?? fitScale(positions, room);
  // the frame is as tall as the diagram, up to most of the screen
  const height = Math.max(280, planeSize(Object.values(positions)).height * scale + 18);
  const suggested = model.relations.filter((relation) => relation.status === "suggested" && relation.cardinality !== "n:m");
  const canBuild = model.relations.some((relation) => relation.status === "accepted" && relation.cardinality !== "n:m");
  // a selection that no longer exists (a relation just accepted gets a new id) shows the list
  const current =
    selected &&
    (selected.kind === "relation" ? model.relations : model.tables).some((item) => item.id === selected.id)
      ? selected
      : null;

  return (
    <>
      <header className="flex flex-wrap items-end justify-between gap-x-6 gap-y-3">
        <div className="min-w-0 space-y-1">
          <h1 className="text-xl font-semibold text-ink-1">{t("model.title")}</h1>
          <p className="max-w-[72ch] text-[13px] text-ink-2">{summary(model)}</p>
          {(model.notes ?? []).map((note, index) => (
            <p key={index} className="max-w-[72ch] text-xs text-ink-3">
              <code className="font-mono text-[11px]">{note.code}</code> {note.text}
            </p>
          ))}
        </div>
        <div className="flex flex-wrap gap-2">
          {suggested.length > 0 && (
            <Button icon={Check} disabled={acceptAll.isPending} onClick={() => acceptAll.mutate(suggested)}>
              {t("model.acceptAll", { count: String(suggested.length) })}
            </Button>
          )}
          <Button variant="primary" icon={Table2} disabled={!canBuild} title={canBuild ? undefined : t("model.buildNeedsRelation")} onClick={() => setBuilding(true)}>
            {t("model.build")}
          </Button>
        </div>
      </header>
      {acceptAll.isError && <ErrorNote>{acceptAll.error.message}</ErrorNote>}

      {phone ? (
        <PhoneList projectId={projectId} model={model} selected={current} onSelect={setSelected} />
      ) : (
        <div className="grid min-w-0 grid-cols-[minmax(0,1fr)] gap-4 lg:grid-cols-[minmax(0,1fr)_340px]">
          <div className="min-w-0 space-y-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <Legend />
              <div className="flex items-center gap-0.5">
                <Button variant="ghost" icon={ZoomOut} iconOnly aria-label={t("model.zoomOut")} title={t("model.zoomOut")} onClick={() => setZoom(Math.max(0.5, Math.round((scale - 0.1) * 10) / 10))} />
                <Button variant="ghost" icon={ZoomIn} iconOnly aria-label={t("model.zoomIn")} title={t("model.zoomIn")} onClick={() => setZoom(Math.min(1.25, Math.round((scale + 0.1) * 10) / 10))} />
                <Button variant="ghost" icon={Maximize} iconOnly aria-label={t("model.zoomFit")} title={t("model.zoomFit")} onClick={() => setZoom(null)} />
              <Button
                variant="ghost"
                icon={LayoutGrid}
                disabled={layout.isPending}
                onClick={() => {
                  setMoved(auto);
                  layout.mutate({ positions: auto });
                }}
              >
                {t("model.autoLayout")}
              </Button>
              </div>
            </div>
            <div ref={measure} style={{ height }} className="max-h-[clamp(420px,calc(100dvh-300px),780px)] overflow-auto overscroll-contain rounded-lg border border-line bg-surface-0">
              <ModelCanvas
                scale={scale}
                ready={room !== undefined}
                model={model}
                positions={positions}
                selected={current}
                onSelect={setSelected}
                onMove={(root, point) => {
                  setMoved({ ...moved, [root]: point });
                  layout.mutate({ positions: { [root]: point } });
                }}
              />
            </div>
          </div>
          <aside aria-label={t("model.panel.label")} className="min-w-0 lg:max-h-[clamp(420px,calc(100dvh-260px),820px)] lg:overflow-y-auto lg:pr-1">
            <ModelPanel projectId={projectId} model={model} selected={current} onSelect={setSelected} />
          </aside>
        </div>
      )}

      <Builds projectId={projectId} builds={model.builds} />
      {building && <BuildDialog projectId={projectId} model={model} open={building} onOpenChange={setBuilding} />}
    </>
  );
}

function Legend() {
  const item = "inline-flex items-center gap-1.5";
  return (
    <ul className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-3">
      <li className={item}>
        <svg width="22" height="6" aria-hidden>
          <line x1="1" y1="3" x2="21" y2="3" stroke="var(--ink-3)" strokeWidth="1.5" />
        </svg>
        {t("model.legend.accepted")}
      </li>
      <li className={item}>
        <svg width="22" height="6" aria-hidden>
          <line x1="1" y1="3" x2="21" y2="3" stroke="var(--ink-3)" strokeWidth="1.5" strokeDasharray="5 4" />
        </svg>
        {t("model.legend.suggested")}
      </li>
      <li className={item}>
        <svg width="22" height="6" aria-hidden>
          <line x1="1" y1="3" x2="21" y2="3" stroke="var(--bad)" strokeWidth="1.5" strokeDasharray="5 4" />
        </svg>
        {t("model.legend.many")}
      </li>
      <li className="max-sm:hidden">{t("model.legend.marks")}</li>
    </ul>
  );
}

/** 041: below 640 px the diagram is a list (tables, then relations); a choice opens its detail in
 *  a full-screen sheet. */
function PhoneList({ projectId, model, selected, onSelect }: { projectId: string; model: Model; selected: Selection; onSelect: (selection: Selection) => void }) {
  const tables = [...model.tables].sort((a, b) => Number(b.role === "fact") - Number(a.role === "fact") || (a.depth ?? 9) - (b.depth ?? 9));
  return (
    <div className="flex flex-col gap-5">
      <p className="text-xs text-ink-3">{t("model.phone.hint")}</p>
      <section className="space-y-2" aria-labelledby="model-tables">
        <h2 id="model-tables" className="text-xs font-semibold text-ink-1">
          {t("model.list.tables")}
        </h2>
        <ul className="divide-y divide-line rounded-md border border-line bg-surface-1">
          {tables.map((table) => (
            <li key={table.id}>
              <button type="button" onClick={() => onSelect({ kind: "table", id: table.id })} className="flex w-full items-baseline gap-3 px-3 py-2.5 text-left">
                <span className="min-w-0 flex-1 truncate text-[13px] font-medium text-ink-1">{table.name}</span>
                <span className="shrink-0 text-xs text-ink-3 tabular-nums">{t("model.card.rows", { rows: formatCount(table.rows) })}</span>
                <span className={`shrink-0 text-xs ${table.role === "fact" ? "text-accent" : "text-ink-2"}`}>{roleLabel(table)}</span>
              </button>
            </li>
          ))}
        </ul>
      </section>
      <RelationList projectId={projectId} model={model} onSelect={onSelect} />
      <Dialog open={selected !== null} onOpenChange={(open) => !open && onSelect(null)} title={t("model.panel.label")}>
        <ModelPanel projectId={projectId} model={model} selected={selected} onSelect={onSelect} />
      </Dialog>
    </div>
  );
}
