import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { useMemo, useState } from "react";

import { formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { Button, buttonClass } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { Checkbox, ErrorNote, Field, Select, TextInput } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { modelApi, modelKeys, useBuild, type BuildPreview, type Model, type ModelTable } from "./api";

type Reached = { table: ModelTable; through: string | null; keys: Set<string> };

/** The dimensions reached from a fact along accepted relations (snowflake up to 3 links, R-BLD-01),
 *  in the order the build joins them. */
export function reachedFrom(model: Model, factId: string): Reached[] {
  const byId = new Map(model.tables.map((table) => [table.id, table]));
  const links = model.relations.filter((relation) => relation.status === "accepted" && relation.cardinality !== "n:m" && !relation.problem);
  const out: Reached[] = [];
  const seen = new Set([factId]);
  let frontier: { id: string; depth: number }[] = [{ id: factId, depth: 0 }];
  while (frontier.length > 0) {
    const next: { id: string; depth: number }[] = [];
    for (const { id, depth } of frontier) {
      for (const link of links.filter((relation) => relation.child_id === id)) {
        const parent = byId.get(link.parent_id);
        if (!parent || seen.has(parent.id) || depth >= 3) continue;
        seen.add(parent.id);
        const keys = new Set([
          ...link.parent_columns,
          ...links.filter((other) => other.child_id === parent.id).flatMap((other) => other.child_columns),
        ]);
        out.push({ table: parent, through: depth === 0 ? null : (byId.get(id)?.name ?? null), keys });
        next.push({ id: parent.id, depth: depth + 1 });
      }
    }
    frontier = next;
  }
  return out;
}

type DialogProps = {
  projectId: string;
  model: Model;
  open: boolean;
  onOpenChange: (open: boolean) => void;
};

export function BuildDialog({ projectId, model, open, onOpenChange }: DialogProps) {
  const facts = model.facts.length ? model.facts : model.tables.map((table) => table.id);
  const [factId, setFactId] = useState(facts[0] ?? "");
  const [edits, setEdits] = useState<Record<string, string[]>>({});
  const [name, setName] = useState<string | null>(null);
  const build = useBuild(projectId);
  const reached = useMemo(() => reachedFrom(model, factId), [model, factId]);
  const picks = useMemo(() => {
    const out: Record<string, string[]> = {};
    for (const { table, keys } of reached) {
      out[table.id] = edits[table.id] ?? table.columns.filter((column) => column.default_pick && !keys.has(column.name)).map((column) => column.name);
    }
    return out;
  }, [reached, edits]);
  const body = { fact_id: factId, picks };
  const preview = useQuery({
    queryKey: modelKeys.preview(projectId, body),
    queryFn: () => modelApi.preview(projectId, body),
    enabled: open && Boolean(factId) && reached.length > 0,
    placeholderData: keepPreviousData,
    retry: false,
    staleTime: 10_000,
  });
  const shownName = name ?? preview.data?.name ?? "";
  const refused = preview.data?.refused ?? [];
  const created = build.data?.dataset_id;

  function toggle(tableId: string, column: string, on: boolean) {
    const current = picks[tableId] ?? [];
    setEdits({ ...edits, [tableId]: on ? [...current, column] : current.filter((item) => item !== column) });
  }

  const footer = created ? (
    <>
      <Link to="/d/$datasetId/ml" params={{ datasetId: created }} className={buttonClass("secondary")}>
        {t("model.builds.ml")}
      </Link>
      <Link to="/d/$datasetId/kpi" params={{ datasetId: created }} className={buttonClass("secondary")}>
        {t("model.builds.kpi")}
      </Link>
      <Link to="/d/$datasetId" params={{ datasetId: created }} className={buttonClass("primary")}>
        {t("model.aggregate.open")}
      </Link>
    </>
  ) : (
    <>
      <Button variant="ghost" onClick={() => onOpenChange(false)}>
        {t("model.dialog.cancel")}
      </Button>
      <Button
        variant="primary"
        disabled={build.isPending || refused.length > 0 || !preview.data || reached.length === 0}
        onClick={() => build.mutate({ ...body, name: shownName.trim() || null })}
      >
        {build.isPending ? t("model.dialog.building") : t("model.dialog.build")}
      </Button>
    </>
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange} title={t("model.dialog.title")} description={t("model.dialog.body")} footer={footer} size="lg">
      <div className="flex flex-col gap-5">
        <div className="grid grid-cols-[minmax(0,1fr)] gap-3 sm:grid-cols-2">
          <Field label={t("model.dialog.fact")}>
            <Select
              value={factId}
              onChange={(event) => {
                setFactId(event.target.value);
                setEdits({});
                setName(null);
              }}
            >
              {facts.map((id) => (
                <option key={id} value={id}>
                  {model.tables.find((table) => table.id === id)?.name ?? id}
                </option>
              ))}
            </Select>
          </Field>
          <Field label={t("model.dialog.name")}>
            <TextInput value={shownName} maxLength={200} onChange={(event) => setName(event.target.value)} />
          </Field>
        </div>

        {created && (
          <p role="status" className="text-[13px] text-ok">
            {t("model.dialog.done", { name: shownName })}
          </p>
        )}
        <Counts preview={preview.data} loading={preview.isFetching && !preview.data} />
        {preview.isError && <ErrorNote>{preview.error.message}</ErrorNote>}
        {refused.map((reason, index) => (
          <ErrorNote key={index}>{t("model.dialog.refused", { text: `${reason.text} (${reason.code})` })}</ErrorNote>
        ))}
        {build.isError && <ErrorNote>{build.error.message}</ErrorNote>}

        <section className="space-y-3">
          <div className="space-y-1">
            <h3 className="text-xs font-semibold text-ink-1">{t("model.dialog.columns")}</h3>
            <p className="text-xs text-ink-3">{t("model.dialog.defaultsHelp")}</p>
          </div>
          {reached.length === 0 && <p className="text-xs text-ink-2">{t("model.buildNeedsRelation")}</p>}
          <div className="grid grid-cols-[minmax(0,1fr)] gap-3 md:grid-cols-2 lg:grid-cols-3">
            {reached.map(({ table, through, keys }) => {
              const offered = table.columns.filter((column) => !keys.has(column.name));
              const chosen = new Set(picks[table.id] ?? []);
              const step = preview.data?.steps.find((item) => item.parent_id === table.id);
              const unmatched = step ? step.check.orphan_rows + step.check.child_empty : 0;
              return (
                <fieldset key={table.id} className="min-w-0 rounded-md border border-line bg-surface-1">
                  <legend className="sr-only">{table.name}</legend>
                  <div className="flex items-baseline gap-2 border-b border-line px-3 py-2">
                    <span className="min-w-0 flex-1 truncate text-[13px] font-medium text-ink-1">{table.name}</span>
                    <span className="shrink-0 text-xs text-ink-3 tabular-nums">
                      {t("model.dialog.picked", { picked: String(chosen.size), total: String(offered.length) })}
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-x-3 gap-y-0.5 px-3 pt-2 text-xs text-ink-3">
                    {through && <span>{t("model.dialog.through", { name: through })}</span>}
                    {unmatched > 0 && step && (
                      <span className="text-warn tabular-nums">
                        {t("model.dialog.unmatched", { count: formatCount(unmatched), parent: table.name })}
                      </span>
                    )}
                    <span className="ml-auto flex gap-2">
                      <button type="button" className="hover:text-accent" onClick={() => setEdits({ ...edits, [table.id]: offered.map((column) => column.name) })}>
                        {t("model.dialog.all")}
                      </button>
                      <button type="button" className="hover:text-accent" onClick={() => setEdits({ ...edits, [table.id]: [] })}>
                        {t("model.dialog.none")}
                      </button>
                      <button
                        type="button"
                        className="hover:text-accent"
                        onClick={() => {
                          const rest = { ...edits };
                          delete rest[table.id];
                          setEdits(rest);
                        }}
                      >
                        {t("model.dialog.defaults")}
                      </button>
                    </span>
                  </div>
                  <ul className="max-h-56 space-y-0.5 overflow-y-auto px-3 py-2">
                    {offered.map((column) => (
                      <li key={column.name}>
                        <Checkbox
                          checked={chosen.has(column.name)}
                          onChange={(event) => toggle(table.id, column.name, event.target.checked)}
                          className="w-full py-0.5 text-xs"
                          label={
                            <span className="flex min-w-0 items-baseline gap-2">
                              <span className="truncate font-mono text-ink-1">{column.name}</span>
                              <span className="shrink-0 text-ink-3">{t(`model.category.${column.category}` as MessageKey)}</span>
                            </span>
                          }
                        />
                      </li>
                    ))}
                  </ul>
                </fieldset>
              );
            })}
          </div>
        </section>
      </div>
    </Dialog>
  );
}

function Counts({ preview, loading }: { preview: BuildPreview | undefined; loading: boolean }) {
  if (loading || !preview) return <Loading label={t("model.dialog.counting")} className="h-16 rounded-md" compact />;
  const same = preview.rows_after === preview.rows_before && preview.refused.length === 0;
  return (
    <div className="space-y-1.5">
      <dl className="flex flex-wrap gap-x-8 gap-y-2">
        {[
          [t("model.dialog.before"), preview.rows_before],
          [t("model.dialog.after"), preview.rows_after],
          [t("model.dialog.columnsOut"), preview.columns.length],
        ].map(([label, value]) => (
          <div key={String(label)} className="space-y-0.5">
            <dt className="text-xs text-ink-3">{label}</dt>
            <dd className="text-lg font-semibold text-ink-1 tabular-nums">{formatCount(Number(value))}</dd>
          </div>
        ))}
      </dl>
      {same && <p className="text-xs text-ok">{t("model.dialog.same")}</p>}
    </div>
  );
}
