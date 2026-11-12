import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { Merge } from "lucide-react";
import { useMemo, useState } from "react";

import { useColumns, useDatasets } from "../../api/hooks";
import { formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { Button, buttonClass } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { Checkbox, ErrorNote, Field, Select, TextInput } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { useProjects } from "../projects/api";
import { joinApi, joinKeys, useCreateJoin, type JoinBody, type JoinKey, type JoinMode, type JoinPreview, type JoinType } from "./api";

const MODES: JoinMode[] = ["lookup", "expand", "summarise"];
const JOINS: JoinType[] = ["left", "inner", "right", "full"];
const MANUAL = "manual";

/** The other datasets, this project's first (089, R-JN-01: a project is where tables belong together). */
export function useJoinCandidates(datasetId: string) {
  const datasets = useDatasets();
  const projects = useProjects();
  return useMemo(() => {
    const ready = (datasets.data ?? []).filter((d) => d.id !== datasetId && d.status === "ready");
    const project = projects.find((p) => p.dataset_ids.includes(datasetId));
    const inProject = new Set(project?.dataset_ids ?? []);
    return {
      project: project?.name ?? null,
      same: ready.filter((d) => inProject.has(d.id)),
      others: ready.filter((d) => !inProject.has(d.id)),
    };
  }, [datasets.data, projects, datasetId]);
}

/** "Join…" next to Export on the Data page: opens the dialog. */
export function JoinButton({ datasetId, name }: { datasetId: string; name: string }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Button variant="secondary" onClick={() => setOpen(true)}>
        <Merge size={14} strokeWidth={1.5} aria-hidden />
        {t("data.join.button")}
      </Button>
      {open && <JoinDialog datasetId={datasetId} name={name} open={open} onOpenChange={setOpen} />}
    </>
  );
}

export function keyLabel(key: JoinKey): string {
  return t("data.join.keyOption", {
    left: key.left_columns.join(" + "),
    right: key.right_columns.join(" + "),
    cardinality: key.cardinality,
    share: `${Math.round(key.matched_share * 100)} %`,
  });
}

type DialogProps = { datasetId: string; name: string; open: boolean; onOpenChange: (open: boolean) => void };

/** Source on the left (the two tables and how their keys meet), the result on the right; nothing is
 *  written before "Create dataset" (089). */
export function JoinDialog({ datasetId, name, open, onOpenChange }: DialogProps) {
  const candidates = useJoinCandidates(datasetId);
  const [otherId, setOtherId] = useState("");
  const [keyIndex, setKeyIndex] = useState("0");
  const [manual, setManual] = useState<{ left: string; right: string }>({ left: "", right: "" });
  const [mode, setMode] = useState<JoinMode | null>(null);
  const [join, setJoin] = useState<JoinType | null>(null);
  const [picks, setPicks] = useState<string[] | null>(null);
  const [newName, setNewName] = useState<string | null>(null);
  const create = useCreateJoin(datasetId);

  const suggestion = useQuery({
    queryKey: joinKeys.suggest(datasetId, otherId),
    queryFn: () => joinApi.suggest(datasetId, otherId),
    enabled: open && Boolean(otherId),
    retry: false,
    staleTime: 30_000,
  });
  const keys = suggestion.data?.keys ?? [];
  const chosenKey = keyIndex === MANUAL ? undefined : keys[Number(keyIndex)];
  const left = chosenKey?.left_columns ?? (manual.left ? [manual.left] : []);
  const right = chosenKey?.right_columns ?? (manual.right ? [manual.right] : []);
  const offered = suggestion.data?.other_columns ?? [];
  const defaults = offered.filter((c) => c.default_pick).map((c) => c.name);
  const body: JoinBody | null =
    otherId && left.length > 0 && right.length === left.length
      ? { other_id: otherId, left_columns: left, right_columns: right, mode, join_type: join, columns: picks }
      : null;
  const preview = useQuery({
    queryKey: joinKeys.preview(datasetId, body ?? { other_id: "", left_columns: [], right_columns: [] }),
    queryFn: () => joinApi.preview(datasetId, body as JoinBody),
    enabled: open && body !== null,
    placeholderData: keepPreviousData,
    retry: false,
    staleTime: 10_000,
  });
  const result = body ? preview.data : undefined;
  const refused = result?.refused ?? [];
  const created = create.data?.dataset_id;
  const otherName = [...candidates.same, ...candidates.others].find((d) => d.id === otherId)?.name ?? "";
  const shownName = newName ?? result?.suggested_name ?? "";

  function pickOther(id: string) {
    setOtherId(id);
    setKeyIndex("0");
    setManual({ left: "", right: "" });
    setMode(null);
    setJoin(null);
    setPicks(null);
    setNewName(null);
  }

  const footer = created ? (
    <>
      <Link to="/d/$datasetId/ml" params={{ datasetId: created }} className={buttonClass("secondary")}>
        {t("data.join.openMl")}
      </Link>
      <Link to="/d/$datasetId" params={{ datasetId: created }} className={buttonClass("primary")} onClick={() => onOpenChange(false)}>
        {t("data.join.open")}
      </Link>
    </>
  ) : (
    <>
      <Button variant="ghost" onClick={() => onOpenChange(false)}>
        {t("data.join.cancel")}
      </Button>
      <Button
        variant="primary"
        disabled={!body || !result || refused.length > 0 || create.isPending}
        onClick={() => body && create.mutate({ ...body, mode: result?.mode ?? mode, name: shownName.trim() || null })}
      >
        {create.isPending
          ? t("data.join.creating")
          : t("data.join.create", { rows: formatCount(result?.rows_after ?? 0) })}
      </Button>
    </>
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange} title={t("data.join.title", { name })} description={t("data.join.body")} footer={footer} size="lg">
      <div className="flex flex-col gap-5">
        <div className="grid grid-cols-[minmax(0,1fr)] gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <Field label={t("data.join.other")}>
            <Select value={otherId} onChange={(event) => pickOther(event.target.value)}>
              <option value="">{t("data.join.pick")}</option>
              {candidates.same.length > 0 && (
                <optgroup label={t("data.join.thisProject", { project: candidates.project ?? "" })}>
                  {candidates.same.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.name}
                    </option>
                  ))}
                </optgroup>
              )}
              {candidates.others.length > 0 && (
                <optgroup label={t("data.join.otherDatasets")}>
                  {candidates.others.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.name}
                    </option>
                  ))}
                </optgroup>
              )}
            </Select>
          </Field>
          {otherId && (
            <Field label={t("data.join.keys")}>
              <Select value={keyIndex} disabled={suggestion.isFetching} onChange={(event) => setKeyIndex(event.target.value)}>
                {keys.map((key, index) => (
                  <option key={index} value={String(index)}>
                    {keyLabel(key)}
                  </option>
                ))}
                <option value={MANUAL}>{t("data.join.manual")}</option>
              </Select>
            </Field>
          )}
          {result && (
            <Field label={t("data.join.mode")}>
              <Select value={mode ?? result.mode} onChange={(event) => setMode(event.target.value as JoinMode)}>
                {MODES.map((value) => {
                  const reason = result.modes.find((m) => m.mode === value)?.reason;
                  const text = t(`data.join.mode.${value}` as MessageKey, { other: otherName, name });
                  return (
                    <option key={value} value={value} disabled={Boolean(reason)}>
                      {reason ? `${text} — ${reason}` : text}
                    </option>
                  );
                })}
              </Select>
            </Field>
          )}
          {otherId && keyIndex === MANUAL && (
            <ManualKeys
              datasetId={datasetId}
              otherColumns={offered.map((c) => c.name)}
              value={manual}
              onChange={setManual}
            />
          )}
          {result && result.mode !== "summarise" && (
            <Field label={t("data.join.type")}>
              <Select value={join ?? result.join_type} onChange={(event) => setJoin(event.target.value as JoinType)}>
                {JOINS.map((value) => (
                  <option key={value} value={value}>
                    {t(`data.join.type.${value}` as MessageKey)}
                    {value === result.join_type && join === null ? ` ${t("data.join.recommended")}` : ""}
                  </option>
                ))}
              </Select>
            </Field>
          )}
          {result && (
            <Field label={t("data.join.name")}>
              <TextInput value={shownName} maxLength={200} onChange={(event) => setNewName(event.target.value)} />
            </Field>
          )}
        </div>

        {suggestion.isError && <ErrorNote>{suggestion.error.message}</ErrorNote>}
        {otherId && suggestion.data && keys.length === 0 && <p className="text-[13px] text-ink-2">{t("data.join.noKeys")}</p>}
        {preview.isError && body && <ErrorNote>{preview.error.message}</ErrorNote>}
        {refused.map((reason, index) => (
          <ErrorNote key={index}>{`${reason.text} (${reason.code})`}</ErrorNote>
        ))}
        {create.isError && <ErrorNote>{create.error.message}</ErrorNote>}
        {created && (
          <p role="status" className="text-[13px] text-ok">
            {t("data.join.done", { name: shownName })}
          </p>
        )}

        {otherId && (suggestion.isFetching || (body && preview.isFetching && !result)) && <Loading label={t("data.join.checking")} />}
        {result && (
          <div className="grid grid-cols-[minmax(0,1fr)] gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
            <Source result={result} name={name} other={otherName} chosen={chosenKey} />
            <Result
              result={result}
              offered={result.mode === "summarise" ? [] : offered.map((c) => c.name).filter((c) => !right.includes(c))}
              picks={picks ?? defaults}
              onPicks={setPicks}
            />
          </div>
        )}
      </div>
    </Dialog>
  );
}

function ManualKeys({
  datasetId,
  otherColumns,
  value,
  onChange,
}: {
  datasetId: string;
  otherColumns: string[];
  value: { left: string; right: string };
  onChange: (value: { left: string; right: string }) => void;
}) {
  const own = (useColumns(datasetId, true).data ?? []).map((c) => c.name);
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-2">
      <Field label={t("data.join.leftColumn")}>
        <Select value={value.left} onChange={(event) => onChange({ ...value, left: event.target.value })}>
          <option value="">—</option>
          {own.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </Select>
      </Field>
      <Field label={t("data.join.rightColumn")}>
        <Select value={value.right} onChange={(event) => onChange({ ...value, right: event.target.value })}>
          <option value="">—</option>
          {otherColumns.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </Select>
      </Field>
    </div>
  );
}

/** Left pane: the two tables and how their keys meet, with every rule's reason. */
function Source({ result, name, other, chosen }: { result: JoinPreview; name: string; other: string; chosen: JoinKey | undefined }) {
  const check = result.check;
  const missing = check.orphan_rows + check.child_empty;
  return (
    <section className="min-w-0 space-y-3 rounded-md border border-line p-3" aria-label={t("data.join.source")}>
      <h3 className="text-xs font-semibold text-ink-1">{t("data.join.source")}</h3>
      <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 text-xs tabular-nums">
        <dt className="text-ink-3">{name}</dt>
        <dd className="text-ink-1">{t("data.join.rowsOf", { rows: formatCount(result.rows_before) })}</dd>
        <dt className="text-ink-3">{other}</dt>
        <dd className="text-ink-1">{t("data.join.rowsOf", { rows: formatCount(result.mode === "lookup" ? check.parent_rows : check.child_rows) })}</dd>
        <dt className="text-ink-3">{t("data.join.matched")}</dt>
        <dd className="text-ink-1">
          {formatCount(check.matched_rows)} / {formatCount(check.child_rows)}
        </dd>
      </dl>
      {missing > 0 && (
        <p className="text-xs text-warn">
          {t("data.join.unmatched", { count: formatCount(missing), examples: check.orphans.map((o) => o.value).join(", ") })}
        </p>
      )}
      <ul className="space-y-1 text-xs text-ink-2">
        {[...(chosen?.reasons ?? []), ...result.reasons, ...result.warnings].map((reason, index) => (
          <li key={index}>
            {reason.text} <span className="text-ink-3">({reason.code})</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

/** Right pane: the first rows of the result, the added columns marked; the columns to add. */
function Result({ result, offered, picks, onPicks }: { result: JoinPreview; offered: string[]; picks: string[]; onPicks: (picks: string[]) => void }) {
  const added = new Set(result.added);
  return (
    <section className="min-w-0 space-y-3 rounded-md border border-line p-3" aria-label={t("data.join.result")}>
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h3 className="text-xs font-semibold text-ink-1">{t("data.join.result")}</h3>
        <span className="text-xs text-ink-2 tabular-nums">
          {t("data.join.counts", {
            before: formatCount(result.rows_before),
            after: formatCount(result.rows_after),
            columns: String(result.columns.length),
            added: String(result.added.length),
          })}
        </span>
      </div>
      {offered.length > 0 && (
        <details className="text-xs">
          <summary className="cursor-pointer text-accent select-none hover:underline">
            {t("data.join.columns", { picked: String(picks.length), total: String(offered.length) })}
          </summary>
          <ul className="mt-2 grid max-h-40 grid-cols-[repeat(auto-fill,minmax(min(100%,12rem),1fr))] gap-x-3 gap-y-0.5 overflow-y-auto">
            {offered.map((column) => (
              <li key={column} className="min-w-0">
                <Checkbox
                  checked={picks.includes(column)}
                  onChange={(event) => onPicks(event.target.checked ? [...picks, column] : picks.filter((c) => c !== column))}
                  className="w-full py-0.5"
                  label={<span className="truncate font-mono">{column}</span>}
                />
              </li>
            ))}
          </ul>
        </details>
      )}
      <div className="max-h-80 overflow-auto rounded border border-line">
        <table className="w-full text-xs tabular-nums">
          <thead className="sticky top-0 bg-surface-2">
            <tr>
              {result.columns.map((column) => (
                <th key={column} scope="col" className={`px-2 py-1 text-left font-medium whitespace-nowrap ${added.has(column) ? "text-accent" : "text-ink-2"}`}>
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {result.rows.map((row, index) => (
              <tr key={index} className="border-t border-line">
                {row.map((cell, column) => (
                  <td key={column} className={`max-w-48 truncate px-2 py-1 ${cell === null ? "text-ink-3" : "text-ink-1"}`}>
                    {cell ?? "—"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
