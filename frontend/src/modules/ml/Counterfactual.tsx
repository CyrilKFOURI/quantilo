import { useState } from "react";

import type { MlCounterfactual, MlRun } from "../../api/client";
import { useMlCounterfactual, useMlPredictionRowCounterfactual } from "../../api/hooks";
import { formatNumber } from "../../charts/figures";
import { t } from "../../lib/i18n";
import { Checkbox, ErrorNote, Select, TextInput } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { Cell } from "../kpi/Customize";

const MAX_CHANGES = 3;

/** “What would change it?” under one row's waterfall (125, Advanced): folded, and not computed
 * before the user opens it. */
export function WhatWouldChange({
  datasetId,
  run,
  row,
  predictionId,
}: {
  datasetId: string;
  run: Pick<MlRun, "id" | "task">;
  row: number;
  /** 125 bis: a row of new data (Predict on new data), judged against that data (R-XAI-21). */
  predictionId?: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <details className="border-t border-line pt-2 text-xs" onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary className="cursor-pointer text-[13px] text-accent select-none hover:underline">{t("ml.whatif.open")}</summary>
      {open && <Search datasetId={datasetId} run={run} row={row} predictionId={predictionId} />}
    </details>
  );
}

function Search({ datasetId, run, row, predictionId }: { datasetId: string; run: Pick<MlRun, "id" | "task">; row: number; predictionId?: string }) {
  const [editable, setEditable] = useState<string[] | null>(null);
  const [maxChanges, setMaxChanges] = useState(MAX_CHANGES);
  const [targetClass, setTargetClass] = useState<string | null>(null);
  const [targetValue, setTargetValue] = useState<number | null>(null);
  const body = { editable, target_class: targetClass, target_value: targetValue, max_changes: maxChanges };
  const onRun = useMlCounterfactual(datasetId, run.id, predictionId ? null : { row, ...body });
  const onNew = useMlPredictionRowCounterfactual(datasetId, predictionId ?? "", row, predictionId ? body : null);
  const found = predictionId ? onNew : onRun;
  const data = found.data;
  const allowed = editable ?? data?.columns.filter((c) => c.editable).map((c) => c.column) ?? [];
  const toggle = (column: string, on: boolean) =>
    setEditable(on ? [...new Set([...allowed, column])] : allowed.filter((c) => c !== column));
  return (
    <div className="mt-2 space-y-3">
      <p className="max-w-prose text-ink-2">{t("ml.whatif.hint")}</p>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-[repeat(3,minmax(0,1fr))]">
        <Cell label={t("ml.whatif.maxChanges")} why={t("ml.whatif.maxChangesWhy")} changed={maxChanges !== MAX_CHANGES} onReset={() => setMaxChanges(MAX_CHANGES)}>
          <Select aria-label={t("ml.whatif.maxChanges")} value={maxChanges} onChange={(e) => setMaxChanges(Number(e.target.value))} className="w-full text-xs">
            {[1, 2, 3].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </Select>
        </Cell>
        {data && run.task !== "regression" && (
          <Cell label={t("ml.whatif.goal")} why={t("ml.whatif.goalClassWhy")} changed={targetClass !== null} onReset={() => setTargetClass(null)}>
            <Select aria-label={t("ml.whatif.goal")} value={data.target_class ?? ""} onChange={(e) => setTargetClass(e.target.value)} className="w-full text-xs">
              {data.classes.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </Select>
          </Cell>
        )}
        {data && run.task === "regression" && (
          <Cell label={t("ml.whatif.goalValue")} why={t("ml.whatif.goalValueWhy")} changed={targetValue !== null} onReset={() => setTargetValue(null)}>
            <TextInput
              aria-label={t("ml.whatif.goalValue")}
              type="number"
              key={data.target_value ?? "none"}
              defaultValue={data.target_value ?? ""}
              onBlur={(e) => {
                const value = Number(e.target.value);
                if (e.target.value !== "" && Number.isFinite(value)) setTargetValue(value);
              }}
              className="w-full text-xs"
            />
          </Cell>
        )}
      </div>
      {data && (
        <fieldset className="space-y-1">
          <legend className="mb-1 flex items-center gap-2 text-[11px] text-ink-2">
            {t("ml.whatif.columns")}
            {editable !== null && (
              <button type="button" onClick={() => setEditable(null)} className="text-ink-3 hover:text-accent">
                {t("kpi.resetAuto")}
              </button>
            )}
          </legend>
          <ul className="grid grid-cols-[repeat(auto-fill,minmax(min(100%,13rem),1fr))] gap-x-3 gap-y-1">
            {data.columns.map((c) => (
              <li key={c.column} className="min-w-0">
                <Checkbox
                  checked={allowed.includes(c.column)}
                  onChange={(e) => toggle(c.column, e.target.checked)}
                  label={
                    <span className="min-w-0 truncate font-mono text-ink-1" title={c.locked ? t("ml.whatif.locked", { reason: c.locked }) : undefined}>
                      {c.column}
                      {c.locked && <span className="ml-1 font-sans text-[11px] text-ink-3">{t("ml.whatif.lockedShort", { reason: c.locked })}</span>}
                    </span>
                  }
                />
              </li>
            ))}
          </ul>
          {data.columns.some((c) => c.locked) && <p className="max-w-prose text-[11px] text-ink-3">{t("ml.whatif.lockedRule")}</p>}
        </fieldset>
      )}
      {found.isError ? (
        <ErrorNote>{found.error.message}</ErrorNote>
      ) : !data ? (
        <Loading label={t("ml.whatif.computing")} compact className="h-24 rounded-md" />
      ) : (
        <Answer found={data} stale={found.isPlaceholderData} />
      )}
    </div>
  );
}

function Answer({ found, stale }: { found: MlCounterfactual; stale: boolean }) {
  const changed = new Set(found.changes.map((c) => c.column));
  return (
    <div className={`space-y-2 ${stale ? "opacity-60" : ""}`} aria-busy={stale}>
      <p className={`max-w-prose text-[13px] ${found.reached ? "text-ink-1" : "text-warn"}`}>{found.sentence}</p>
      {found.changes.length > 0 && (
        <p className="text-ink-2">
          {t("ml.whatif.output", { output: found.output, before: formatNumber(found.before), after: formatNumber(found.after) })}
        </p>
      )}
      <p className={`max-w-prose ${found.plausible === false ? "text-warn" : "text-ink-2"}`}>{found.plausibility}</p>
      <div className="max-h-72 overflow-auto rounded-md border border-line">
        <table className="w-full border-collapse text-[13px] tabular-nums" aria-label={t("ml.whatif.table")}>
          <thead className="sticky top-0 bg-surface-2 text-left text-xs text-ink-3">
            <tr>
              <th className="px-3 py-1.5 font-normal">{t("ml.feature")}</th>
              <th className="px-3 py-1.5 font-normal">{t("ml.whatif.thisRow")}</th>
              <th className="px-3 py-1.5 font-normal">{t("ml.whatif.changed")}</th>
              {found.nearest_row !== null && <th className="px-3 py-1.5 font-normal">{t("ml.whatif.nearest", { row: String(found.nearest_row + 1) })}</th>}
            </tr>
          </thead>
          <tbody>
            {found.table.map((r) => (
              <tr key={r.column} className={`border-t border-line ${changed.has(r.column) ? "bg-accent-soft" : ""}`}>
                <td className="max-w-0 truncate px-3 py-1 font-mono text-xs text-ink-1" title={r.column}>
                  {r.column}
                </td>
                <td className="px-3 py-1 text-ink-2">{r.row}</td>
                <td className={`px-3 py-1 ${changed.has(r.column) ? "font-medium text-ink-1" : "text-ink-3"}`}>{r.changed}</td>
                {found.nearest_row !== null && <td className="px-3 py-1 text-ink-2">{r.nearest ?? "—"}</td>}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
