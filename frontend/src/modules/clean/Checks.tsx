import { Link } from "@tanstack/react-router";
import { ArrowRight, CircleAlert, CircleCheck, CircleX, Pencil, Plus, ShieldCheck, Table2, Trash2, X } from "lucide-react";
import { useState } from "react";

import { useColumns } from "../../api/hooks";
import { describe as describeFilter } from "../../features/filters/conditions";
import { formatCount, formatRelativeTime } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { Button, buttonClass } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { ErrorNote, Field, Select, TextInput } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { useAddCheck, useCheckRows, useChecks, useSaveChecks, type CheckKind, type CheckResult, type CleanCheck } from "./api";
import { ARITY, blankCheck, CHECK_KINDS, checkFilter, describeCheck, nextCheckId } from "./checkRules";
import { Conditions } from "./StepFields";

const STATUS = { pass: CircleCheck, fail: CircleX, error: CircleAlert } as const;
const STATUS_TONE = { pass: "text-ok", fail: "text-warn", error: "text-warn" } as const;
const COMPARE: CleanCheck["op"][] = ["lt", "le", "eq", "ne", "ge", "gt"];
const field = "h-8 w-full min-w-0 text-[13px]";

/** "rows 1, 2, 3 …" (numbers from 1, as the file counts them). */
function rowsText(examples: number[], failing: number): string {
  const shown = examples.map((row) => formatCount(row + 1)).join(", ");
  return failing > examples.length ? `${shown}…` : shown;
}

/** N31: the Checks tab. What must hold in the data, declared once and run again at each import and
 *  each new version, like dbt tests; a check never changes the data (no silent fix): it says how
 *  many rows fail and shows them. */
export function ChecksPanel({ datasetId, columns }: { datasetId: string; columns: string[] }) {
  const found = useChecks(datasetId);
  const save = useSaveChecks(datasetId);
  const [editing, setEditing] = useState<CleanCheck | null>(null);
  const [seeing, setSeeing] = useState<{ check: CleanCheck; result: CheckResult } | null>(null);
  if (found.isError) return <ErrorNote>{found.error.message}</ErrorNote>;
  if (!found.data) return <Loading label={t("clean.checks.loading")} className="h-32 rounded-lg" compact />;
  const { checks, report } = found.data;
  const results = new Map((report?.results ?? []).map((result) => [result.id, result]));
  const write = (next: CleanCheck[]) => save.mutate(next, { onSuccess: () => setEditing(null) });
  const known = editing ? checks.some((check) => check.id === editing.id) : false;
  return (
    <section aria-label={t("clean.checks.label")} className="space-y-3">
      <p className="max-w-prose text-[13px] text-ink-2">{t("clean.checks.intro")}</p>
      {report && <Summary report={report} ranAt={found.data.ran_at} trigger={found.data.trigger} />}
      {checks.length === 0 && !editing && <p className="rounded-lg border border-dashed border-line px-3 py-6 text-center text-[13px] text-ink-2">{t("clean.checks.none")}</p>}
      {checks.length > 0 && (
        <ul className="divide-y divide-line rounded-lg border border-line bg-surface-1">
          {checks.map((check) =>
            editing?.id === check.id ? (
              <li key={check.id} className="p-3">
                <CheckForm check={editing} columns={columns} busy={save.isPending} onChange={setEditing} onCancel={() => setEditing(null)} onSave={() => write(checks.map((other) => (other.id === check.id ? editing : other)))} />
              </li>
            ) : (
              <CheckRow
                key={check.id}
                check={check}
                result={results.get(check.id)}
                onSee={(result) => setSeeing({ check, result })}
                onEdit={() => setEditing(check)}
                onRemove={() => write(checks.filter((other) => other.id !== check.id))}
              />
            ),
          )}
        </ul>
      )}
      {editing && !known ? (
        <div className="rounded-lg border border-line bg-surface-1 p-3">
          <CheckForm check={editing} columns={columns} busy={save.isPending} onChange={setEditing} onCancel={() => setEditing(null)} onSave={() => write([...checks, editing])} />
        </div>
      ) : (
        !editing && (
          <Button icon={Plus} onClick={() => setEditing(blankCheck("not_empty", nextCheckId(checks), columns))}>
            {t("clean.checks.add")}
          </Button>
        )
      )}
      {save.error && <ErrorNote>{save.error.message}</ErrorNote>}
      {found.data.history.length > 1 && <History runs={found.data.history} />}
      {seeing && <FailingRowsDialog datasetId={datasetId} check={seeing.check} result={seeing.result} onClose={() => setSeeing(null)} />}
    </section>
  );
}

function Summary({ report, ranAt, trigger }: { report: NonNullable<ReturnType<typeof useChecks>["data"]>["report"]; ranAt: string | null; trigger: string | null }) {
  if (!report) return null;
  const parts = [
    t("clean.checks.pass", { count: formatCount(report.passed) }),
    report.failed > 0 && t(report.failed === 1 ? "clean.checks.failOne" : "clean.checks.fail", { count: formatCount(report.failed), rows: formatCount(report.failing_rows) }),
    report.errors > 0 && t("clean.checks.errors", { count: formatCount(report.errors) }),
  ].filter(Boolean);
  return (
    <p className="text-[13px] text-ink-1 tabular-nums" role="status">
      <span className="font-medium">{t("clean.checks.count", { count: formatCount(report.results.length) })}</span>
      {": "}
      {parts.join(" · ")}
      {ranAt && <span className="text-xs text-ink-3"> · {t(`clean.checks.ran.${trigger ?? "opened"}` as MessageKey, { when: formatRelativeTime(ranAt) })}</span>}
    </p>
  );
}

function CheckRow({ check, result, onSee, onEdit, onRemove }: { check: CleanCheck; result: CheckResult | undefined; onSee: (result: CheckResult) => void; onEdit: () => void; onRemove: () => void }) {
  const label = describeCheck(check);
  const Icon = result ? STATUS[result.status] : CircleCheck;
  const status = !check.enabled
    ? t("clean.checks.off")
    : !result
      ? ""
      : result.status === "pass"
        ? t("clean.checks.passes", { rows: formatCount(result.rows) })
        : result.status === "fail"
          ? t("clean.checks.fails", { count: formatCount(result.failing), rows: formatCount(result.rows), list: rowsText(result.examples, result.failing) })
          : t("clean.checks.cannot", { message: result.message ?? "" });
  return (
    <li className="flex flex-wrap items-start gap-x-3 gap-y-1 px-3 py-2">
      <Icon size={15} strokeWidth={1.5} className={`mt-0.5 shrink-0 ${result ? STATUS_TONE[result.status] : "text-ink-3"}`} aria-hidden />
      <div className="min-w-0 flex-1 basis-64">
        <p className="text-[13px] text-ink-1">
          {label}
          {result && <span className="sr-only"> · {t(`clean.checks.status.${result.status}` as MessageKey)}</span>}
        </p>
        <p className="text-xs text-ink-2 tabular-nums">
          {status}
          {check.rule && <span className="text-ink-3"> · {t("clean.checks.from", { rule: check.rule })}</span>}
        </p>
      </div>
      <span className="flex shrink-0 items-center gap-1">
        {result?.status === "fail" && (
          <Button variant="ghost" icon={Table2} onClick={() => onSee(result)} aria-label={`${t("clean.checks.see")}: ${label}`}>
            {t("clean.checks.see")}
          </Button>
        )}
        <Button variant="ghost" icon={Pencil} iconOnly aria-label={`${t("clean.checks.edit")}: ${label}`} title={t("clean.checks.edit")} onClick={onEdit} />
        <Button variant="ghost" icon={Trash2} iconOnly aria-label={`${t("clean.checks.remove")}: ${label}`} title={t("clean.checks.remove")} onClick={onRemove} />
      </span>
    </li>
  );
}

function History({ runs }: { runs: NonNullable<ReturnType<typeof useChecks>["data"]>["history"] }) {
  return (
    <details className="text-xs text-ink-2">
      <summary className="cursor-pointer text-ink-3">{t("clean.checks.history", { count: formatCount(runs.length) })}</summary>
      <ul className="mt-1 space-y-0.5 tabular-nums">
        {runs.map((run) => (
          <li key={run.ran_at}>
            {formatRelativeTime(run.ran_at)} · {t(`clean.checks.trigger.${run.trigger}` as MessageKey)} ·{" "}
            {run.failed ? t("clean.checks.historyFail", { count: formatCount(run.failed), rows: formatCount(run.failing_rows) }) : t("clean.checks.historyPass")}
          </li>
        ))}
      </ul>
    </details>
  );
}

/** One check's settings: what it tests, on which columns, and its own name (optional). */
function CheckForm({ check, columns, busy, onChange, onCancel, onSave }: { check: CleanCheck; columns: string[]; busy: boolean; onChange: (check: CleanCheck) => void; onCancel: () => void; onSave: () => void }) {
  const set = (patch: Partial<CleanCheck>) => onChange({ ...check, ...patch });
  const [least, most] = ARITY[check.kind];
  const ready =
    check.columns.length >= least &&
    check.columns.length <= most &&
    (check.kind !== "in_list" || check.values.length > 0) &&
    (check.kind !== "between" || check.min !== null || check.max !== null) &&
    (check.kind !== "pattern" || Boolean(check.pattern)) &&
    (check.kind !== "condition" || check.conditions.length > 0);
  return (
    <form
      aria-label={t("clean.checks.form")}
      className="space-y-3"
      onSubmit={(event) => {
        event.preventDefault();
        if (ready) onSave();
      }}
    >
      <div className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,12rem),1fr))] gap-3">
        <Field label={t("clean.checks.kind")}>
          <Select value={check.kind} onChange={(event) => onChange({ ...blankCheck(event.target.value as CheckKind, check.id, check.columns.length ? check.columns : columns), name: check.name })} className={field}>
            {CHECK_KINDS.map((kind) => (
              <option key={kind} value={kind}>
                {t(`clean.checks.kind.${kind}` as MessageKey)}
              </option>
            ))}
          </Select>
        </Field>
        <Field label={t("clean.checks.name")}>
          <TextInput value={check.name ?? ""} maxLength={120} placeholder={describeCheck({ ...check, name: null })} onChange={(event) => set({ name: event.target.value || null })} className={field} />
        </Field>
      </div>
      <p className="text-xs text-ink-3">{t(`clean.checks.kind.${check.kind}.hint` as MessageKey)}</p>
      {check.kind === "condition" ? (
        <div className="space-y-2">
          <div className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,12rem),1fr))] gap-3">
            <Field label={t("clean.checks.expect")}>
              <Select value={check.expect} onChange={(event) => set({ expect: event.target.value as CleanCheck["expect"] })} className={field}>
                <option value="all">{t("clean.checks.expect.all")}</option>
                <option value="none">{t("clean.checks.expect.none")}</option>
              </Select>
            </Field>
            <Field label={t("clean.field.combine")}>
              <Select value={check.combine} onChange={(event) => set({ combine: event.target.value as CleanCheck["combine"] })} className={field}>
                <option value="all">{t("clean.combine.all")}</option>
                <option value="any">{t("clean.combine.any")}</option>
              </Select>
            </Field>
          </div>
          <Conditions value={check.conditions} columns={columns} onChange={(conditions) => set({ conditions })} />
        </div>
      ) : (
        <ColumnsField check={check} columns={columns} onChange={(picked) => set({ columns: picked })} />
      )}
      {check.kind === "in_list" && (
        <Field label={t("clean.checks.values")}>
          <textarea
            className="min-h-16 rounded-md border border-line bg-surface-1 px-2 py-1.5 text-[13px] text-ink-1 outline-none focus-visible:border-accent"
            value={check.values.join("\n")}
            onChange={(event) => set({ values: event.target.value.split(/\r?\n/).map((value) => value.trim()).filter(Boolean) })}
          />
        </Field>
      )}
      {check.kind === "between" && (
        <div className="grid grid-cols-2 gap-3">
          <Field label={t("clean.checks.min")}>
            <TextInput value={check.min ?? ""} onChange={(event) => set({ min: event.target.value || null })} className={field} />
          </Field>
          <Field label={t("clean.checks.max")}>
            <TextInput value={check.max ?? ""} onChange={(event) => set({ max: event.target.value || null })} className={field} />
          </Field>
        </div>
      )}
      {check.kind === "pattern" && (
        <Field label={t("clean.checks.pattern")}>
          <TextInput value={check.pattern ?? ""} placeholder="[0-9]{5}" onChange={(event) => set({ pattern: event.target.value || null })} className={`${field} font-mono`} />
        </Field>
      )}
      {check.kind === "compare" && (
        <Field label={t("clean.checks.op")}>
          <Select value={check.op} onChange={(event) => set({ op: event.target.value as CleanCheck["op"] })} className={field}>
            {COMPARE.map((op) => (
              <option key={op} value={op}>
                {t(`clean.checks.compare.${op}` as MessageKey)}
              </option>
            ))}
          </Select>
        </Field>
      )}
      {check.kind === "sum" && (
        <Field label={t("clean.checks.tolerance")}>
          <TextInput type="number" min={0} step="any" value={String(check.tolerance)} onChange={(event) => set({ tolerance: Math.max(0, Number(event.target.value) || 0) })} className={field} />
        </Field>
      )}
      <div className="flex flex-wrap items-center gap-2">
        <Button type="submit" variant="primary" disabled={!ready || busy}>
          {t("clean.checks.save")}
        </Button>
        <Button variant="ghost" onClick={onCancel}>
          {t("import.cancel")}
        </Button>
      </div>
    </form>
  );
}

/** The columns of a check, one select each, in the order the kind reads them. */
function ColumnsField({ check, columns, onChange }: { check: CleanCheck; columns: string[]; onChange: (columns: string[]) => void }) {
  const [least, most] = ARITY[check.kind];
  const labels = (index: number) => {
    if (check.kind === "compare" || check.kind === "determines") return t(index === 0 ? "clean.checks.columnA" : "clean.checks.columnB");
    if (check.kind === "sum") return t(index === 0 ? "clean.checks.total" : "clean.checks.part");
    return t("clean.field.column");
  };
  return (
    <div className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,12rem),1fr))] gap-3">
      {check.columns.map((name, index) => (
        <Field key={index} label={labels(index)}>
          <span className="flex items-center gap-1">
            <Select value={name} onChange={(event) => onChange(check.columns.map((other, i) => (i === index ? event.target.value : other)))} className={field}>
              {[...new Set([name, ...columns])].map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </Select>
            {check.columns.length > least && (
              <button type="button" aria-label={t("clean.checks.removeColumn")} title={t("clean.checks.removeColumn")} onClick={() => onChange(check.columns.filter((_, i) => i !== index))} className="inline-flex size-7 shrink-0 items-center justify-center rounded-md text-ink-3 hover:bg-surface-2 hover:text-ink-1">
                <X size={13} aria-hidden />
              </button>
            )}
          </span>
        </Field>
      ))}
      {(check.columns.length < least || check.columns.length < most) && (
        <span className="flex items-end">
          <button type="button" onClick={() => onChange([...check.columns, columns.find((name) => !check.columns.includes(name)) ?? columns[0] ?? ""])} className="inline-flex h-8 items-center gap-1 text-xs text-accent hover:underline">
            <Plus size={12} aria-hidden />
            {t("clean.checks.addColumn")}
          </button>
        </span>
      )}
    </div>
  );
}

/** The rows failing a check, its columns first (row numbers from 1); Open these rows in Data when
 *  the check can be written as a filter of the Data page (N41's way). */
function FailingRowsDialog({ datasetId, check, result, onClose }: { datasetId: string; check: CleanCheck; result: CheckResult; onClose: () => void }) {
  const rows = useCheckRows(datasetId, check.id);
  const columns = useColumns(datasetId, true);
  const types = Object.fromEntries((columns.data ?? []).map((column) => [column.name, column.type]));
  const conditions = columns.data ? checkFilter(check, types) : null;
  return (
    <Dialog open onOpenChange={(open) => !open && onClose()} title={t("clean.checks.rowsTitle")} description={describeCheck(check)} size="lg">
      <div className="space-y-3">
        <p className="text-[13px] text-ink-1 tabular-nums">{t("clean.checks.rowsCount", { count: formatCount(result.failing), rows: formatCount(result.rows) })}</p>
        {rows.isError ? (
          <ErrorNote>{rows.error.message}</ErrorNote>
        ) : !rows.data ? (
          <Loading label={t("clean.checks.loading")} className="h-32 rounded-lg" compact />
        ) : (
          <div className="space-y-1">
            <p className="text-xs text-ink-3">{t("clean.preview.first", { count: String(rows.data.rows.length) })}</p>
            <div className="max-h-[28rem] overflow-auto rounded-lg border border-line bg-surface-1">
              <table data-keep-line className="w-full border-collapse text-[12px] tabular-nums">
                <thead className="sticky top-0 bg-surface-1">
                  <tr className="border-b border-line text-left text-[11px] text-ink-3">
                    <th scope="col" className="px-2 py-1.5 text-right font-normal">
                      {t("clean.preview.row")}
                    </th>
                    {rows.data.columns.map((name) => (
                      <th key={name} scope="col" className={`px-2 py-1.5 font-normal ${check.columns.includes(name) ? "text-ink-1" : ""}`}>
                        {name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.data.rows.map(({ row, values }) => (
                    <tr key={row} className="border-b border-line last:border-0">
                      <td className="px-2 py-1 text-right text-ink-3">{formatCount(row + 1)}</td>
                      {values.map((value, index) => (
                        <td key={index} className="max-w-60 truncate px-2 py-1 font-mono whitespace-nowrap text-ink-1">
                          {value ?? <span className="text-ink-3">{t("clean.empty")}</span>}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-t border-line pt-3">
          {conditions ? (
            <>
              <Link to="/d/$datasetId/data" params={{ datasetId }} search={{ filters: { dataset: datasetId, conditions } }} className={buttonClass("secondary")}>
                {t("clean.rows.data")}
                <ArrowRight size={14} strokeWidth={1.5} aria-hidden />
              </Link>
              <p className="min-w-0 flex-1 basis-64 text-xs text-ink-3">{t("clean.checks.dataHint", { filter: conditions.map(describeFilter).join(" · ") })}</p>
            </>
          ) : (
            columns.data && <p className="text-xs text-ink-3">{t("clean.checks.noData")}</p>
          )}
        </div>
      </div>
    </Dialog>
  );
}

/** N31 in Suggestions: a declared check that fails is said at the top, as an information (never a
 *  fix), with the way to its rows. Nothing when there are no checks (or an older server). */
export function ChecksNotice({ datasetId, onOpen }: { datasetId: string; onOpen: () => void }) {
  const found = useChecks(datasetId);
  const report = found.data?.report;
  if (!report || report.failed + report.errors === 0) return null;
  const text = report.failed
    ? t(report.failed === 1 ? "clean.checks.notice.one" : "clean.checks.notice", { count: formatCount(report.failed), rows: formatCount(report.failing_rows) })
    : t("clean.checks.notice.errors", { count: formatCount(report.errors) });
  return (
    <p role="status" className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg border border-line bg-surface-1 px-3 py-2 text-[13px] text-ink-1">
      <ShieldCheck size={15} strokeWidth={1.5} className="shrink-0 text-warn" aria-hidden />
      <span className="min-w-0 flex-1 basis-64">{text}</span>
      <Button variant="ghost" onClick={onOpen}>
        {t("clean.checks.open")}
      </Button>
    </p>
  );
}

/** "Make it a check": the check a step or a suggestion stands for, added to the family's checks
 *  (the same one twice is kept once); it then runs at each import. */
export function MakeCheckButton({ datasetId, check, iconOnly = false }: { datasetId: string; check: CleanCheck; iconOnly?: boolean }) {
  const add = useAddCheck(datasetId);
  const label = describeCheck(check);
  if (add.data) return <span className="text-xs text-ok">{t("clean.checks.made", { id: add.data.id })}</span>;
  return (
    <>
      <Button variant="ghost" icon={ShieldCheck} iconOnly={iconOnly} disabled={add.isPending} aria-label={`${t("clean.checks.make")}: ${label}`} title={`${t("clean.checks.make")}: ${label}`} onClick={() => add.mutate(check)}>
        {iconOnly ? undefined : t("clean.checks.make")}
      </Button>
      {add.error && <span className="text-xs text-warn">{add.error.message}</span>}
    </>
  );
}
