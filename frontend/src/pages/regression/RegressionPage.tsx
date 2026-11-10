import { Link, useNavigate, useParams, useSearch } from "@tanstack/react-router";
import { AlertTriangle, CheckCircle2, ChevronDown, Columns3, Info, XCircle } from "lucide-react";
import { Fragment, useMemo, useState, type ReactNode } from "react";

import type {
  AnalysisType,
  Column,
  Regression,
  RegressionCheck,
  RegressionParams,
  RegressionSe,
  RegressionSection,
  RegressionTerm,
} from "../../api/client";
import { useColumns, useDataset, useRegression, useRegressionSuggestion } from "../../api/hooks";
import { formatNumber } from "../../charts/figures";
import { Plot } from "../../charts/Plot";
import { EvidenceBadge } from "../../features/correlations/evidence";
import { NoLongerHolds, noLongerHolds } from "../../features/inference/MultipleTesting";
import { qqFigure } from "../../features/variable/figures";
import { formatP } from "../../features/variable/normality";
import { formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { PHONE, useMedia } from "../../lib/useMedia";
import { Cell } from "../../modules/kpi/Customize";
import { Button } from "../../ui/Button";
import { ErrorNote, Field, Select } from "../../ui/fields";
import { InfoTip } from "../../ui/InfoTip";
import { Loading } from "../../ui/Loading";
import { ParamPanel } from "../../ui/ParamPanel";
import { Popover } from "../../ui/Popover";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { Advanced } from "./Advanced";
import { Sentence } from "./Sentence";
import { influenceFigure, residualsFigure } from "./figures";

export type RegressionSearch = {
  target?: string;
  /** Comma-separated; absent: the strongest links (R-REG-21); "-": none. */
  columns?: string;
  /** Absent: a column of role weight (R-REG-SE-04); "none": no weights. */
  weights?: string;
  cluster?: string;
  se?: string;
  focus?: string;
};

const text = (value: unknown) => (typeof value === "string" && value ? value : undefined);

export function validateRegressionSearch(search: Record<string, unknown>): RegressionSearch {
  return {
    target: text(search.target),
    columns: text(search.columns),
    weights: text(search.weights),
    cluster: text(search.cluster),
    se: text(search.se),
    focus: text(search.focus),
  };
}

const NUMBERS = new Set<AnalysisType>(["numeric_continuous", "numeric_discrete"]);
const TARGETS = new Set<AnalysisType>([...NUMBERS, "boolean"]);
const REGRESSORS = new Set<AnalysisType>([...TARGETS, "categorical_nominal", "categorical_ordinal"]);
const GROUPS = new Set<AnalysisType>(["categorical_nominal", "categorical_ordinal", "identifier", "numeric_discrete", "boolean"]);
export const SE_KINDS: RegressionSe[] = ["classic", "hc0", "hc1", "hc2", "hc3", "hac", "cluster", "cluster2", "dk"];
const NO_COLUMNS = "-";
const NO_WEIGHTS = "none";
const PLOT_HEIGHT = 280;
const STATUS: Record<RegressionCheck["status"], { icon: typeof Info; tone: string; label: MessageKey }> = {
  ok: { icon: CheckCircle2, tone: "text-ok", label: "regression.status.ok" },
  warn: { icon: AlertTriangle, tone: "text-warn", label: "regression.status.warn" },
  bad: { icon: XCircle, tone: "text-bad", label: "regression.status.bad" },
  info: { icon: Info, tone: "text-ink-3", label: "regression.status.info" },
};
const SECTIONS: Record<string, RegressionSection> = {
  standard_errors: "standard-errors",
  influence: "influence",
  terms: "terms",
  panel: "panel",
};

const splitColumns = (value: string | undefined) =>
  value === undefined ? undefined : value === NO_COLUMNS ? [] : value.split(",").filter(Boolean);

export function RegressionPage() {
  const phone = useMedia(PHONE);
  const { datasetId } = useParams({ from: "/d/$datasetId/regression" });
  const search = useSearch({ from: "/d/$datasetId/regression" });
  const navigate = useNavigate({ from: "/d/$datasetId/regression" });
  const dataset = useDataset(datasetId);
  const columns = useColumns(datasetId, dataset.data?.status === "ready");
  const asked = splitColumns(search.columns);
  const suggestion = useRegressionSuggestion(
    datasetId,
    search.target,
    dataset.data?.status === "ready" && (!search.target || asked === undefined || Boolean(search.focus)),
  );
  // R-REGUI-01: the address first, the rules fill what it leaves out.
  const target = search.target ?? suggestion.data?.target?.column ?? undefined;
  const chosen = asked ?? suggestion.data?.columns.map((pick) => pick.column);
  const params: RegressionParams | null =
    target && chosen
      ? {
          target,
          columns: chosen,
          weights: search.weights === NO_WEIGHTS ? "" : search.weights,
          cluster: search.cluster,
          se: (search.se as RegressionSe | undefined) ?? "auto",
          focus: search.focus,
        }
      : null;
  const regression = useRegression(datasetId, params);
  const [advanced, setAdvanced] = useState(false);
  const [opened, setOpened] = useState<RegressionSection[]>([]);
  const [how, setHow] = useState(false);
  const [standardized, setStandardized] = useState(false);

  const usable = useMemo(
    () => (columns.data ?? []).filter((c) => !c.excluded).sort((a, b) => a.position - b.position),
    [columns.data],
  );
  const reasons = useMemo(() => {
    const map = new Map<string, string>();
    if (asked === undefined) for (const pick of suggestion.data?.columns ?? []) map.set(pick.column, pick.reason);
    return map;
  }, [asked, suggestion.data]);

  const set = (patch: RegressionSearch) =>
    void navigate({ search: (previous) => ({ ...previous, ...patch }), replace: true });
  const setColumns = (next: string[]) => set({ columns: next.length ? next.join(",") : NO_COLUMNS, focus: next.includes(search.focus ?? "") ? search.focus : undefined });
  const open = (section: RegressionSection) => {
    setAdvanced(true);
    setOpened((current) => (current.includes(section) ? current : [...current, section]));
    requestAnimationFrame(() => document.getElementById(`regression-${section}`)?.scrollIntoView({ behavior: "smooth", block: "start" }));
  };
  const toggle = (section: RegressionSection) =>
    setOpened((current) => (current.includes(section) ? current.filter((s) => s !== section) : [...current, section]));

  const result = regression.data;
  const settings = (
    <Settings
      usable={usable}
      target={target}
      chosen={chosen ?? []}
      reasons={reasons}
      targetReason={search.target ? null : (suggestion.data?.target?.reason ?? null)}
      onTarget={(name) => set({ target: name, columns: undefined, focus: undefined })}
      onColumns={setColumns}
    />
  );
  const howCells = result && (
    <HowCells
      result={result}
      search={search}
      usable={usable}
      onChange={set}
    />
  );
  const controls = search.focus && target && chosen ? extraControls(suggestion.data?.columns.map((p) => p.column) ?? [], chosen) : [];

  return (
    <div className="flex flex-col gap-5 px-(--gutter) py-6">
      <header className="max-w-prose">
        <h1 className="text-xl font-semibold tracking-tight">{t("regression.title")}</h1>
        <p className="mt-1 text-ink-2">{t("regression.description")}</p>
      </header>

      {phone ? (
        <ParamPanel
          label={t("regression.settings")}
          summary={[target ?? t("regression.pickTarget"), chosen && t("regression.columnsCount", { count: String(chosen.length) }), result?.se.label]}
          layout="stack"
        >
          {settings}
          {howCells}
        </ParamPanel>
      ) : (
        <div className="space-y-3">
          <div className="flex flex-wrap items-end gap-3">
            {settings}
            {result && (
              <button
                type="button"
                aria-expanded={how}
                onClick={() => setHow(!how)}
                className="inline-flex h-(--control-h) items-center gap-1.5 rounded-md px-2 text-[13px] text-ink-2 hover:bg-surface-2 hover:text-ink-1"
              >
                {t("regression.how")}
                <span className="text-ink-3">
                  {result.model_label} · {result.se.label}
                  {result.weights ? ` · ${result.weights}` : ""}
                </span>
                <ChevronDown size={14} strokeWidth={1.5} aria-hidden className={how ? "rotate-180" : ""} />
              </button>
            )}
          </div>
          {how && howCells && <div className="rounded-lg border border-line bg-surface-1 p-3">{howCells}</div>}
        </div>
      )}

      {controls.length > 0 && target && chosen && (
        <p className="flex flex-wrap items-center gap-2 text-[13px] text-ink-2">
          <Sentence text={t("regression.addControls", { target: `\`${target}\``, columns: controls.map((c) => `\`${c}\``).join(", ") })} />
          <Button variant="secondary" onClick={() => setColumns([...chosen, ...controls])}>
            {t("regression.addControlsAction")}
          </Button>
        </p>
      )}

      {suggestion.isError && <ErrorNote>{suggestion.error.message}</ErrorNote>}
      {regression.isError ? (
        <ErrorNote>{regression.error.message}</ErrorNote>
      ) : !params ? (
        suggestion.isPending || columns.isPending ? (
          <Loading label={t("regression.choosing")} className="h-[420px] rounded-lg" />
        ) : (
          <p className="rounded-lg border border-dashed border-line px-4 py-10 text-center text-ink-2">{t("regression.pick")}</p>
        )
      ) : !result ? (
        <Loading label={t("regression.loading")} className="h-[520px] rounded-lg" />
      ) : (
        <>
          <Conclusion result={result} />
          <QualityBand result={result} />
          <Coefficients result={result} standardized={standardized} onStandardized={setStandardized} />
          <Checks result={result} datasetId={datasetId} onOpen={open} />
          <Charts result={result} />
          <section className="rounded-lg border border-line bg-surface-1">
            <button
              type="button"
              aria-expanded={advanced}
              onClick={() => setAdvanced(!advanced)}
              className="flex w-full items-center gap-2 px-4 py-3 text-left"
            >
              <span className="text-sm font-semibold">{t("regression.advanced")}</span>
              <span className="text-xs text-ink-3">{t("regression.advancedHint")}</span>
              <ChevronDown size={15} strokeWidth={1.5} aria-hidden className={`ml-auto text-ink-3 ${advanced ? "rotate-180" : ""}`} />
            </button>
            {advanced && (
              <Advanced
                datasetId={datasetId}
                params={params}
                result={result}
                opened={opened}
                onToggle={toggle}
                onUse={(kind) => set({ se: kind })}
              />
            )}
          </section>
        </>
      )}
    </div>
  );
}

/** The suggested columns not in the model yet, after "Add controls" from Two variables. */
function extraControls(suggested: string[], chosen: string[]): string[] {
  return suggested.filter((name) => !chosen.includes(name)).slice(0, 4);
}

function Settings({
  usable,
  target,
  chosen,
  reasons,
  targetReason,
  onTarget,
  onColumns,
}: {
  usable: Column[];
  target: string | undefined;
  chosen: string[];
  reasons: Map<string, string>;
  targetReason: string | null;
  onTarget: (name: string) => void;
  onColumns: (names: string[]) => void;
}) {
  const targets = usable.filter((c) => TARGETS.has(c.type));
  const regressors = usable.filter((c) => REGRESSORS.has(c.type) && c.name !== target);
  return (
    <>
      <Field label={t("regression.target")}>
        <Select value={target ?? ""} onChange={(event) => onTarget(event.target.value)} title={targetReason ?? undefined} className="w-56 max-sm:w-full">
          {!target && <option value="">{t("regression.pickTarget")}</option>}
          {targets.map((column) => (
            <option key={column.name} value={column.name}>
              {column.name}
            </option>
          ))}
        </Select>
      </Field>
      <Field label={t("regression.columns")}>
        <ColumnsChooser options={regressors.map((c) => c.name)} chosen={chosen} reasons={reasons} onChange={onColumns} />
      </Field>
    </>
  );
}

function ColumnsChooser({
  options,
  chosen,
  reasons,
  onChange,
}: {
  options: string[];
  chosen: string[];
  reasons: Map<string, string>;
  onChange: (names: string[]) => void;
}) {
  const [open, setOpen] = useState(false);
  const toggle = (name: string) =>
    onChange(chosen.includes(name) ? chosen.filter((item) => item !== name) : [...chosen, name]);
  const label = chosen.length ? chosen.slice(0, 3).join(", ") + (chosen.length > 3 ? ` +${chosen.length - 3}` : "") : t("regression.noColumns");
  return (
    <Popover
      open={open}
      onOpenChange={setOpen}
      label={t("regression.columns")}
      align="start"
      trigger={
        <button
          type="button"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          className="inline-flex h-(--control-h) w-72 max-w-full items-center gap-1.5 rounded-md border border-line bg-surface-1 px-2.5 text-left text-[13px] text-ink-1 hover:bg-surface-2 max-sm:w-full"
        >
          <Columns3 size={14} strokeWidth={1.5} aria-hidden className="shrink-0 text-ink-3" />
          <span className="truncate font-mono text-xs">{label}</span>
        </button>
      }
    >
      <div className="max-h-80 w-72 overflow-y-auto p-2">
        {options.map((name) => (
          <label key={name} title={reasons.get(name)} className="flex cursor-pointer items-center gap-2 rounded px-1 py-1 text-[13px] hover:bg-surface-2">
            <input type="checkbox" className="size-3.5 accent-(--accent)" checked={chosen.includes(name)} onChange={() => toggle(name)} />
            <span className="truncate font-mono text-xs">{name}</span>
            {reasons.has(name) && <span className="ml-auto shrink-0 text-[11px] text-ink-3">{t("regression.suggested")}</span>}
          </label>
        ))}
      </div>
    </Popover>
  );
}

/** The rules' choices (045 cells): model, standard errors, weights, grouped by, each with its
 *  reason and a reset to Auto. */
function HowCells({
  result,
  search,
  usable,
  onChange,
}: {
  result: Regression;
  search: RegressionSearch;
  usable: Column[];
  onChange: (patch: RegressionSearch) => void;
}) {
  const field = "h-8 w-full min-w-0 text-[13px]";
  const weights = usable.filter((c) => NUMBERS.has(c.type) && c.name !== result.target);
  const groups = usable.filter((c) => GROUPS.has(c.type) && c.name !== result.target);
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
      <Cell label={t("regression.model")} why={result.model_reason} changed={false} onReset={() => undefined}>
        <p className="flex h-8 items-center truncate text-[13px] text-ink-1" title={result.model_reason}>
          {result.model_label}
        </p>
      </Cell>
      <Cell label={t("regression.se")} why={result.se.reason} changed={Boolean(search.se)} onReset={() => onChange({ se: undefined })}>
        <Select aria-label={t("regression.se")} value={search.se ?? "auto"} onChange={(e) => onChange({ se: e.target.value === "auto" ? undefined : e.target.value })} className={field}>
          <option value="auto">{t("regression.auto", { choice: result.se.label })}</option>
          {SE_KINDS.map((kind) => (
            <option key={kind} value={kind}>
              {t(`regression.seKind.${kind}` as MessageKey)}
            </option>
          ))}
        </Select>
      </Cell>
      <Cell label={t("regression.weights")} why={result.weights ? result.model_reason : t("regression.weightsWhy")} changed={search.weights !== undefined} onReset={() => onChange({ weights: undefined })}>
        <Select aria-label={t("regression.weights")} value={search.weights ?? ""} onChange={(e) => onChange({ weights: e.target.value || undefined })} className={field}>
          <option value="">{t("regression.auto", { choice: result.weights ?? t("regression.none") })}</option>
          <option value={NO_WEIGHTS}>{t("regression.none")}</option>
          {weights.map((c) => (
            <option key={c.name} value={c.name}>
              {c.name}
            </option>
          ))}
        </Select>
      </Cell>
      <Cell label={t("regression.cluster")} why={t("regression.clusterWhy")} changed={Boolean(search.cluster)} onReset={() => onChange({ cluster: undefined })}>
        <Select aria-label={t("regression.cluster")} value={search.cluster ?? ""} onChange={(e) => onChange({ cluster: e.target.value || undefined })} className={field}>
          <option value="">{t("regression.none")}</option>
          {groups.map((c) => (
            <option key={c.name} value={c.name}>
              {c.name}
            </option>
          ))}
        </Select>
      </Cell>
    </div>
  );
}

function Conclusion({ result }: { result: Regression }) {
  return (
    <section className="max-w-prose space-y-1.5">
      <p className="text-[15px] leading-relaxed text-ink-1">
        <Sentence text={result.sentence} />{" "}
        {result.multiple.method !== "none" && <InfoTip label={t("inference.why")} text={result.multiple.reason} />}
      </p>
      <p className="flex flex-wrap items-center gap-x-1.5 text-xs text-ink-3">
        <span>
          {t("regression.seLine", { model: result.model_label, se: result.se.label })}
          {result.se.lag ? ` · ${t("regression.lag", { lag: String(result.se.lag) })}` : ""}
          {result.se.clusters ? ` · ${t("regression.clusters", { count: formatCount(result.se.clusters) })}` : ""}
        </span>
        <InfoTip label={t("regression.why")} text={result.se.reason} />
      </p>
      {result.structure.message && (
        <p className="text-xs text-ink-3">
          <Sentence text={result.structure.message} />
        </p>
      )}
      {result.hints.map((hint) => (
        <p key={hint} className="text-xs text-ink-3">
          <Sentence text={hint} />
        </p>
      ))}
    </section>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="min-w-0">
      <dt className="flex items-center gap-1 text-[11px] text-ink-3">
        {label}
        {hint && <InfoTip label={label} text={hint} />}
      </dt>
      <dd className="text-[15px] text-ink-1 tabular-nums">{value}</dd>
    </div>
  );
}

const r2 = (value: number | null) => (value === null ? "—" : value.toFixed(3));

function QualityBand({ result }: { result: Regression }) {
  const q = result.quality;
  const lost = q.loaded - q.rows;
  return (
    <section className="rounded-lg border border-line bg-surface-1 px-4 py-3">
      <dl className="grid grid-cols-3 gap-x-4 gap-y-3 lg:grid-cols-6">
        <Stat label={t("regression.rowsUsed")} value={formatCount(q.rows)} />
        {q.r2 !== null ? (
          <>
            <Stat label="R²" value={r2(q.r2)} />
            <Stat label={t("regression.adjR2")} value={r2(q.adj_r2)} />
          </>
        ) : (
          <Stat label={t("regression.pseudoR2")} value={r2(q.pseudo_r2)} hint={t("regression.pseudoR2Hint")} />
        )}
        <Stat label={t("regression.oosR2")} value={r2(q.oos_r2)} hint={t("regression.oosHint", { how: q.oos_how })} />
        <Stat label="AIC" value={formatNumber(q.aic)} />
        <Stat label="BIC" value={formatNumber(q.bic)} />
      </dl>
      {(lost > 0 || q.sampled) && (
        <p className="mt-2 text-xs text-ink-3">
          {q.sampled && t("regression.sampled", { rows: formatCount(q.loaded), total: formatCount(q.total) })}{" "}
          {lost > 0 && (
            <Sentence
              text={t("regression.lost", {
                count: formatCount(lost),
                detail: q.missing.map((m) => t("regression.lostColumn", { rows: formatCount(m.rows), column: `\`${m.column}\`` })).join(", "),
              })}
            />
          )}
        </p>
      )}
    </section>
  );
}

function Coefficients({
  result,
  standardized,
  onStandardized,
}: {
  result: Regression;
  standardized: boolean;
  onStandardized: (value: boolean) => void;
}) {
  const logit = result.model === "logit";
  const value = (term: RegressionTerm): [number | null, number | null, number | null] => {
    if (standardized) return [term.standardized, term.std_low, term.std_high];
    if (logit && term.marginal !== null) {
      const points = (v: number | null) => (v === null ? null : v * 100);
      return [points(term.marginal), points(term.marginal_low), points(term.marginal_high)];
    }
    return [term.estimate, term.ci_low, term.ci_high];
  };
  return (
    <section>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-baseline gap-x-3">
          <h2 className="text-sm font-semibold">{t("regression.coefficients")}</h2>
          <span className="text-xs text-ink-3">{logit ? t("regression.coefficientsLogit") : t("regression.coefficientsHint")}</span>
        </div>
        <SegmentedControl
          label={t("regression.scale")}
          value={standardized ? "std" : "unit"}
          options={[
            { value: "unit", label: logit ? t("regression.points") : t("regression.perUnit") },
            { value: "std", label: t("regression.standardized") },
          ]}
          onChange={(next) => onStandardized(next === "std")}
        />
      </div>
      <div className="overflow-x-auto rounded-md border border-line">
        {/* 041 (R-REGUI-03): a phone keeps the term, the estimate with its interval below it, adj. p and the evidence. */}
        <table className="w-full border-collapse text-[13px] max-sm:[&_tr>:nth-child(3)]:hidden max-sm:[&_tr>:nth-child(4)]:hidden max-sm:[&_tr>:nth-child(7)]:hidden">
          <thead className="bg-surface-2 text-left text-xs text-ink-3">
            <tr>
              <th scope="col" className="px-3 py-1.5 font-normal">{t("regression.term")}</th>
              <th scope="col" className="px-3 py-1.5 text-right font-normal">
                {standardized ? t("regression.perSd") : logit ? t("regression.ame") : t("regression.estimate")}
              </th>
              <th scope="col" className="px-3 py-1.5 text-right font-normal">{t("regression.ci")}</th>
              <th scope="col" className="px-3 py-1.5 text-right font-normal">p</th>
              <th scope="col" className="px-3 py-1.5 text-right font-normal">adj. p</th>
              <th scope="col" className="px-3 py-1.5 font-normal">{t("evidence.title")}</th>
              <th scope="col" className="px-3 py-1.5 text-right font-normal">{logit ? t("regression.odds") : t("regression.stdError")}</th>
            </tr>
          </thead>
          <tbody>
            {result.terms.map((term, index) => {
              const [estimate, low, high] = value(term);
              const intercept = index === 0;
              if (intercept && standardized) return null;
              return (
                <tr key={term.term} title={term.sentence} className={`border-t border-line ${term.evidence === "none" && !intercept ? "text-ink-2" : ""}`}>
                  <td className="max-w-64 px-3 py-1.5 font-mono text-xs break-words">{term.term}</td>
                  <td className={`px-3 py-1.5 text-right whitespace-nowrap tabular-nums ${(estimate ?? 0) < 0 ? "text-bad" : "text-ink-1"}`}>
                    {formatNumber(estimate)}
                    <span className="block text-[11px] text-ink-3 sm:hidden">
                      [{formatNumber(low)} ; {formatNumber(high)}]
                    </span>
                  </td>
                  <td className="px-3 py-1.5 text-right whitespace-nowrap text-ink-2 tabular-nums">
                    [{formatNumber(low)} ; {formatNumber(high)}]
                  </td>
                  <td className="px-3 py-1.5 text-right text-ink-3 tabular-nums">{formatP(term.p_value)}</td>
                  <td className="px-3 py-1.5 text-right text-ink-2 tabular-nums">
                    <span className="whitespace-nowrap">{intercept ? "—" : formatP(term.q_value)}</span>
                    {noLongerHolds(term.p_value, term.q_value) && <NoLongerHolds />}
                  </td>
                  <td className="px-3 py-1.5">{!intercept && <EvidenceBadge level={term.evidence} />}</td>
                  <td className="px-3 py-1.5 text-right text-ink-3 tabular-nums">
                    {logit ? (intercept ? "—" : formatNumber(term.odds_ratio)) : formatNumber(term.std_error)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {result.left_out.length > 0 && (
        <p className="mt-1.5 text-xs text-ink-3">
          {t("regression.leftOutCount", { count: String(result.left_out.length) })}{" "}
          {result.left_out.map((item, i) => (
            <Fragment key={item.term}>
              {i > 0 && "; "}
              <code className="font-mono">{item.term}</code>: <Sentence text={item.reason} />
            </Fragment>
          ))}
        </p>
      )}
    </section>
  );
}

function Checks({
  result,
  datasetId,
  onOpen,
}: {
  result: Regression;
  datasetId: string;
  onOpen: (section: RegressionSection) => void;
}) {
  return (
    <section>
      <h2 className="mb-2 text-sm font-semibold">{t("regression.checks")}</h2>
      <ul className="divide-y divide-line rounded-md border border-line bg-surface-1">
        {result.checks.map((check) => {
          const status = STATUS[check.status];
          const Icon = status.icon;
          return (
            <li key={check.key} className="grid grid-cols-[1rem_minmax(0,1fr)] gap-x-3 gap-y-1 px-3 py-2.5 sm:grid-cols-[4.5rem_minmax(0,12rem)_minmax(0,1fr)_8rem] sm:items-start">
              <span className={`inline-flex items-center gap-1 pt-px text-xs font-medium ${status.tone}`}>
                <Icon size={14} strokeWidth={1.75} aria-hidden />
                <span className="sr-only sm:not-sr-only">{t(status.label)}</span>
              </span>
              <span className="text-[13px] font-medium text-ink-1">{check.title}</span>
              <p className="col-start-2 text-[13px] text-ink-2 sm:col-start-auto">
                <Sentence text={check.sentence} />
              </p>
              <span className="col-start-2 sm:col-start-auto">
                <CheckAction check={check} datasetId={datasetId} target={result.target} onOpen={onOpen} />
              </span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function CheckAction({
  check,
  datasetId,
  target,
  onOpen,
}: {
  check: RegressionCheck;
  datasetId: string;
  target: string;
  onOpen: (section: RegressionSection) => void;
}): ReactNode {
  const style = "text-xs whitespace-nowrap text-accent hover:underline";
  const section = check.action ? SECTIONS[check.action] : undefined;
  if (section)
    return (
      <button type="button" className={style} onClick={() => onOpen(section)}>
        {t("regression.try")}
      </button>
    );
  if (check.action === "compare" && check.column)
    return (
      <Link to="/d/$datasetId/compare" params={{ datasetId }} search={(previous) => ({ ...previous, x: check.column ?? undefined, y: target, z: undefined, color: undefined })} className={style}>
        {t("regression.seeCurve")}
      </Link>
    );
  if (check.action === "correlations")
    return (
      <Link to="/d/$datasetId/correlations" params={{ datasetId }} className={style}>
        {t("regression.seeCorrelations")}
      </Link>
    );
  return null;
}

function Charts({ result }: { result: Regression }) {
  const plots = result.plots;
  const residuals = useMemo(() => (plots ? residualsFigure(plots) : null), [plots]);
  const qq = useMemo(
    () => (plots ? qqFigure(plots.qq_theoretical, plots.qq_sample, { slope: 1, intercept: 0 }, t("regression.normalQuantiles")) : null),
    [plots],
  );
  const influence = useMemo(() => (plots ? influenceFigure(plots) : null), [plots]);
  if (!plots || !residuals || !qq || !influence) return null;
  const sentence = (key: string) => result.checks.find((c) => c.key === key)?.sentence ?? "";
  const cards: [string, string, typeof residuals][] = [
    [t("regression.residualsTitle"), sentence("spread"), residuals],
    [t("regression.qqTitle"), sentence("normality"), qq],
    [t("regression.influenceTitle"), sentence("influence"), influence],
  ];
  return (
    <section className="space-y-1.5">
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
        {cards.map(([title, conclusion, figure]) => (
          <figure key={title} className="min-w-0 rounded-lg border border-line bg-surface-1 p-3">
            <figcaption className="mb-1">
              <span className="block text-[13px] font-medium text-ink-1">{title}</span>
              <span className="block text-xs text-ink-2">
                <Sentence text={conclusion} />
              </span>
            </figcaption>
            <Plot figure={figure} height={PLOT_HEIGHT} label={title} />
          </figure>
        ))}
      </div>
      {plots.points < plots.of && (
        <p className="text-xs text-ink-3">{t("regression.plotSample", { points: formatCount(plots.points), rows: formatCount(plots.of) })}</p>
      )}
    </section>
  );
}
