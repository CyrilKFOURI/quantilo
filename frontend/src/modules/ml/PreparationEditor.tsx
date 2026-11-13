import { RotateCcw } from "lucide-react";
import { useState } from "react";

import type { MlChoicesInput, MlClasses, MlColumnChoice, MlDecision, MlPreparation, MlPreparationPreview, MlTask, MlValidation } from "../../api/client";
import { useMlPreparation } from "../../api/hooks";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { Checkbox, ErrorNote, Select, TextInput } from "../../ui/fields";
import { InfoTip } from "../../ui/InfoTip";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { Tabs } from "../../ui/Tabs";
import { Loading } from "../../ui/Loading";
import { TextRepresentation } from "./TextRepresentation";

type Key = "impute" | "transform" | "encoding" | "scaling";
type Tab = "columns" | "selection" | "missing" | "imbalance" | "preview";
type Global = Exclude<keyof MlChoicesInput, "columns" | "selection">;

const KEYS: Key[] = ["impute", "transform", "encoding", "scaling"];
const METHODS = ["variance", "correlation", "vif", "mutual_info", "univariate", "l1", "rfe", "permutation"] as const;
const label = (key: string, value: string) => t(`ml.prep.${key}.${value}` as MessageKey);
/** Causal smoothed lags (063): chosen in Advanced › More only. */
const LAGS = ["past_mean", "past_ewm"];
const isLag = (value: string | null | undefined) => value != null && LAGS.includes(value);

/** The value a decision applies for a setting, as the options name it ("none" = nothing). */
function current(decision: MlDecision, key: Key): string {
  return decision[key] ?? "none";
}

/** One line that says what the preparation does. */
export function preparationSummary(preparation: MlPreparation): string {
  const used = preparation.decisions.filter((d) => d.use);
  const parts = [
    t("ml.preparationSummary", {
      used: String(used.length),
      inputs: String(preparation.inputs),
      excluded: String(preparation.excluded),
    }),
  ];
  const logs = used.filter((d) => d.transform === "log1p").length;
  const onehot = used.filter((d) => d.encoding === "onehot").length;
  if (logs) parts.push(t("ml.prep.summary.log", { count: String(logs) }));
  if (onehot) parts.push(t("ml.prep.summary.onehot", { count: String(onehot) }));
  const texts = used.filter((d) => d.type === "text").length;
  if (texts) parts.push(t("ml.prep.summary.text", { count: String(texts) }));
  const strategy = preparation.balance?.strategy;
  if (strategy && strategy !== "none") parts.push(label("imbalance", strategy).toLowerCase());
  if (preparation.dropped_rows) parts.push(t("ml.prep.summary.dropped", { count: String(preparation.dropped_rows) }));
  // R-SEL-06: columns left out as leaks are named where the user looks, not only in the editor
  const leaks = preparation.decisions.filter((d) => d.leak && !d.use).map((d) => d.column);
  if (leaks.length) parts.push(t("ml.prep.summary.leaks", { count: String(leaks.length), columns: leaks.slice(0, 4).join(", ") + (leaks.length > 4 ? "…" : "") }));
  return parts.join(" · ");
}

export function PreparationEditor({
  open,
  onOpenChange,
  datasetId,
  target,
  task,
  model,
  settings,
  choices,
  onChoices,
  classes = null,
  validation = null,
}: {
  classes?: MlClasses | null;
  /** A panel validation leaves its period and entity out (R-SEL-09): the preview shows it. */
  validation?: MlValidation | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  datasetId: string;
  target: string;
  task: MlTask;
  model: string | null;
  settings: Record<string, unknown> | null;
  choices: MlChoicesInput;
  onChoices: (next: MlChoicesInput) => void;
}) {
  const [tab, setTab] = useState<Tab>("columns");
  const preview = useMlPreparation(datasetId, target, model, settings, choices, open, classes, validation);
  const preparation = preview.data?.preparation;

  const setColumn = (column: string, key: keyof MlColumnChoice, value: string | boolean | undefined) => {
    const next: Partial<MlColumnChoice> = { ...(choices.columns?.[column] ?? {}) };
    if (value === undefined) delete next[key];
    else Object.assign(next, { [key]: value });
    const columns = { ...(choices.columns ?? {}) };
    if (Object.keys(next).length) columns[column] = next;
    else delete columns[column];
    onChoices({ ...choices, columns });
  };
  const setGlobal = (key: Global, value: unknown) => onChoices({ ...choices, [key]: value });
  const custom = Boolean(preparation?.custom);

  const tabs: { value: Tab; label: string; badge?: string }[] = [
    { value: "columns", label: t("ml.prep.tab.columns") },
    { value: "selection", label: t("ml.prep.tab.selection"), badge: choices.selection?.length ? String(choices.selection.length) : undefined },
    { value: "missing", label: t("ml.prep.tab.missing") },
    ...(task === "regression" ? [] : [{ value: "imbalance" as const, label: t("ml.prep.tab.imbalance") }]),
    { value: "preview", label: t("ml.prep.tab.preview"), badge: preview.data ? String(preview.data.columns.length) : undefined },
  ];

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      size="lg"
      title={t("ml.prep.title")}
      description={preparation ? preparationSummary(preparation) : t("ml.prep.loading")}
      footer={
        <>
          <span className="mr-auto text-xs text-ink-3" aria-live="polite">
            {preview.isFetching ? t("ml.prep.updating") : t("ml.prep.appliesNext")}
          </span>
          <Button onClick={() => onOpenChange(false)}>{t("ml.prep.done")}</Button>
        </>
      }
    >
      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-3">
          <SegmentedControl
            label={t("ml.prep.mode")}
            value={custom ? "custom" : "auto"}
            options={[
              { value: "auto", label: t("ml.prep.automatic") },
              { value: "custom", label: t("ml.prep.custom") },
            ]}
            onChange={(mode) => mode === "auto" && onChoices({})}
          />
          <Button variant="ghost" onClick={() => onChoices({})} disabled={!custom}>
            <RotateCcw size={14} strokeWidth={1.5} aria-hidden />
            {t("ml.prep.resetAll")}
          </Button>
          <p className="text-xs text-ink-3">{custom ? t("ml.prep.customHint") : t("ml.prep.autoHint")}</p>
        </div>
        {preview.isError && <ErrorNote>{preview.error.message}</ErrorNote>}
        {preparation && preparation.warnings.length > 0 && (
          <ul className="space-y-1 rounded-md border border-warn/40 bg-warn/8 px-3 py-2 text-xs text-ink-1" role="status">
            {preparation.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        )}
        {!preview.data ? (
          <Loading label={t("loading.preview")} className="h-64 rounded-lg" />
        ) : (
          <Tabs label={t("ml.prep.title")} value={tab} items={tabs} onChange={setTab}>
            <div className={preview.isFetching ? "opacity-70 transition-opacity" : undefined}>
              {tab === "columns" && <ColumnsTab preparation={preview.data.preparation} target={target} onColumn={setColumn} onGlobal={setGlobal} />}
              {tab === "selection" && <SelectionTab preparation={preview.data.preparation} choices={choices} onChoices={onChoices} />}
              {tab === "missing" && <MissingTab preparation={preview.data.preparation} onGlobal={setGlobal} />}
              {tab === "imbalance" && <ImbalanceTab preparation={preview.data.preparation} onGlobal={setGlobal} />}
              {tab === "preview" && <PreviewTab preview={preview.data} />}
            </div>
          </Tabs>
        )}
      </div>
    </Dialog>
  );
}

// --- Columns -------------------------------------------------------------------------------------

type OnColumn = (column: string, key: keyof MlColumnChoice, value: string | boolean | undefined) => void;

function ColumnsTab({
  preparation,
  target,
  onColumn,
  onGlobal,
}: {
  preparation: MlPreparation;
  target: string;
  onColumn: OnColumn;
  onGlobal: (key: Global, value: unknown) => void;
}) {
  const rows = preparation.decisions.filter((d) => d.column !== target);
  const used = rows.filter((d) => d.use);
  const excluded = rows.filter((d) => !d.use);
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[760px] border-collapse text-[13px]">
        <thead className="text-left text-xs text-ink-3">
          <tr>
            <th className="py-1.5 pr-3 font-normal">{t("ml.prep.column")}</th>
            <th className="py-1.5 pr-3 font-normal">{t("ml.prep.use")}</th>
            {KEYS.map((key) => (
              <th key={key} className="py-1.5 pr-3 font-normal">
                {t(`ml.prep.head.${key}` as MessageKey)}
              </th>
            ))}
            <th className="py-1.5 font-normal">
              <span className="sr-only">{t("ml.prep.why")}</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {used.map((d) => (
            <UsedRow key={d.column} decision={d} onColumn={onColumn} />
          ))}
          {excluded.map((d) => (
            <ExcludedRow key={d.column} decision={d} onColumn={onColumn} />
          ))}
        </tbody>
      </table>
      <AdvancedEncodings preparation={preparation} onColumn={onColumn} onTargetLag={(value) => onGlobal("target_lag", value)} target={target} />
    </div>
  );
}

/** Causal smoothed lags (063, R-LAG-01): niche and rarely worth it, so they live in
 * Advanced › More, folded twice and not rendered before the user opens them. */
export function AdvancedEncodings({
  preparation,
  onColumn,
  onTargetLag,
  target,
}: {
  preparation: MlPreparation;
  onColumn: OnColumn;
  /** 063 bis (R-LAG-03): the target's own past as an input, ordered rows only. */
  onTargetLag?: (value: "none" | "past_mean" | "past_ewm") => void;
  target?: string;
}) {
  const [more, setMore] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  return (
    <details className="mt-3 border-t border-line pt-3 text-xs" onToggle={(event) => setAdvanced(event.currentTarget.open)}>
      <summary className="cursor-pointer text-[13px] text-accent select-none hover:underline">{t("ml.prep.advanced")}</summary>
      {preparation.folds && <p className="mt-2 max-w-prose text-ink-2">{preparation.folds.why}</p>}
      {advanced && <TextRepresentation preparation={preparation} onColumn={onColumn} />}
      <details className="mt-2" onToggle={(event) => setMore(event.currentTarget.open)}>
        <summary className="cursor-pointer text-[13px] text-accent select-none hover:underline">{t("ml.prep.more")}</summary>
        {more && <LagChoices preparation={preparation} onColumn={onColumn} />}
        {more && onTargetLag && <TargetLag preparation={preparation} target={target ?? ""} onChange={onTargetLag} />}
      </details>
    </details>
  );
}

function LagChoices({ preparation, onColumn }: { preparation: MlPreparation; onColumn: OnColumn }) {
  const numbers = preparation.decisions.filter((d) => d.use && d.options.transform?.some((o) => isLag(o.value)));
  const reason = numbers[0]?.options.transform?.find((o) => isLag(o.value))?.reason ?? null;
  return (
    <div className="mt-2 space-y-2">
      <p className="max-w-prose text-ink-2">{t("ml.prep.lags.intro")}</p>
      {reason && <p className="max-w-prose text-ink-3">{reason}</p>}
      {numbers.length === 0 ? (
        <p className="text-ink-3">{t("ml.prep.lags.empty")}</p>
      ) : (
        <ul className="grid grid-cols-[repeat(auto-fill,minmax(min(100%,14rem),1fr))] gap-2">
          {numbers.map((d) => (
            <li key={d.column} className="flex min-w-0 items-center gap-2">
              <span className="min-w-0 truncate font-mono text-ink-1">{d.column}</span>
              <Select
                aria-label={t("ml.prep.lags.for", { column: d.column })}
                value={isLag(d.transform) ? (d.transform ?? "") : ""}
                disabled={Boolean(reason)}
                onChange={(event) => onColumn(d.column, "transform", event.target.value || undefined)}
                className="ml-auto h-7 max-w-36 text-xs"
              >
                <option value="">{t("ml.prep.lags.off")}</option>
                {LAGS.map((value) => (
                  <option key={value} value={value}>
                    {label("transform", value)}
                  </option>
                ))}
              </Select>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** 063 bis (R-LAG-03): the target's mean (or EWMA) over its previous periods, shifted by one
 *  period plus the embargo; refused without ordered rows (R-LAG-02), with the reason. */
function TargetLag({ preparation, target, onChange }: { preparation: MlPreparation; target: string; onChange: (value: "none" | "past_mean" | "past_ewm") => void }) {
  const value = preparation.choices?.target_lag ?? "none";
  const said = (preparation.warnings ?? []).find((w) => w.startsWith("R-LAG-03") || w.startsWith("R-LAG-02 · Past values of the target"));
  const ordered = Boolean(preparation.order);
  return (
    <div className="mt-3 space-y-1">
      <p className="max-w-prose text-ink-2">{t("ml.prep.targetLag.intro", { target })}</p>
      <div className="flex items-center gap-2">
        <span className="font-mono text-ink-1">{target}</span>
        <Select
          aria-label={t("ml.prep.targetLag.label", { target })}
          value={value}
          disabled={!ordered && value === "none"}
          title={!ordered ? t("ml.prep.targetLag.needsOrder") : undefined}
          onChange={(event) => onChange(event.target.value as "none" | "past_mean" | "past_ewm")}
          className="h-7 max-w-40 text-xs"
        >
          <option value="none">{t("ml.prep.lags.off")}</option>
          {LAGS.map((lag) => (
            <option key={lag} value={lag}>
              {label("transform", lag)}
            </option>
          ))}
        </Select>
      </div>
      {!ordered && <p className="max-w-prose text-ink-3">{t("ml.prep.targetLag.needsOrder")}</p>}
      {said && <p className="max-w-prose text-ink-2">{said}</p>}
    </div>
  );
}

function TypeTag({ type }: { type: string }) {
  return <span className="ml-1.5 text-[11px] text-ink-3">{t(`ml.prep.type.${type}` as MessageKey)}</span>;
}

function UsedRow({ decision, onColumn }: { decision: MlDecision; onColumn: OnColumn }) {
  const manualUse = decision.manual.includes("use");
  return (
    <tr className="border-t border-line align-middle">
      <td className="py-1.5 pr-3">
        <span className="font-mono text-xs text-ink-1">{decision.column}</span>
        <TypeTag type={decision.type} />
        {decision.leak && <span className="ml-1.5 rounded-full bg-bad/10 px-1.5 py-0.5 text-[10px] text-bad">{t("ml.prep.leakBadge")}</span>}
      </td>
      <td className="py-1.5 pr-3">
        <Checkbox
          aria-label={t("ml.prep.useColumn", { column: decision.column })}
          checked
          onChange={() => onColumn(decision.column, "use", manualUse ? undefined : false)}
        />
      </td>
      {KEYS.map((key) => (
        <td key={key} className="py-1.5 pr-3">
          <ChoiceCell decision={decision} settingKey={key} onColumn={onColumn} />
          {key === "impute" && <IndicatorBox decision={decision} onColumn={onColumn} />}
        </td>
      ))}
      <td className="py-1.5">
        <InfoTip label={t("ml.prep.whyFor", { column: decision.column })} text={decision.why.join(" ")} />
      </td>
    </tr>
  );
}

function ChoiceCell({ decision, settingKey, onColumn }: { decision: MlDecision; settingKey: Key; onColumn: OnColumn }) {
  // lags are chosen in Advanced › More; the cell only shows one once it is chosen
  const options = decision.options[settingKey]?.filter((o) => !isLag(o.value) || o.value === decision.transform);
  const auto = decision.auto[settingKey] ?? "none";
  if (!options || options.length === 0) {
    if (auto === "none") return <span className="text-ink-3">—</span>;
    return <span className="text-xs text-ink-3">{label(settingKey, auto)}</span>;
  }
  const manual = decision.manual.includes(settingKey);
  const pick = (value: string) => onColumn(decision.column, settingKey, value === "" || value === auto ? undefined : value);
  return (
    <span className="inline-flex items-center gap-1">
      {manual && <span className="size-1.5 shrink-0 rounded-full bg-accent" aria-hidden />}
      <Select
        aria-label={t("ml.prep.settingFor", { setting: t(`ml.prep.head.${settingKey}` as MessageKey), column: decision.column })}
        value={manual ? current(decision, settingKey) : ""}
        onChange={(event) => pick(event.target.value)}
        className={`h-7 max-w-44 text-xs ${manual ? "text-ink-1" : "text-ink-3"}`}
      >
        <option value="">{t("ml.prep.auto", { value: label(settingKey, auto) })}</option>
        {options.map((o) => (
          <option key={o.value} value={o.value} disabled={Boolean(o.reason)}>
            {o.reason ? `${label(settingKey, o.value)} — ${o.reason}` : label(settingKey, o.value)}
          </option>
        ))}
      </Select>
      {manual && (
        <button
          type="button"
          onClick={() => pick("")}
          aria-label={t("ml.prep.resetFor", { column: decision.column })}
          className="inline-flex size-5 items-center justify-center rounded text-ink-3 hover:bg-surface-2 hover:text-ink-1"
        >
          <RotateCcw size={12} strokeWidth={1.5} aria-hidden />
        </button>
      )}
    </span>
  );
}

function IndicatorBox({ decision, onColumn }: { decision: MlDecision; onColumn: OnColumn }) {
  const yes = decision.options.indicator?.find((o) => o.value === "yes");
  if (!yes || decision.missing === 0) return null;
  const auto = decision.auto.indicator === "yes";
  return (
    <Checkbox
      className="mt-1 text-[11px] text-ink-2"
      label={t("ml.prep.indicator", { share: `${Math.round(decision.missing * 100) || "<1"} %` })}
      title={yes.reason ?? t("ml.prep.indicatorHint")}
      disabled={Boolean(yes.reason)}
      checked={decision.indicator}
      onChange={(event) => onColumn(decision.column, "indicator", event.target.checked === auto ? undefined : event.target.checked)}
    />
  );
}

function ExcludedRow({ decision, onColumn }: { decision: MlDecision; onColumn: OnColumn }) {
  const [confirming, setConfirming] = useState(false);
  const byYou = decision.manual.includes("use");
  const reason = byYou ? t("ml.prep.leftOutByYou") : (decision.why[0] ?? "");
  return (
    <tr className="border-t border-line text-ink-3">
      <td className="py-1.5 pr-3">
        <span className="font-mono text-xs">{decision.column}</span>
        <TypeTag type={decision.type} />
      </td>
      <td className="py-1.5 pr-3">
        <Checkbox
          aria-label={t("ml.prep.useColumn", { column: decision.column })}
          checked={false}
          disabled={!decision.reactivable}
          onChange={() => (byYou ? onColumn(decision.column, "use", undefined) : decision.leak ? setConfirming(true) : onColumn(decision.column, "use", true))}
        />
      </td>
      <td colSpan={KEYS.length} className={`py-1.5 pr-3 text-xs ${decision.leak ? "text-bad" : ""}`}>
        {confirming ? (
          <span className="inline-flex flex-wrap items-center gap-2 text-ink-1">
            {t("ml.prep.leakConfirm")}
            <Button variant="primary" onClick={() => onColumn(decision.column, "use", true)}>
              {t("ml.prep.useAnyway")}
            </Button>
            <Button variant="ghost" onClick={() => setConfirming(false)}>
              {t("ml.prep.cancel")}
            </Button>
          </span>
        ) : (
          <>
            {t("ml.excluded")} · {reason}
          </>
        )}
      </td>
      <td className="py-1.5">
        <InfoTip label={t("ml.prep.whyFor", { column: decision.column })} text={decision.why.join(" ")} />
      </td>
    </tr>
  );
}

// --- Selection -----------------------------------------------------------------------------------

function SelectionTab({ preparation, choices, onChoices }: { preparation: MlPreparation; choices: MlChoicesInput; onChoices: (next: MlChoicesInput) => void }) {
  const chosen = choices.selection ?? [];
  const toggle = (method: (typeof METHODS)[number], on: boolean) =>
    onChoices({ ...choices, selection: on ? [...chosen, { method, value: null }] : chosen.filter((s) => s.method !== method) });
  const setValue = (method: string, value: number | null) =>
    onChoices({ ...choices, selection: chosen.map((s) => (s.method === method ? { ...s, value } : s)) });
  return (
    <div className="space-y-3">
      <p className="text-xs text-ink-2">{t("ml.prep.selectionHint")}</p>
      <ul className="divide-y divide-line rounded-md border border-line">
        {METHODS.map((method) => {
          const index = chosen.findIndex((s) => s.method === method);
          const report = index >= 0 ? preparation.selection[index] : undefined;
          const choice = index >= 0 ? chosen[index] : undefined;
          return (
            <li key={method} className="flex flex-wrap items-center gap-3 px-3 py-2">
              <Checkbox
                className="min-w-56 text-[13px] text-ink-1"
                label={t(`ml.prep.method.${method}` as MessageKey)}
                checked={index >= 0}
                onChange={(event) => toggle(method, event.target.checked)}
              />
              {method !== "rfe" && (
                <NumberField
                  label={t(`ml.prep.methodValue.${method}` as MessageKey)}
                  value={choice?.value ?? null}
                  placeholder={report?.value ?? undefined}
                  disabled={index < 0}
                  onCommit={(value) => setValue(method, value)}
                />
              )}
              <span className="flex-1 text-xs text-ink-3">{t(`ml.prep.methodWhat.${method}` as MessageKey)}</span>
              {report && (
                <span className="text-xs text-ink-1 tabular-nums">
                  {t("ml.prep.kept", { kept: String(report.kept), of: String(report.of) })}
                </span>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/** A number typed freely and applied on blur or Enter (each change recomputes the preview). */
function NumberField({
  label: text,
  value,
  placeholder,
  disabled,
  percent = false,
  onCommit,
}: {
  label: string;
  value: number | null;
  placeholder?: number;
  disabled?: boolean;
  percent?: boolean;
  onCommit: (value: number | null) => void;
}) {
  const scale = percent ? 100 : 1;
  const shown = value === null ? "" : String(+(value * scale).toPrecision(6));
  const commit = (raw: string) => {
    const number = raw.trim() === "" ? null : Number(raw);
    if (number !== null && !Number.isFinite(number)) return;
    const next = number === null ? null : number / scale;
    if (next !== value) onCommit(next);
  };
  return (
    <label className="flex items-center gap-1.5 text-xs text-ink-2">
      {text}
      <TextInput
        key={shown}
        type="number"
        step="any"
        min={0}
        defaultValue={shown}
        placeholder={placeholder === undefined ? undefined : String(+(placeholder * scale).toPrecision(6))}
        disabled={disabled}
        onBlur={(event) => commit(event.target.value)}
        onKeyDown={(event) => event.key === "Enter" && commit(event.currentTarget.value)}
        className="h-7 w-20 text-xs"
      />
      {percent && "%"}
    </label>
  );
}

// --- Missing & outliers --------------------------------------------------------------------------

function MissingTab({ preparation, onGlobal }: { preparation: MlPreparation; onGlobal: (key: Global, value: unknown) => void }) {
  const c = preparation.choices;
  const row = "flex flex-wrap items-center justify-between gap-3 py-2";
  return (
    <div className="max-w-2xl divide-y divide-line text-[13px]">
      <label className={row}>
        <span>
          {t("ml.prep.numericImpute")}
          <span className="block text-xs text-ink-3">{t("ml.prep.numericImputeHint")}</span>
        </span>
        <Select aria-label={t("ml.prep.numericImpute")} value={c.numeric_impute} onChange={(e) => onGlobal("numeric_impute", e.target.value)} className="h-7 text-xs">
          {(["auto", "mean", "median", "constant", "knn", "iterative"] as const).map((v) => (
            <option key={v} value={v}>
              {label("numericImpute", v)}
            </option>
          ))}
        </Select>
      </label>
      <div className={row}>
        <span>
          {t("ml.prep.indicatorShare")}
          <span className="block text-xs text-ink-3">{t("ml.prep.indicatorShareHint")}</span>
        </span>
        <NumberField label="" percent value={c.indicator_share} onCommit={(v) => onGlobal("indicator_share", v ?? 0.05)} />
      </div>
      <div className={row}>
        <span>
          {t("ml.prep.rare")}
          <span className="block text-xs text-ink-3">{t("ml.prep.rareHint")}</span>
        </span>
        <span className="flex gap-3">
          <NumberField label="" percent value={c.rare_share} onCommit={(v) => onGlobal("rare_share", v ?? 0.01)} />
          <NumberField label={t("ml.prep.rareRows")} value={c.rare_rows} onCommit={(v) => onGlobal("rare_rows", Math.round(v ?? 10))} />
        </span>
      </div>
      <div className={row}>
        <span>
          {t("ml.prep.oneHotLimit")}
          <span className="block text-xs text-ink-3">{t("ml.prep.oneHotLimitHint")}</span>
        </span>
        <NumberField label="" value={c.one_hot_limit} onCommit={(v) => onGlobal("one_hot_limit", Math.round(v ?? 15))} />
      </div>
      <div className={row}>
        <span>
          {t("ml.prep.smoothing")}
          <span className="block text-xs text-ink-3">{t("ml.prep.smoothingHint")}</span>
        </span>
        <NumberField label="" value={c.target_smoothing} onCommit={(v) => onGlobal("target_smoothing", v ?? 10)} />
      </div>
      <label className={row}>
        <span>
          {t("ml.prep.outliers")}
          <span className="block text-xs text-ink-3">
            {preparation.dropped_rows
              ? t("ml.prep.droppedRows", { count: preparation.dropped_rows.toLocaleString("en-US") })
              : t("ml.prep.outliersHint")}
          </span>
        </span>
        <Select aria-label={t("ml.prep.outliers")} value={c.outliers} onChange={(e) => onGlobal("outliers", e.target.value)} className="h-7 text-xs">
          {(["none", "clip_percentiles", "clip_iqr", "drop_rows"] as const).map((v) => (
            <option key={v} value={v}>
              {label("outliers", v)}
            </option>
          ))}
        </Select>
      </label>
    </div>
  );
}

// --- Imbalance -----------------------------------------------------------------------------------

function ImbalanceTab({ preparation, onGlobal }: { preparation: MlPreparation; onGlobal: (key: Global, value: unknown) => void }) {
  const balance = preparation.balance;
  const classes = balance ? Object.keys(balance.before) : [];
  return (
    <div className="max-w-2xl space-y-3 text-[13px]">
      <label className="flex items-center justify-between gap-3">
        {t("ml.prep.imbalanceStrategy")}
        <Select aria-label={t("ml.prep.imbalanceStrategy")} value={preparation.choices.imbalance} onChange={(e) => onGlobal("imbalance", e.target.value)} className="h-7 text-xs">
          {(["auto", "none", "weights", "oversample", "undersample"] as const).map((v) => (
            <option key={v} value={v}>
              {label("imbalance", v)}
            </option>
          ))}
        </Select>
      </label>
      {balance && (
        <>
          <p className="text-xs text-ink-2">
            <span className="font-mono text-ink-3">{balance.rule}</span> · {balance.why}
          </p>
          <table className="w-full border-collapse tabular-nums">
            <thead className="text-left text-xs text-ink-3">
              <tr>
                <th className="py-1 font-normal">{t("ml.class")}</th>
                <th className="py-1 text-right font-normal">{t("ml.prep.before")}</th>
                <th className="py-1 text-right font-normal">{t("ml.prep.after")}</th>
              </tr>
            </thead>
            <tbody>
              {classes.map((name) => (
                <tr key={name} className="border-t border-line">
                  <td className="py-1">{name}</td>
                  <td className="py-1 text-right">{(balance.before[name] ?? 0).toLocaleString("en-US")}</td>
                  <td className="py-1 text-right">{(balance.after[name] ?? 0).toLocaleString("en-US")}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-xs text-ink-3">{t("ml.prep.imbalanceHint")}</p>
        </>
      )}
    </div>
  );
}

// --- Preview -------------------------------------------------------------------------------------

function cell(value: number | null): string {
  if (value === null) return "—";
  if (Number.isInteger(value)) return String(value);
  return value.toPrecision(4);
}

function PreviewTab({ preview }: { preview: MlPreparationPreview }) {
  return (
    <div className="space-y-2">
      <p className="text-xs text-ink-2">
        {t("ml.prep.previewHint", {
          inputs: String(preview.columns.length),
          rows: preview.train_rows.toLocaleString("en-US"),
        })}
      </p>
      <div className="max-h-[50dvh] overflow-auto rounded-md border border-line">
        <table className="border-collapse text-xs tabular-nums">
          <thead className="sticky top-0 bg-surface-1 text-left text-ink-3">
            <tr>
              <th className="px-2 py-1.5 font-normal">{t("ml.prep.row")}</th>
              {preview.columns.map((name) => (
                <th key={name} className="px-2 py-1.5 font-mono font-normal whitespace-nowrap">
                  {name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.rows.map((values, i) => (
              <tr key={preview.row_numbers[i] ?? i} className="border-t border-line">
                <td className="px-2 py-1 text-ink-3">{preview.row_numbers[i]}</td>
                {values.map((value, j) => (
                  <td key={preview.columns[j] ?? j} className="px-2 py-1 text-right">
                    {cell(value)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
