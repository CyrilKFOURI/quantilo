import * as RadixDialog from "@radix-ui/react-dialog";
import { RotateCcw, X } from "lucide-react";
import { useState, type ReactNode } from "react";

import type { MlCandidate, MlModel, MlPanelCheck, MlSetting, MlValidation } from "../../api/client";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Checkbox, Select, TextInput } from "../../ui/fields";
import { InfoTip } from "../../ui/InfoTip";
import { Tabs } from "../../ui/Tabs";
import { LITE } from "../../lite/flag";
import { disabledOptions, isVisible, type Values } from "./rules";

/** Training rows by default: 200,000, and 50,000 in the browser version (R-WEB-01, 102). */
export const DEFAULT_ROWS = LITE ? 50_000 : 200_000;
const FIXED_ROWS = LITE ? 10_000 : 50_000;

export const DEFAULT_VALIDATION: MlValidation = {
  method: "split",
  test_size: 0.2,
  folds: 5,
  repeats: 3,
  group: null,
  time_column: null,
  metric: null,
  seed: 42,
  max_rows: DEFAULT_ROWS,
  threshold: "default",
  period_column: null,
  entity_column: null,
  embargo: 0,
  window: "expanding",
  window_periods: null,
  sample_periods: null,
  criterion: "mean",
  period_threshold: null,
};

export const PANEL = "panel_walk_forward";

/** Panel walk-forward filled from what the page recognised (035): period, entity, embargo. */
export function panelValidation(validation: MlValidation, check: MlPanelCheck | undefined): MlValidation {
  return {
    ...validation,
    method: PANEL,
    period_column: validation.period_column ?? check?.period ?? null,
    entity_column: validation.entity_column ?? check?.entity ?? null,
    embargo: Math.max(validation.embargo, check?.embargo ?? 0),
    threshold: "default",
  };
}

const CRITERIA = ["mean", "sharpe", "worst"] as const;

/** How the scores of the periods make the run's score (mean, mean / std, worst period). */
export function CriterionSelect({ validation, onValidation }: { validation: MlValidation; onValidation: (next: MlValidation) => void }) {
  return (
    <Select
      aria-label={t("ml.validation.criterion")}
      value={validation.criterion}
      onChange={(e) => onValidation({ ...validation, criterion: e.target.value as MlValidation["criterion"] })}
      className="h-7 max-w-44 text-xs"
    >
      {CRITERIA.map((c) => (
        <option key={c} value={c}>
          {t(`ml.validation.criterion.${c}` as MessageKey)}
        </option>
      ))}
    </Select>
  );
}

export const metricName = (metric: string) => t(`ml.metric.${metric}` as MessageKey);

/** One setting: plain label, its field, why the default, the library's name; a dot and a reset
 *  button once changed; options a rule forbids are disabled with the reason. */
export function SettingField({
  setting,
  value,
  changed,
  disabled,
  onChange,
  onReset,
  technical = false,
}: {
  setting: MlSetting;
  value: unknown;
  changed: boolean;
  disabled?: Record<string, string>;
  onChange: (value: unknown) => void;
  onReset: () => void;
  technical?: boolean;
}) {
  const label = (
    <span className="flex min-w-0 items-center gap-1 text-xs text-ink-2">
      {changed && <span className="size-1.5 shrink-0 rounded-full bg-accent" aria-hidden />}
      <span className="truncate">{setting.label}</span>
      {setting.why && <InfoTip label={setting.label} text={setting.why} />}
    </span>
  );
  let field: ReactNode;
  if (setting.kind === "select") {
    field = (
      <Select aria-label={setting.label} value={String(value)} onChange={(event) => onChange(event.target.value)} className="h-7 max-w-44 text-xs">
        {setting.options.map((option) => {
          const reason = disabled?.[option];
          const name = setting.option_labels[option] ?? option;
          return (
            <option key={option} value={option} disabled={Boolean(reason)}>
              {reason ? `${name} — ${reason}` : name}
            </option>
          );
        })}
      </Select>
    );
  } else if (setting.kind === "bool") {
    const reason = disabled?.[String(!value)];
    field = <Checkbox aria-label={setting.label} checked={Boolean(value)} disabled={Boolean(reason)} title={reason} onChange={(event) => onChange(event.target.checked)} />;
  } else if (setting.kind === "text") {
    field = (
      <TextInput key={String(value)} aria-label={setting.label} defaultValue={String(value)} onBlur={(event) => onChange(event.target.value)} className="h-7 w-28 text-xs" />
    );
  } else {
    field = (
      <TextInput
        key={String(value)}
        aria-label={setting.label}
        type="number"
        defaultValue={String(value)}
        min={setting.minimum ?? undefined}
        max={setting.maximum ?? undefined}
        step={setting.kind === "int" ? 1 : "any"}
        onBlur={(event) => event.target.value !== "" && onChange(Number(event.target.value))}
        onKeyDown={(event) => event.key === "Enter" && event.currentTarget.blur()}
        className="h-7 w-24 text-xs"
      />
    );
  }
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-0.5">
      {label}
      <span className="flex items-center gap-1">
        {field}
        {changed && (
          <button type="button" onClick={onReset} aria-label={t("ml.settings.resetOne", { setting: setting.label })} className="inline-flex size-5 items-center justify-center rounded text-ink-3 hover:bg-surface-2 hover:text-ink-1">
            <RotateCcw size={12} strokeWidth={1.5} aria-hidden />
          </button>
        )}
      </span>
      {technical && setting.param && <span className="col-span-2 font-mono text-[10px] text-ink-3">{setting.param}</span>}
    </div>
  );
}

type Tab = "model" | "optimizer" | "validation" | "rows";

export function SettingsDrawer({
  open,
  onOpenChange,
  model,
  values,
  chosen,
  onChange,
  onReset,
  validation,
  onValidation,
  metrics,
  candidates,
  task,
  onPreparation,
  onTrain,
  training,
  error,
  panel,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  model: MlModel;
  values: Values;
  chosen: Values;
  onChange: (key: string, value: unknown) => void;
  onReset: (key?: string) => void;
  validation: MlValidation;
  onValidation: (next: MlValidation) => void;
  metrics: string[];
  candidates: MlCandidate[];
  task: string;
  onPreparation: () => void;
  onTrain: () => void;
  training: boolean;
  error: string | null;
  panel?: MlPanelCheck | undefined;
}) {
  const [tab, setTab] = useState<Tab>("model");
  const [expert, setExpert] = useState(false);
  const disabled = disabledOptions(model, values);
  const shown = (s: MlSetting) => isVisible(s, values) && (expert || s.level !== "expert");
  const modelSettings = model.settings.filter((s) => s.group !== "Optimizer" && shown(s));
  const optimizer = model.settings.filter((s) => s.group === "Optimizer" && shown(s));
  const groups = [...new Set(modelSettings.map((s) => s.group))];
  const active = tab === "optimizer" && !expert ? "model" : tab;
  const field = (s: MlSetting) => (
    <SettingField
      key={s.key}
      setting={s}
      value={values[s.key]}
      changed={s.key in chosen}
      disabled={disabled[s.key]}
      onChange={(value) => onChange(s.key, value)}
      onReset={() => onReset(s.key)}
      technical
    />
  );
  return (
    <RadixDialog.Root open={open} onOpenChange={onOpenChange}>
      <RadixDialog.Portal>
        <RadixDialog.Overlay className="fixed inset-0 z-40 bg-ink-1/20" />
        <RadixDialog.Content className="fixed top-0 right-0 bottom-0 z-50 flex w-[min(480px,100vw)] flex-col border-l border-line bg-surface-1 shadow-[0_12px_40px_-12px_rgb(0_0_0/0.35)]">
          <header className="flex items-center justify-between gap-3 border-b border-line px-5 py-3">
            <div className="min-w-0">
              <RadixDialog.Title className="truncate text-sm font-semibold">{t("ml.settings.title", { model: model.label })}</RadixDialog.Title>
              <RadixDialog.Description className="text-xs text-ink-3">{t("ml.settings.description")}</RadixDialog.Description>
            </div>
            <span className="flex items-center gap-2">
              <Checkbox className="text-xs text-ink-2" label={t("ml.settings.expert")} checked={expert} onChange={(e) => setExpert(e.target.checked)} />
              <Button variant="ghost" onClick={() => onReset()} disabled={Object.keys(chosen).length === 0}>
                {t("ml.settings.resetAll")}
              </Button>
              <RadixDialog.Close aria-label={t("dialog.close")} className="inline-flex size-7 items-center justify-center rounded-md text-ink-3 hover:bg-surface-2 hover:text-ink-1">
                <X size={16} strokeWidth={1.5} aria-hidden />
              </RadixDialog.Close>
            </span>
          </header>
          <div className="min-h-0 flex-1 overflow-y-auto px-5 py-3">
            <Tabs
              label={t("ml.settings.title", { model: model.label })}
              value={active}
              onChange={setTab}
              items={[
                { value: "model", label: t("ml.settings.tab.model") },
                ...(expert ? [{ value: "optimizer" as const, label: t("ml.settings.tab.optimizer") }] : []),
                { value: "validation", label: t("ml.settings.tab.validation") },
                { value: "rows", label: t("ml.settings.tab.rows") },
              ]}
            >
              {active === "model" && (
                <div className="space-y-4">
                  {groups.map((group) => (
                    <section key={group} className="space-y-2.5">
                      <h3 className="text-[11px] font-medium tracking-wide text-ink-3 uppercase">{group}</h3>
                      {modelSettings.filter((s) => s.group === group).map(field)}
                    </section>
                  ))}
                  <section className="space-y-1 border-t border-line pt-3">
                    <h3 className="text-[11px] font-medium tracking-wide text-ink-3 uppercase">{t("ml.preparation")}</h3>
                    <Button variant="ghost" onClick={onPreparation}>
                      {t("ml.settings.openPreparation")}
                    </Button>
                  </section>
                </div>
              )}
              {active === "optimizer" && (
                <div className="space-y-2.5">
                  {optimizer.length ? optimizer.map(field) : <p className="text-xs text-ink-3">{t("ml.settings.noOptimizer")}</p>}
                </div>
              )}
              {active === "validation" && (
                <ValidationTab validation={validation} onValidation={onValidation} metrics={metrics} candidates={candidates} task={task} panel={panel} />
              )}
              {active === "rows" && <RowsTab validation={validation} onValidation={onValidation} />}
            </Tabs>
          </div>
          <footer className="flex items-center gap-3 border-t border-line px-5 py-3">
            {error && <span className="flex-1 text-xs text-bad">{error}</span>}
            <Button variant="primary" className="ml-auto" onClick={onTrain} disabled={training || Boolean(error)}>
              {t("ml.train")}
            </Button>
          </footer>
        </RadixDialog.Content>
      </RadixDialog.Portal>
    </RadixDialog.Root>
  );
}

function Row({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-0.5 text-xs">
      <span className="text-ink-2">{label}</span>
      {children}
      {hint && <span className="col-span-2 text-[11px] text-ink-3">{hint}</span>}
    </label>
  );
}

function NumberBox({ value, min, max, step = 1, label, onChange }: { value: number; min: number; max: number; step?: number; label: string; onChange: (v: number) => void }) {
  return (
    <TextInput
      key={value}
      aria-label={label}
      type="number"
      min={min}
      max={max}
      step={step}
      defaultValue={value}
      onBlur={(e) => {
        const next = Number(e.target.value);
        if (Number.isFinite(next)) onChange(Math.min(max, Math.max(min, next)));
      }}
      className="h-7 w-24 text-xs"
    />
  );
}

const METHODS = ["split", "kfold", "repeated_kfold", "group_kfold", "time_split", "loo", PANEL] as const;
const STEPS = [2, 3, 4, 8] as const;

function ValidationTab({
  validation,
  onValidation,
  metrics,
  candidates,
  task,
  panel,
}: {
  validation: MlValidation;
  onValidation: (next: MlValidation) => void;
  metrics: string[];
  candidates: MlCandidate[];
  task: string;
  panel?: MlPanelCheck | undefined;
}) {
  const set = (patch: Partial<MlValidation>) => onValidation({ ...validation, ...patch });
  const dates = candidates.filter((c) => c.type === "datetime");
  const method = validation.method;
  const pick = (next: MlValidation["method"]) =>
    next === PANEL ? onValidation(panelValidation(validation, panel)) : set({ method: next });
  // on a panel, numbers are judged by their rank correlation within each period
  const auto = method === PANEL && task === "regression" ? "spearman" : (metrics[0] ?? "");
  return (
    <div className="space-y-3">
      <Row label={t("ml.validation.method")} hint={t(`ml.validation.methodHint.${method}` as MessageKey)}>
        <Select aria-label={t("ml.validation.method")} value={method} onChange={(e) => pick(e.target.value as MlValidation["method"])} className="h-7 max-w-48 text-xs">
          {METHODS.map((m) => (
            <option key={m} value={m} disabled={m === "time_split" && dates.length === 0}>
              {t(`ml.validation.method.${m}` as MessageKey)}
            </option>
          ))}
        </Select>
      </Row>
      {(method === "split" || method === "time_split") && (
        <Row label={t("ml.validation.testSize")}>
          <NumberBox label={t("ml.validation.testSize")} value={Math.round(validation.test_size * 100)} min={10} max={40} onChange={(v) => set({ test_size: v / 100 })} />
        </Row>
      )}
      {method === PANEL && <PanelFields validation={validation} set={set} candidates={candidates} panel={panel} />}
      {(method === "kfold" || method === "repeated_kfold" || method === "group_kfold" || method === PANEL) && (
        <Row label={t("ml.validation.folds")}>
          <NumberBox label={t("ml.validation.folds")} value={validation.folds} min={3} max={10} onChange={(v) => set({ folds: v })} />
        </Row>
      )}
      {method === "repeated_kfold" && (
        <Row label={t("ml.validation.repeats")}>
          <NumberBox label={t("ml.validation.repeats")} value={validation.repeats} min={2} max={10} onChange={(v) => set({ repeats: v })} />
        </Row>
      )}
      {method === "group_kfold" && (
        <Row label={t("ml.validation.group")} hint={t("ml.validation.groupHint")}>
          <Select aria-label={t("ml.validation.group")} value={validation.group ?? ""} onChange={(e) => set({ group: e.target.value || null })} className="h-7 max-w-44 text-xs">
            <option value="">—</option>
            {candidates.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name}
              </option>
            ))}
          </Select>
        </Row>
      )}
      {method === "time_split" && (
        <Row label={t("ml.validation.timeColumn")}>
          <Select aria-label={t("ml.validation.timeColumn")} value={validation.time_column ?? ""} onChange={(e) => set({ time_column: e.target.value || null })} className="h-7 max-w-44 text-xs">
            <option value="">—</option>
            {dates.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name}
              </option>
            ))}
          </Select>
        </Row>
      )}
      <Row label={t("ml.validation.metric")} hint={t(task === "regression" ? "ml.validation.metricHintRegression" : "ml.validation.metricHintClasses")}>
        <Select aria-label={t("ml.validation.metric")} value={validation.metric ?? ""} onChange={(e) => set({ metric: e.target.value || null })} className="h-7 max-w-44 text-xs">
          <option value="">{t("ml.validation.metricAuto", { metric: metricName(auto) })}</option>
          {metrics.map((m) => (
            <option key={m} value={m}>
              {metricName(m)}
            </option>
          ))}
        </Select>
      </Row>
      {task === "binary" && (method === "split" || method === "time_split") && (
        <Row label={t("ml.validation.threshold")} hint={t("ml.validation.thresholdHint")}>
          <Select
            aria-label={t("ml.validation.threshold")}
            value={validation.threshold}
            onChange={(e) => set({ threshold: e.target.value as MlValidation["threshold"] })}
            className="h-7 max-w-44 text-xs"
          >
            {(["default", "f1", "balanced_accuracy"] as const).map((v) => (
              <option key={v} value={v}>
                {t(`ml.validation.threshold.${v}` as MessageKey)}
              </option>
            ))}
          </Select>
        </Row>
      )}
      {method === PANEL && (
        <Row label={t("ml.validation.criterion")} hint={t("ml.validation.criterionHint")}>
          <CriterionSelect validation={validation} onValidation={onValidation} />
        </Row>
      )}
      <Row label={t("ml.validation.seed")}>
        <NumberBox label={t("ml.validation.seed")} value={validation.seed} min={0} max={2 ** 31 - 1} onChange={(v) => set({ seed: Math.round(v) })} />
      </Row>
    </div>
  );
}

/** Panel walk-forward (035): the period and entity, the embargo (with the one the target calls
 *  for), the training window and the periods trained on. */
function PanelFields({
  validation,
  set,
  candidates,
  panel,
}: {
  validation: MlValidation;
  set: (patch: Partial<MlValidation>) => void;
  candidates: MlCandidate[];
  panel?: MlPanelCheck | undefined;
}) {
  const found = new Set(panel?.period_candidates ?? []);
  const periods = [...candidates].sort((a, b) => Number(found.has(b.name)) - Number(found.has(a.name)));
  const suggested = panel && panel.period === validation.period_column ? panel.embargo : null;
  const sample = validation.sample_periods === null ? "auto" : String(validation.sample_periods);
  return (
    <>
      <Row label={t("ml.validation.period")} hint={t("ml.validation.periodHint")}>
        <Select aria-label={t("ml.validation.period")} value={validation.period_column ?? ""} onChange={(e) => set({ period_column: e.target.value || null })} className="h-7 max-w-44 text-xs">
          <option value="">—</option>
          {periods.map((c) => (
            <option key={c.name} value={c.name}>
              {c.name}
            </option>
          ))}
        </Select>
      </Row>
      <Row label={t("ml.validation.entity")} hint={t("ml.validation.entityHint")}>
        <Select aria-label={t("ml.validation.entity")} value={validation.entity_column ?? ""} onChange={(e) => set({ entity_column: e.target.value || null })} className="h-7 max-w-44 text-xs">
          <option value="">{t("ml.validation.entityNone")}</option>
          {candidates
            .filter((c) => c.name !== validation.period_column)
            .map((c) => (
              <option key={c.name} value={c.name}>
                {c.name}
              </option>
            ))}
        </Select>
      </Row>
      <Row label={t("ml.validation.embargo")} hint={t("ml.validation.embargoHint")}>
        <NumberBox label={t("ml.validation.embargo")} value={validation.embargo} min={0} max={1_000} onChange={(v) => set({ embargo: Math.round(v) })} />
      </Row>
      {suggested !== null && suggested > validation.embargo && (
        <p className="flex flex-wrap items-center gap-2 text-[11px] text-warn" role="status">
          <span className="min-w-0 flex-1">{panel?.embargo_reason}</span>
          <Button variant="ghost" onClick={() => set({ embargo: suggested })}>
            {t("ml.validation.embargoUse", { embargo: String(suggested) })}
          </Button>
        </p>
      )}
      <Row label={t("ml.validation.window")}>
        <Select aria-label={t("ml.validation.window")} value={validation.window} onChange={(e) => set({ window: e.target.value as MlValidation["window"] })} className="h-7 max-w-48 text-xs">
          <option value="expanding">{t("ml.validation.window.expanding")}</option>
          <option value="rolling">{t("ml.validation.window.rolling")}</option>
        </Select>
      </Row>
      {validation.window === "rolling" && (
        <Row label={t("ml.validation.windowPeriods")} hint={t("ml.validation.windowPeriodsHint")}>
          <TextInput
            key={validation.window_periods ?? "auto"}
            aria-label={t("ml.validation.windowPeriods")}
            type="number"
            min={1}
            defaultValue={validation.window_periods ?? ""}
            onBlur={(e) => set({ window_periods: e.target.value === "" ? null : Math.max(1, Math.round(Number(e.target.value))) })}
            className="h-7 w-24 text-xs"
          />
        </Row>
      )}
      <Row label={t("ml.validation.samplePeriods")} hint={t("ml.validation.samplePeriodsHint")}>
        <Select
          aria-label={t("ml.validation.samplePeriods")}
          value={sample}
          onChange={(e) => set({ sample_periods: e.target.value === "auto" ? null : Number(e.target.value) })}
          className="h-7 max-w-48 text-xs"
        >
          <option value="auto">{t("ml.validation.samplePeriods.auto")}</option>
          <option value="1">{t("ml.validation.samplePeriods.all")}</option>
          {STEPS.map((n) => (
            <option key={n} value={String(n)}>
              {t("ml.validation.samplePeriods.some", { n: String(n) })}
            </option>
          ))}
        </Select>
      </Row>
    </>
  );
}

function RowsTab({ validation, onValidation }: { validation: MlValidation; onValidation: (next: MlValidation) => void }) {
  const mode = validation.max_rows === null ? "all" : validation.max_rows === DEFAULT_ROWS ? "default" : "fixed";
  const hint = LITE ? `${t("ml.rows.hint")} ${t("ml.rows.hintBrowser")}` : t("ml.rows.hint");
  return (
    <div className="space-y-3">
      <Row label={t("ml.rows.training")} hint={validation.method === PANEL ? `${hint} ${t("ml.rows.panelHint")}` : hint}>
        <Select
          aria-label={t("ml.rows.training")}
          value={mode}
          onChange={(e) => onValidation({ ...validation, max_rows: e.target.value === "all" ? null : e.target.value === "default" ? DEFAULT_ROWS : FIXED_ROWS })}
          className="h-7 text-xs"
        >
          <option value="default">{LITE ? t("ml.rows.defaultBrowser") : t("ml.rows.default")}</option>
          {/* R-WEB-01: every row is left to the desktop app */}
          {!LITE && <option value="all">{t("ml.rows.all")}</option>}
          <option value="fixed">{t("ml.rows.fixed")}</option>
        </Select>
      </Row>
      {mode === "fixed" && validation.max_rows !== null && (
        <Row label={t("ml.rows.count")}>
          <NumberBox label={t("ml.rows.count")} value={validation.max_rows} min={100} max={LITE ? DEFAULT_ROWS : 100_000_000} onChange={(v) => onValidation({ ...validation, max_rows: Math.round(v) })} />
        </Row>
      )}
    </div>
  );
}
