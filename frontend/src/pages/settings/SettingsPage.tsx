import { RotateCcw } from "lucide-react";
import { useState, type ReactNode } from "react";

import type { DetectionSettings, KeepPolicy } from "../../api/client";
import {
  useClearCaches,
  useGpu,
  useHealth,
  useRefreshGpu,
  useSettings,
  useStorage,
  useUpdateSettings,
} from "../../api/hooks";
import { PaletteSelect, PaletteSwatches } from "../../features/theme/PaletteSelect";
import { ThemeSwitcher } from "../../features/theme/ThemeSwitcher";
import { ALPHAS, FACTORY_DEFAULTS, PAGE_SIZES, TOP_NS, useAnalysisDefaults } from "../../lib/defaults";
import { formatBytes } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { DEFAULT_PALETTE, palettePreference } from "../../lib/palettes";
import { densityPreference, motionPreference, usePreference } from "../../lib/preferences";
import { useCleanSettings, useUpdateCleanSettings, type CleanMode } from "../../modules/clean/api";
import { CLEAN_ENABLED } from "../../modules/clean/flag";
import { LegalLinks } from "../../features/legal/LegalLinks";
import { Button } from "../../ui/Button";
import { ErrorNote, Select, TextInput } from "../../ui/fields";
import { PageHeader } from "../../ui/PageHeader";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { Loading } from "../../ui/Loading";

const DETECTION_DEFAULTS: DetectionSettings = { parse_rate: 0.95, empty_rate: 0.95, discrete_max: 20, identifier_unique: 0.95 };
const RATES: { key: keyof DetectionSettings; label: MessageKey; hint: MessageKey }[] = [
  { key: "parse_rate", label: "settings.parseRate", hint: "settings.parseRateHint" },
  { key: "identifier_unique", label: "settings.identifierUnique", hint: "settings.identifierUniqueHint" },
  { key: "empty_rate", label: "settings.emptyRate", hint: "settings.emptyRateHint" },
];

export function SettingsPage() {
  const [density, setDensity] = usePreference(densityPreference);
  const [motion, setMotion] = usePreference(motionPreference);
  const [palette, setPalette] = usePreference(palettePreference);
  const [defaults, setDefaults] = useAnalysisDefaults();

  return (
    <div className="px-(--gutter) py-8">
      <PageHeader title={t("settings.title")} />

      <Section title={t("settings.appearance")}>
        <SettingRow label={t("theme.label")} hint={t("settings.themeHint")}>
          <ThemeSwitcher />
        </SettingRow>
        <SettingRow
          label={t("density.label")}
          hint={t("settings.densityHint")}
          onReset={density !== densityPreference.defaultValue ? () => setDensity(densityPreference.defaultValue) : undefined}
        >
          <SegmentedControl
            label={t("density.label")}
            value={density}
            onChange={setDensity}
            options={[
              { value: "compact", label: t("density.compact") },
              { value: "comfortable", label: t("density.comfortable") },
            ]}
          />
        </SettingRow>
        <SettingRow
          label={t("settings.motion")}
          hint={t("settings.motionHint")}
          onReset={motion !== motionPreference.defaultValue ? () => setMotion(motionPreference.defaultValue) : undefined}
        >
          <SegmentedControl
            label={t("settings.motion")}
            value={motion}
            onChange={setMotion}
            options={[
              { value: "system", label: t("settings.motionSystem") },
              { value: "reduce", label: t("settings.motionReduce") },
              { value: "full", label: t("settings.motionFull") },
            ]}
          />
        </SettingRow>
        <SettingRow
          label={t("palette.label")}
          hint={t("settings.paletteHint")}
          onReset={palette !== DEFAULT_PALETTE ? () => setPalette(DEFAULT_PALETTE) : undefined}
        >
          <span className="flex w-64 max-w-[55vw] flex-col items-stretch gap-1.5">
            <PaletteSelect value={palette} onChange={(id) => setPalette(id ?? DEFAULT_PALETTE)} className="w-full" />
            <PaletteSwatches id={palette} />
          </span>
        </SettingRow>
      </Section>

      <Section title={t("settings.analysis")} intro={t("settings.analysisIntro")}>
        <SettingRow
          label={t("settings.topN")}
          hint={t("settings.topNHint")}
          onReset={defaults.topN !== FACTORY_DEFAULTS.topN ? () => setDefaults({ topN: FACTORY_DEFAULTS.topN }) : undefined}
        >
          <Select aria-label={t("settings.topN")} value={defaults.topN} onChange={(event) => setDefaults({ topN: Number(event.target.value) })} className="w-24">
            {TOP_NS.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </Select>
        </SettingRow>
        <SettingRow
          label={t("settings.pageSize")}
          hint={t("settings.pageSizeHint")}
          onReset={defaults.pageSize !== FACTORY_DEFAULTS.pageSize ? () => setDefaults({ pageSize: FACTORY_DEFAULTS.pageSize }) : undefined}
        >
          <Select aria-label={t("settings.pageSize")} value={defaults.pageSize} onChange={(event) => setDefaults({ pageSize: Number(event.target.value) })} className="w-24">
            {PAGE_SIZES.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </Select>
        </SettingRow>
        <SettingRow
          label={t("settings.alpha")}
          hint={t("settings.alphaHint")}
          onReset={defaults.alpha !== FACTORY_DEFAULTS.alpha ? () => setDefaults({ alpha: FACTORY_DEFAULTS.alpha }) : undefined}
        >
          <SegmentedControl
            label={t("settings.alpha")}
            value={defaults.alpha}
            onChange={(alpha) => setDefaults({ alpha })}
            options={ALPHAS.map((value) => ({ value, label: value }))}
          />
        </SettingRow>
      </Section>

      <DetectionSection />
      {CLEAN_ENABLED && <CleaningSection />}
      <ComputeSection />
      <StorageSection />
      <AboutSection />
    </div>
  );
}

/** Thresholds are edited as a whole and saved with one button: they change future detections. */
function DetectionSection() {
  const settings = useSettings();
  const save = useUpdateSettings();
  const saved = settings.data?.detection;
  const [draft, setDraft] = useState<{ base: DetectionSettings | undefined; values: DetectionSettings | null }>({
    base: undefined,
    values: null,
  });
  // A new server value (first load, after saving) resets the draft.
  const values = draft.base === saved ? (draft.values ?? saved) : saved;
  const edit = (patch: Partial<DetectionSettings>) => values && setDraft({ base: saved, values: { ...values, ...patch } });
  const changed = Boolean(values && saved && JSON.stringify(values) !== JSON.stringify(saved));
  const isFactory = values && JSON.stringify(values) === JSON.stringify(DETECTION_DEFAULTS);

  return (
    <Section title={t("settings.detection")} intro={t("settings.detectionIntro")}>
      {!values ? (
        <Loading label={t("loading.default")} className="h-40 rounded-md" />
      ) : (
        <>
          <SettingRow label={t("settings.discreteMax")} hint={t("settings.discreteMaxHint")}>
            <TextInput
              type="number"
              min={2}
              max={1000}
              aria-label={t("settings.discreteMax")}
              value={values.discrete_max}
              onChange={(event) => edit({ discrete_max: Number(event.target.value) })}
              className="w-24 text-right"
            />
          </SettingRow>
          {RATES.map(({ key, label, hint }) => (
            <SettingRow key={key} label={t(label)} hint={t(hint)}>
              <div className="flex items-center gap-1.5">
                <TextInput
                  type="number"
                  min={50}
                  max={100}
                  step={1}
                  aria-label={t(label)}
                  value={Math.round(values[key] * 100)}
                  onChange={(event) => edit({ [key]: Number(event.target.value) / 100 })}
                  className="w-20 text-right"
                />
                <span className="text-ink-3">%</span>
              </div>
            </SettingRow>
          ))}
          {save.isError && <ErrorNote>{save.error.message}</ErrorNote>}
          <div className="flex items-center justify-end gap-2 pt-3">
            {!isFactory && (
              <Button variant="ghost" onClick={() => setDraft({ base: saved, values: DETECTION_DEFAULTS })}>
                {t("settings.factory")}
              </Button>
            )}
            <Button variant="primary" disabled={!changed || save.isPending} onClick={() => save.mutate({ detection: values })}>
              {save.isSuccess && !changed ? t("settings.saved") : t("settings.save")}
            </Button>
          </div>
        </>
      )}
    </Section>
  );
}

const CLEAN_MODES: { value: CleanMode; label: MessageKey }[] = [
  { value: "suggest", label: "clean.settings.suggest" },
  { value: "auto", label: "clean.settings.auto" },
  { value: "off", label: "clean.settings.off" },
];

/** Cleaning (065): what Clean does with a new dataset. Default "Suggest only" until the precision
 *  of the automatic rules is measured (M4). */
function CleaningSection() {
  const settings = useCleanSettings();
  const save = useUpdateCleanSettings();
  const mode = settings.data?.mode ?? "suggest";
  if (settings.isError) return null; // the server's Clean module is off (QUANTILO_CLEAN_ENABLED)
  return (
    <Section title={t("clean.settings.title")} intro={t("clean.settings.intro")}>
      <SettingRow
        label={t("clean.settings.mode")}
        hint={t("clean.settings.modeHint")}
        onReset={mode !== "suggest" ? () => save.mutate("suggest") : undefined}
      >
        <Select
          aria-label={t("clean.settings.mode")}
          value={mode}
          disabled={!settings.data || save.isPending}
          onChange={(event) => save.mutate(event.target.value as CleanMode)}
          className="w-52 max-w-[55vw]"
        >
          {CLEAN_MODES.map(({ value, label }) => (
            <option key={value} value={value}>
              {t(label)}
            </option>
          ))}
        </Select>
      </SettingRow>
      {save.isError && <ErrorNote>{save.error.message}</ErrorNote>}
    </Section>
  );
}

/** The graphics card found and the user's choice; used by the deep models (023b). */
function ComputeSection() {
  const settings = useSettings();
  const save = useUpdateSettings();
  const gpu = useGpu();
  const refresh = useRefreshGpu();
  const status = gpu.data;
  const ready = status?.state === "ready";
  const stored = settings.data?.compute?.use_gpu;
  const useGpuNow = stored ?? ready;
  return (
    <Section title={t("settings.compute")} intro={t("settings.computeIntro")}>
      <SettingRow label={t("settings.gpuFound")} hint={status?.message ?? "…"}>
        <Button onClick={() => refresh.mutate()} disabled={refresh.isPending}>
          {t("settings.checkAgain")}
        </Button>
      </SettingRow>
      <SettingRow label={t("settings.useGpu")} hint={ready ? t("settings.useGpuHint") : t("settings.useGpuNone")}>
        <SegmentedControl
          label={t("settings.useGpu")}
          value={useGpuNow ? "on" : "off"}
          onChange={(value) => settings.data && save.mutate({ compute: { use_gpu: value === "on" } })}
          options={[
            { value: "on", label: t("settings.on") },
            { value: "off", label: t("settings.off") },
          ]}
        />
      </SettingRow>
    </Section>
  );
}

function StorageSection() {
  const storage = useStorage();
  const clear = useClearCaches();
  const settings = useSettings();
  const save = useUpdateSettings();
  const keep: KeepPolicy = settings.data?.versions?.keep ?? "under_limit";
  const limit = storage.data ? formatBytes(storage.data.keep_limit_bytes ?? 0).replace(".0 ", " ") : "…";
  return (
    <Section title={t("settings.storage")}>
      <SettingRow label={t("settings.workspace")} hint={t("settings.workspaceHint")}>
        <code className="max-w-80 truncate rounded bg-surface-2 px-1.5 py-0.5 font-mono text-xs text-ink-1" title={storage.data?.workspace}>
          {storage.data?.workspace ?? "…"}
        </code>
      </SettingRow>
      <SettingRow label={t("settings.used")} hint={t("settings.usedHint")}>
        <span className="text-ink-1 tabular-nums">{storage.data ? formatBytes(storage.data.total_bytes) : "…"}</span>
      </SettingRow>
      {storage.data && storage.data.datasets.length > 0 && (
        <table className="mt-1 mb-3 w-full text-[13px]">
          <caption className="sr-only">{t("settings.perDataset")}</caption>
          <tbody>
            {storage.data.datasets.map((dataset) => (
              <tr key={dataset.id} className="border-t border-line">
                <td className="py-1.5 text-ink-2">{dataset.name}</td>
                <td className="py-1.5 text-right text-ink-3 tabular-nums">{formatBytes(dataset.bytes)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <SettingRow
        label={t("settings.keepVersion")}
        hint={t("settings.keepVersionHint", { size: storage.data ? formatBytes(storage.data.versions_bytes ?? 0) : "…" })}
      >
        <SegmentedControl
          label={t("settings.keepVersion")}
          value={keep}
          options={[
            { value: "always", label: t("settings.keepAlways") },
            { value: "under_limit", label: t("settings.keepUnder", { size: limit }) },
            { value: "never", label: t("settings.keepNever") },
          ]}
          onChange={(value) => settings.data && save.mutate({ versions: { keep: value } })}
        />
      </SettingRow>
      <SettingRow label={t("settings.caches")} hint={t("settings.cachesHint")}>
        <Button onClick={() => clear.mutate()} disabled={clear.isPending}>
          {clear.isSuccess ? t("settings.cachesCleared") : t("settings.clearCaches")}
        </Button>
      </SettingRow>
    </Section>
  );
}

function AboutSection() {
  const health = useHealth();
  return (
    <>
      <Section title={t("settings.about")}>
        <SettingRow label={t("settings.version")} hint={t("settings.licence")}>
          <span className="font-mono text-xs text-ink-1">{health.data?.version ?? "…"}</span>
        </SettingRow>
      </Section>
      {/* 081: the legal footer of the Settings page */}
      <LegalLinks statement className="mt-3 max-w-2xl" />
    </>
  );
}

function Section({ title, intro, children }: { title: string; intro?: string; children: ReactNode }) {
  return (
    <section className="mt-8 max-w-2xl">
      <h2 className="border-b border-line pb-2 text-sm font-medium">{title}</h2>
      {intro && <p className="mt-2 text-xs text-ink-3">{intro}</p>}
      <dl className="divide-y divide-line">{children}</dl>
    </section>
  );
}

function SettingRow({ label, hint, onReset, children }: { label: string; hint: string; onReset?: () => void; children: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-6 py-3">
      <div className="min-w-0">
        <dt className="text-ink-1">{label}</dt>
        <dd className="text-xs text-ink-3">{hint}</dd>
      </div>
      <dd className="flex shrink-0 items-center gap-2">
        {onReset && (
          <button
            type="button"
            onClick={onReset}
            aria-label={t("settings.resetOne", { name: label })}
            title={t("settings.reset")}
            className="inline-flex size-7 items-center justify-center rounded-md text-ink-3 hover:bg-surface-2 hover:text-ink-1"
          >
            <RotateCcw size={13} strokeWidth={1.75} aria-hidden />
          </button>
        )}
        {children}
      </dd>
    </div>
  );
}
