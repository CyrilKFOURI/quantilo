import { Check, ChevronRight, Download, Lock, Maximize2, RotateCcw, Shuffle } from "lucide-react";
import { useMemo, useState } from "react";

import { formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { PHONE, useMedia } from "../../lib/useMedia";
import { BeforeAfter, type CompareMode } from "../../ui/BeforeAfter";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { ErrorNote, Field, Select } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { usePrepare, useSavePrepare, useStepPreview, type PrepareState, type StepPreview } from "./api";
import type { ImageRow } from "./model";
import {
  delta,
  isOn,
  measureText,
  pickSamples,
  sameRecipe,
  SHOWN_MEASURES,
  STEPS,
  stepInfo,
  stepsOn,
  toggled,
  type Recipe,
  type Sample,
  type StepId,
  type StepInfo,
} from "./prepModel";
import type { VisionSearch } from "./search";

const SIZES = [128, 160, 192, 224, 256, 288, 320, 384, 448, 512];
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

type PrepareProps = {
  datasetId: string;
  rows: readonly ImageRow[];
  search: VisionSearch;
  set: (patch: VisionSearch) => void;
};

/**
 * 110 (study § 5): how the images enter a model, one step at a time, each shown before → after on
 * six images chosen by a rule that is said (R-VPR-05). The order is fixed (R-VPR-01); the
 * pointed steps are folded under Advanced; the recipe is saved as settings only (R-VPR-03).
 */
export function Prepare({ datasetId, rows, search, set }: PrepareProps) {
  const server = usePrepare(datasetId, true);
  if (server.isError) return <ErrorNote>{(server.error as Error).message}</ErrorNote>;
  if (!server.data) return <Loading label={t("vision.prepare.loading")} className="h-[480px] rounded-md" />;
  return <PrepareBody datasetId={datasetId} rows={rows} search={search} set={set} server={server.data} />;
}

function PrepareBody({ datasetId, rows, search, set, server }: PrepareProps & { server: PrepareState }) {
  const phone = useMedia(PHONE);
  const saved = server.recipe as unknown as Recipe;
  const [draft, setDraft] = useState<Recipe | null>(null);
  const recipe = draft ?? saved;
  const dirty = draft !== null && !sameRecipe(draft, saved);
  const save = useSavePrepare(datasetId);
  const step: StepId = search.step ?? "resize";
  const info = stepInfo(step);
  const [shuffle, setShuffle] = useState(0);
  // a phone compares in one frame (the slider), so a sample fits on the screen (§ 9.3)
  const [mode, setMode] = useState<CompareMode>(() => (phone ? "slider" : "side"));
  const [enlarged, setEnlarged] = useState<Sample | null>(null);
  const samples = useMemo(() => pickSamples(rows, info, shuffle), [rows, info, shuffle]);
  const change = (next: Recipe) => setDraft(next);

  const sentence = t("vision.prepare.sentence", {
    size: String(recipe.size),
    fit: t(recipe.fit === "pad" ? "vision.prepare.fit.padShort" : "vision.prepare.fit.cropShort"),
    steps: stepsOn(recipe) === 1 ? t("vision.prepare.oneStep") : t("vision.prepare.manySteps", { count: formatCount(stepsOn(recipe)) }),
  });

  const download = () => {
    const blob = new Blob([JSON.stringify({ ...recipe, kind: "vision-prepare", version: 1 }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "prepare.json";
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-4">
      <header className="space-y-1">
        <p className="text-[15px] leading-snug font-medium text-pretty text-ink-1">{sentence}</p>
        <p className="text-xs text-ink-2">
          {t(`vision.prepare.task.${server.task}` as MessageKey)} {t("vision.prepare.never")}
        </p>
      </header>

      <div className="flex flex-wrap items-center gap-2">
        <Button variant="primary" icon={Check} onClick={() => save.mutate(recipe as unknown as Record<string, unknown>, { onSuccess: () => setDraft(null) })} disabled={!dirty || save.isPending}>
          {save.isPending ? t("vision.prepare.saving") : t("vision.prepare.save")}
        </Button>
        {dirty && (
          <Button variant="ghost" onClick={() => setDraft(null)}>
            {t("vision.prepare.discard")}
          </Button>
        )}
        <Button variant="ghost" icon={RotateCcw} onClick={() => change(server.default as unknown as Recipe)} disabled={sameRecipe(recipe, server.default as unknown as Recipe)}>
          {t("vision.prepare.reset")}
        </Button>
        <Button variant="ghost" icon={Download} onClick={download}>
          {t("vision.prepare.export")}
        </Button>
        {save.isError && <span className="text-xs text-bad">{(save.error as Error).message}</span>}
        <span className="ml-auto text-xs text-ink-3" aria-live="polite">
          {dirty ? t("vision.prepare.unsaved") : server.saved ? t("vision.prepare.saved") : t("vision.prepare.saveDefault")}
        </span>
      </div>

      <div className={phone ? "space-y-4" : "grid grid-cols-[264px_minmax(0,1fr)] items-start gap-5"}>
        {phone ? (
          <PhoneSteps recipe={recipe} step={step} onStep={(id) => set({ step: id === "resize" ? undefined : id })} onToggle={(id, on) => change(toggled(recipe, id, on))} />
        ) : (
          <Pipeline recipe={recipe} step={step} onStep={(id) => set({ step: id === "resize" ? undefined : id })} onToggle={(id, on) => change(toggled(recipe, id, on))} borders={server.borders} phone={phone} />
        )}

        <section className="min-w-0 space-y-3" aria-labelledby="prepare-step-title">
          <div className="space-y-1">
            <h2 id="prepare-step-title" className="flex flex-wrap items-baseline gap-x-2 text-sm font-semibold text-ink-1">
              {t(`vision.prepare.step.${step}` as MessageKey)}
              <span className="text-xs font-normal text-ink-3">
                {info.always ? t("vision.prepare.always") : isOn(recipe, step) ? t("vision.prepare.on") : t("vision.prepare.offPreview")}
              </span>
            </h2>
            <p className="max-w-[72ch] text-[13px] text-pretty text-ink-2">{t(`vision.prepare.reason.${step}` as MessageKey)}</p>
          </div>

          <Settings step={step} recipe={recipe} onChange={change} task={server.task} />

          {info.preview ? (
            <>
              <div className="flex flex-wrap items-center gap-2 border-t border-line pt-3">
                <p className="min-w-0 basis-full text-xs text-ink-2">{step === "augment" ? t("vision.prepare.samplesAugment") : t("vision.prepare.samplesRule", { measure: info.measure ? t(`vision.prepare.measure.${info.measure}` as MessageKey) : t("vision.prepare.measure.none") })}</p>
                <SegmentedControl<CompareMode>
                  label={t("beforeAfter.modes")}
                  value={mode}
                  onChange={setMode}
                  options={[
                    { value: "side", label: t("beforeAfter.mode.side") },
                    { value: "slider", label: t("beforeAfter.mode.slider") },
                    { value: "toggle", label: t("beforeAfter.mode.toggle") },
                    ...(info.geometric ? [] : [{ value: "difference" as const, label: t("beforeAfter.mode.difference") }]),
                  ]}
                />
                <Button variant="ghost" icon={Shuffle} onClick={() => setShuffle((value) => value + 1)}>
                  {t("vision.prepare.shuffle")}
                </Button>
              </div>
              {info.geometric && <p className="text-xs text-ink-3">{t("vision.prepare.noDifference")}</p>}
              <ul className={phone ? "grid grid-cols-1 gap-5" : "grid grid-cols-1 gap-x-5 gap-y-6 xl:grid-cols-2"}>
                {samples.map((sample, index) => (
                  <li key={`${sample.row.key}:${sample.seed}:${index}`} className="min-w-0">
                    <SamplePreview
                      datasetId={datasetId}
                      sample={sample}
                      step={step}
                      info={info}
                      recipe={recipe}
                      mode={mode}
                      onEnlarge={() => setEnlarged(sample)}
                    />
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <NormaliseTable />
          )}
        </section>
      </div>

      {enlarged && (
        <Dialog open onOpenChange={(open) => !open && setEnlarged(null)} title={`${t(`vision.prepare.step.${step}` as MessageKey)}: ${enlarged.row.name}`} size="lg">
          <SamplePreview datasetId={datasetId} sample={enlarged} step={step} info={info} recipe={recipe} mode={mode} large />
        </Dialog>
      )}
    </div>
  );
}

// --- the pipeline --------------------------------------------------------------------------------

function Pipeline({
  recipe,
  step,
  onStep,
  onToggle,
  borders,
  phone,
}: {
  recipe: Recipe;
  step: StepId;
  onStep: (id: StepId) => void;
  onToggle: (id: StepId, on: boolean) => void;
  borders: PrepareState["borders"];
  phone: boolean;
}) {
  const advanced = STEPS.filter((s) => s.level === "advanced");
  const advancedOn = advanced.filter((s) => isOn(recipe, s.id)).length;
  const [open, setOpen] = useState(advancedOn > 0 || advanced.some((s) => s.id === step));
  // always the recipe's own order (R-VPR-01); the Advanced steps appear in their places when shown
  const shown = STEPS.filter((s) => open || s.level === "first");
  return (
    <nav aria-label={t("vision.prepare.pipeline")} className={phone ? "space-y-1" : "sticky top-2 space-y-1"}>
      <ol className="relative space-y-0.5 before:absolute before:top-4 before:bottom-4 before:left-4 before:w-px before:bg-line">
        {shown.map((info) => (
          <StepRow key={info.id} info={info} recipe={recipe} active={info.id === step} onStep={onStep} onToggle={onToggle} suggest={info.id === "trim" && borders.suggest} />
        ))}
      </ol>
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
        className="flex min-h-9 w-full items-center gap-1.5 rounded-md px-2 text-left text-xs text-ink-2 hover:bg-surface-2 pointer-coarse:min-h-11"
      >
        <ChevronRight size={14} strokeWidth={1.5} aria-hidden className={`shrink-0 text-ink-3 motion-safe:transition-transform ${open ? "rotate-90" : ""}`} />
        <span className="font-medium text-ink-1">{open ? t("vision.prepare.hideAdvanced") : t("vision.prepare.showAdvanced")}</span>
        <span className="tabular-nums">{t("vision.prepare.advancedCount", { on: String(advancedOn), all: String(advanced.length) })}</span>
        {borders.suggest && !recipe.trim && !open && <span className="ml-auto text-warn">{t("vision.prepare.suggested")}</span>}
      </button>
    </nav>
  );
}

/** On a phone (041): the steps in one native list, in their order, and the switch of the chosen
 *  one, so the previews stay on the first screen. */
function PhoneSteps({ recipe, step, onStep, onToggle }: { recipe: Recipe; step: StepId; onStep: (id: StepId) => void; onToggle: (id: StepId, on: boolean) => void }) {
  const info = stepInfo(step);
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_auto] items-end gap-3 rounded-md border border-line bg-surface-1 p-3">
      <Field label={t("vision.prepare.pipeline")}>
        <Select value={step} onChange={(event) => onStep(event.target.value as StepId)}>
          {STEPS.map((s) => (
            <option key={s.id} value={s.id}>
              {t(`vision.prepare.step.${s.id}` as MessageKey)}
              {s.always ? "" : ` (${isOn(recipe, s.id) ? t("vision.prepare.sum.on") : t("vision.prepare.sum.off")})`}
              {s.level === "advanced" ? `, ${t("vision.prepare.advancedTag")}` : ""}
            </option>
          ))}
        </Select>
      </Field>
      {info.always ? (
        <span className="inline-flex min-h-11 items-center gap-1.5 text-xs text-ink-3">
          <Lock size={12} strokeWidth={1.5} aria-hidden />
          {t("vision.prepare.always")}
        </span>
      ) : (
        <label className="inline-flex min-h-11 cursor-pointer items-center gap-2 text-[13px] text-ink-1">
          <input type="checkbox" checked={isOn(recipe, step)} onChange={(event) => onToggle(step, event.target.checked)} className="size-4 accent-(--accent)" />
          {t("vision.prepare.use")}
        </label>
      )}
    </div>
  );
}

function StepRow({ info, recipe, active, onStep, onToggle, suggest }: { info: StepInfo; recipe: Recipe; active: boolean; onStep: (id: StepId) => void; onToggle: (id: StepId, on: boolean) => void; suggest: boolean }) {
  const on = isOn(recipe, info.id);
  const name = t(`vision.prepare.step.${info.id}` as MessageKey);
  return (
    <li className={`relative flex items-center gap-1 rounded-md ${active ? "bg-surface-2" : "hover:bg-surface-2"}`}>
      {info.always ? (
        <span className="inline-flex size-8 shrink-0 items-center justify-center bg-inherit text-ink-3" title={t("vision.prepare.always")} aria-hidden>
          <Lock size={12} strokeWidth={1.5} />
        </span>
      ) : (
        <label className="relative inline-flex size-8 shrink-0 cursor-pointer items-center justify-center pointer-coarse:size-11">
          <input type="checkbox" checked={on} onChange={(event) => onToggle(info.id, event.target.checked)} aria-label={t("vision.prepare.toggle", { name })} className="size-3.5 accent-(--accent)" />
        </label>
      )}
      <button
        type="button"
        aria-current={active ? "step" : undefined}
        onClick={() => onStep(info.id)}
        className="flex min-h-9 min-w-0 flex-1 flex-col items-start justify-center py-1 pr-2 text-left pointer-coarse:min-h-11"
      >
        <span className={`text-[13px] ${active ? "font-medium text-ink-1" : on || info.always ? "text-ink-1" : "text-ink-2"}`}>
          {name}
          {info.level === "advanced" && <span className="ml-1.5 text-xs text-ink-3">{t("vision.prepare.advancedTag")}</span>}
        </span>
        <span className="text-xs text-ink-3">{suggest && !on ? t("vision.prepare.bordersSuggest") : summary(info.id, recipe)}</span>
      </button>
    </li>
  );
}

function summary(id: StepId, recipe: Recipe): string {
  switch (id) {
    case "decode":
      return t("vision.prepare.sum.decode");
    case "resize":
      return t("vision.prepare.sum.resize", { size: String(recipe.size), fit: t(recipe.fit === "pad" ? "vision.prepare.fit.padShort" : "vision.prepare.fit.cropShort") });
    case "denoise":
      return recipe.denoise === "off" ? t("vision.prepare.sum.off") : recipe.denoise === "median" ? t("vision.prepare.sum.median") : t("vision.prepare.sum.gaussian", { sigma: String(recipe.sigma) });
    case "equalise":
      return recipe.equalise === "off" ? t("vision.prepare.sum.off") : recipe.equalise === "global" ? t("vision.prepare.sum.global") : t("vision.prepare.sum.clahe", { clip: String(recipe.clahe_clip) });
    case "augment":
      return isOn(recipe, "augment") ? augmentSummary(recipe) : t("vision.prepare.sum.off");
    case "normalise":
      return t("vision.prepare.sum.normalise");
    default:
      return isOn(recipe, id) ? t("vision.prepare.sum.on") : t("vision.prepare.sum.off");
  }
}

function augmentSummary(recipe: Recipe): string {
  const parts = [
    recipe.flip ? t("vision.prepare.aug.flip") : null,
    recipe.rotate > 0 ? t("vision.prepare.aug.rotate", { deg: String(recipe.rotate) }) : null,
    recipe.jitter > 0 ? t("vision.prepare.aug.jitter", { j: String(recipe.jitter) }) : null,
    recipe.crop_scale < 1 ? t("vision.prepare.aug.crop", { s: String(recipe.crop_scale) }) : null,
  ].filter(Boolean);
  return parts.join(", ");
}

// --- the settings of a step ----------------------------------------------------------------------

function Settings({ step, recipe, onChange, task }: { step: StepId; recipe: Recipe; onChange: (next: Recipe) => void; task: PrepareState["task"] }) {
  const number = (key: keyof Recipe, values: readonly number[], label: MessageKey) => (
    <Field label={t(label)}>
      <Select value={String(recipe[key])} onChange={(event) => onChange({ ...recipe, [key]: Number(event.target.value) })}>
        {values.map((value) => (
          <option key={value} value={value}>
            {value}
          </option>
        ))}
      </Select>
    </Field>
  );
  const grid = "grid grid-cols-2 gap-3 sm:grid-cols-3";
  switch (step) {
    case "resize":
      return (
        <div className={grid}>
          {number("size", SIZES, "vision.prepare.set.size")}
          <div className="space-y-1">
            <span className="block text-xs text-ink-2">{t("vision.prepare.set.fit")}</span>
            <SegmentedControl<Recipe["fit"]>
              label={t("vision.prepare.set.fit")}
              value={recipe.fit}
              onChange={(fit) => onChange({ ...recipe, fit })}
              options={[
                { value: "crop", label: t("vision.prepare.fit.crop") },
                { value: "pad", label: t("vision.prepare.fit.pad") },
              ]}
            />
          </div>
          <p className="col-span-2 self-end text-xs text-ink-3 sm:col-span-1">{t("vision.prepare.oddRule")}</p>
        </div>
      );
    case "denoise":
      return (
        <div className={grid}>
          <Field label={t("vision.prepare.set.method")}>
            <Select value={recipe.denoise === "off" ? "gaussian" : recipe.denoise} onChange={(event) => onChange({ ...recipe, denoise: event.target.value as Recipe["denoise"] })}>
              <option value="gaussian">{t("vision.prepare.denoise.gaussian")}</option>
              <option value="median">{t("vision.prepare.denoise.median")}</option>
            </Select>
          </Field>
          {recipe.denoise !== "median" && number("sigma", [0.5, 1, 1.5, 2, 3], "vision.prepare.set.sigma")}
        </div>
      );
    case "equalise":
      return (
        <div className={grid}>
          <Field label={t("vision.prepare.set.method")}>
            <Select value={recipe.equalise === "off" ? "clahe" : recipe.equalise} onChange={(event) => onChange({ ...recipe, equalise: event.target.value as Recipe["equalise"] })}>
              <option value="clahe">{t("vision.prepare.equalise.clahe")}</option>
              <option value="global">{t("vision.prepare.equalise.global")}</option>
            </Select>
          </Field>
          {recipe.equalise !== "global" && number("clahe_clip", [1, 1.5, 2, 3, 4], "vision.prepare.set.clip")}
          {recipe.equalise !== "global" && number("clahe_tiles", [4, 8, 16], "vision.prepare.set.tiles")}
        </div>
      );
    case "sharpen":
      return (
        <div className={grid}>
          {number("sharpen_radius", [0.5, 1, 2, 3], "vision.prepare.set.radius")}
          {number("sharpen_amount", [0.5, 1, 1.5, 2], "vision.prepare.set.amount")}
        </div>
      );
    case "augment":
      return (
        <div className="space-y-2">
          <div className={grid}>
            <Field label={t("vision.prepare.set.flip")}>
              <Select value={recipe.flip ? "on" : "off"} onChange={(event) => onChange({ ...recipe, flip: event.target.value === "on" })}>
                <option value="on">{t("vision.prepare.yes")}</option>
                <option value="off">{t("vision.prepare.no")}</option>
              </Select>
            </Field>
            {number("rotate", [0, 5, 10, 15, 20], "vision.prepare.set.rotate")}
            {number("jitter", [0, 0.1, 0.2, 0.3], "vision.prepare.set.jitter")}
            {number("crop_scale", [1, 0.9, 0.8, 0.6], "vision.prepare.set.cropScale")}
          </div>
          {task === "anomaly" && <p className="text-xs text-warn">{t("vision.prepare.anomalyWarning")}</p>}
        </div>
      );
    default:
      return null;
  }
}

function NormaliseTable() {
  const channels = ["R", "G", "B"];
  return (
    <div className="max-w-md space-y-2 border-t border-line pt-3">
      <table className="w-full text-[13px] tabular-nums">
        <caption className="mb-1 text-left text-xs text-ink-2">{t("vision.prepare.normaliseCaption")}</caption>
        <thead>
          <tr className="text-xs text-ink-3">
            <th className="py-1 text-left font-normal">{t("vision.prepare.channel")}</th>
            <th className="py-1 text-right font-normal">{t("vision.prepare.mean")}</th>
            <th className="py-1 text-right font-normal">{t("vision.prepare.std")}</th>
          </tr>
        </thead>
        <tbody>
          {channels.map((channel, i) => (
            <tr key={channel} className="border-t border-line">
              <td className="py-1 text-ink-1">{channel}</td>
              <td className="py-1 text-right text-ink-1">{IMAGENET_MEAN[i]?.toFixed(3)}</td>
              <td className="py-1 text-right text-ink-1">{IMAGENET_STD[i]?.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs text-ink-3">{t("vision.prepare.normaliseNote")}</p>
    </div>
  );
}

// --- one sample, before → after ------------------------------------------------------------------

function SamplePreview({
  datasetId,
  sample,
  step,
  info,
  recipe,
  mode,
  onEnlarge,
  large = false,
}: {
  datasetId: string;
  sample: Sample;
  step: StepId;
  info: StepInfo;
  recipe: Recipe;
  mode: CompareMode;
  onEnlarge?: () => void;
  large?: boolean;
}) {
  const preview = useStepPreview(datasetId, sample.row.id, step, recipe as unknown as Record<string, unknown>, sample.seed);
  const data = preview.data;
  const why =
    sample.why === "typical"
      ? t("vision.prepare.why.typical")
      : sample.why === "random"
        ? step === "augment"
          ? t("vision.prepare.why.draw", { n: String((sample.seed % 6) + 1) })
          : t("vision.prepare.why.random")
        : t(`vision.prepare.why.${sample.why}` as MessageKey, { measure: info.measure ? t(`vision.prepare.measure.${info.measure}` as MessageKey) : "" });

  if (preview.isError) return <ErrorNote>{(preview.error as Error).message}</ErrorNote>;
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline gap-2 text-xs">
        <span className="min-w-0 flex-1 truncate text-ink-1" title={sample.row.key}>
          {sample.row.name}
        </span>
        <span className="shrink-0 text-ink-3">{why}</span>
        {onEnlarge && (
          <button type="button" onClick={onEnlarge} className="relative inline-flex size-6 shrink-0 items-center justify-center rounded text-ink-3 hover:bg-surface-2 hover:text-ink-1 pointer-coarse:size-11" aria-label={t("vision.prepare.enlarge", { name: sample.row.name })} title={t("vision.prepare.enlarge", { name: sample.row.name })}>
            <Maximize2 size={13} strokeWidth={1.5} aria-hidden />
          </button>
        )}
      </div>
      {data ? (
        <BeforeAfter
          label={t("vision.prepare.compareLabel", { name: sample.row.name, step: t(`vision.prepare.step.${step}` as MessageKey) })}
          mode={large ? undefined : mode}
          defaultMode={mode}
          showModes={large}
          wide={large}
          differenceBlocked={info.geometric ? t("vision.prepare.noDifference") : null}
          busy={preview.isFetching}
          panels={[
            { src: data.before, label: t("vision.prepare.before"), alt: t("vision.prepare.beforeAlt", { name: sample.row.name }), caption: <Caption data={data} side="before" /> },
            { src: data.after, label: t("vision.prepare.after"), alt: t("vision.prepare.afterAlt", { name: sample.row.name }), caption: <Caption data={data} side="after" /> },
          ]}
        />
      ) : (
        <Loading label={t("vision.prepare.computing")} compact className={large ? "h-[420px] rounded-md" : "aspect-[2/1] rounded-md"} />
      )}
      {data && data.notes.length > 0 && (
        <ul className="space-y-0.5 text-xs text-ink-2">
          {data.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Caption({ data, side }: { data: StepPreview; side: "before" | "after" }) {
  const size = side === "before" ? data.before_size : data.after_size;
  const measures = side === "before" ? data.before_measures : data.after_measures;
  const resized = (data.before_size[0] ?? 0) !== (data.after_size[0] ?? 0) || (data.before_size[1] ?? 0) !== (data.after_size[1] ?? 0);
  return (
    <span className="block text-ink-2 tabular-nums">
      <span className="block">{t("vision.prepare.size", { w: formatCount(size[0] ?? 0), h: formatCount(size[1] ?? 0) })}</span>
      <span className="flex flex-wrap gap-x-2.5">
        {SHOWN_MEASURES.map((measure) => {
          const change = side === "after" ? delta(data.before_measures[measure], data.after_measures[measure], measure === "sharpness" ? 0 : 2) : null;
          const comparable = !(measure === "sharpness" && resized); // R-VPR-09
          return (
            <span key={measure} title={!comparable && side === "after" ? t("vision.prepare.notComparable") : undefined}>
              <span className="text-ink-3">{t(`vision.prepare.short.${measure}` as MessageKey)}</span> {measureText(measure, measures[measure])}
              {side === "after" && change !== null && change !== "0" && comparable && <span className="text-ink-1"> ({change})</span>}
              {side === "after" && !comparable && <span className="text-ink-3"> (n/c)</span>}
            </span>
          );
        })}
      </span>
    </span>
  );
}
