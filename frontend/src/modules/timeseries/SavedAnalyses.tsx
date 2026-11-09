import { useNavigate } from "@tanstack/react-router";
import { Bookmark, BookmarkPlus, Trash2 } from "lucide-react";
import { useState } from "react";

import type { DriverSpec, SeriesSetup } from "../../api/client";
import { useAnalyses, useAnalysisActions, type SavedAnalysis, type SeriesConfig } from "../../api/recipes";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { ErrorNote, Field, TextInput } from "../../ui/fields";
import { Popover } from "../../ui/Popover";
import { useMethods } from "./Methods";
import type { TimeSeriesSearch } from "./search";

/** The page's configuration as an analysis: what the bar shows, as resolved on the data. */
export function configOf(
  search: TimeSeriesSearch,
  setup: SeriesSetup,
  models: string[],
  horizon: number | null,
  drivers: DriverSpec[] = [],
): SeriesConfig {
  return {
    time: setup.time,
    value: setup.value || null,
    group: setup.group,
    series: setup.series,
    period: search.period ? setup.period : "auto",
    aggregate: search.agg ? setup.aggregate : "auto",
    window: search.window ?? null,
    align: (search.align ?? "trailing") as SeriesConfig["align"],
    gaps: (search.gaps ?? "leave") as SeriesConfig["gaps"],
    season: search.season ?? null,
    models,
    horizon,
    drivers: drivers.length ? drivers : null,
  };
}

/** The page's URL for a saved analysis. */
export function searchOf(config: SavedAnalysis["config"]): Partial<TimeSeriesSearch> {
  const optional = <T,>(value: T | null | undefined) => value ?? undefined;
  return {
    time: optional(config.time),
    value: optional(config.value),
    group: optional(config.group),
    series: optional(config.series),
    period: config.period === "auto" ? undefined : config.period,
    agg: config.aggregate === "auto" ? undefined : config.aggregate,
    window: optional(config.window),
    align: config.align === "trailing" ? undefined : config.align,
    gaps: config.gaps === "leave" ? undefined : config.gaps,
    season: optional(config.season),
  };
}

/** "Save analysis" and the analyses kept on this dataset (030), beside the page title. */
export function SavedAnalyses({
  datasetId,
  search,
  setup,
  drivers,
  onDrivers,
}: {
  datasetId: string;
  search: TimeSeriesSearch;
  setup: SeriesSetup | undefined;
  /** The forecast drivers (028 A4): saved with the analysis, given back when one is opened. */
  drivers: DriverSpec[];
  onDrivers: (drivers: DriverSpec[]) => void;
}) {
  const analyses = useAnalyses(datasetId);
  const { save, remove } = useAnalysisActions(datasetId);
  const [models] = useMethods("forecast");
  const [saving, setSaving] = useState(false);
  const [listing, setListing] = useState(false);
  const [name, setName] = useState("");
  const [horizon, setHorizon] = useState("");
  const navigate = useNavigate();
  const count = analyses.data?.length ?? 0;

  const open = () => {
    if (!setup) return;
    const period = t(`time.period.${setup.period}` as never);
    setName(`${setup.value || t("ts.saved.rows")} per ${period.toLowerCase()}`);
    setHorizon("");
    setSaving(true);
  };
  const steps = Number(horizon);
  const validHorizon = horizon === "" || (Number.isInteger(steps) && steps >= 1 && steps <= 500);
  const submit = () => {
    if (!setup || !validHorizon) return;
    save.mutate(
      { name: name.trim(), config: configOf(search, setup, models, horizon ? steps : null, drivers) },
      { onSuccess: () => setSaving(false) },
    );
  };

  return (
    <div className="mt-3 flex items-center gap-2">
      <Button icon={BookmarkPlus} onClick={open} disabled={!setup}>
        {t("ts.saved.save")}
      </Button>
      <Popover
        open={listing}
        onOpenChange={setListing}
        label={t("ts.saved.list")}
        align="start"
        trigger={
          <Button
            icon={Bookmark}
            disabled={count === 0}
            aria-expanded={listing}
            onClick={() => setListing(!listing)}
          >
            {t("ts.saved.listCount", { count: String(count) })}
          </Button>
        }
      >
        <ul className="space-y-0.5 p-1">
          {(analyses.data ?? []).map((analysis) => (
            <li key={analysis.id} className="flex items-center gap-1">
              <button
                type="button"
                className="min-w-0 flex-1 rounded-md px-2 py-1.5 text-left hover:bg-surface-2"
                onClick={() => {
                  setListing(false);
                  onDrivers(analysis.config.drivers ?? []);
                  void navigate({ to: ".", search: searchOf(analysis.config) as never, replace: true });
                }}
              >
                <span className="block truncate text-[13px] text-ink-1">{analysis.name}</span>
                <span className="block truncate text-xs text-ink-3">
                  {analysis.origin ? t("ts.saved.fromRecipe", { name: analysis.origin }) : analysis.created_at.slice(0, 10)}
                </span>
              </button>
              <Button
                variant="ghost"
                icon={Trash2}
                iconOnly
                aria-label={t("ts.saved.delete", { name: analysis.name })}
                title={t("ts.saved.deleteShort")}
                onClick={() => remove.mutate(analysis.id)}
              />
            </li>
          ))}
        </ul>
      </Popover>
      {saving && (
        <Dialog
          open
          onOpenChange={setSaving}
          title={t("ts.saved.title")}
          description={t("ts.saved.description")}
          size="sm"
          footer={
            <div className="flex justify-end gap-2">
              <Button onClick={() => setSaving(false)}>{t("recipes.cancel")}</Button>
              <Button variant="primary" disabled={!name.trim() || !validHorizon || save.isPending} onClick={submit}>
                {t("ts.saved.confirm")}
              </Button>
            </div>
          }
        >
          <div className="space-y-3">
            <Field label={t("ts.saved.name")}>
              <TextInput value={name} maxLength={200} onChange={(event) => setName(event.target.value)} />
            </Field>
            <Field label={t("ts.saved.horizon")}>
              <TextInput
                inputMode="numeric"
                value={horizon}
                placeholder={t("ts.saved.horizonAuto")}
                onChange={(event) => setHorizon(event.target.value.trim())}
              />
            </Field>
            <p className="text-xs text-ink-2">
              {t("ts.saved.models", { models: models.map((key) => t(`tsm.${key}` as never)).join(", ") })}
            </p>
            {drivers.length > 0 && (
              <p className="text-xs text-ink-2">
                {t("ts.saved.drivers", { drivers: drivers.map((driver) => driver.column).join(", ") })}
              </p>
            )}
            {save.isError && <ErrorNote>{save.error.message}</ErrorNote>}
          </div>
        </Dialog>
      )}
    </div>
  );
}
