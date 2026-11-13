import { ChevronDown, ChevronRight, Plus, X } from "lucide-react";

import type {
  DriverSpec,
  ForecastJob,
  HolidayOffer,
  SeriesDrivers,
} from "../../api/client";
import { formatNumber } from "../../charts/figures";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { Select, TextInput } from "../../ui/fields";
import { LOCKED } from "./Methods";

type Result = NonNullable<ForecastJob["result"]>;
type Candidate = SeriesDrivers["candidates"][number];
type Future = DriverSpec["future"];

/** The models that use drivers (028 A4); the others ignore them. */
export const DRIVER_MODELS = ["sarimax", "tslm", "gbm_lags", "prophet_like"];
const MAX_DRIVERS = 6;

const modelName = (key: string) => t(`tsm.${key}` as MessageKey);
const r = (value: number | null) => (value === null ? "—" : value.toFixed(2));

/** The drivers worth sending: columns still offered, calendar parts valid for the period,
 * public holidays when the period allows them and a country is chosen (028 A5). */
export function usableDrivers(
  drivers: DriverSpec[],
  candidates: SeriesDrivers | undefined,
): DriverSpec[] {
  if (!candidates) return [];
  const columns = new Set(candidates.candidates.map((c) => c.column));
  return drivers.filter((d) =>
    d.kind === "calendar"
      ? candidates.calendar.includes(d.column)
      : d.kind === "holidays"
        ? Boolean(candidates.holidays?.usable && d.column)
        : columns.has(d.column),
  );
}

/** The drivers kept by the page, holidays still waiting for a country included (shown in the
 * panel, not sent). */
export function shownDrivers(
  drivers: DriverSpec[],
  candidates: SeriesDrivers | undefined,
): DriverSpec[] {
  const usable = usableDrivers(drivers, candidates);
  return drivers.filter(
    (d) =>
      usable.includes(d) ||
      (d.kind === "holidays" && !d.column && candidates?.holidays?.usable),
  );
}

const HOLIDAY_NAME = /^Public holidays \((.+)\)$/;
/** A driver's name as shown: the holidays' label in the reader's language. */
export function driverLabel(name: string): string {
  const found = HOLIDAY_NAME.exec(name);
  return found ? t("ts.holidays.name", { code: found[1] ?? "" }) : name;
}

/** Why this country is proposed, or why none is (coded rule, R-TS-HOL-01). */
export function guessText(offer: HolidayOffer): string {
  const { guess } = offer;
  const country = offer.countries.find((c) => c.code === guess.code);
  return t(`ts.holidays.rule.${guess.rule}` as MessageKey, {
    country: country ? `${country.name} (${country.code})` : (guess.code ?? ""),
    column: guess.column ?? "",
    value: guess.value ?? "",
  });
}

/** Why the calendar cannot be used here. */
function holidaysBlocked(offer: HolidayOffer): string | null {
  if (offer.usable) return null;
  return offer.countries.length
    ? t("ts.holidays.needsPeriod")
    : t("ts.holidays.missing");
}

const spec = (
  column: string,
  kind: DriverSpec["kind"],
  known = false,
): DriverSpec => ({ column, kind, known, future: "file", value: null });

/** A column as a new driver: known in advance when the file already has its future values. */
export function newDriver(candidate: Candidate, flag = false): DriverSpec {
  if (flag) return spec(candidate.column, "flag", true);
  return spec(candidate.column, "column", candidate.future_periods > 0);
}

/** "Revenue and Visits move together: use Visits as a driver. [Add driver]" */
export function DriverSuggestion({
  candidates,
  drivers,
  onAdd,
}: {
  candidates: SeriesDrivers | undefined;
  drivers: DriverSpec[];
  onAdd: (driver: DriverSpec) => void;
}) {
  const chosen = new Set(drivers.map((d) => d.column));
  const best = candidates?.candidates.find(
    (c) => c.suggested && !c.flag && !chosen.has(c.column),
  );
  if (!best || drivers.length >= MAX_DRIVERS) return null;
  return (
    <p className="flex flex-wrap items-center gap-2 text-[13px] text-ink-2">
      <span>
        {t("ts.drivers.suggest", {
          what: candidates?.setup.value ?? "",
          name: best.column,
          r: r(best.correlation),
        })}
      </span>
      <Button variant="ghost" icon={Plus} onClick={() => onAdd(newDriver(best))}>
        {t("ts.drivers.suggestAdd")}
      </Button>
    </p>
  );
}

/** The Drivers panel (collapsed by default): outside variables for the driver models. */
export function DriversPanel({
  candidates,
  drivers,
  onChange,
  open,
  onOpenChange,
  models,
  onAddModel,
  horizon,
}: {
  candidates: SeriesDrivers | undefined;
  drivers: DriverSpec[];
  onChange: (drivers: DriverSpec[]) => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  models: string[];
  onAddModel: (key: string) => void;
  horizon: number;
}) {
  const Chevron = open ? ChevronDown : ChevronRight;
  const usedBy = models.filter((key) => DRIVER_MODELS.includes(key));
  const addable = DRIVER_MODELS.find((key) => !LOCKED.has(key));
  const byColumn = new Map(
    candidates?.candidates.map((c) => [c.column, c]) ?? [],
  );
  const chosen = new Set(
    drivers
      .filter((d) => d.kind === "column" || d.kind === "flag")
      .map((d) => d.column),
  );
  const holidays = candidates?.holidays ?? null;
  const holidaysOn = drivers.some((d) => d.kind === "holidays");
  const blocked = holidays ? holidaysBlocked(holidays) : null;
  const free = candidates?.candidates.filter((c) => !chosen.has(c.column)) ?? [];
  const full = drivers.length >= MAX_DRIVERS;
  const update = (at: number, next: DriverSpec) =>
    onChange(drivers.map((d, i) => (i === at ? next : d)));
  const remove = (at: number) => onChange(drivers.filter((_, i) => i !== at));
  const toggleCalendar = (part: string) => {
    const on = drivers.some((d) => d.kind === "calendar" && d.column === part);
    if (on)
      onChange(
        drivers.filter((d) => !(d.kind === "calendar" && d.column === part)),
      );
    else if (!full) onChange([...drivers, spec(part, "calendar", true)]);
  };

  return (
    <div className="rounded-md border border-line">
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 px-3 py-1.5">
        <button
          type="button"
          aria-expanded={open}
          onClick={() => onOpenChange(!open)}
          className="inline-flex items-center gap-1.5 text-[13px] font-medium text-ink-1"
        >
          <Chevron size={14} strokeWidth={1.5} aria-hidden />
          {t("ts.drivers.title")}
          {drivers.length > 0 && (
            <span className="text-ink-3 tabular-nums">· {drivers.length}</span>
          )}
        </button>
        <span className="text-xs text-ink-3">{t("ts.drivers.usedByModels")}</span>
      </div>
      {open && (
        <div className="space-y-2 border-t border-line px-3 py-2 text-xs">
          <p className="text-ink-2">{t("ts.drivers.explain")}</p>
          {drivers.length > 0 && (
            <ul className="divide-y divide-line rounded-md border border-line bg-surface-1">
              {drivers.map((driver, at) => (
                <DriverRow
                  key={`${driver.kind}:${driver.kind === "holidays" ? "" : driver.column}`}
                  driver={driver}
                  candidate={byColumn.get(driver.column)}
                  holidays={holidays}
                  horizon={horizon}
                  onChange={(next) => update(at, next)}
                  onRemove={() => remove(at)}
                />
              ))}
            </ul>
          )}
          {candidates && candidates.calendar.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="mr-1 text-ink-3">{t("ts.drivers.calendar")}</span>
              {candidates.calendar.map((part) => {
                const on = drivers.some(
                  (d) => d.kind === "calendar" && d.column === part,
                );
                return (
                  <button
                    key={part}
                    type="button"
                    aria-pressed={on}
                    disabled={!on && full}
                    onClick={() => toggleCalendar(part)}
                    className={`inline-flex h-6 items-center gap-1 rounded-full border px-2.5 disabled:opacity-50 ${on ? "border-accent bg-accent-soft text-ink-1" : "border-line text-ink-2 hover:bg-surface-2 hover:text-ink-1"}`}
                  >
                    {!on && <Plus size={11} strokeWidth={2} aria-hidden />}
                    {t(`ts.drivers.cal.${part}` as MessageKey)}
                  </button>
                );
              })}
              <span className="text-ink-3">{t("ts.drivers.calendarHint")}</span>
            </div>
          )}
          {holidays && (
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="mr-1 text-ink-3">{t("ts.holidays.title")}</span>
              {!holidaysOn && (
                <button
                  type="button"
                  disabled={Boolean(blocked) || full}
                  onClick={() =>
                    onChange([
                      ...drivers,
                      spec(holidays.guess.code ?? "", "holidays", true),
                    ])
                  }
                  className="inline-flex h-6 items-center gap-1 rounded-full border border-line px-2.5 text-ink-2 hover:bg-surface-2 hover:text-ink-1 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <Plus size={11} strokeWidth={2} aria-hidden />
                  {t("ts.holidays.add")}
                </button>
              )}
              <span className={blocked ? "text-ink-3" : "text-ink-2"}>
                {blocked ?? guessText(holidays)}
              </span>
            </div>
          )}
          <div className="flex flex-wrap items-center gap-2">
            {free.length > 0 ? (
              <>
                <AddSelect
                  label={t("ts.drivers.addColumn")}
                  options={free}
                  disabled={full}
                  onPick={(c) => onChange([...drivers, newDriver(c)])}
                />
                {free.some((c) => c.flag) && (
                  <AddSelect
                    label={t("ts.drivers.addFlag")}
                    options={free.filter((c) => c.flag)}
                    disabled={full}
                    onPick={(c) => onChange([...drivers, newDriver(c, true)])}
                  />
                )}
              </>
            ) : (
              candidates && (
                <span className="text-ink-3">{t("ts.drivers.none")}</span>
              )
            )}
            {full && <span className="text-ink-3">{t("ts.drivers.max")}</span>}
          </div>
          {drivers.length > 0 && usedBy.length === 0 && (
            <p className="flex flex-wrap items-center gap-2 text-warn">
              {t("ts.drivers.noModel")}
              {addable && (
                <Button variant="ghost" onClick={() => onAddModel(addable)}>
                  {t("ts.drivers.addModel", { model: modelName(addable) })}
                </Button>
              )}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function AddSelect({
  label,
  options,
  disabled,
  onPick,
}: {
  label: string;
  options: Candidate[];
  disabled: boolean;
  onPick: (candidate: Candidate) => void;
}) {
  return (
    <Select
      aria-label={label}
      value=""
      disabled={disabled}
      onChange={(event) => {
        const found = options.find((c) => c.column === event.target.value);
        if (found) onPick(found);
      }}
      className="h-7 text-xs"
    >
      <option value="">{label}</option>
      {options.map((c) => (
        <option key={c.column} value={c.column}>
          {c.correlation === null
            ? c.column
            : t("ts.drivers.option", { name: c.column, r: r(c.correlation) })}
        </option>
      ))}
    </Select>
  );
}

function DriverRow({
  driver,
  candidate,
  holidays,
  horizon,
  onChange,
  onRemove,
}: {
  driver: DriverSpec;
  candidate: Candidate | undefined;
  holidays: HolidayOffer | null;
  horizon: number;
  onChange: (driver: DriverSpec) => void;
  onRemove: () => void;
}) {
  const kind = driver.kind;
  const name =
    kind === "calendar"
      ? t(`ts.drivers.cal.${driver.column}` as MessageKey)
      : kind === "holidays"
        ? t("ts.holidays.title")
        : driver.column;
  const future = driver.future;
  return (
    <li className="flex flex-wrap items-center gap-x-3 gap-y-1.5 px-2.5 py-1.5">
      <span
        className={`min-w-24 text-[13px] text-ink-1 ${kind === "calendar" || kind === "holidays" ? "" : "font-mono"}`}
      >
        {name}
      </span>
      {kind === "column" && (
        <SegmentedControl
          label={t("ts.drivers.when", { name })}
          value={driver.known ? "known" : "past"}
          options={[
            { value: "past", label: t("ts.drivers.pastOnly") },
            { value: "known", label: t("ts.drivers.known") },
          ]}
          onChange={(value) =>
            onChange({ ...driver, known: value === "known", future })
          }
        />
      )}
      {kind === "column" && !driver.known && (
        <span className="text-ink-3">
          {t("ts.drivers.lagged", { lag: String(horizon) })}
        </span>
      )}
      {kind === "column" && driver.known && (
        <>
          <label className="flex items-center gap-1.5 text-ink-2">
            {t("ts.drivers.future")}
            <Select
              value={future}
              onChange={(event) =>
                onChange({ ...driver, future: event.target.value as Future })
              }
              className="h-7 text-xs"
            >
              <option value="file">
                {t("ts.drivers.fromFile", {
                  count: String(candidate?.future_periods ?? 0),
                  horizon: String(horizon),
                })}
              </option>
              <option value="constant">{t("ts.drivers.constant")}</option>
              <option value="last">{t("ts.drivers.last")}</option>
            </Select>
          </label>
          {future === "constant" && (
            <TextInput
              type="number"
              aria-label={t("ts.drivers.constantLabel", { name })}
              value={driver.value ?? ""}
              onChange={(event) =>
                onChange({
                  ...driver,
                  value:
                    event.target.value === ""
                      ? null
                      : Number(event.target.value),
                })
              }
              className="h-7 w-24 text-xs"
            />
          )}
          {(future === "last" ||
            (future === "file" && !candidate?.future_periods)) && (
            <span className="text-warn">{t("ts.drivers.lastWarn")}</span>
          )}
        </>
      )}
      {kind === "flag" && (
        <span className="text-ink-3">{t("ts.drivers.flagHint")}</span>
      )}
      {kind === "calendar" && (
        <span className="text-ink-3">{t("ts.drivers.calendarHint")}</span>
      )}
      {kind === "holidays" && holidays && (
        <HolidaysChoice driver={driver} offer={holidays} onChange={onChange} />
      )}
      <button
        type="button"
        aria-label={t("ts.drivers.remove", { name })}
        onClick={onRemove}
        className="ml-auto rounded-full p-0.5 text-ink-3 hover:bg-surface-2 hover:text-ink-1"
      >
        <X size={12} strokeWidth={2} aria-hidden />
      </button>
    </li>
  );
}

/** Country and subdivision of the public-holiday driver (028 A5). */
function HolidaysChoice({
  driver,
  offer,
  onChange,
}: {
  driver: DriverSpec;
  offer: HolidayOffer;
  onChange: (driver: DriverSpec) => void;
}) {
  const country = offer.countries.find((c) => c.code === driver.column);
  const subdivisions = country?.subdivisions ?? [];
  return (
    <>
      <Select
        aria-label={t("ts.holidays.country")}
        value={driver.column}
        onChange={(event) =>
          onChange({ ...driver, column: event.target.value, subdiv: null })
        }
        className="h-7 max-w-56 text-xs"
      >
        {!driver.column && <option value="">{t("ts.holidays.choose")}</option>}
        {offer.countries.map((c) => (
          <option key={c.code} value={c.code}>
            {c.name} ({c.code})
          </option>
        ))}
      </Select>
      {subdivisions.length > 0 && (
        <Select
          aria-label={t("ts.holidays.subdiv")}
          value={driver.subdiv ?? ""}
          onChange={(event) =>
            onChange({ ...driver, subdiv: event.target.value || null })
          }
          className="h-7 max-w-48 text-xs"
        >
          <option value="">{t("ts.holidays.wholeCountry")}</option>
          {subdivisions.map((s) => (
            <option key={s.code} value={s.code}>
              {s.name ? `${s.code} · ${s.name}` : s.code}
            </option>
          ))}
        </Select>
      )}
      <span className="basis-full text-ink-3">
        {offer.weekly ? t("ts.holidays.hintWeek") : t("ts.holidays.hintDay")}{" "}
        {t("ts.holidays.separate")}
      </span>
    </>
  );
}

const VERDICT_TONE: Record<string, string> = {
  helps: "text-ok",
  unclear: "text-ink-2",
  no_help: "text-warn",
};

/** Driver by driver, for each model that used them: effect, and the MASE without it. */
export function DriverResults({ result }: { result: Result }) {
  const models = result.results.filter(
    (m) => m.uses_drivers && m.drivers.length > 0,
  );
  if (!models.length) return null;
  return (
    <div className="space-y-1.5">
      <h3 className="text-[13px] font-medium">{t("ts.drivers.results")}</h3>
      <div className="overflow-x-auto rounded-md border border-line">
        <table className="w-full border-collapse text-[13px] tabular-nums">
          <thead className="bg-surface-2 text-left text-xs text-ink-3">
            <tr>
              <th className="px-3 py-1.5 font-normal">{t("ts.drivers.driver")}</th>
              <th className="px-3 py-1.5 font-normal">{t("ts.forecast.model")}</th>
              <th className="px-3 py-1.5 text-right font-normal">
                {t("ts.drivers.effect")}
              </th>
              <th className="px-3 py-1.5 text-right font-normal">
                {t("ts.drivers.pValue")}
              </th>
              <th className="px-3 py-1.5 text-right font-normal">
                {t("ts.drivers.maseWithout")}
              </th>
              <th className="px-3 py-1.5 font-normal">{t("ts.drivers.verdict")}</th>
            </tr>
          </thead>
          <tbody>
            {models.flatMap((model) =>
              model.drivers.map((effect) => (
                <tr key={`${model.key}:${effect.driver}`} className="border-t border-line">
                  <td className="px-3 py-1.5 text-ink-1">{driverLabel(effect.driver)}</td>
                  <td className="px-3 py-1.5 text-ink-2">{modelName(model.key)}</td>
                  <td className="px-3 py-1.5 text-right">
                    {effect.coefficient !== null
                      ? t("ts.drivers.perUnit", {
                          value: `${effect.coefficient >= 0 ? "+" : "−"}${formatNumber(Math.abs(effect.coefficient))}`,
                        })
                      : effect.importance !== null
                        ? t("ts.drivers.importance", {
                            value: effect.importance.toFixed(0),
                          })
                        : "—"}
                  </td>
                  <td className="px-3 py-1.5 text-right">
                    {effect.p_value === null
                      ? "—"
                      : effect.p_value < 0.001
                        ? "< 0.001"
                        : effect.p_value.toFixed(3)}
                  </td>
                  <td className="px-3 py-1.5 text-right">
                    {formatNumber(effect.mase_without)}
                    <span className="text-ink-3">
                      {" "}
                      ({t("ts.drivers.withIt", {
                        mase: formatNumber(model.scores?.mase),
                      })})
                    </span>
                  </td>
                  <td
                    className={`px-3 py-1.5 ${effect.verdict ? VERDICT_TONE[effect.verdict] : "text-ink-3"}`}
                  >
                    {effect.verdict
                      ? t(`ts.drivers.v.${effect.verdict}` as MessageKey)
                      : "—"}
                  </td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-ink-3">{t("ts.drivers.resultsHint")}</p>
    </div>
  );
}
