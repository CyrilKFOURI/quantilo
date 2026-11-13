import { Pin as PinIcon, X } from "lucide-react";
import { useId, useState, type ReactNode } from "react";

import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { ErrorNote, Select, TextInput } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { usePinActions, usePinComparison, usePins, type PinLabel } from "./api";
import { Comparison } from "./Comparison";
import { COLORS, colorVar, nextColor, parseRows } from "./rows";

const SHOWN = 40; // pinned rows listed before "+N more"

/** One colour of the palette, as a swatch (the label is always written beside it). */
export function Swatch({ color, className = "size-3" }: { color: number; className?: string }) {
  return <span aria-hidden className={`inline-block shrink-0 rounded-sm ${className}`} style={{ background: colorVar(color) }} />;
}

/** The eight colours of the palette as a radio group (R-PIN-03). */
function Colors({ value, onChange, label }: { value: number; onChange: (color: number) => void; label: string }) {
  return (
    <span role="radiogroup" aria-label={label} className="inline-flex flex-wrap items-center gap-1">
      {COLORS.map((color) => (
        <button
          key={color}
          type="button"
          role="radio"
          aria-checked={value === color}
          aria-label={t("pins.color", { n: String(color) })}
          onClick={() => onChange(color)}
          className={`inline-flex size-6 items-center justify-center rounded-md border ${value === color ? "border-ink-1" : "border-transparent hover:border-line"}`}
        >
          <Swatch color={color} className="size-4" />
        </button>
      ))}
    </span>
  );
}

/** Pin rows with a label and a colour, manage the labels, and ask what distinguishes the pinned
 *  rows (099). Shown under Advanced on Data, Unusual rows and Clean; `selected` are the rows
 *  ticked in the page's table (from 0), and row numbers can be typed as shown (from 1). */
export function PinPanel({
  datasetId,
  selected = [],
  onClearSelection,
  hint,
  footer,
}: {
  datasetId: string;
  selected?: number[];
  onClearSelection?: () => void;
  /** How rows are picked on this page ("Click row numbers in the table"). */
  hint?: string;
  footer?: ReactNode;
}) {
  const pins = usePins(datasetId);
  const actions = usePinActions(datasetId);
  const [typed, setTyped] = useState("");
  const [label, setLabel] = useState("");
  const [chosen, setChosen] = useState<number | null>(null);
  const [asked, setAsked] = useState<string | null | undefined>(undefined);
  const [compareOn, setCompareOn] = useState("");
  const comparison = usePinComparison(datasetId, asked);
  const listId = useId();

  const labels = pins.data?.labels ?? [];
  const existing = labels.find((item) => item.label === label.trim());
  // Derived, never set in an effect: an existing label brings its colour, a new one the next free.
  const color = chosen ?? existing?.color ?? nextColor(labels);
  const parsed = parseRows(typed);
  const rows = [...new Set([...selected, ...(parsed ?? [])])].sort((a, b) => a - b);
  const pinned = pins.data?.pins ?? [];
  const found = pinned.filter((pin) => pin.found);
  const lost = pinned.filter((pin) => !pin.found);

  const submit = () =>
    actions.pin.mutate(
      { rows, label: label.trim(), color },
      {
        onSuccess: () => {
          setTyped("");
          setChosen(null);
          onClearSelection?.();
        },
      },
    );
  const error = actions.pin.error ?? actions.unpin.error ?? actions.changeLabel.error;

  return (
    <div className="space-y-4">
      <p className="max-w-prose text-[13px] text-ink-2">{t("pins.intro")}</p>
      <form
        aria-label={t("pins.form")}
        className="params-grid items-end"
        onSubmit={(event) => {
          event.preventDefault();
          if (rows.length && label.trim()) submit();
        }}
      >
        <label className="flex min-w-0 flex-col gap-1">
          <span className="text-xs text-ink-3">{t("pins.rows")}</span>
          <TextInput aria-label={t("pins.rows")} value={typed} onChange={(event) => setTyped(event.target.value)} placeholder={t("pins.rowsPlaceholder")} aria-invalid={parsed === null} className="h-8 w-full text-[13px]" />
          <span className="text-xs text-ink-3" aria-live="polite">
            {parsed === null ? t("pins.rowsInvalid") : selected.length > 0 ? t("pins.selected", { count: formatCount(selected.length) }) : (hint ?? t("pins.rowsHint"))}
          </span>
        </label>
        <label className="flex min-w-0 flex-col gap-1">
          <span className="text-xs text-ink-3">{t("pins.label")}</span>
          <TextInput aria-label={t("pins.label")} value={label} onChange={(event) => setLabel(event.target.value)} maxLength={60} list={listId} placeholder={t("pins.labelPlaceholder")} className="h-8 w-full text-[13px]" />
          <datalist id={listId}>
            {labels.map((item) => (
              <option key={item.label} value={item.label} />
            ))}
          </datalist>
          <span className="text-xs text-ink-3">{existing ? t("pins.labelExisting", { rows: formatCount(existing.rows) }) : t("pins.labelNew")}</span>
        </label>
        <div className="flex min-w-0 flex-col gap-1">
          <span className="text-xs text-ink-3">{t("pins.colorLabel")}</span>
          <Colors value={color} onChange={setChosen} label={t("pins.colorLabel")} />
          <span>
            <Button variant="primary" type="submit" icon={PinIcon} disabled={!rows.length || !label.trim() || actions.pin.isPending} className="mt-1">
              {rows.length === 0 ? t("pins.pin") : rows.length === 1 ? t("pins.pinOne") : t("pins.pinMany", { count: formatCount(rows.length) })}
            </Button>
          </span>
        </div>
      </form>
      {error && <ErrorNote>{error.message}</ErrorNote>}

      {pins.isError ? (
        <ErrorNote>{pins.error.message}</ErrorNote>
      ) : !pins.data ? (
        <Loading label={t("pins.loading")} className="h-16 rounded-md" compact />
      ) : pinned.length === 0 ? (
        <p className="text-xs text-ink-3">{t("pins.none")}</p>
      ) : (
        <div className="space-y-3">
          <ul aria-label={t("pins.labels")} className="divide-y divide-line rounded-md border border-line">
            {labels.map((item) => (
              <LabelRow
                key={item.label}
                item={item}
                rows={found.filter((pin) => pin.label === item.label).map((pin) => ({ key: pin.key, row: pin.row }))}
                onColor={(next) => actions.changeLabel.mutate({ label: item.label, color: next })}
                onRename={(to) => to !== item.label && actions.changeLabel.mutate({ label: item.label, to })}
                onUnpin={(keys) => actions.unpin.mutate(keys)}
                allKeys={pinned.filter((pin) => pin.label === item.label).map((pin) => pin.key)}
              />
            ))}
          </ul>
          {lost.length > 0 && (
            <p className="flex flex-wrap items-center gap-2 text-xs text-warn" role="status">
              {lost.length === 1 ? t("pins.lostOne") : t("pins.lostMany", { count: formatCount(lost.length) })}
              <Button variant="ghost" onClick={() => actions.unpin.mutate(lost.map((pin) => pin.key))}>
                {t("pins.unpinLost")}
              </Button>
            </p>
          )}
          <section aria-label={t("pins.compare.title")} className="space-y-3 border-t border-line pt-3">
            <div className="flex flex-wrap items-center gap-2">
              <Select aria-label={t("pins.compare.which")} value={compareOn} onChange={(event) => setCompareOn(event.target.value)} className="h-8 max-w-60 text-[13px]">
                <option value="">{t("pins.compare.all")}</option>
                {labels.map((item) => (
                  <option key={item.label} value={item.label}>
                    {item.label}
                  </option>
                ))}
              </Select>
              <Button onClick={() => setAsked(compareOn || null)} disabled={comparison.isFetching}>
                {t("pins.compare.title")}
              </Button>
            </div>
            {comparison.isError ? (
              <ErrorNote>{comparison.error.message}</ErrorNote>
            ) : comparison.isFetching ? (
              <Loading label={t("pins.compare.loading")} className="h-24 rounded-md" compact />
            ) : (
              comparison.data && <Comparison found={comparison.data} />
            )}
          </section>
        </div>
      )}
      {pins.data?.parent && <p className="max-w-prose text-xs text-ink-2">{pins.data.parent}</p>}
      {footer}
    </div>
  );
}

function LabelRow({
  item,
  rows,
  allKeys,
  onColor,
  onRename,
  onUnpin,
}: {
  item: PinLabel;
  rows: { key: string; row: number }[];
  allKeys: string[];
  onColor: (color: number) => void;
  onRename: (to: string) => void;
  onUnpin: (keys: string[]) => void;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const commit = () => {
    const value = (draft ?? "").trim();
    setDraft(null);
    if (value) onRename(value);
  };
  return (
    <li className="space-y-1.5 px-3 py-2">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <Swatch color={item.color} />
        <TextInput
          aria-label={t("pins.rename", { label: item.label })}
          value={draft ?? item.label}
          maxLength={60}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={commit}
          onKeyDown={(event) => event.key === "Enter" && commit()}
          className="h-7 w-44 text-[13px] font-medium"
        />
        <span className="text-xs text-ink-3 tabular-nums">{item.rows === 1 ? t("pins.oneRow") : t("pins.rowsCount", { count: formatCount(item.rows) })}</span>
        <Colors value={item.color} onChange={onColor} label={t("pins.colorOf", { label: item.label })} />
        <Button variant="ghost" className="ml-auto" onClick={() => onUnpin(allKeys)}>
          {t("pins.unpinLabel")}
        </Button>
      </div>
      <ul aria-label={t("pins.rowsOf", { label: item.label })} className="flex flex-wrap gap-1">
        {rows.slice(0, SHOWN).map((pin) => (
          <li key={pin.key}>
            <button
              type="button"
              onClick={() => onUnpin([pin.key])}
              aria-label={t("pins.unpinRow", { row: formatCount(pin.row + 1) })}
              title={t("pins.unpinRow", { row: formatCount(pin.row + 1) })}
              className="inline-flex items-center gap-1 rounded border border-line px-1.5 py-0.5 text-xs text-ink-2 tabular-nums hover:bg-surface-2 hover:text-ink-1"
            >
              {formatCount(pin.row + 1)}
              <X size={11} aria-hidden />
            </button>
          </li>
        ))}
        {rows.length > SHOWN && <li className="px-1 text-xs text-ink-3">{t("pins.more", { count: formatCount(rows.length - SHOWN) })}</li>}
      </ul>
    </li>
  );
}
