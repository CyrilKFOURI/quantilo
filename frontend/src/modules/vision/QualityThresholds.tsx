import { useQuery } from "@tanstack/react-query";
import { ChevronDown, RotateCcw, SlidersHorizontal } from "lucide-react";
import { useMemo, useState } from "react";

import { request } from "../../api/client";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { ErrorNote, TextInput } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { usePreviewQuality, type QualityRule } from "./api";

/** The integer thresholds (a count of pixels, bits, images or neighbours). */
const INTEGERS = new Set(["min_side", "near_phash", "near_dhash", "class_warn", "class_exclude", "knn_k", "min_relative"]);

type Props = {
  datasetId: string;
  rules: readonly QualityRule[];
  /** The draft's thresholds: only those that differ from the defaults. */
  draft: Readonly<Record<string, number>>;
  onUse: (thresholds: Record<string, number>) => void;
};

function useDefaults(enabled: boolean) {
  return useQuery({
    queryKey: ["image-quality-thresholds"],
    queryFn: () => request<Record<string, number>>("GET", "/images/thresholds"),
    enabled,
    staleTime: Infinity,
  });
}

/**
 * c3, R-VQT-01 and R-VQT-02: every threshold of the R-VQ rules, by rule, folded (Advanced). Preview
 * runs the rules again on the table, nothing written; "Use these thresholds" puts them in the draft,
 * saved as a version with the decisions. Nothing is asked before the section is opened.
 */
export function QualityThresholds(props: Props) {
  const [open, setOpen] = useState(false);
  const changed = Object.keys(props.draft).length;
  return (
    <details className="group rounded-md border border-line bg-surface-1" onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary className="flex min-h-9 cursor-pointer list-none items-center gap-1.5 px-3 text-[13px] text-ink-1 pointer-coarse:min-h-11">
        <ChevronDown size={15} strokeWidth={1.5} aria-hidden className="text-ink-3 transition-transform duration-150 group-not-open:-rotate-90" />
        <SlidersHorizontal size={14} strokeWidth={1.5} aria-hidden className="text-ink-3" />
        {t("vision.thresholds.title")}
        <span className="text-xs text-ink-3">{changed > 0 ? t("vision.thresholds.changed", { count: formatCount(changed) }) : t("vision.thresholds.defaults")}</span>
      </summary>
      {open && <Body {...props} />}
    </details>
  );
}

function Body({ datasetId, rules, draft, onUse }: Props) {
  const defaults = useDefaults(true);
  if (defaults.isError) return <ErrorNote>{(defaults.error as Error).message}</ErrorNote>;
  if (!defaults.data) return <Loading label={t("vision.thresholds.loading")} compact className="m-3 h-16 rounded-md" />;
  return <Editor datasetId={datasetId} rules={rules} draft={draft} onUse={onUse} defaults={defaults.data} />;
}

function Editor({ datasetId, rules, draft, onUse, defaults }: Props & { defaults: Record<string, number> }) {
  const [values, setValues] = useState<Record<string, string>>(() =>
    Object.fromEntries(Object.entries(defaults).map(([name, value]) => [name, String(draft[name] ?? value)])),
  );
  const preview = usePreviewQuality(datasetId);
  const groups = useMemo(() => {
    const owned = new Set<string>();
    const byRule = rules
      .filter((rule) => Object.keys(rule.parameters).length > 0)
      .map((rule) => {
        const names = Object.keys(rule.parameters).filter((name) => !owned.has(name));
        names.forEach((name) => owned.add(name));
        return { code: rule.code, title: rule.title, names };
      })
      .filter((group) => group.names.length > 0);
    const rest = Object.keys(defaults).filter((name) => !owned.has(name));
    return rest.length > 0 ? [...byRule, { code: "", title: t("vision.thresholds.setWide"), names: rest }] : byRule;
  }, [rules, defaults]);

  const invalid = Object.entries(values).filter(([, text]) => text.trim() === "" || !Number.isFinite(Number(text)) || Number(text) < 0);
  // only what differs from the defaults goes to the draft
  const overrides: Record<string, number> = {};
  for (const [name, text] of Object.entries(values)) {
    const value = INTEGERS.has(name) ? Math.round(Number(text)) : Number(text);
    if (Number.isFinite(value) && value !== defaults[name]) overrides[name] = value;
  }
  const same = JSON.stringify(Object.entries(overrides).sort()) === JSON.stringify(Object.entries(draft).sort());
  const result = preview.data;

  return (
    <div className="space-y-4 border-t border-line px-3 py-3">
      <p className="max-w-prose text-[13px] text-ink-2">{t("vision.thresholds.help")}</p>
      <div className="grid grid-cols-1 gap-x-6 gap-y-4 sm:grid-cols-2 lg:grid-cols-3">
        {groups.map((group) => (
          <fieldset key={group.code || "set"} className="min-w-0 space-y-2">
            <legend className="mb-1 flex items-baseline gap-1.5 text-xs text-ink-1">
              {group.code && <span className="font-mono text-ink-3">{group.code}</span>}
              {group.title}
            </legend>
            {group.names.map((name) => (
              <label key={name} className="grid grid-cols-[minmax(0,1fr)_5.5rem] items-center gap-2 text-xs">
                <span className="min-w-0 truncate font-mono text-ink-2" title={name}>
                  {name}
                </span>
                <TextInput
                  type="number"
                  inputMode="decimal"
                  min={0}
                  step={INTEGERS.has(name) ? 1 : "any"}
                  value={values[name] ?? ""}
                  aria-invalid={invalid.some(([bad]) => bad === name)}
                  onChange={(event) => setValues({ ...values, [name]: event.target.value })}
                  className="w-full text-right tabular-nums"
                  title={t("vision.thresholds.default", { value: String(defaults[name]) })}
                />
              </label>
            ))}
          </fieldset>
        ))}
      </div>
      {invalid.length > 0 && <p className="text-xs text-bad">{t("vision.thresholds.invalid")}</p>}
      <div className="flex flex-wrap gap-2">
        <Button disabled={invalid.length > 0 || preview.isPending} onClick={() => preview.mutate(overrides)}>
          {preview.isPending ? t("vision.thresholds.previewing") : t("vision.thresholds.preview")}
        </Button>
        <Button variant="primary" disabled={invalid.length > 0 || same} onClick={() => onUse(overrides)}>
          {t("vision.thresholds.use")}
        </Button>
        <Button variant="ghost" icon={RotateCcw} onClick={() => setValues(Object.fromEntries(Object.entries(defaults).map(([name, value]) => [name, String(value)])))}>
          {t("vision.thresholds.reset")}
        </Button>
      </div>
      {preview.isError && <ErrorNote>{(preview.error as Error).message}</ErrorNote>}
      {result && (
        <section aria-live="polite" className="space-y-2">
          <p className="text-[13px] font-medium text-ink-1">
            {t("vision.thresholds.with")} {result.quality.sentence}{" "}
            <span className="font-normal text-ink-2">{t("vision.thresholds.now", { flagged: formatCount(result.current_flagged) })}</span>
          </p>
          <table className="w-full max-w-lg text-xs tabular-nums">
            <caption className="sr-only">{t("vision.thresholds.compare")}</caption>
            <thead>
              <tr className="border-b border-line text-ink-3">
                <th scope="col" className="py-1 text-left font-normal">
                  {t("vision.thresholds.rule")}
                </th>
                <th scope="col" className="py-1 text-right font-normal">
                  {t("vision.thresholds.nowColumn")}
                </th>
                <th scope="col" className="py-1 text-right font-normal">
                  {t("vision.thresholds.withColumn")}
                </th>
              </tr>
            </thead>
            <tbody>
              {[...new Set([...Object.keys(result.current_counts), ...Object.keys(result.counts)])].sort().map((code) => {
                const before = result.current_counts[code] ?? 0;
                const after = result.counts[code] ?? 0;
                return (
                  <tr key={code} className="border-b border-line">
                    <th scope="row" className="py-1 text-left font-normal text-ink-1">
                      <span className="font-mono text-ink-3">{code}</span> {rules.find((rule) => rule.code === code)?.title ?? ""}
                    </th>
                    <td className="py-1 text-right text-ink-2">{formatCount(before)}</td>
                    <td className={`py-1 text-right ${after !== before ? "font-medium text-ink-1" : "text-ink-2"}`}>{formatCount(after)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {result.quality.notes.length > 0 && (
            <ul className="list-disc space-y-0.5 pl-5 text-xs text-ink-2">
              {result.quality.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          )}
        </section>
      )}
    </div>
  );
}
