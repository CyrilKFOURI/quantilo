import { TriangleAlert } from "lucide-react";
import { useMemo, useState } from "react";

import type { CategoricalProfile, DateProfile, IdentifierProfile, TextProfile } from "../../api/client";
import { formatNumber } from "../../charts/figures";
import { Plot } from "../../charts/Plot";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { Tabs } from "../../ui/Tabs";
import { calendarFigure, horizontalBars } from "./figures";
import { Metrics, type Metric } from "./Metrics";
import { pLabel } from "./normality";

function pct(part: number, whole: number): string | undefined {
  if (!whole || !part) return undefined;
  const value = (100 * part) / whole;
  return value > 0 && value < 0.1 ? "<0.1%" : `${value.toFixed(value < 10 ? 1 : 0)}%`;
}

export function categoricalMetrics(profile: CategoricalProfile): Metric[] {
  const present = profile.rows - profile.missing;
  return [
    {
      label: t("cat.top"),
      value: profile.top?.label ?? "—",
      note: profile.top ? pct(profile.top.count, present) : undefined,
    },
    { label: t("stat.distinct"), value: formatCount(profile.distinct) },
    { label: t("stat.missing"), value: formatCount(profile.missing), note: pct(profile.missing, profile.rows) },
    {
      label: t("cat.entropy"),
      value: profile.entropy === null ? "—" : `${formatNumber(profile.entropy)} bits`,
      explain: t("cat.explainEntropy"),
    },
    { label: t("cat.evenness"), value: formatNumber(profile.normalized_entropy), explain: t("cat.explainEvenness") },
    { label: t("cat.herfindahl"), value: formatNumber(profile.herfindahl), explain: t("cat.explainHerfindahl") },
    { label: t("cat.rare"), value: formatCount(profile.rare_count), explain: t("cat.rareHint") },
  ];
}

export function CategoricalPanel({ profile }: { profile: CategoricalProfile }) {
  const present = profile.rows - profile.missing;
  return (
    <div className="space-y-3">
      {profile.imbalanced && profile.top && (
        <p className="flex items-start gap-2 rounded-md border border-line bg-surface-1 px-3 py-2 text-ink-1">
          <TriangleAlert size={15} strokeWidth={1.5} className="mt-0.5 shrink-0 text-warn" aria-hidden />
          {t("cat.imbalance", { label: profile.top.label, share: pct(profile.top.count, present) ?? "" })}
        </p>
      )}
      <Metrics items={categoricalMetrics(profile)} />
      {profile.chi2_p !== null && (
        <p className="text-ink-2">
          <span className="text-ink-3">{t("cat.uniform")} · </span>
          {profile.chi2_p < 0.05
            ? t("cat.uniformNo", { p: pLabel(profile.chi2_p) })
            : t("cat.uniformYes", { p: pLabel(profile.chi2_p) })}
        </p>
      )}
      {profile.rare.length > 0 && (
        <p className="text-xs text-ink-3">
          {t("cat.rare")}: <span className="font-mono text-ink-2">{profile.rare.map((bar) => bar.label).join(" · ")}</span>
        </p>
      )}
    </div>
  );
}

export function dateMetrics(profile: DateProfile): Metric[] {
  return [
    { label: t("date.range"), value: profile.min && profile.max ? `${profile.min.slice(0, 10)} → ${profile.max.slice(0, 10)}` : "—" },
    { label: t("date.granularity"), value: profile.granularity },
    { label: t("date.step"), value: profile.step ?? "—" },
    { label: t("date.periods"), value: formatCount(profile.periods) },
    {
      label: t("date.gaps"),
      value: formatCount(profile.gap_count),
      tone: profile.gap_count ? "text-warn" : "text-ok",
    },
    { label: t("stat.missing"), value: formatCount(profile.missing), note: pct(profile.missing, profile.rows) },
    { label: t("date.time"), value: profile.has_time ? t("date.yes") : t("date.no") },
  ];
}

export function DatePanel({ profile }: { profile: DateProfile }) {
  const calendar = useMemo(() => calendarFigure(profile.calendar), [profile.calendar]);
  return (
    <div className="space-y-4">
      <Metrics items={dateMetrics(profile)} />
      {profile.gap_count === 0 ? (
        <p className="text-ink-2">{t("date.noGaps")}</p>
      ) : (
        <p className="text-ink-2">
          <span className="text-ink-3">{t("date.gapList")}: </span>
          <span className="font-mono text-xs">{profile.gaps.join(" · ")}</span>
        </p>
      )}
      {profile.calendar.length > 0 && (
        <figure className="rounded-md border border-line p-2">
          <figcaption className="mb-1 px-1 text-xs text-ink-3">{t("date.calendar")}</figcaption>
          <Plot figure={calendar} height={180} label={t("date.calendar")} />
        </figure>
      )}
    </div>
  );
}

type TextTab = "words" | "examples";
type ExpressionSize = "1" | "2" | "3";

function languageMetric(profile: TextProfile): Metric {
  const label = t("text.language");
  if (!profile.language) return { label, value: "—", explain: `${t("text.explainLanguage")} ${t("text.languageNone")}.` };
  const mixed = profile.language === "mixed";
  const note = mixed
    ? profile.languages.map((bar) => `${bar.label} ${bar.count}%`).join(", ")
    : t("text.ofTexts", { share: `${Math.round(100 * (profile.language_share ?? 0))}%` });
  return {
    label,
    value: mixed ? t("columns.mixedLanguages") : (profile.language_label ?? profile.language),
    note,
    explain: t("text.explainLanguage"),
  };
}

export function textMetrics(profile: TextProfile): Metric[] {
  return [
    { label: t("stat.distinct"), value: formatCount(profile.distinct) },
    { label: t("stat.missing"), value: formatCount(profile.missing), note: pct(profile.missing, profile.rows) },
    {
      label: t("text.lengths"),
      value: `${formatCount(profile.min_length)} – ${formatCount(profile.max_length)}`,
      note: profile.mean_length !== null ? `avg ${formatNumber(profile.mean_length)}` : undefined,
    },
    {
      label: t("text.wordsPerText"),
      value: formatNumber(profile.median_words),
      note: profile.p90_words !== null ? t("text.p90", { value: formatNumber(profile.p90_words) }) : undefined,
    },
    { label: t("text.distinctWords"), value: formatCount(profile.distinct_words) },
    languageMetric(profile),
  ];
}

/** Free text, short text or address (050): figures, then Words and Examples tabs. */
export function TextPanel({ profile }: { profile: TextProfile }) {
  const [tab, setTab] = useState<TextTab>("words");
  const tabs = [
    { value: "words" as const, label: t("text.tab.words") },
    { value: "examples" as const, label: t("text.tab.examples"), badge: profile.repeated_count ? formatCount(profile.repeated_count) : undefined },
  ];
  return (
    <div className="space-y-4">
      <Metrics items={textMetrics(profile)} />
      <Tabs label={t("text.tabs")} value={tab} items={tabs} onChange={setTab}>
        {tab === "words" ? <TextWords profile={profile} /> : <TextExamples profile={profile} />}
      </Tabs>
    </div>
  );
}

function TextWords({ profile }: { profile: TextProfile }) {
  const [size, setSize] = useState<ExpressionSize>("1");
  const bars = { "1": profile.words, "2": profile.bigrams, "3": profile.trigrams }[size];
  const figure = useMemo(() => horizontalBars(bars, (tokens) => tokens.text), [bars]);
  const top = bars[0];
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <SegmentedControl
          label={t("text.size")}
          value={size}
          onChange={setSize}
          options={[
            { value: "1", label: t("text.size.1") },
            { value: "2", label: t("text.size.2") },
            { value: "3", label: t("text.size.3") },
          ]}
        />
        {profile.sampled_texts !== null && (
          <p className="text-xs text-ink-3">{t("text.sampled", { count: formatCount(profile.sampled_texts) })}</p>
        )}
      </div>
      {top && <p className="text-ink-1">{t("text.topExpression", { label: top.label, count: formatCount(top.count) })}</p>}
      <figure className="rounded-md border border-line p-2">
        <figcaption className="mb-1 px-1 text-xs text-ink-3">{t("text.expressions")}</figcaption>
        {bars.length ? (
          <Plot figure={figure} height={Math.max(160, bars.length * 20)} label={t("text.expressions")} />
        ) : (
          <p className="px-1 py-4 text-ink-3">{t("text.noWords")}</p>
        )}
      </figure>
    </div>
  );
}

function TextExamples({ profile }: { profile: TextProfile }) {
  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2">
        <ExampleList title={t("text.shortest")} texts={profile.shortest} />
        <ExampleList title={t("text.longest")} texts={profile.longest} />
      </div>
      {profile.repeated_count === 0 ? (
        <p className="text-ink-2">{t("text.noRepeated")}</p>
      ) : (
        <>
          <p className="text-ink-2">
            {t("text.repeatedNote", {
              count: formatCount(profile.repeated_count),
              pct: pct(profile.repeated_rows, profile.rows - profile.missing) ?? "0%",
            })}
          </p>
          <CountTable title={t("text.repeated")} rows={profile.repeated} />
        </>
      )}
    </div>
  );
}

function ExampleList({ title, texts }: { title: string; texts: string[] }) {
  return (
    <section className="min-w-0">
      <h3 className="mb-1.5 text-xs font-medium text-ink-3">{title}</h3>
      <ul className="divide-y divide-line rounded-md border border-line">
        {texts.map((text) => (
          <li key={text} className="line-clamp-3 px-3 py-1.5 text-[13px] break-words text-ink-1" title={text}>
            {text}
          </li>
        ))}
      </ul>
    </section>
  );
}

export function IdentifierPanel({ profile }: { profile: IdentifierProfile }) {
  return (
    <div className="space-y-4">
      <Metrics
        items={[
          { label: t("id.unique"), value: `${profile.unique_pct.toFixed(profile.unique_pct === 100 ? 0 : 2)}%` },
          { label: t("stat.distinct"), value: formatCount(profile.distinct) },
          {
            label: t("id.repeated"),
            value: formatCount(profile.duplicated_values),
            tone: profile.duplicated_values ? "text-warn" : "text-ok",
          },
          { label: t("stat.missing"), value: formatCount(profile.missing), note: pct(profile.missing, profile.rows) },
          { label: t("id.length"), value: `${formatCount(profile.min_length)} – ${formatCount(profile.max_length)}` },
        ]}
      />
      {profile.duplicated_values === 0 ? (
        <p className="text-ok">{t("id.allUnique")}</p>
      ) : (
        <CountTable title={t("id.repeated")} rows={profile.duplicates} />
      )}
      <CountTable title={`${t("id.patterns")} · ${t("id.patternHint")}`} rows={profile.patterns} mono />
    </div>
  );
}

function CountTable({ title, rows, mono = false }: { title: string; rows: { label: string; count: number }[]; mono?: boolean }) {
  return (
    <section>
      <h3 className="mb-1.5 text-xs font-medium text-ink-3">{title}</h3>
      <div className="max-h-72 overflow-y-auto rounded-md border border-line">
        <table className="w-full border-collapse text-[13px]">
          <thead className="sticky top-0 bg-surface-2 text-left text-xs text-ink-3">
            <tr>
              <th scope="col" className="px-3 py-1.5 font-normal">{t("id.value")}</th>
              <th scope="col" className="px-3 py-1.5 text-right font-normal">{t("id.times")}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.label} className="border-t border-line">
                <td className={`max-w-0 truncate px-3 py-1 ${mono ? "font-mono text-xs" : ""}`} title={row.label}>
                  {row.label}
                </td>
                <td className="px-3 py-1 text-right tabular-nums">{formatCount(row.count)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
