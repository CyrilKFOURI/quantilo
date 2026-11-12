import { useState } from "react";

import type { Column, Topics } from "../../api/client";
import { useActiveFilters } from "../../api/filters";
import { useKeepTopics, useTopics } from "../../api/hooks";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { ErrorNote, Select } from "../../ui/fields";
import { InfoTip } from "../../ui/InfoTip";
import { Loading } from "../../ui/Loading";
import { Folded } from "./Folded";

const COUNTS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
const TOPIC_ROLES = new Set([null, "free_text"]); // R-TOP-01

/** 059 (R-TOP-01): topics are offered on free text only. */
export function offersTopics(column: Column): boolean {
  return column.type === "text" && TOPIC_ROLES.has(column.role ?? null);
}

/** Advanced › More › Topics of a free text column (R-TOP-01): slow (NMF), so nothing is asked
 *  before the user opens both layers and clicks "Find topics". */
export function TopicsSection({ datasetId, column }: { datasetId: string; column: Column }) {
  return (
    <Folded title={t("topics.advanced")} hint={t("topics.advancedHint")}>
      <Folded title={t("topics.more")} hint={t("topics.moreHint")} nested>
        <TopicsBody datasetId={datasetId} column={column.name} />
      </Folded>
    </Folded>
  );
}

function TopicsBody({ datasetId, column }: { datasetId: string; column: string }) {
  const filters = useActiveFilters(datasetId);
  const [k, setK] = useState<number | null>(null);
  const [asked, setAsked] = useState(false);
  const topics = useTopics(datasetId, column, k, filters, asked);
  const keep = useKeepTopics(datasetId, column);
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col gap-1 text-xs text-ink-3">
          {t("topics.count")}
          <Select
            value={k === null ? "" : String(k)}
            onChange={(event) => setK(event.target.value ? Number(event.target.value) : null)}
            className="w-32"
          >
            <option value="">{t("topics.countAuto")}</option>
            {COUNTS.map((count) => (
              <option key={count} value={count}>
                {count}
              </option>
            ))}
          </Select>
        </label>
        {!asked && (
          <Button variant="primary" onClick={() => setAsked(true)}>
            {t("topics.find")}
          </Button>
        )}
        <p className="max-w-prose text-xs text-ink-3">{t("topics.hint")}</p>
      </div>
      {asked &&
        (topics.isError ? (
          <ErrorNote>{topics.error.message}</ErrorNote>
        ) : !topics.data ? (
          <Loading label={t("topics.loading")} className="h-40 rounded-lg" />
        ) : (
          <TopicsResult
            topics={topics.data}
            column={column}
            busy={keep.isPending}
            onKeep={(on) => keep.mutate({ k, filters, keep: on })}
          />
        ))}
      {keep.isError && <ErrorNote>{keep.error.message}</ErrorNote>}
    </div>
  );
}

function TopicsResult({
  topics,
  column,
  busy,
  onKeep,
}: {
  topics: Topics;
  column: string;
  busy: boolean;
  onKeep: (on: boolean) => void;
}) {
  return (
    <section className="space-y-3" aria-label={t("topics.title")}>
      <p className="text-sm font-medium text-ink-1">{topics.sentence}</p>
      <p className="text-xs text-ink-2">
        {/* the tip first: at the left, its box never runs off the page */}
        <InfoTip label={t("topics.rulesLabel")} text={topics.reasons.join(" · ")} />{" "}
        {t("topics.agreement", {
          agreement: `${Math.round(topics.agreement * 100)}%`,
          none: `${Math.round(topics.no_topic * 100)}%`,
        })}
        {topics.sampled && ` ${t("topics.sampled", { count: formatCount(topics.texts) })}`}
      </p>
      <div className="flex flex-wrap items-center gap-2">
        {topics.kept ? (
          <>
            <span className="text-xs text-ok">{t("topics.kept", { column: `${column} › Topic` })}</span>
            <Button variant="ghost" disabled={busy} onClick={() => onKeep(false)}>
              {t("topics.forget")}
            </Button>
          </>
        ) : (
          <Button disabled={busy} onClick={() => onKeep(true)}>
            {t("topics.keep")}
          </Button>
        )}
        <span className="text-xs text-ink-3">{t("topics.keepHint")}</span>
      </div>
      <ol className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {topics.topics.map((topic) => (
          <li key={topic.number} className="min-w-0 rounded-lg border border-line bg-surface-1 p-3">
            <h3 className="flex items-baseline justify-between gap-2 text-sm font-semibold">
              <span className="min-w-0 truncate" title={topic.label}>
                {topic.label}
              </span>
              <span className="shrink-0 text-xs font-normal text-ink-3 tabular-nums">
                {`${Math.round(topic.share * 100)}%`}
              </span>
            </h3>
            <div className="mt-1 h-1.5 rounded-sm bg-surface-2" aria-hidden>
              <div className="h-1.5 rounded-sm bg-accent" style={{ width: `${Math.max(1, topic.share * 100)}%` }} />
            </div>
            <p className="mt-2 flex flex-wrap gap-1">
              {topic.words.map((word) => (
                <span key={word.word} className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[11px] text-ink-1">
                  {word.word}
                </span>
              ))}
            </p>
            {topic.examples.length > 0 && (
              <ul className="mt-2 space-y-1 text-xs text-ink-2">
                {topic.examples.map((example, index) => (
                  <li key={index} className="line-clamp-3">
                    “{example}”
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ol>
    </section>
  );
}
