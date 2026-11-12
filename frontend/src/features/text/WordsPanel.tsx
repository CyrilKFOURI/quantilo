import type { DistinctiveWords, GroupWords } from "../../api/client";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { InfoTip } from "../../ui/InfoTip";

const percent = (share: number) => `${Math.round(share * 100)}%`;

/** 059: the words that mark each group of texts (Compare, Segments, Changes). A conclusion
 *  above, then one card per group (3 per line on a laptop): each word with two bars, the
 *  share of the group's texts that use it and the share of the other texts. */
export function WordsPanel({ words, title }: { words: DistinctiveWords; title?: string }) {
  const widest = Math.max(
    0.01,
    ...words.groups.flatMap((g) => g.words.map((w) => Math.max(w.share, w.elsewhere))),
  );
  return (
    <section className="min-w-0 space-y-3" aria-label={title ?? t("words.title")}>
      <p className="text-sm font-medium text-ink-1">{words.sentence}</p>
      <p className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-3">
        {/* the tip first: at the left, its box never runs off the page */}
        <InfoTip label={t("words.rulesLabel")} text={words.rules.join(" · ")} />
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className="inline-block h-2 w-3 rounded-sm bg-accent" />
          {t("words.inGroup")}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className="inline-block h-2 w-3 rounded-sm bg-ink-3" />
          {t("words.elsewhere")}
        </span>
        {words.sampled && <span>{t("words.sampled", { count: formatCount(words.texts) })}</span>}
        {words.groups_left > 0 && <span>{t("words.groupsLeft", { count: String(words.groups_left) })}</span>}
      </p>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {words.groups.map((group) => (
          <GroupCard key={group.group} group={group} by={words.by} widest={widest} />
        ))}
      </div>
    </section>
  );
}

function GroupCard({ group, by, widest }: { group: GroupWords; by: string; widest: number }) {
  return (
    <article className="min-w-0 rounded-lg border border-line bg-surface-1 p-3">
      <h3 className="flex items-baseline justify-between gap-2 text-sm font-semibold">
        <span className="min-w-0 truncate" title={`${by} = ${group.group}`}>
          {group.group}
        </span>
        <span className="shrink-0 text-xs font-normal text-ink-3">
          {t("words.texts", { count: formatCount(group.texts) })}
        </span>
      </h3>
      {group.words.length === 0 ? (
        <p className="mt-2 text-xs text-ink-3">{group.note ?? t("words.none")}</p>
      ) : (
        <ul className="mt-2 space-y-1.5">
          {group.words.map((word) => (
            <li
              key={word.word}
              className="grid grid-cols-[minmax(0,7rem)_minmax(0,1fr)_auto] items-center gap-2 text-xs"
              title={t("words.wordTitle", {
                word: word.word,
                share: percent(word.share),
                elsewhere: percent(word.elsewhere),
                z: word.score.toFixed(1),
              })}
            >
              <span className="truncate font-mono text-ink-1">{word.word}</span>
              <span className="flex flex-col gap-0.5" aria-hidden>
                <span className="h-1.5 rounded-sm bg-accent" style={{ width: `${(word.share / widest) * 100}%` }} />
                <span
                  className="h-1.5 rounded-sm bg-ink-3"
                  style={{ width: `${Math.max(1, (word.elsewhere / widest) * 100)}%` }}
                />
              </span>
              <span className="text-right text-ink-2 tabular-nums">
                {percent(word.share)} <span className="text-ink-3">/ {percent(word.elsewhere)}</span>
              </span>
            </li>
          ))}
        </ul>
      )}
    </article>
  );
}
