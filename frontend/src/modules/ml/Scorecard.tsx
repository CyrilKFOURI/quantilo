import { useState } from "react";

import type { InterpretAvailability, MlClassCards, MlGap, MlOptimalTree, MlRashomonSet, MlRun, MlScorecard } from "../../api/client";
import { useInterpretAvailable, useMlClassCards, useMlOptimal, useMlScorecard, useMlSparseTree } from "../../api/hooks";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { ErrorNote, Select } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { Cell } from "../kpi/Customize";
import { SurrogateTree } from "./ExplainAdvanced";
import { ScoreBand } from "./ResultParts";

const CONDITIONS = 5;
const POINTS = 5;
const DEPTH = 4;
const LEAVES = 12;
const pct = (value: number | null | undefined, digits = 0) =>
  value === null || value === undefined || !Number.isFinite(value) ? "—" : `${(value * 100).toFixed(digits)} %`;
const num = (value: number | null | undefined, digits = 3) =>
  value === null || value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);
const signed = (points: number) => (points > 0 ? `+${points}` : String(points));

/** R-XAI-05: a scorecard gives the risk of one event (two classes); 121 bis and 122 bis open the
 *  tab, as *Readable model*, to three classes or more (one card per class, R-XAI-17) and to a
 *  number (the sparse tree, R-XAI-19). */
export function hasScorecard(run: MlRun): boolean {
  if (run.task === "binary") return Boolean(run.classification?.positive);
  return run.task === "multiclass" || run.task === "regression";
}

function Block({ title, hint, children }: { title: string; hint: string; children: React.ReactNode }) {
  return (
    <section className="space-y-2">
      <h4 className="text-[13px] font-medium text-ink-1">{title}</h4>
      <p className="max-w-prose text-xs text-ink-2">{hint}</p>
      {children}
    </section>
  );
}

/** The gap to the run's model on the same held-out rows (R-XAI-07): the sentence, then the band. */
function GapBand({ gap, readable }: { gap: MlGap; readable: string }) {
  return (
    <div className="space-y-2">
      <p className="max-w-prose text-[15px] leading-snug font-medium text-ink-1">{gap.sentence}</p>
      <ScoreBand
        items={[
          { label: t("ml.scorecard.modelMetric", { metric: gap.metric }), value: num(gap.model), note: t("ml.scorecard.rows", { rows: gap.rows.toLocaleString("en-US") }) },
          { label: t("ml.scorecard.readableMetric", { readable, metric: gap.metric }), value: num(gap.readable) },
          {
            label: t("ml.scorecard.gap"),
            value: `${(gap.gap * 100).toFixed(1)} pts`,
            note: t("ml.scorecard.interval", { low: (gap.low * 100).toFixed(1), high: (gap.high * 100).toFixed(1) }),
            explain: t("ml.scorecard.gapHelp"),
          },
          ...(gap.model_ece !== null && gap.readable_ece !== null
            ? [{ label: t("ml.scorecard.ece"), value: `${num(gap.readable_ece)} · ${num(gap.model_ece)}`, note: t("ml.scorecard.readableModel", { readable }), explain: t("ml.scorecard.eceHelp") }]
            : []),
        ]}
      />
    </div>
  );
}

function CardTable({ card }: { card: MlScorecard }) {
  return (
    <table className="w-full border-collapse text-[13px] tabular-nums" aria-label={t("ml.scorecard.card")}>
      <thead className="text-left text-xs text-ink-3">
        <tr>
          <th className="py-1.5 pr-3 font-normal">{t("ml.scorecard.condition")}</th>
          <th className="px-3 py-1.5 text-right font-normal">{t("ml.scorecard.points")}</th>
          <th className="py-1.5 pl-3 text-right font-normal">{t("ml.scorecard.share")}</th>
        </tr>
      </thead>
      <tbody>
        {card.lines.map((line) => (
          <tr key={line.condition} className="border-t border-line">
            <td className="py-1.5 pr-3 font-mono text-xs text-ink-1">{line.condition}</td>
            <td className="px-3 py-1.5 text-right font-semibold text-ink-1">{signed(line.points)}</td>
            <td className="py-1.5 pl-3 text-right text-ink-2">{pct(line.share)}</td>
          </tr>
        ))}
      </tbody>
      <tfoot>
        <tr className="border-t border-line">
          <td colSpan={3} className="py-1.5 text-xs text-ink-2">
            {t("ml.scorecard.howTo")}
          </td>
        </tr>
      </tfoot>
    </table>
  );
}

function RiskTable({ card, positive }: { card: MlScorecard; positive: string }) {
  return (
    <div className="max-h-[320px] overflow-auto">
      <table className="w-full border-collapse text-[13px] tabular-nums" aria-label={t("ml.scorecard.risks")}>
        <thead className="sticky top-0 bg-surface-0 text-left text-xs text-ink-3">
          <tr>
            <th className="py-1.5 pr-3 font-normal">{t("ml.scorecard.total")}</th>
            <th className="px-3 py-1.5 text-right font-normal">{t("ml.scorecard.risk", { positive })}</th>
            <th className="px-3 py-1.5 text-right font-normal">{t("ml.scorecard.observed")}</th>
            <th className="py-1.5 pl-3 text-right font-normal">{t("ml.scorecard.rowsHeader")}</th>
          </tr>
        </thead>
        <tbody>
          {card.steps.map((step) => (
            <tr key={step.total} className={`border-t border-line ${step.rows ? "" : "text-ink-3"}`}>
              <td className="py-1.5 pr-3">{signed(step.total)}</td>
              <td className="px-3 py-1.5 text-right font-medium">{pct(step.risk, 1)}</td>
              <td className="px-3 py-1.5 text-right text-ink-2">{step.rows ? pct(step.observed, 1) : "—"}</td>
              <td className="py-1.5 pl-3 text-right text-ink-2">{step.rows.toLocaleString("en-US")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** The Scorecard tab (121, 122): computed when the tab opens; the tree under Advanced, the optional
 *  solvers under Advanced › More, each asked for only once open. Three classes or more: one card
 *  per class behind its button (121 bis); a number: the sparse tree in front (122 bis). */
export function ScorecardTab({ run, datasetId }: { run: MlRun; datasetId: string }) {
  if (run.task === "regression") return <NumberTab run={run} datasetId={datasetId} />;
  if (run.task === "multiclass") return <ClassesTab run={run} datasetId={datasetId} />;
  return <BinaryTab run={run} datasetId={datasetId} />;
}

function Settings({
  conditions,
  points,
  monotone,
  onConditions,
  onPoints,
  onMonotone,
}: {
  conditions: number;
  points: number;
  monotone: boolean;
  onConditions: (n: number) => void;
  onPoints: (n: number) => void;
  onMonotone: (on: boolean) => void;
}) {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-[repeat(3,minmax(0,1fr))]">
      <Cell label={t("ml.scorecard.conditions")} why={t("ml.scorecard.conditionsWhy")} changed={conditions !== CONDITIONS} onReset={() => onConditions(CONDITIONS)}>
        <Select aria-label={t("ml.scorecard.conditions")} value={conditions} onChange={(e) => onConditions(Number(e.target.value))} className="w-full text-xs">
          {[3, 4, 5, 6, 7, 8, 10].map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </Select>
      </Cell>
      <Cell label={t("ml.scorecard.pointsRange")} why={t("ml.scorecard.pointsWhy")} changed={points !== POINTS} onReset={() => onPoints(POINTS)}>
        <Select aria-label={t("ml.scorecard.pointsRange")} value={points} onChange={(e) => onPoints(Number(e.target.value))} className="w-full text-xs">
          {[3, 5, 10].map((n) => (
            <option key={n} value={n}>
              {`−${n} to +${n}`}
            </option>
          ))}
        </Select>
      </Cell>
      <Cell label={t("ml.scorecard.monotone")} why={t("ml.scorecard.monotoneWhy")} changed={monotone} onReset={() => onMonotone(false)}>
        <Select aria-label={t("ml.scorecard.monotone")} value={monotone ? "on" : "off"} onChange={(e) => onMonotone(e.target.value === "on")} className="w-full text-xs">
          <option value="off">{t("ml.scorecard.monotoneOff")}</option>
          <option value="on">{t("ml.scorecard.monotoneOn")}</option>
        </Select>
      </Cell>
    </div>
  );
}

function BinaryTab({ run, datasetId }: { run: MlRun; datasetId: string }) {
  const [conditions, setConditions] = useState(CONDITIONS);
  const [points, setPoints] = useState(POINTS);
  const [monotone, setMonotone] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const found = useMlScorecard(datasetId, run.id, conditions, points, true, monotone);
  const positive = run.classification?.positive ?? "";
  return (
    <div className="space-y-5">
      <p className="max-w-prose text-xs text-ink-2">{t("ml.scorecard.intro", { positive })}</p>
      <Settings conditions={conditions} points={points} monotone={monotone} onConditions={setConditions} onPoints={setPoints} onMonotone={setMonotone} />
      {found.isError ? (
        <ErrorNote>{found.error.message}</ErrorNote>
      ) : !found.data ? (
        <Loading label={t("ml.scorecard.computing")} className="h-48 rounded-md" />
      ) : (
        <div className={`space-y-5 ${found.isPlaceholderData ? "opacity-60" : ""}`}>
          <GapBand gap={found.data.gap} readable={t("ml.scorecard.theCard")} />
          <div className="grid gap-x-6 gap-y-4 @3xl:grid-cols-2">
            <figure className="min-w-0 border-t border-line pt-2">
              <figcaption className="mb-1 text-xs font-medium text-ink-2">{found.data.best.sentence}</figcaption>
              <CardTable card={found.data.best} />
            </figure>
            <figure className="min-w-0 border-t border-line pt-2">
              <figcaption className="mb-1 text-xs font-medium text-ink-2">{t("ml.scorecard.risksCaption", { positive })}</figcaption>
              <RiskTable card={found.data.best} positive={positive} />
            </figure>
          </div>
          {(found.data.shape ?? []).map((said) => (
            <p key={said} className="max-w-prose text-xs text-ink-2">
              {said}
            </p>
          ))}
          <ul className="space-y-0.5 text-xs text-ink-3">
            {found.data.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
            <li>{t("ml.scorecard.method", { seconds: found.data.seconds.toFixed(1) })}</li>
          </ul>
          {found.data.others.length > 0 && (
            <details className="text-xs">
              <summary className="cursor-pointer text-[13px] text-accent select-none hover:underline">
                {t("ml.scorecard.others", { count: String(found.data.others.length) })}
              </summary>
              <ul className="mt-3 grid gap-4 @3xl:grid-cols-2">
                {found.data.others.map((other) => (
                  <li key={other.lines.map((l) => l.condition).join("|")} className="min-w-0 border-t border-line pt-2">
                    <p className="mb-1 text-xs text-ink-2">{other.sentence}</p>
                    <CardTable card={other} />
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      )}
      <details className="border-t border-line pt-3" open={advanced} onToggle={(e) => setAdvanced(e.currentTarget.open)}>
        <summary className="cursor-pointer text-[13px] text-accent select-none hover:underline">{t("ml.scorecard.advanced")}</summary>
        {advanced && <Advanced run={run} datasetId={datasetId} />}
      </details>
    </div>
  );
}

/** 121 bis (R-XAI-17): three classes or more, one card per class, built on request (one search
 *  per class). The sparse tree stays under Advanced. */
function ClassesTab({ run, datasetId }: { run: MlRun; datasetId: string }) {
  const [conditions, setConditions] = useState(CONDITIONS);
  const [points, setPoints] = useState(POINTS);
  const [monotone, setMonotone] = useState(false);
  const [asked, setAsked] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const found = useMlClassCards(datasetId, run.id, conditions, points, monotone, asked);
  return (
    <div className="space-y-5">
      <p className="max-w-prose text-xs text-ink-2">{t("ml.scorecard.classesIntro")}</p>
      <Settings conditions={conditions} points={points} monotone={monotone} onConditions={setConditions} onPoints={setPoints} onMonotone={setMonotone} />
      {!asked ? (
        <Button onClick={() => setAsked(true)}>{t("ml.scorecard.classesRun")}</Button>
      ) : found.isError ? (
        <ErrorNote>{found.error.message}</ErrorNote>
      ) : !found.data ? (
        <Loading label={t("ml.scorecard.classesComputing")} className="h-48 rounded-md" />
      ) : (
        <ClassesResult found={found.data} stale={found.isPlaceholderData} />
      )}
      <details className="border-t border-line pt-3" open={advanced} onToggle={(e) => setAdvanced(e.currentTarget.open)}>
        <summary className="cursor-pointer text-[13px] text-accent select-none hover:underline">{t("ml.scorecard.advanced")}</summary>
        {advanced && <Advanced run={run} datasetId={datasetId} />}
      </details>
    </div>
  );
}

function ClassesResult({ found, stale }: { found: MlClassCards; stale: boolean }) {
  return (
    <div className={`space-y-4 ${stale ? "opacity-60" : ""}`}>
      <p className="max-w-prose text-[15px] leading-snug font-medium text-ink-1">{found.sentence}</p>
      <ScoreBand
        items={[
          { label: t("ml.scorecard.cardsAccuracy"), value: pct(found.accuracy), note: t("ml.scorecard.rows", { rows: found.checked.toLocaleString("en-US") }) },
          { label: t("ml.scorecard.modelAccuracy"), value: pct(found.model_accuracy) },
        ]}
      />
      <ul className="grid gap-4 @3xl:grid-cols-2">
        {found.classes.map((one) => (
          <li key={one.label} className="min-w-0 border-t border-line pt-2">
            <p className="mb-1 text-xs font-medium text-ink-1">
              {t("ml.scorecard.classCard", { label: one.label, auc: num(one.auc), model: num(one.model_auc) })}
            </p>
            <CardTable card={one.card} />
          </li>
        ))}
      </ul>
      <ul className="space-y-0.5 text-xs text-ink-3">
        {found.notes.map((note) => (
          <li key={note}>{note}</li>
        ))}
        <li>{t("ml.scorecard.method", { seconds: found.seconds.toFixed(1) })}</li>
      </ul>
    </div>
  );
}

/** 122 bis (R-XAI-19): a number has no card; its sparse tree is in front, judged by R², and the
 *  optimal trees are greyed with their reason (GOSDT and TreeFARMS learn classes only). */
function NumberTab({ run, datasetId }: { run: MlRun; datasetId: string }) {
  return (
    <div className="space-y-5">
      <p className="max-w-prose text-xs text-ink-2">{t("ml.scorecard.numberIntro")}</p>
      <TreeBlock run={run} datasetId={datasetId} />
      <details className="border-t border-line pt-3 text-xs">
        <summary className="cursor-pointer text-[13px] text-accent select-none hover:underline">{t("ml.scorecard.more")}</summary>
        <div className="mt-3 space-y-1">
          <button type="button" disabled className="cursor-not-allowed rounded-md border border-line px-3 py-1.5 text-xs text-ink-3 coarse:min-h-11" title={t("ml.scorecard.optimalNumber")}>
            {t("ml.scorecard.gosdtRun")}
          </button>
          <p className="max-w-prose text-xs text-ink-2">{t("ml.scorecard.optimalNumber")}</p>
        </div>
      </details>
    </div>
  );
}

/** Scorecard › Advanced (122): the sparse CART tree, then More. */
function Advanced({ run, datasetId }: { run: MlRun; datasetId: string }) {
  const [more, setMore] = useState(false);
  return (
    <div className="mt-3 space-y-6">
      <TreeBlock run={run} datasetId={datasetId} />
      <details className="text-xs" onToggle={(e) => setMore(e.currentTarget.open)}>
        <summary className="cursor-pointer text-[13px] text-accent select-none hover:underline">{t("ml.scorecard.more")}</summary>
        {more && <More run={run} datasetId={datasetId} />}
      </details>
    </div>
  );
}

function TreeBlock({ run, datasetId }: { run: MlRun; datasetId: string }) {
  const [depth, setDepth] = useState(DEPTH);
  const [leaves, setLeaves] = useState(LEAVES);
  const found = useMlSparseTree(datasetId, run.id, depth, leaves, true);
  return (
    <Block title={t("ml.scorecard.tree")} hint={t("ml.scorecard.treeHint")}>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-[repeat(3,minmax(0,1fr))]">
        <Cell label={t("ml.scorecard.depth")} why={t("ml.scorecard.depthWhy")} changed={depth !== DEPTH} onReset={() => setDepth(DEPTH)}>
          <Select aria-label={t("ml.scorecard.depth")} value={depth} onChange={(e) => setDepth(Number(e.target.value))} className="w-full text-xs">
            {[1, 2, 3, 4, 5, 6].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </Select>
        </Cell>
        <Cell label={t("ml.scorecard.leaves")} why={t("ml.scorecard.leavesWhy")} changed={leaves !== LEAVES} onReset={() => setLeaves(LEAVES)}>
          <Select aria-label={t("ml.scorecard.leaves")} value={leaves} onChange={(e) => setLeaves(Number(e.target.value))} className="w-full text-xs">
            {[4, 6, 8, 12, 16, 20].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </Select>
        </Cell>
      </div>
      {found.isError ? (
        <ErrorNote>{found.error.message}</ErrorNote>
      ) : !found.data ? (
        <Loading label={t("ml.scorecard.treeComputing")} compact className="h-32 rounded-md" />
      ) : (
        <div className={`space-y-3 ${found.isPlaceholderData ? "opacity-60" : ""}`}>
          {found.data.gap && <GapBand gap={found.data.gap} readable={t("ml.scorecard.theTree")} />}
          <p className="max-w-prose text-[13px] text-ink-1">{found.data.sentence}</p>
          <SurrogateTree nodes={found.data.nodes} label={t("ml.scorecard.tree")} />
        </div>
      )}
    </Block>
  );
}

/** Advanced › More (122): GOSDT and TreeFARMS, greyed with their reason when absent (R-XAI-10). */
function More({ run, datasetId }: { run: MlRun; datasetId: string }) {
  const available = useInterpretAvailable(true);
  if (available.isError) return <ErrorNote>{available.error.message}</ErrorNote>;
  if (!available.data) return <Loading label={t("ml.scorecard.checking")} compact className="mt-3 h-16 rounded-md" />;
  const by = new Map(available.data.map((a) => [a.key, a]));
  return (
    <div className="mt-3 space-y-5">
      <Optional run={run} datasetId={datasetId} kind="gosdt" availability={by.get("gosdt")} hint={t("ml.scorecard.gosdtHint")} />
      <Optional run={run} datasetId={datasetId} kind="treefarms" availability={by.get("treefarms")} hint={t("ml.scorecard.rashomonHint")} />
    </div>
  );
}

function Optional({
  run,
  datasetId,
  kind,
  availability,
  hint,
}: {
  run: MlRun;
  datasetId: string;
  kind: "gosdt" | "treefarms";
  availability: InterpretAvailability | undefined;
  hint: string;
}) {
  const [asked, setAsked] = useState(false);
  const usable = Boolean(availability?.available);
  const found = useMlOptimal(datasetId, run.id, kind, DEPTH, usable && asked);
  const label = availability?.label ?? kind;
  return (
    <Block title={label} hint={hint}>
      {!usable ? (
        <div className="space-y-1">
          <button type="button" disabled className="cursor-not-allowed rounded-md border border-line px-3 py-1.5 text-xs text-ink-3 coarse:min-h-11" title={availability?.reason ?? undefined}>
            {t(kind === "gosdt" ? "ml.scorecard.gosdtRun" : "ml.scorecard.rashomonRun")}
          </button>
          <p className="max-w-prose text-xs text-ink-2">{availability?.reason}</p>
          {availability?.install && (
            <p className="text-xs text-ink-3">
              {t("ml.scorecard.install")} <code className="font-mono text-ink-2">{availability.install}</code>
            </p>
          )}
        </div>
      ) : !asked ? (
        <button type="button" onClick={() => setAsked(true)} className="rounded-md border border-line px-3 py-1.5 text-xs text-ink-1 hover:bg-surface-2 coarse:min-h-11">
          {t(kind === "gosdt" ? "ml.scorecard.gosdtRun" : "ml.scorecard.rashomonRun")}
        </button>
      ) : found.isError ? (
        <ErrorNote>{found.error.message}</ErrorNote>
      ) : !found.data ? (
        <Loading label={t("ml.scorecard.solving")} compact className="h-24 rounded-md" />
      ) : kind === "gosdt" ? (
        <OptimalResult found={found.data as MlOptimalTree} />
      ) : (
        <RashomonResult found={found.data as MlRashomonSet} />
      )}
    </Block>
  );
}

function OptimalResult({ found }: { found: MlOptimalTree }) {
  return (
    <div className="space-y-3">
      {found.gap && <GapBand gap={found.gap} readable={t("ml.scorecard.theOptimal")} />}
      <p className="max-w-prose text-[13px] text-ink-1">{found.sentence}</p>
      {found.read_back && <p className="max-w-prose text-xs text-ink-3">{t("ml.scorecard.readBack")}</p>}
      <SurrogateTree nodes={found.nodes} label={t("ml.scorecard.theOptimal")} />
    </div>
  );
}

function RashomonResult({ found }: { found: MlRashomonSet }) {
  return (
    <div className="space-y-2">
      <p className="max-w-prose text-[13px] text-ink-1">{found.sentence}</p>
      <ul className="max-w-md space-y-0.5 text-[13px] tabular-nums">
        {found.columns.map((c) => (
          <li key={c.column} className="flex justify-between gap-3 border-t border-line py-1">
            <span className="text-ink-1">{c.column}</span>
            <span className="text-ink-2">{t("ml.scorecard.usedBy", { share: pct(c.share) })}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
