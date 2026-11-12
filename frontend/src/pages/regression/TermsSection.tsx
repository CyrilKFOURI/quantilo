import { CheckCircle2, Info, X } from "lucide-react";
import { useMemo, useState } from "react";

import type { AnalysisType, Column, Regression, RegressionParams, RegressionTerms } from "../../api/client";
import { useRegressionPartial, useRegressionSection } from "../../api/hooks";
import { formatNumber } from "../../charts/figures";
import { Plot } from "../../charts/Plot";
import { formatP } from "../../features/variable/normality";
import { formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { Cell } from "../../modules/kpi/Customize";
import { Button } from "../../ui/Button";
import { ErrorNote, Field, Select } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { addedVariableFigure, componentFigure, effectCurveFigure } from "./figures";
import { Sentence } from "./Sentence";

/** The kinds of term (070), in the order the menu offers them. */
export const TERM_KINDS = ["spline", "square", "cube", "log", "interaction", "trend", "month", "weekday", "reference"] as const;
type Kind = (typeof TERM_KINDS)[number];
export type Coding = "reference" | "deviation";

const NUMBERS = new Set<AnalysisType>(["numeric_continuous", "numeric_discrete"]);
const CATEGORIES = new Set<AnalysisType>(["categorical_nominal", "categorical_ordinal", "boolean"]);
const DATES = new Set<AnalysisType>(["datetime"]);
const CURVE_HEIGHT = 260;
const th = "px-3 py-1.5 font-normal";
const td = "px-3 py-1.5 tabular-nums text-right";
const field = "h-8 w-full min-w-0 text-[13px]";

/** A term as the user reads it: "Curve of `age`", "`a` × `b`". */
export function termLabel(spec: string): string {
  const [kind, rest = ""] = spec.split(/:(.*)/s);
  const [first = "", second = ""] = rest.split("|");
  if (kind === "interaction") return t("regression.terms.label.interaction", { a: `\`${first}\``, b: `\`${second}\`` });
  if (kind === "reference") return t("regression.terms.label.reference", { column: `\`${first}\``, level: second });
  if (!TERM_KINDS.includes(kind as Kind)) return spec;
  return t(`regression.terms.label.${kind}` as MessageKey, { column: `\`${first}\`` });
}

type TermsProps = {
  datasetId: string;
  params: RegressionParams;
  result: Regression;
  usable: Column[];
  terms: string[];
  onTerms: (next: string[]) => void;
  center: boolean;
  onCenter: (next: boolean) => void;
  coding: Coding;
  onCoding: (next: Coding) => void;
};

/** Advanced › Terms (068, 070): the terms beyond the columns as they are (curves, logs,
 *  interactions, dates, references), centering and contrasts; their joint tests, effect
 *  curves and slopes; the references, columns left out and VIF; the partial plots on demand. */
export function TermsSection(props: TermsProps) {
  const { datasetId, params, result, terms, onTerms } = props;
  const query = useRegressionSection(datasetId, "terms", params, true);
  const data: RegressionTerms | undefined = query.data;
  return (
    <div className="space-y-4 text-[13px]">
      <TermBuilder {...props} />
      {terms.length > 0 && (
        <div>
          <h3 className="mb-1 text-xs font-medium text-ink-2">{t("regression.terms.inUse")}</h3>
          <ul className="flex flex-wrap gap-1.5">
            {terms.map((spec) => {
              const note = data?.notes.find((n) => n.key === spec);
              const refused = note?.status === "refused";
              return (
                <li key={spec} title={note?.reason} className={`inline-flex max-w-full items-center gap-1 rounded-md border px-2 py-0.5 text-xs ${refused ? "border-warn text-warn" : "border-line text-ink-1"}`}>
                  <span className="min-w-0 truncate">
                    <Sentence text={termLabel(spec)} />
                  </span>
                  <button type="button" aria-label={t("regression.terms.remove", { term: spec })} onClick={() => onTerms(terms.filter((s) => s !== spec))} className="shrink-0 text-ink-3 hover:text-ink-1 pointer-coarse:p-2">
                    <X size={12} strokeWidth={1.75} aria-hidden />
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      )}
      {!data ? (
        query.error ? (
          <ErrorNote>{query.error.message}</ErrorNote>
        ) : (
          <Loading label={t("regression.section.loading")} compact className="h-32 rounded-md" />
        )
      ) : (
        <TermsBody data={data} datasetId={datasetId} params={params} result={result} />
      )}
    </div>
  );
}

function TermBuilder({ result, usable, terms, onTerms, center, onCenter, coding, onCoding }: TermsProps) {
  const [kind, setKind] = useState<Kind>("spline");
  const [first, setFirst] = useState("");
  const [second, setSecond] = useState("");
  const others = usable.filter((c) => c.name !== result.target);
  const names = (types: Set<AnalysisType>) => others.filter((c) => types.has(c.type)).map((c) => c.name);
  const levels = useMemo(() => {
    const found = new Map<string, string[]>();
    for (const ref of result.references) found.set(ref.column, [ref.level]);
    for (const term of result.terms) if (term.level !== null && term.group === term.column) found.get(term.column)?.push(term.level);
    return found;
  }, [result]);
  const firstOptions =
    kind === "log"
      ? names(NUMBERS)
      : kind === "spline" || kind === "square" || kind === "cube"
        ? names(new Set([...NUMBERS, ...DATES]))
        : kind === "trend" || kind === "month" || kind === "weekday"
          ? names(DATES)
          : kind === "reference"
            ? [...levels.keys()]
            : names(new Set([...NUMBERS, ...CATEGORIES, ...DATES]));
  const firstNow = firstOptions.includes(first) ? first : "";
  const secondOptions = kind === "interaction" ? firstOptions.filter((n) => n !== firstNow) : kind === "reference" ? (levels.get(firstNow) ?? []) : [];
  const secondNow = secondOptions.includes(second) ? second : "";
  const spec =
    !firstNow
      ? null
      : kind === "interaction"
        ? secondNow && `interaction:${firstNow}|${secondNow}`
        : kind === "reference"
          ? secondNow && `reference:${firstNow}|${secondNow}`
          : `${kind}:${firstNow}`;
  const already = Boolean(spec && terms.includes(spec));
  const add = () => {
    if (!spec || already) return;
    // a reference replaces the previous one of the same column (R-REG-TERM-06)
    const kept = kind === "reference" ? terms.filter((s) => !s.startsWith(`reference:${firstNow}|`)) : terms;
    onTerms([...kept, spec]);
    setFirst("");
    setSecond("");
  };
  return (
    <div className="space-y-3">
      <div>
        <h3 className="text-xs font-medium text-ink-2">{t("regression.terms.add")}</h3>
        <p className="max-w-prose text-xs text-ink-3">{t("regression.terms.addHint")}</p>
      </div>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-[minmax(0,11rem)_minmax(0,14rem)_minmax(0,14rem)_auto] sm:items-end">
        <Field label={t("regression.terms.kind")}>
          <Select aria-label={t("regression.terms.kind")} value={kind} onChange={(event) => setKind(event.target.value as Kind)} className={field}>
            {TERM_KINDS.map((k) => (
              <option key={k} value={k}>
                {t(`regression.terms.kind.${k}` as MessageKey)}
              </option>
            ))}
          </Select>
        </Field>
        <Field label={t("regression.terms.column")}>
          <Select aria-label={t("regression.terms.column")} value={firstNow} onChange={(event) => setFirst(event.target.value)} className={field}>
            <option value="">{firstOptions.length ? t("regression.terms.pick") : t("regression.terms.noColumn")}</option>
            {firstOptions.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </Select>
        </Field>
        {kind === "interaction" || kind === "reference" ? (
          <Field label={kind === "interaction" ? t("regression.terms.with") : t("regression.terms.level")}>
            <Select aria-label={kind === "interaction" ? t("regression.terms.with") : t("regression.terms.level")} value={secondNow} onChange={(event) => setSecond(event.target.value)} className={field} disabled={!firstNow}>
              <option value="">{t("regression.terms.pick")}</option>
              {secondOptions.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </Select>
          </Field>
        ) : (
          <span className="hidden sm:block" />
        )}
        <Button variant="secondary" onClick={add} disabled={!spec || already}>
          {t("regression.terms.addAction")}
        </Button>
      </div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <Cell label={t("regression.terms.center")} why={t("regression.terms.centerWhy")} changed={center} onReset={() => onCenter(false)}>
          <Select aria-label={t("regression.terms.center")} value={center ? "on" : "off"} onChange={(event) => onCenter(event.target.value === "on")} className={field}>
            <option value="off">{t("regression.terms.centerOff")}</option>
            <option value="on">{t("regression.terms.centerOn")}</option>
          </Select>
        </Cell>
        <Cell label={t("regression.terms.coding")} why={t("regression.terms.codingWhy")} changed={coding === "deviation"} onReset={() => onCoding("reference")}>
          <Select aria-label={t("regression.terms.coding")} value={coding} onChange={(event) => onCoding(event.target.value as Coding)} className={field}>
            <option value="reference">{t("regression.terms.codingReference")}</option>
            <option value="deviation">{t("regression.terms.codingDeviation")}</option>
          </Select>
        </Cell>
      </div>
    </div>
  );
}

function TermsBody({ data, datasetId, params, result }: { data: RegressionTerms; datasetId: string; params: RegressionParams; result: Regression }) {
  const refused = data.notes.filter((n) => n.status !== "used" && !data.specs.includes(n.key));
  return (
    <>
      <p className="text-ink-2">
        {data.sentence} {data.condition !== null && t("regression.condition", { value: data.condition.toFixed(1) })}
      </p>
      {refused.length > 0 && (
        <ul className="space-y-0.5 text-xs text-ink-2">
          {refused.map((note) => (
            <li key={note.key}>
              <Sentence text={note.label} /> · <Sentence text={note.reason} /> <span className="text-ink-3">({note.rule})</span>
            </li>
          ))}
        </ul>
      )}
      {data.tests.length > 0 && <Tests data={data} />}
      {data.slopes.length > 0 && <Slopes data={data} />}
      {data.contrasts.length > 0 && <Contrasts data={data} />}
      {data.curves.length > 0 && <Curves data={data} />}
      <References data={data} />
      <Shape datasetId={datasetId} params={params} numbers={data.numbers} model={result.model} />
    </>
  );
}

function Tests({ data }: { data: RegressionTerms }) {
  return (
    <div>
      <h3 className="mb-1 text-xs font-medium text-ink-2">{t("regression.terms.tests")}</h3>
      <ul className="space-y-1">
        {data.tests.map((test) => {
          const clear = test.status === "ok";
          const Icon = clear ? CheckCircle2 : Info;
          return (
            <li key={`${test.key}-${test.question}`} className="grid grid-cols-[1rem_minmax(0,1fr)] gap-x-2 text-ink-2 sm:grid-cols-[1rem_minmax(0,18rem)_minmax(0,1fr)]">
              <Icon size={14} strokeWidth={1.75} aria-label={clear ? t("regression.terms.clear") : t("regression.terms.notClear")} className={`mt-0.5 ${clear ? "text-ok" : "text-ink-3"}`} />
              <span className="text-ink-1">
                <Sentence text={test.question} />
              </span>
              <span className="col-start-2 sm:col-start-auto">
                <Sentence text={test.sentence} />
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function Slopes({ data }: { data: RegressionTerms }) {
  return (
    <div>
      <h3 className="mb-1 text-xs font-medium text-ink-2">{t("regression.terms.slopes")}</h3>
      <div className="overflow-x-auto rounded-md border border-line">
        <table className="w-full border-collapse">
          <thead className="bg-surface-2 text-left text-xs text-ink-3">
            <tr>
              <th scope="col" className={th}>{t("regression.terms.slopeOf")}</th>
              <th scope="col" className={th}>{t("regression.terms.slopeAt")}</th>
              <th scope="col" className={`${th} text-right`}>{t("regression.terms.slope")}</th>
              <th scope="col" className={`${th} text-right`}>{t("regression.ci")}</th>
              <th scope="col" className={`${th} text-right`}>p</th>
            </tr>
          </thead>
          <tbody>
            {data.slopes.map((slope) => (
              <tr key={`${slope.term}-${slope.at}`} className="border-t border-line">
                <td className="px-3 py-1.5 font-mono text-xs">{slope.column}</td>
                <td className="px-3 py-1.5 text-ink-2">
                  <Sentence text={slope.at} />
                </td>
                <td className={`${td} text-ink-1`}>{formatNumber(slope.slope)}</td>
                <td className={`${td} whitespace-nowrap text-ink-2`}>
                  [{formatNumber(slope.low)} ; {formatNumber(slope.high)}]
                </td>
                <td className={`${td} text-ink-3`}>{formatP(slope.p_value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Contrasts({ data }: { data: RegressionTerms }) {
  return (
    <div>
      <h3 className="mb-1 text-xs font-medium text-ink-2">{t("regression.terms.levels")}</h3>
      <div className="overflow-x-auto rounded-md border border-line">
        <table className="w-full border-collapse">
          <thead className="bg-surface-2 text-left text-xs text-ink-3">
            <tr>
              <th scope="col" className={th}>{t("regression.term")}</th>
              <th scope="col" className={`${th} text-right`}>{t("regression.terms.againstAverage")}</th>
              <th scope="col" className={`${th} text-right`}>{t("regression.ci")}</th>
              <th scope="col" className={`${th} text-right`}>p</th>
            </tr>
          </thead>
          <tbody>
            {data.contrasts.map((level) => (
              <tr key={`${level.column}-${level.level}`} className="border-t border-line">
                <td className="px-3 py-1.5 font-mono text-xs">
                  {level.column} = {level.level}
                  {level.derived && <span className="ml-1 font-sans text-[11px] text-ink-3">({t("regression.terms.derived")})</span>}
                </td>
                <td className={`${td} ${level.estimate < 0 ? "text-bad" : "text-ink-1"}`}>{formatNumber(level.estimate)}</td>
                <td className={`${td} whitespace-nowrap text-ink-2`}>
                  [{formatNumber(level.low)} ; {formatNumber(level.high)}]
                </td>
                <td className={`${td} text-ink-3`}>{formatP(level.p_value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Curves({ data }: { data: RegressionTerms }) {
  const figures = useMemo(() => data.curves.map((curve) => [curve, effectCurveFigure(curve)] as const), [data.curves]);
  return (
    <div>
      <h3 className="mb-1 text-xs font-medium text-ink-2">{t("regression.terms.curves")}</h3>
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
        {figures.map(([curve, figure]) => (
          <figure key={curve.column} className="min-w-0 rounded-lg border border-line bg-surface-1 p-3">
            <figcaption className="mb-1 text-xs text-ink-2">
              <Sentence text={curve.sentence} />
            </figcaption>
            <Plot figure={figure} height={CURVE_HEIGHT} label={t("regression.terms.curveOf", { column: curve.column })} />
          </figure>
        ))}
      </div>
    </div>
  );
}

function References({ data }: { data: RegressionTerms }) {
  return (
    <>
      {data.references.length > 0 && (
        <div>
          <h3 className="mb-1 text-xs font-medium text-ink-2">{t("regression.references")}</h3>
          <ul className="space-y-0.5 text-ink-2">
            {data.references.map((ref) => (
              <li key={ref.column}>
                <code className="font-mono text-xs text-ink-1">{ref.column}</code> = {ref.level} · {ref.reason}
                {ref.others > 0 && ` · ${t("regression.others", { count: String(ref.others) })}`}
              </li>
            ))}
          </ul>
        </div>
      )}
      {data.left_out.length > 0 && (
        <div>
          <h3 className="mb-1 text-xs font-medium text-ink-2">{t("regression.leftOut")}</h3>
          <ul className="space-y-0.5 text-ink-2">
            {data.left_out.map((item) => (
              <li key={item.term}>
                <code className="font-mono text-xs text-ink-1">{item.term}</code> · <Sentence text={item.reason} />
              </li>
            ))}
          </ul>
        </div>
      )}
      {data.vifs.length > 0 && (
        <div className="overflow-x-auto rounded-md border border-line">
          <table className="w-full border-collapse">
            <thead className="bg-surface-2 text-left text-xs text-ink-3">
              <tr>
                <th scope="col" className={th}>{t("regression.term")}</th>
                <th scope="col" className={`${th} text-right`}>VIF</th>
                <th scope="col" className={th}>{t("regression.partner")}</th>
                <th scope="col" className={`${th} text-right`}>r</th>
              </tr>
            </thead>
            <tbody>
              {data.vifs.map((vif) => (
                <tr key={vif.term} className="border-t border-line">
                  <td className="px-3 py-1.5 font-mono text-xs">{vif.term}</td>
                  <td className={`${td} ${(vif.vif ?? 0) > 10 ? "text-bad" : "text-ink-1"}`}>{vif.vif === null ? "∞" : vif.vif.toFixed(1)}</td>
                  <td className="px-3 py-1.5 font-mono text-xs text-ink-2">{vif.partner ?? "—"}</td>
                  <td className={`${td} text-ink-2`}>{vif.correlation === null ? "—" : vif.correlation.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

const LEAST_SQUARES = new Set<string>(["ols", "wls", "log_ols", "boxcox"]);

/** R-REG-TERM-11: the partial plots of one column, asked for only once it is chosen. */
function Shape({ datasetId, params, numbers, model }: { datasetId: string; params: RegressionParams; numbers: string[]; model: string }) {
  const [column, setColumn] = useState("");
  const chosen = numbers.includes(column) ? column : "";
  const query = useRegressionPartial(datasetId, params, chosen || null);
  const plot = query.data;
  const figures = useMemo(
    () =>
      plot?.available
        ? ([
            [t("regression.terms.partialTitle"), addedVariableFigure(plot)],
            [t("regression.terms.componentTitle"), componentFigure(plot)],
          ] as const)
        : null,
    [plot],
  );
  if (!LEAST_SQUARES.has(model) || numbers.length === 0) return null;
  return (
    <div className="space-y-2 border-t border-line pt-3">
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-[minmax(0,16rem)_minmax(0,1fr)] sm:items-end">
        <Field label={t("regression.terms.shape")}>
          <Select aria-label={t("regression.terms.shape")} value={chosen} onChange={(event) => setColumn(event.target.value)} className={field}>
            <option value="">{t("regression.terms.pick")}</option>
            {numbers.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </Select>
        </Field>
        <p className="text-xs text-ink-3">{t("regression.terms.shapeHint")}</p>
      </div>
      {chosen &&
        (query.error ? (
          <ErrorNote>{query.error.message}</ErrorNote>
        ) : !plot ? (
          <Loading label={t("regression.section.loading")} compact className="h-32 rounded-md" />
        ) : !plot.available || !figures ? (
          <p className="text-ink-2">
            <Sentence text={plot.reason} />
          </p>
        ) : (
          <>
            <p className="text-ink-2">
              <Sentence text={plot.sentence} />
            </p>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              {figures.map(([title, figure]) => (
                <figure key={title} className="min-w-0 rounded-lg border border-line bg-surface-1 p-3">
                  <figcaption className="mb-1 text-[13px] font-medium text-ink-1">{title}</figcaption>
                  <Plot figure={figure} height={CURVE_HEIGHT} label={title} />
                </figure>
              ))}
            </div>
            {plot.points < plot.of && <p className="text-xs text-ink-3">{t("regression.terms.partialSample", { points: formatCount(plot.points), rows: formatCount(plot.of) })}</p>}
          </>
        ))}
    </div>
  );
}
