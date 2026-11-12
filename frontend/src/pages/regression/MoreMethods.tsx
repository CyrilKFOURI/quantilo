import { ChevronDown, X } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";

import type {
  RegressionComponents,
  RegressionCox,
  RegressionCrossSection,
  RegressionCurves,
  RegressionMeasurement,
  RegressionMethodItem,
  RegressionMethodKind,
  RegressionMethodOptions,
  RegressionMethods,
  RegressionMixed,
  RegressionOutputs,
  RegressionParams,
  RegressionTobit,
} from "../../api/client";
import { useRegressionMethod, useRegressionMethods } from "../../api/hooks";
import { formatNumber } from "../../charts/figures";
import { Plot } from "../../charts/Plot";
import { formatP } from "../../features/variable/normality";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { ErrorNote, Field, Select } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { curvesFigure } from "./figures";
import { Sentence } from "./Sentence";

const th = "px-3 py-1.5 font-normal";
const td = "px-3 py-1.5 tabular-nums text-right";
const field = "h-8 w-full min-w-0 text-[13px]";
const RULE = "__rule";
const NONE = "__none";
const CURVE_HEIGHT = 280;
const SHOWN_CURVES = 3;

const interval = (low: number | null | undefined, high: number | null | undefined) => `[${formatNumber(low ?? null)} ; ${formatNumber(high ?? null)}]`;

function Pending({ error }: { error: Error | null }) {
  return error ? <ErrorNote>{error.message}</ErrorNote> : <Loading label={t("regression.section.loading")} compact className="h-32 rounded-md" />;
}

function Notes({ notes }: { notes: string[] }) {
  return (
    <>
      {notes.map((note) => (
        <p key={note} className="text-xs text-ink-3">
          <Sentence text={note} />
        </p>
      ))}
    </>
  );
}

function Table({ head, children }: { head: ReactNode; children: ReactNode }) {
  return (
    <div className="overflow-x-auto rounded-md border border-line">
      <table className="w-full border-collapse text-[13px]">
        <thead className="bg-surface-2 text-left text-xs text-ink-3">
          <tr>{head}</tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

/** A column picked by a rule, another one, or none: the native list says which is in use. */
function PickSelect({
  label,
  value,
  onChange,
  rule,
  choices,
  allowNone = true,
}: {
  label: string;
  value: string | undefined;
  onChange: (next: string | undefined) => void;
  rule: string | null;
  choices: string[];
  allowNone?: boolean;
}) {
  const current = value === undefined ? RULE : value === "" ? NONE : value;
  return (
    <Field label={label}>
      <Select
        aria-label={label}
        value={current}
        onChange={(event) => {
          const next = event.target.value;
          onChange(next === RULE ? undefined : next === NONE ? "" : next);
        }}
        className={field}
      >
        <option value={RULE}>{rule ? t("regression.methods.byRule", { column: rule }) : t("regression.methods.noRule")}</option>
        {allowNone && <option value={NONE}>{t("regression.methods.none")}</option>}
        {choices
          .filter((name) => name !== rule)
          .map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
      </Select>
    </Field>
  );
}

/** Advanced › More (074–076): the methods a rule proposes for these data first, each with its
 *  reason, then those that apply without a rule; the ones that do not apply say why. Each method
 *  is a fold of its own, asked for only once it is opened (onion rule). */
export function MoreMethods({ datasetId, params }: { datasetId: string; params: RegressionParams }) {
  const query = useRegressionMethods(datasetId, params, true);
  const [opened, setOpened] = useState<RegressionMethodKind[]>([]);
  const data: RegressionMethods | undefined = query.data;
  const toggle = (kind: RegressionMethodKind) => setOpened((current) => (current.includes(kind) ? current.filter((k) => k !== kind) : [...current, kind]));
  return (
    <section id="regression-methods" className="space-y-2 border-t border-line pt-3 text-[13px]" aria-label={t("regression.methods.title")}>
      <div>
        <h3 className="text-[13px] font-medium text-ink-1">{t("regression.methods.title")}</h3>
        <p className="max-w-prose text-xs text-ink-2">{t("regression.methods.hint")}</p>
      </div>
      {!data ? (
        <Pending error={query.error} />
      ) : (
        <>
          <div className="space-y-2">
            {data.methods
              .filter((method) => method.available)
              .map((method) => (
                <MethodFold key={method.kind} method={method} open={opened.includes(method.kind)} onToggle={() => toggle(method.kind)}>
                  <MethodBody kind={method.kind} datasetId={datasetId} params={params} offered={data} />
                </MethodFold>
              ))}
          </div>
          {data.methods.some((method) => !method.available) && (
            <ul className="space-y-0.5 text-xs text-ink-3" aria-label={t("regression.methods.notHere")}>
              {data.methods
                .filter((method) => !method.available)
                .map((method) => (
                  <li key={method.kind}>
                    <span className="text-ink-2">{method.title}</span> · <Sentence text={method.reason} />
                  </li>
                ))}
            </ul>
          )}
        </>
      )}
    </section>
  );
}

function MethodFold({ method, open, onToggle, children }: { method: RegressionMethodItem; open: boolean; onToggle: () => void; children: ReactNode }) {
  return (
    <div id={`regression-method-${method.kind}`} className="rounded-md border border-line">
      <button type="button" aria-expanded={open} onClick={onToggle} className="flex w-full items-start gap-2 px-3 py-2 text-left">
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-baseline gap-2">
            <span className="text-[13px] font-medium text-ink-1">{method.title}</span>
            {method.proposed && <span className="text-xs text-accent">{t("regression.methods.proposed")}</span>}
          </span>
          <span className="block text-xs text-ink-3">
            <Sentence text={method.reason} />
          </span>
        </span>
        <ChevronDown size={14} strokeWidth={1.5} aria-hidden className={`mt-0.5 shrink-0 text-ink-3 ${open ? "rotate-180" : ""}`} />
      </button>
      {open && <div className="space-y-3 px-3 pb-3">{children}</div>}
    </div>
  );
}

type BodyProps = { datasetId: string; params: RegressionParams; offered: RegressionMethods };

function MethodBody({ kind, ...props }: BodyProps & { kind: RegressionMethodKind }) {
  switch (kind) {
    case "cross_section":
      return <CrossSection {...props} />;
    case "outputs":
      return <Outputs {...props} />;
    case "components":
      return <Components {...props} />;
    case "measurement":
      return <Measurement {...props} />;
    case "curves":
      return <Curves {...props} />;
    case "survival":
      return <Survival {...props} />;
    case "tobit":
      return <Tobit {...props} />;
    case "mixed":
      return <Mixed {...props} />;
  }
}

function useMethod<K extends RegressionMethodKind>(kind: K, { datasetId, params }: BodyProps, options: RegressionMethodOptions = {}, enabled = true) {
  return useRegressionMethod(datasetId, kind, params, options, enabled);
}

// --- 074 ------------------------------------------------------------------------------------------

function CrossSection(props: BodyProps) {
  const query = useMethod("cross_section", props);
  const data: RegressionCrossSection | undefined = query.data;
  if (!data) return <Pending error={query.error} />;
  if (!data.available) return <p className="text-ink-2">{data.reason}</p>;
  return (
    <>
      <p className="max-w-prose text-ink-2">
        <Sentence text={data.sentence} />
      </p>
      <Table
        head={
          <>
            <th scope="col" className={th}>{t("regression.term")}</th>
            <th scope="col" className={`${th} text-right`}>{t("regression.methods.fmMean")}</th>
            <th scope="col" className={`${th} text-right`}>{t("regression.methods.nwCi")}</th>
            <th scope="col" className={`${th} text-right`}>p</th>
            <th scope="col" className={`${th} text-right`}>{t("regression.methods.pooled")}</th>
          </>
        }
      >
        {data.terms.map((term) => (
          <tr key={term.term} className="border-t border-line">
            <td className="px-3 py-1.5 font-mono text-xs">{term.term}</td>
            <td className={`${td} text-ink-1`}>{formatNumber(term.estimate)}</td>
            <td className={`${td} whitespace-nowrap text-ink-2`}>{interval(term.ci_low, term.ci_high)}</td>
            <td className={`${td} text-ink-2`}>{formatP(term.p_value)}</td>
            <td className={`${td} text-ink-2`}>{formatNumber(term.pooled)}</td>
          </tr>
        ))}
      </Table>
      <p className="text-xs text-ink-3">
        <Sentence text={data.lag_reason} />
      </p>
      {data.ics.length > 0 && (
        <div className="space-y-1.5">
          <h4 className="text-xs font-medium text-ink-2">{t("regression.methods.ic")}</h4>
          <p className="max-w-prose text-ink-2">
            <Sentence text={data.ic_sentence} />
          </p>
          <Table
            head={
              <>
                <th scope="col" className={th}>{t("regression.methods.column")}</th>
                <th scope="col" className={`${th} text-right`}>{t("regression.methods.meanIc")}</th>
                <th scope="col" className={`${th} text-right`}>ICIR</th>
                <th scope="col" className={`${th} text-right`}>{t("regression.methods.nwT")}</th>
                <th scope="col" className={`${th} text-right`}>{t("regression.methods.positive")}</th>
              </>
            }
          >
            {data.ics.map((ic) => (
              <tr key={ic.column} className="border-t border-line">
                <td className={`px-3 py-1.5 font-mono text-xs ${ic.ranks ? "text-ink-1" : "text-ink-3"}`}>{ic.column}</td>
                <td className={`${td} text-ink-1`}>{ic.mean.toFixed(3)}</td>
                <td className={`${td} text-ink-2`}>{formatNumber(ic.icir)}</td>
                <td className={`${td} ${ic.ranks ? "text-ink-1" : "text-ink-2"}`}>{formatNumber(ic.t)}</td>
                <td className={`${td} text-ink-2`}>{Math.round(ic.positive * 100)} %</td>
              </tr>
            ))}
          </Table>
        </div>
      )}
      <Notes notes={data.notes} />
    </>
  );
}

function Outputs(props: BodyProps) {
  const [chosen, setChosen] = useState<string[]>([]);
  const picked = chosen.filter((name) => props.offered.numbers.includes(name));
  const query = useMethod("outputs", props, { outputs: picked }, picked.length > 0);
  const data: RegressionOutputs | undefined = query.data;
  return (
    <>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Field label={t("regression.methods.addOutcome")}>
          <Select aria-label={t("regression.methods.addOutcome")} value="" onChange={(event) => event.target.value && setChosen([...picked, event.target.value])} className={field}>
            <option value="">{t("regression.methods.addOutcome")}</option>
            {props.offered.numbers
              .filter((name) => !picked.includes(name))
              .map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
          </Select>
        </Field>
        {picked.length > 0 && (
          <ul className="flex flex-wrap items-end gap-1.5 sm:col-span-2">
            {picked.map((name) => (
              <li key={name}>
                <button
                  type="button"
                  onClick={() => setChosen(picked.filter((item) => item !== name))}
                  aria-label={t("regression.methods.removeOutcome", { name })}
                  className="inline-flex h-8 items-center gap-1 rounded-md border border-line bg-surface-2 px-2 font-mono text-xs text-ink-1 hover:text-accent"
                >
                  {name}
                  <X size={12} strokeWidth={1.75} aria-hidden />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
      {picked.length === 0 ? (
        <p className="text-xs text-ink-3">{t("regression.methods.outcomesWaiting")}</p>
      ) : !data ? (
        <Pending error={query.error} />
      ) : !data.available ? (
        <p className="text-ink-2">{data.reason}</p>
      ) : (
        <>
          <p className="max-w-prose text-ink-2">
            <Sentence text={data.sentence} />
          </p>
          <Table
            head={
              <>
                <th scope="col" className={th}>{t("regression.term")}</th>
                {data.outputs.map((name) => (
                  <th key={name} scope="col" className={`${th} text-right font-mono`}>
                    {name}
                  </th>
                ))}
              </>
            }
          >
            {data.terms.map((term, i) => (
              <tr key={term} className="border-t border-line">
                <td className="px-3 py-1.5 font-mono text-xs">{term}</td>
                {(data.coefficients[i] ?? []).map((cell, j) => (
                  <td key={data.outputs[j] ?? j} className={`${td} ${cell.p_value !== null && cell.p_value < 0.05 ? "text-ink-1" : "text-ink-3"}`} title={`SE ${formatNumber(cell.std_error)} · p ${formatP(cell.p_value)}`}>
                    {formatNumber(cell.estimate)}
                  </td>
                ))}
              </tr>
            ))}
          </Table>
          <Table
            head={
              <>
                <th scope="col" className={th}>{t("regression.methods.jointTerm")}</th>
                <th scope="col" className={`${th} text-right`}>{t("regression.methods.wilks")}</th>
                <th scope="col" className={`${th} text-right`}>{t("regression.methods.pillai")}</th>
                <th scope="col" className={`${th} text-right`}>p</th>
              </>
            }
          >
            {data.tests.map((test) => (
              <tr key={test.term} className="border-t border-line">
                <td className="px-3 py-1.5 font-mono text-xs">{test.term}</td>
                <td className={`${td} text-ink-2`}>{test.wilks.toFixed(3)}</td>
                <td className={`${td} text-ink-2`}>{test.pillai.toFixed(3)}</td>
                <td className={`${td} text-ink-1`}>{formatP(test.pillai_p)}</td>
              </tr>
            ))}
          </Table>
          <p className={data.grs ? "max-w-prose text-ink-2" : "text-xs text-ink-3"}>
            <Sentence text={data.grs ? data.grs.sentence : data.grs_reason} />
          </p>
          <Notes notes={data.notes} />
        </>
      )}
    </>
  );
}

function Components(props: BodyProps) {
  const query = useMethod("components", props);
  const data: RegressionComponents | undefined = query.data;
  if (!data) return <Pending error={query.error} />;
  if (!data.available) return <p className="text-ink-2">{data.reason}</p>;
  return (
    <>
      <p className="max-w-prose text-ink-2">
        <Sentence text={data.sentence} />
      </p>
      <p className="text-xs text-ink-3">
        <Sentence text={data.kept_reason} />
      </p>
      <Table
        head={
          <>
            <th scope="col" className={th}>{t("regression.methods.component")}</th>
            <th scope="col" className={`${th} text-right`}>{t("regression.methods.share")}</th>
            <th scope="col" className={`${th} text-right`}>{t("regression.estimate")}</th>
            <th scope="col" className={`${th} text-right`}>p</th>
          </>
        }
      >
        {data.components.map((component) => (
          <tr key={component.component} className="border-t border-line">
            <td className="px-3 py-1.5 text-xs">
              <span className="font-mono text-ink-1">{component.component}</span> <span className="text-ink-3">{component.label}</span>
            </td>
            <td className={`${td} text-ink-2`}>{Math.round(component.share * 100)} %</td>
            <td className={`${td} text-ink-1`}>{formatNumber(component.estimate)}</td>
            <td className={`${td} text-ink-2`}>{formatP(component.p_value)}</td>
          </tr>
        ))}
      </Table>
      <Table
        head={
          <>
            <th scope="col" className={th}>{t("regression.methods.column")}</th>
            <th scope="col" className={`${th} text-right`}>{t("regression.methods.throughComponents")}</th>
            <th scope="col" className={`${th} text-right`}>{t("regression.methods.leastSquares")}</th>
          </>
        }
      >
        {data.effects.map((effect) => (
          <tr key={effect.term} className="border-t border-line">
            <td className="px-3 py-1.5 font-mono text-xs">{effect.term}</td>
            <td className={`${td} whitespace-nowrap text-ink-1`}>
              {formatNumber(effect.estimate)} <span className="text-ink-3">{interval(effect.ci_low, effect.ci_high)}</span>
            </td>
            <td className={`${td} whitespace-nowrap text-ink-2`}>
              {formatNumber(effect.least_squares)} <span className="text-ink-3">{interval(effect.ls_low, effect.ls_high)}</span>
            </td>
          </tr>
        ))}
      </Table>
      <Notes notes={data.notes} />
    </>
  );
}

// --- 075 ------------------------------------------------------------------------------------------

function Measurement(props: BodyProps) {
  const [sigma, setSigma] = useState<string | undefined>(undefined);
  const [sigmaX, setSigmaX] = useState<string | undefined>(undefined);
  const query = useMethod("measurement", props, { sigma, sigma_x: sigmaX });
  const data: RegressionMeasurement | undefined = query.data;
  const { offered } = props;
  return (
    <>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <PickSelect label={t("regression.methods.sigma")} value={sigma} onChange={setSigma} rule={offered.sigma?.column ?? null} choices={offered.numbers} />
        <PickSelect label={t("regression.methods.sigmaX")} value={sigmaX} onChange={setSigmaX} rule={offered.sigma_x?.column ?? null} choices={offered.numbers} />
      </div>
      {!data ? (
        <Pending error={query.error} />
      ) : !data.available ? (
        <p className="text-ink-2">{data.reason}</p>
      ) : (
        <>
          {data.uncertainty ? (
            <div className="space-y-1.5">
              <h4 className="text-xs font-medium text-ink-2">{t("regression.methods.uncertainties")}</h4>
              <p className="max-w-prose text-ink-2">
                <Sentence text={data.uncertainty.sentence} />
              </p>
              <p className={`max-w-prose text-xs ${data.uncertainty.absolute ? "text-ink-3" : "text-warn"}`}>
                <Sentence text={data.uncertainty.absolute_reason} />
              </p>
              <Table
                head={
                  <>
                    <th scope="col" className={th}>{t("regression.term")}</th>
                    <th scope="col" className={`${th} text-right`}>{t("regression.estimate")}</th>
                    <th scope="col" className={`${th} text-right`}>{t("regression.methods.absoluteSe")}</th>
                    <th scope="col" className={`${th} text-right`}>{t("regression.methods.scaledSe")}</th>
                    <th scope="col" className={`${th} text-right`}>{t("regression.ci")}</th>
                  </>
                }
              >
                {data.uncertainty.terms.map((term) => (
                  <tr key={term.term} className="border-t border-line">
                    <td className="px-3 py-1.5 font-mono text-xs">{term.term}</td>
                    <td className={`${td} text-ink-1`}>{formatNumber(term.estimate)}</td>
                    <td className={`${td} ${data.uncertainty?.absolute ? "text-ink-1" : "text-ink-3"}`}>{formatNumber(term.absolute_se)}</td>
                    <td className={`${td} ${data.uncertainty?.absolute ? "text-ink-3" : "text-ink-1"}`}>{formatNumber(term.scaled_se)}</td>
                    <td className={`${td} whitespace-nowrap text-ink-2`}>{interval(term.ci_low, term.ci_high)}</td>
                  </tr>
                ))}
              </Table>
              <p className="text-ink-2">
                <Sentence text={data.uncertainty.pull_sentence} />
              </p>
              {data.uncertainty.dropped > 0 && <p className="text-xs text-ink-3">{t("regression.methods.dropped", { count: formatCount(data.uncertainty.dropped) })}</p>}
            </div>
          ) : (
            <p className="text-xs text-ink-3">
              <Sentence text={data.uncertainty_reason} />
            </p>
          )}
          {data.errors ? (
            <div className="space-y-1.5">
              <h4 className="text-xs font-medium text-ink-2">{t("regression.methods.errorsX")}</h4>
              <p className="max-w-prose text-ink-2">
                <Sentence text={data.errors.sentence} />
              </p>
              {data.errors.ratio_reason && (
                <p className="text-xs text-ink-3">
                  <Sentence text={data.errors.ratio_reason} />
                </p>
              )}
              {data.errors.lines.length > 0 && (
                <Table
                  head={
                    <>
                      <th scope="col" className={th}>{t("regression.methods.method")}</th>
                      <th scope="col" className={`${th} text-right`}>{t("regression.methods.slope")}</th>
                      <th scope="col" className={`${th} text-right`}>{t("regression.methods.intercept")}</th>
                    </>
                  }
                >
                  {data.errors.lines.map((line) => (
                    <tr key={line.method} className="border-t border-line" title={line.reason}>
                      <td className="px-3 py-1.5 text-xs text-ink-1">{line.label}</td>
                      <td className={`${td} whitespace-nowrap text-ink-1`}>
                        {formatNumber(line.slope)} <span className="text-ink-3">± {formatNumber(line.slope_se)}</span>
                      </td>
                      <td className={`${td} whitespace-nowrap text-ink-2`}>
                        {formatNumber(line.intercept)} <span className="text-ink-3">± {formatNumber(line.intercept_se)}</span>
                      </td>
                    </tr>
                  ))}
                </Table>
              )}
              {data.errors.tls.length > 0 && (
                <Table
                  head={
                    <>
                      <th scope="col" className={th}>{t("regression.term")}</th>
                      <th scope="col" className={`${th} text-right`}>TLS</th>
                      <th scope="col" className={`${th} text-right`}>{t("regression.methods.leastSquares")}</th>
                    </>
                  }
                >
                  {data.errors.tls.map((term) => (
                    <tr key={term.term} className="border-t border-line">
                      <td className="px-3 py-1.5 font-mono text-xs">{term.term}</td>
                      <td className={`${td} text-ink-1`}>{formatNumber(term.estimate)}</td>
                      <td className={`${td} text-ink-2`}>{formatNumber(term.least_squares)}</td>
                    </tr>
                  ))}
                </Table>
              )}
              <Notes notes={data.errors.notes} />
            </div>
          ) : (
            <p className="text-xs text-ink-3">
              <Sentence text={data.errors_reason} />
            </p>
          )}
        </>
      )}
    </>
  );
}

function Curves(props: BodyProps) {
  const query = useMethod("curves", props);
  const data: RegressionCurves | undefined = query.data;
  const figure = useMemo(() => (data?.available && data.fits.length ? curvesFigure(data, SHOWN_CURVES, data.column ?? "x", props.params.target ?? "y") : null), [data, props.params.target]);
  if (!data) return <Pending error={query.error} />;
  if (!data.available) return <p className="text-ink-2">{data.reason}</p>;
  return (
    <>
      {data.sentence && (
        <p className="max-w-prose text-ink-2">
          <Sentence text={data.sentence} />
        </p>
      )}
      {figure && (
        <figure className="min-w-0 rounded-lg border border-line bg-surface-1 p-3">
          <Plot figure={figure} height={CURVE_HEIGHT} label={t("regression.methods.curvesChart")} />
        </figure>
      )}
      <Table
        head={
          <>
            <th scope="col" className={th}>{t("regression.methods.shape")}</th>
            <th scope="col" className={`${th} text-right`}>ΔAIC</th>
            <th scope="col" className={`${th} text-right`}>R²</th>
            <th scope="col" className={th}>{t("regression.methods.parameters")}</th>
          </>
        }
      >
        {data.fits.map((fit) => (
          <tr key={fit.kind} className="border-t border-line">
            <td className="px-3 py-1.5 text-xs">
              <span className={fit.kind === data.best ? "font-medium text-ink-1" : "text-ink-2"}>{fit.label}</span> <span className="font-mono text-ink-3">{fit.formula}</span>
            </td>
            <td className={`${td} text-ink-2`}>{fit.delta_aic === null ? "—" : fit.delta_aic.toFixed(1)}</td>
            <td className={`${td} text-ink-2`}>{fit.r2 === null ? "—" : fit.r2.toFixed(3)}</td>
            <td className="px-3 py-1.5 text-xs text-ink-2">
              {fit.error ?? fit.parameters.map((p) => `${p.name} ${formatNumber(p.estimate)} ${interval(p.ci_low, p.ci_high)}`).join(" · ")}
            </td>
          </tr>
        ))}
      </Table>
      {data.skipped.map((reason) => (
        <p key={reason} className="text-xs text-ink-3">
          {reason}
        </p>
      ))}
      {data.weighted && <p className="text-xs text-ink-3">{t("regression.methods.curvesWeighted", { column: data.weighted })}</p>}
      {data.sampled && <p className="text-xs text-ink-3">{t("regression.methods.curvesSampled", { rows: formatCount(data.rows) })}</p>}
    </>
  );
}

// --- 076 ------------------------------------------------------------------------------------------

function Survival(props: BodyProps) {
  const [event, setEvent] = useState<string | undefined>(undefined);
  const query = useMethod("survival", props, { event });
  const data: RegressionCox | undefined = query.data;
  const { offered } = props;
  return (
    <>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <PickSelect label={t("regression.methods.event")} value={event} onChange={setEvent} rule={offered.event?.column ?? null} choices={offered.events} allowNone={false} />
      </div>
      {!data ? (
        <Pending error={query.error} />
      ) : !data.available ? (
        <p className="text-ink-2">{data.reason}</p>
      ) : (
        <>
          <p className="max-w-prose text-ink-2">
            <Sentence text={data.sentence} />
          </p>
          <p className="text-xs text-ink-3">
            <Sentence text={data.event_reason} />
          </p>
          <Table
            head={
              <>
                <th scope="col" className={th}>{t("regression.term")}</th>
                <th scope="col" className={`${th} text-right`}>{t("regression.methods.hazardRatio")}</th>
                <th scope="col" className={`${th} text-right`}>{t("regression.ci")}</th>
                <th scope="col" className={`${th} text-right`}>p</th>
                <th scope="col" className={`${th} text-right`}>{t("regression.methods.schoenfeld")}</th>
              </>
            }
          >
            {data.terms.map((term) => (
              <tr key={term.term} className="border-t border-line" title={term.sentence}>
                <td className="px-3 py-1.5 font-mono text-xs">{term.term}</td>
                <td className={`${td} text-ink-1`}>{formatNumber(term.hazard_ratio)}</td>
                <td className={`${td} whitespace-nowrap text-ink-2`}>{interval(term.ci_low, term.ci_high)}</td>
                <td className={`${td} text-ink-2`}>{formatP(term.p_value)}</td>
                <td className={`${td} ${term.proportional === false ? "text-warn" : "text-ink-2"}`}>{formatP(term.ph_p)}</td>
              </tr>
            ))}
          </Table>
          <p className="max-w-prose text-ink-2">
            <Sentence text={data.ph_sentence} />
          </p>
          <Notes notes={data.notes} />
        </>
      )}
    </>
  );
}

function Tobit(props: BodyProps) {
  const query = useMethod("tobit", props);
  const data: RegressionTobit | undefined = query.data;
  if (!data) return <Pending error={query.error} />;
  if (!data.available) return <p className="text-ink-2">{data.reason}</p>;
  return (
    <>
      <p className="max-w-prose text-ink-2">
        <Sentence text={data.sentence} />
      </p>
      <Table
        head={
          <>
            <th scope="col" className={th}>{t("regression.term")}</th>
            <th scope="col" className={`${th} text-right`}>{t("regression.methods.latent")}</th>
            <th scope="col" className={`${th} text-right`}>{t("regression.methods.observed")}</th>
            <th scope="col" className={`${th} text-right`}>{t("regression.methods.leastSquares")}</th>
          </>
        }
      >
        {data.terms.map((term) => (
          <tr key={term.term} className="border-t border-line" title={`p ${formatP(term.p_value)}`}>
            <td className="px-3 py-1.5 font-mono text-xs">{term.term}</td>
            <td className={`${td} whitespace-nowrap text-ink-1`}>
              {formatNumber(term.estimate)} <span className="text-ink-3">{interval(term.ci_low, term.ci_high)}</span>
            </td>
            <td className={`${td} text-ink-1`}>{formatNumber(term.marginal)}</td>
            <td className={`${td} text-ink-2`}>{formatNumber(term.least_squares)}</td>
          </tr>
        ))}
      </Table>
      <p className="text-xs text-ink-3">{t("regression.methods.sigmaValue", { value: formatNumber(data.sigma), se: formatNumber(data.sigma_se) })}</p>
      <Notes notes={data.notes} />
    </>
  );
}

function Mixed(props: BodyProps) {
  const [slope, setSlope] = useState("");
  const chosen = props.offered.slopes.includes(slope) ? slope : "";
  const query = useMethod("mixed", props, chosen ? { slope: chosen } : {});
  const data: RegressionMixed | undefined = query.data;
  return (
    <>
      {props.offered.slopes.length > 0 && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <Field label={t("regression.methods.slopeBy")}>
            <Select aria-label={t("regression.methods.slopeBy")} value={chosen} onChange={(event) => setSlope(event.target.value)} className={field}>
              <option value="">{t("regression.methods.noSlope")}</option>
              {props.offered.slopes.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </Select>
          </Field>
        </div>
      )}
      {!data ? (
        <Pending error={query.error} />
      ) : !data.available ? (
        <p className="text-ink-2">{data.reason}</p>
      ) : (
        <>
          <p className="max-w-prose text-ink-2">
            <Sentence text={data.sentence} />
          </p>
          <p className="text-xs text-ink-3">
            <Sentence text={data.group_reason} />
          </p>
          <Table
            head={
              <>
                <th scope="col" className={th}>{t("regression.term")}</th>
                <th scope="col" className={`${th} text-right`}>{t("regression.methods.mixedFixed")}</th>
                <th scope="col" className={`${th} text-right`}>{t("regression.methods.leastSquares")}</th>
              </>
            }
          >
            {data.terms.map((term) => (
              <tr key={term.term} className="border-t border-line" title={`p ${formatP(term.p_value)}`}>
                <td className="px-3 py-1.5 font-mono text-xs">{term.term}</td>
                <td className={`${td} whitespace-nowrap text-ink-1`}>
                  {formatNumber(term.estimate)} <span className="text-ink-3">{interval(term.ci_low, term.ci_high)}</span>
                </td>
                <td className={`${td} whitespace-nowrap text-ink-2`}>
                  {formatNumber(term.least_squares)} <span className="text-ink-3">± {formatNumber(term.ls_se)}</span>
                </td>
              </tr>
            ))}
          </Table>
          <p className="text-xs text-ink-3">
            {t("regression.methods.variances", { between: formatNumber(data.group_variance), within: formatNumber(data.residual_variance), groups: formatCount(data.groups) })}
          </p>
          <Notes notes={data.notes} />
        </>
      )}
    </>
  );
}
