import { ChevronDown } from "lucide-react";
import type { ReactNode } from "react";

import type {
  Regression,
  RegressionFamilies,
  RegressionInfluence,
  RegressionMore,
  RegressionParams,
  RegressionSe,
  RegressionSection,
  RegressionStandardErrors,
  RegressionTerms,
} from "../../api/client";
import { useRegressionSection } from "../../api/hooks";
import { formatNumber } from "../../charts/figures";
import { formatP } from "../../features/variable/normality";
import { formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { ErrorNote } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { Instruments } from "./Instruments";
import { PanelSection } from "./PanelSection";
import { Sentence } from "./Sentence";

type AdvancedProps = {
  datasetId: string;
  params: RegressionParams;
  result: Regression;
  opened: RegressionSection[];
  onToggle: (section: RegressionSection) => void;
  onUse: (kind: RegressionSe) => void;
  /** 069: refit with another family (a niche one from More). */
  onFamily: (family: string) => void;
};

const TITLES: Record<RegressionSection, MessageKey> = {
  "standard-errors": "regression.section.standardErrors",
  terms: "regression.section.terms",
  influence: "regression.section.influence",
  more: "regression.section.more",
  families: "regression.section.families",
  panel: "regression.section.panel",
};
const HINTS: Record<RegressionSection, MessageKey> = {
  "standard-errors": "regression.section.standardErrorsHint",
  terms: "regression.section.termsHint",
  influence: "regression.section.influenceHint",
  more: "regression.section.moreHint",
  families: "regression.section.familiesHint",
  panel: "regression.section.panelHint",
};

/** Advanced (068): each section is asked for when it is opened, More only when More is (onion). */
export function Advanced({ datasetId, params, result, opened, onToggle, onUse, onFamily }: AdvancedProps) {
  const fold = (section: RegressionSection, body: ReactNode, nested = false) => {
    const open = opened.includes(section);
    return (
      <div id={`regression-${section}`} className={nested ? "rounded-md border border-line" : "border-t border-line"}>
        <button
          type="button"
          aria-expanded={open}
          onClick={() => onToggle(section)}
          className="flex w-full items-center gap-2 px-4 py-2.5 text-left"
        >
          <span className="text-[13px] font-medium text-ink-1">{t(TITLES[section])}</span>
          <span className="min-w-0 truncate text-xs text-ink-3">{t(HINTS[section])}</span>
          <ChevronDown size={14} strokeWidth={1.5} aria-hidden className={`ml-auto shrink-0 text-ink-3 ${open ? "rotate-180" : ""}`} />
        </button>
        {open && <div className="px-4 pb-4">{body}</div>}
      </div>
    );
  };
  return (
    <div>
      {fold("standard-errors", <StandardErrorsSection datasetId={datasetId} params={params} onUse={onUse} />)}
      {result.structure.kind === "panel" && fold("panel", <PanelSection datasetId={datasetId} params={params} />)}
      {fold("terms", <TermsSection datasetId={datasetId} params={params} />)}
      {fold("influence", <InfluenceSection datasetId={datasetId} params={params} rows={result.quality.rows} />)}
      <div className="border-t border-line p-3">
        {fold(
          "more",
          <div className="space-y-4">
            <MoreSection datasetId={datasetId} params={params} result={result} />
            {/* 069: the niche families, asked for only once More is open (onion rule) */}
            <FamiliesSection datasetId={datasetId} params={params} onFamily={onFamily} />
          </div>,
          true,
        )}
      </div>
    </div>
  );
}

function useSection<S extends RegressionSection>(datasetId: string, section: S, params: RegressionParams) {
  return useRegressionSection(datasetId, section, params, true);
}

function Pending({ error }: { error: Error | null }) {
  return error ? <ErrorNote>{error.message}</ErrorNote> : <Loading label={t("regression.section.loading")} compact className="h-32 rounded-md" />;
}

const th = "px-3 py-1.5 font-normal";
const td = "px-3 py-1.5 tabular-nums";

function StandardErrorsSection({
  datasetId,
  params,
  onUse,
}: {
  datasetId: string;
  params: RegressionParams;
  onUse: (kind: RegressionSe) => void;
}) {
  const query = useSection(datasetId, "standard-errors", params);
  const data: RegressionStandardErrors | undefined = query.data;
  if (!data) return <Pending error={query.error} />;
  const shown = data.kinds.filter((kind) => kind.available);
  const missing = data.kinds.filter((kind) => !kind.available);
  return (
    <div className="space-y-2">
      <p className="text-[13px] text-ink-2">
        <Sentence text={data.sentence} />
      </p>
      <div className="overflow-x-auto rounded-md border border-line">
        <table className="w-full border-collapse text-[13px]">
          <thead className="bg-surface-2 text-left text-xs text-ink-3">
            <tr>
              <th scope="col" className={th}>{t("regression.term")}</th>
              <th scope="col" className={`${th} text-right`}>{t("regression.estimate")}</th>
              {shown.map((kind) => (
                <th key={kind.kind} scope="col" className={`${th} text-right whitespace-nowrap`} title={kind.reason}>
                  {kind.chosen ? <span className="font-medium text-ink-1">{kind.label}</span> : kind.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.terms.map((term, row) => (
              <tr key={term} className="border-t border-line">
                <td className="px-3 py-1.5 font-mono text-xs">{term}</td>
                <td className={`${td} text-right text-ink-1`}>{formatNumber(data.estimates[row])}</td>
                {shown.map((kind) => (
                  <td key={kind.kind} className={`${td} text-right ${kind.chosen ? "text-ink-1" : "text-ink-2"}`} title={`p ${formatP(kind.p_values[row] ?? null)}`}>
                    {formatNumber(kind.std_errors[row])}
                  </td>
                ))}
              </tr>
            ))}
            <tr className="border-t border-line">
              <td className="px-3 py-1.5" colSpan={2} />
              {shown.map((kind) => (
                <td key={kind.kind} className="px-3 py-1.5 text-right">
                  {kind.chosen ? (
                    <span className="text-xs text-ink-3">{t("regression.inUse")}</span>
                  ) : (
                    <button type="button" className="text-xs text-accent hover:underline" onClick={() => onUse(kind.kind)}>
                      {t("regression.use")}
                    </button>
                  )}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
      {missing.length > 0 && (
        <ul className="space-y-0.5 text-xs text-ink-3">
          {missing.map((kind) => (
            <li key={kind.kind}>
              <span className="text-ink-2">{kind.label}</span> · {kind.reason}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function TermsSection({ datasetId, params }: { datasetId: string; params: RegressionParams }) {
  const query = useSection(datasetId, "terms", params);
  const data: RegressionTerms | undefined = query.data;
  if (!data) return <Pending error={query.error} />;
  return (
    <div className="space-y-3 text-[13px]">
      <p className="text-ink-2">
        {data.sentence} {data.condition !== null && t("regression.condition", { value: data.condition.toFixed(1) })}
      </p>
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
                  <td className={`${td} text-right ${(vif.vif ?? 0) > 10 ? "text-bad" : "text-ink-1"}`}>{vif.vif === null ? "∞" : vif.vif.toFixed(1)}</td>
                  <td className="px-3 py-1.5 font-mono text-xs text-ink-2">{vif.partner ?? "—"}</td>
                  <td className={`${td} text-right text-ink-2`}>{vif.correlation === null ? "—" : vif.correlation.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function InfluenceSection({ datasetId, params, rows }: { datasetId: string; params: RegressionParams; rows: number }) {
  const query = useSection(datasetId, "influence", params);
  const data: RegressionInfluence | undefined = query.data;
  if (!data) return <Pending error={query.error} />;
  if (!data.rows.length) return <p className="text-[13px] text-ink-2">{data.sentence}</p>;
  const large = 2 / Math.sqrt(Math.max(rows, 1)); // |DFBETAS| above 2/√n moves a coefficient
  return (
    <div className="space-y-2">
      <p className="text-[13px] text-ink-2">{data.sentence}</p>
      {/* DFBETAS run one column per coefficient: the table scrolls within its frame. */}
      <div className="overflow-x-auto rounded-md border border-line">
        <table className="w-full border-collapse text-[13px]">
          <thead className="bg-surface-2 text-left text-xs text-ink-3">
            <tr>
              <th scope="col" className={th}>{t("regression.row")}</th>
              <th scope="col" className={`${th} text-right`}>{t("regression.leverage")}</th>
              <th scope="col" className={`${th} text-right`}>Cook</th>
              <th scope="col" className={`${th} text-right`}>{t("regression.studentized")}</th>
              <th scope="col" className={`${th} text-right`}>{t("regression.bonferroni")}</th>
              {data.terms.map((term) => (
                <th key={term} scope="col" className={`${th} text-right font-mono whitespace-nowrap`} title={t("regression.dfbetasHint")}>
                  {term}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.rows.map((row) => (
              <tr key={row.row} className="border-t border-line">
                <td className={`${td} text-ink-1`}>{formatCount(row.row + 1)}</td>
                <td className={`${td} text-right text-ink-2`}>{row.leverage.toFixed(3)}</td>
                <td className={`${td} text-right ${row.cook > data.cook_threshold ? "text-ink-1" : "text-ink-2"}`}>{formatNumber(row.cook)}</td>
                <td className={`${td} text-right text-ink-2`}>{row.studentized.toFixed(2)}</td>
                <td className={`${td} text-right text-ink-2`}>{formatP(row.bonferroni)}</td>
                {row.dfbetas.map((value, i) => (
                  <td key={data.terms[i] ?? i} className={`${td} text-right ${Math.abs(value) > large ? "text-ink-1" : "text-ink-3"}`}>
                    {value.toFixed(2)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function MoreSection({ datasetId, params, result }: { datasetId: string; params: RegressionParams; result: Regression }) {
  const query = useSection(datasetId, "more", params);
  const data: RegressionMore | undefined = query.data;
  if (!data) return <Pending error={query.error} />;
  const moved = data.shifts.slice(1);
  return (
    <div className="space-y-3 text-[13px]">
      <p className="text-ink-2">{data.white_sentence}</p>
      <p className="text-ink-2">{data.shift_sentence}</p>
      {moved.length > 0 && (
        <div className="overflow-x-auto rounded-md border border-line">
          <table className="w-full border-collapse">
            <thead className="bg-surface-2 text-left text-xs text-ink-3">
              <tr>
                <th scope="col" className={th}>{t("regression.term")}</th>
                <th scope="col" className={`${th} text-right`}>{t("regression.allRows")}</th>
                <th scope="col" className={`${th} text-right`}>{t("regression.withoutInfluential", { count: formatCount(data.removed) })}</th>
                <th scope="col" className={`${th} text-right`}>{t("regression.change")}</th>
              </tr>
            </thead>
            <tbody>
              {moved.map((shift) => (
                <tr key={shift.term} className="border-t border-line">
                  <td className="px-3 py-1.5 font-mono text-xs">{shift.term}</td>
                  <td className={`${td} text-right text-ink-1`}>{formatNumber(shift.estimate)}</td>
                  <td className={`${td} text-right text-ink-1`}>{formatNumber(shift.without)}</td>
                  <td className={`${td} text-right ${Math.abs(shift.change ?? 0) > 0.2 ? "text-warn" : "text-ink-2"}`}>
                    {shift.change === null ? "—" : `${(shift.change * 100).toFixed(0)} %`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {!LEAST_SQUARES.has(result.model) && <p className="text-xs text-ink-3">{t("regression.moreLogit")}</p>}
      <Instruments datasetId={datasetId} params={params} result={result} />
    </div>
  );
}

const LEAST_SQUARES = new Set<string>(["ols", "wls", "log_ols", "boxcox"]);

/** 069: probit, Box-Cox, fractional and multinomial logit on the same rows, each with its
 *  strongest effect and "Use"; those the target does not admit say why. */
function FamiliesSection({ datasetId, params, onFamily }: { datasetId: string; params: RegressionParams; onFamily: (family: string) => void }) {
  const query = useSection(datasetId, "families", params);
  const data: RegressionFamilies | undefined = query.data;
  if (!data) return <Pending error={query.error} />;
  const fitted = data.alternatives.filter((a) => a.available);
  const missing = data.alternatives.filter((a) => !a.available);
  const unit = (reads: string) => (reads === "points" ? t("regression.ame") : reads === "ratio" ? t("regression.percent") : t("regression.estimate"));
  return (
    <section className="space-y-2 text-[13px]" aria-label={t("regression.section.families")}>
      <h3 className="text-xs font-medium text-ink-2">{t("regression.section.families")}</h3>
      <p className="text-ink-2">{data.sentence}</p>
      {fitted.map((family) => (
        <div key={family.kind} className="space-y-1.5 rounded-md border border-line p-3">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <span className="font-medium text-ink-1">{family.label}</span>
            <span className="flex items-center gap-3 text-xs text-ink-3 tabular-nums">
              {family.aic !== null && <span>AIC {formatNumber(family.aic)}</span>}
              {family.pseudo_r2 !== null && <span>{t("regression.pseudoR2")} {family.pseudo_r2.toFixed(3)}</span>}
              {family.r2 !== null && <span>R² {family.r2.toFixed(3)}</span>}
              {family.error === null && (
                <button type="button" className="text-accent hover:underline" onClick={() => onFamily(family.kind)}>
                  {t("regression.useFamily")}
                </button>
              )}
            </span>
          </div>
          <p className="text-xs text-ink-3">{family.reason}</p>
          {family.error !== null ? (
            <ErrorNote>{t("regression.familyError", { error: family.error })}</ErrorNote>
          ) : (
            <>
              <p className="text-ink-2">
                <Sentence text={family.sentence} />
              </p>
              {family.note && <p className="text-xs text-ink-3">{family.note}</p>}
              {family.effects.length > 0 && (
                <div className="overflow-x-auto rounded-md border border-line">
                  <table className="w-full border-collapse">
                    <thead className="bg-surface-2 text-left text-xs text-ink-3">
                      <tr>
                        <th scope="col" className={th}>{t("regression.term")}</th>
                        <th scope="col" className={`${th} text-right`}>{unit(family.reads)}</th>
                        <th scope="col" className={`${th} text-right`}>{t("regression.ci")}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {family.effects.map((effect) => (
                        <tr key={effect.term} className="border-t border-line">
                          <td className="px-3 py-1.5 font-mono text-xs break-words">{effect.term}</td>
                          <td className={`${td} text-right text-ink-1`}>{formatNumber(effect.value)}</td>
                          <td className={`${td} text-right whitespace-nowrap text-ink-2`}>
                            [{formatNumber(effect.low)} ; {formatNumber(effect.high)}]
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </div>
      ))}
      {missing.length > 0 && (
        <ul className="space-y-0.5 text-xs text-ink-3">
          {missing.map((family) => (
            <li key={family.kind}>
              <span className="text-ink-2">{family.label}</span> · {family.reason}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
