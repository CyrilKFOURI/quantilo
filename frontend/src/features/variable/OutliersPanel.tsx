import type { OutlierMethod } from "../../api/client";
import { useOutliers } from "../../api/hooks";
import { formatNumber } from "../../charts/figures";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { useStoredState } from "../../lib/useStoredState";
import { Button } from "../../ui/Button";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { useFilters } from "../filters/useFilters";
import { Loading } from "../../ui/Loading";

const METHODS: OutlierMethod[] = ["iqr", "iqr3", "zscore", "robust"];

export function OutliersPanel({ datasetId, column }: { datasetId: string; column: string }) {
  const [method, setMethod] = useStoredState<OutlierMethod>("de.variable.outliers", "iqr");
  const outliers = useOutliers(datasetId, column, method, true);
  const data = outliers.data;
  const rule = data?.rules.find((item) => item.method === method);
  const filters = useFilters(datasetId);
  const filterRows = (negate: boolean) => {
    if (!rule || rule.low === null || rule.high === null) return;
    // Outliers lie outside [low, high]: "show only" = outside (negated range), "exclude" = inside.
    filters.add({ column, op: "between", low: rule.low, high: rule.high, negate });
  };

  if (outliers.isError) return <p className="text-bad">{outliers.error.message}</p>;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-xs text-ink-3">{t("outliers.method")}</span>
        <SegmentedControl
          label={t("outliers.method")}
          value={method}
          options={METHODS.map((value) => ({ value, label: t(`outliers.${value}`) }))}
          onChange={setMethod}
        />
      </div>
      <p className="max-w-prose text-ink-2">{t(`outliers.explain.${method}`)}</p>

      {data ? (
        <>
          <div className="overflow-x-auto rounded-md border border-line">
            <table className="w-full border-collapse text-[13px]">
              <thead className="bg-surface-2 text-left text-xs text-ink-3">
                <tr>
                  <th scope="col" className="px-3 py-1.5 font-normal">{t("outliers.method")}</th>
                  <th scope="col" className="px-3 py-1.5 text-right font-normal">{t("outliers.low")}</th>
                  <th scope="col" className="px-3 py-1.5 text-right font-normal">{t("outliers.high")}</th>
                  <th scope="col" className="px-3 py-1.5 text-right font-normal">{t("outliers.count")}</th>
                  <th scope="col" className="px-3 py-1.5 text-right font-normal">%</th>
                </tr>
              </thead>
              <tbody>
                {data.rules.map((item) => (
                  <tr
                    key={item.method}
                    className={`border-t border-line ${item.method === method ? "bg-accent-soft" : ""}`}
                  >
                    <th scope="row" className="px-3 py-1.5 text-left font-normal text-ink-1">
                      <button type="button" className="hover:underline" onClick={() => setMethod(item.method)}>
                        {t(`outliers.${item.method}`)}
                      </button>
                    </th>
                    <td className="px-3 py-1.5 text-right text-ink-2 tabular-nums">{formatNumber(item.low)}</td>
                    <td className="px-3 py-1.5 text-right text-ink-2 tabular-nums">{formatNumber(item.high)}</td>
                    <td className="px-3 py-1.5 text-right tabular-nums">{formatCount(item.count)}</td>
                    <td className="px-3 py-1.5 text-right text-ink-2 tabular-nums">{item.pct.toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <section>
            <div className="mb-1.5 flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-xs font-medium text-ink-3">{t("outliers.rows")}</h3>
              {rule && rule.count > 0 && (
                <div className="flex gap-1.5">
                  <Button onClick={() => filterRows(true)}>{t("outliers.keep")}</Button>
                  <Button onClick={() => filterRows(false)}>{t("outliers.drop")}</Button>
                </div>
              )}
            </div>
            {data.rows.length === 0 ? (
              <p className="text-ink-2">{t("outliers.none")}</p>
            ) : (
              <>
                <div className="max-h-80 overflow-y-auto rounded-md border border-line">
                  <table className="w-full border-collapse text-[13px]">
                    <thead className="sticky top-0 bg-surface-2 text-left text-xs text-ink-3">
                      <tr>
                        <th scope="col" className="px-3 py-1.5 font-normal">{t("outliers.row")}</th>
                        <th scope="col" className="px-3 py-1.5 text-right font-normal">{t("outliers.value")}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.rows.map((row) => (
                        <tr key={row.row} className="border-t border-line">
                          <td className="px-3 py-1 text-ink-2 tabular-nums">{formatCount(row.row)}</td>
                          <td className="px-3 py-1 text-right text-ink-1 tabular-nums">{formatNumber(row.value)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {rule && rule.count > data.rows.length && (
                  <p className="mt-1.5 text-xs text-ink-3">
                    {t("outliers.more", { shown: formatCount(data.rows.length), total: formatCount(rule.count) })}
                  </p>
                )}
              </>
            )}
          </section>
        </>
      ) : (
        <Loading label={t("loading.outliers")} className="h-40 rounded-md" />
      )}
    </div>
  );
}
