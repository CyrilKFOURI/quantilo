import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";

type Cell = { truth: string; predicted: string };

type ConfusionProps = {
  labels: readonly string[];
  counts: readonly (readonly number[])[];
  normalized: readonly (readonly number[])[];
  mode: "counts" | "rows";
  selected: Cell | null;
  onSelect: (cell: Cell | null) => void;
};

/** A cell's shade: one hue (sequential), light to dark with the share of its row; the diagonal
 *  and the errors use the same scale so a large error stands out as much as a large success. */
function shade(share: number): string | undefined {
  if (!(share > 0)) return undefined;
  return `color-mix(in oklab, var(--chart-1) ${Math.round(6 + share * 46)}%, transparent)`;
}

/**
 * 113: the confusion matrix (rows = truth, columns = prediction), in counts or normalised by row.
 * A cell is a button: it shows those images under the matrix without leaving the tab. On a phone
 * the matrix scrolls in its own frame, never the page.
 */
export function ConfusionMatrix({ labels, counts, normalized, mode, selected, onSelect }: ConfusionProps) {
  return (
    <div className="w-fit max-w-full overflow-x-auto overscroll-x-contain rounded-md border border-line">
      <table className="w-auto min-w-full border-collapse text-[12px] tabular-nums sm:min-w-0">
        <caption className="sr-only">{t("vision.results.confusionCaption")}</caption>
        <thead>
          <tr>
            <th scope="col" className="sticky left-0 z-10 bg-surface-1 px-2 py-1.5 text-left font-normal text-ink-3">
              {t("vision.results.truthPredicted")}
            </th>
            {labels.map((label) => (
              <th key={label} scope="col" className="max-w-40 truncate px-2 py-1.5 text-right font-medium text-ink-2" title={label}>
                {label}
              </th>
            ))}
            <th scope="col" className="px-2 py-1.5 text-right font-normal text-ink-3">
              {t("vision.results.recallCol")}
            </th>
          </tr>
        </thead>
        <tbody>
          {labels.map((truth, i) => {
            const row = counts[i] ?? [];
            const total = row.reduce((a, b) => a + b, 0);
            return (
              <tr key={truth} className="border-t border-line">
                <th scope="row" className="sticky left-0 z-10 max-w-36 truncate bg-surface-1 px-2 py-1.5 text-left font-medium text-ink-2" title={truth}>
                  {truth}
                </th>
                {labels.map((predicted, j) => {
                  const count = row[j] ?? 0;
                  const share = normalized[i]?.[j] ?? 0;
                  const isSelected = selected?.truth === truth && selected.predicted === predicted;
                  const diagonal = i === j;
                  return (
                    <td key={predicted} className="p-0.5">
                      <button
                        type="button"
                        disabled={count === 0}
                        aria-pressed={isSelected}
                        aria-label={t("vision.results.cellLabel", { count: formatCount(count), truth, predicted })}
                        onClick={() => onSelect(isSelected ? null : { truth, predicted })}
                        style={{ backgroundColor: shade(share) }}
                        className={`flex h-9 w-full min-w-14 items-center justify-end rounded-[3px] px-2 text-ink-1 transition-shadow enabled:hover:shadow-[inset_0_0_0_1px_var(--ink-3)] disabled:cursor-default ${
                          isSelected ? "shadow-[inset_0_0_0_2px_var(--accent)]" : ""
                        } ${diagonal ? "font-semibold" : ""} ${count === 0 ? "text-ink-3" : ""}`}
                      >
                        {mode === "counts" ? formatCount(count) : `${Math.round(share * 100)} %`}
                      </button>
                    </td>
                  );
                })}
                <td className="px-2 py-1.5 text-right text-ink-2">{total > 0 ? `${Math.round(((row[i] ?? 0) / total) * 100)} %` : "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
