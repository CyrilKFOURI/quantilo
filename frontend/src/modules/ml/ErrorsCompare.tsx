import type { MlErrorComparison } from "../../api/client";
import { useMlErrorsCompared } from "../../api/hooks";
import { formatNumber } from "../../charts/figures";
import { t } from "../../lib/i18n";
import { ErrorNote } from "../../ui/fields";
import { Loading } from "../../ui/Loading";

type Changed = MlErrorComparison["better"][number];

function List({ title, items }: { title: string; items: Changed[] }) {
  if (!items.length) return null;
  return (
    <div className="space-y-1">
      <h4 className="text-xs text-ink-2">{title}</h4>
      <ul className="divide-y divide-line border-y border-line text-[13px] text-ink-1">
        {items.map((c) => (
          <li key={c.where} className="py-1.5">
            {c.sentence}
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Two runs of the same target (036): the subgroups where the second fixes the first one's
 *  errors, and where it makes them worse, on the held-out rows both predicted. */
export function ErrorsCompared({ datasetId, first, second, names }: { datasetId: string; first: string; second: string; names: [string, string] }) {
  const found = useMlErrorsCompared(datasetId, first, second);
  if (found.isError) return <ErrorNote>{found.error.message}</ErrorNote>;
  if (!found.data) return <Loading label={t("loading.errors")} className="h-16 rounded-md" compact />;
  const c = found.data;
  const show = (value: number) => (c.measure === "misclassified" ? `${(value * 100).toFixed(0)} %` : formatNumber(value));
  return (
    <section className="space-y-2" aria-label={t("ml.errors.compareTitle")}>
      <h3 className="text-[13px] font-medium">{t("ml.errors.compareTitle")}</h3>
      <p className="text-xs text-ink-3">
        {t("ml.errors.compareSummary", {
          first: names[0],
          second: names[1],
          label: c.label.charAt(0).toUpperCase() + c.label.slice(1),
          a: show(c.first),
          b: show(c.second),
          rows: c.rows.toLocaleString("en-US"),
        })}
      </p>
      {c.message && <p className="text-[13px] text-ink-2">{c.message}</p>}
      <List title={t("ml.errors.compareBetter", { second: names[1] })} items={c.better} />
      <List title={t("ml.errors.compareWorse", { second: names[1] })} items={c.worse} />
    </section>
  );
}
