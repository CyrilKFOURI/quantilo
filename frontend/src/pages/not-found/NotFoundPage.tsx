import { Link, useParams, useRouterState } from "@tanstack/react-router";
import { Search } from "lucide-react";
import type { ReactNode } from "react";

import type { Dataset } from "../../api/client";
import { isNotFound, useDataset, useDatasets } from "../../api/hooks";
import { OPEN_PALETTE_EVENT } from "../../features/command/Shortcuts";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { buttonClass } from "../../ui/Button";

/** The datasets to offer instead (081, L-08): ready ones, last opened first, three at most. */
export function recentDatasets(datasets: Dataset[], limit = 3): Dataset[] {
  const stamp = (dataset: Dataset) => dataset.last_opened_at ?? dataset.imported_at ?? dataset.created_at;
  return datasets
    .filter((dataset) => dataset.status === "ready")
    .sort((a, b) => stamp(b).localeCompare(stamp(a)))
    .slice(0, limit);
}

/** An empty chart: axes, ticks and a faint grid with nothing plotted. Decoration only. */
function EmptyAxes() {
  return (
    <svg viewBox="0 0 160 96" width={160} height={96} fill="none" aria-hidden className="text-ink-3">
      {[20, 42, 64].map((y) => (
        <path key={y} d={`M24 ${y}H152`} className="stroke-line" strokeWidth={1} strokeDasharray="3 4" />
      ))}
      <path d="M24 8V84H152" stroke="currentColor" strokeWidth={1.25} strokeLinecap="round" />
      {[56, 88, 120, 152].map((x) => (
        <path key={x} d={`M${x} 84v4`} stroke="currentColor" strokeWidth={1.25} strokeLinecap="round" />
      ))}
      {[20, 42, 64].map((y) => (
        <path key={y} d={`M20 ${y}h4`} stroke="currentColor" strokeWidth={1.25} strokeLinecap="round" />
      ))}
      <circle cx={88} cy={84} r={3} className="fill-accent" />
    </svg>
  );
}

type NotFoundProps = { kind?: "page" | "dataset" };

/** 081: a professional "not found" inside the app's shell. One sentence, no blame, the address
 *  asked for, the way home first, then search and the datasets opened last. */
export function NotFoundPage({ kind = "page" }: NotFoundProps) {
  const address = useRouterState({ select: (state) => state.location.pathname });
  const datasets = useDatasets();
  const recent = recentDatasets(datasets.data ?? []);

  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-6 px-(--gutter) py-10 sm:py-16">
      <EmptyAxes />
      <header>
        <h1 className="text-xl font-semibold tracking-tight text-ink-1">
          {kind === "dataset" ? t("notFound.datasetTitle") : t("notFound.title")}
        </h1>
        <p className="mt-1.5 max-w-prose text-ink-2">
          {kind === "dataset" ? t("notFound.datasetBody") : t("notFound.body")}
        </p>
        <p className="mt-2 text-xs text-ink-3">
          {t("notFound.address")} <code className="font-mono break-all text-ink-2">{address}</code>
        </p>
      </header>
      <div className="flex flex-wrap items-center gap-2">
        <Link to="/" className={buttonClass("primary")}>
          {t("notFound.back")}
        </Link>
        <button
          type="button"
          onClick={() => window.dispatchEvent(new Event(OPEN_PALETTE_EVENT))}
          className={buttonClass("secondary")}
        >
          <Search size={15} strokeWidth={1.5} aria-hidden />
          {t("notFound.search")}
          <kbd className="rounded border border-line px-1 font-mono text-[11px] text-ink-3 max-sm:hidden">Ctrl K</kbd>
        </button>
        <Link to="/help" className={buttonClass("ghost")}>
          {t("notFound.guide")}
        </Link>
      </div>
      {recent.length > 0 && (
        <section aria-labelledby="not-found-recent">
          <h2 id="not-found-recent" className="text-xs text-ink-3">
            {t("notFound.recent")}
          </h2>
          <ul className="mt-2 divide-y divide-line rounded-md border border-line bg-surface-1">
            {recent.map((dataset) => (
              <li key={dataset.id}>
                <Link
                  to="/d/$datasetId"
                  params={{ datasetId: dataset.id }}
                  className="flex min-w-0 items-baseline justify-between gap-4 px-3 py-2 hover:bg-surface-2"
                >
                  <span className="min-w-0 truncate text-ink-1">{dataset.name}</span>
                  {dataset.row_count !== null && (
                    <span className="shrink-0 font-mono text-xs text-ink-3">
                      {t("notFound.rows", { count: formatCount(dataset.row_count) })}
                    </span>
                  )}
                </Link>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

/** Every /d/$datasetId page: an id the API does not know shows "Dataset not found" instead of the
 *  page's own error (L-09). The top bar and the sidebar stay. */
export function DatasetGuard({ children }: { children: ReactNode }) {
  const { datasetId } = useParams({ strict: false });
  const dataset = useDataset(datasetId);
  if (datasetId && isNotFound(dataset.error)) return <NotFoundPage kind="dataset" />;
  return children;
}
