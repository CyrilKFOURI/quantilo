import { Link } from "@tanstack/react-router";
import { Menu, Search, X } from "lucide-react";

import { useDataset } from "../../api/hooks";
import { OPEN_PALETTE_EVENT } from "../../features/command/Shortcuts";
import { closeCurrentDataset, useCurrentDataset } from "../../features/datasets/current";
import { CleanedVersionChip } from "../../features/lineage/CleanedVersionChip";
import { FilterButton, FilterChips } from "../../features/filters/FilterBar";
import { t } from "../../lib/i18n";
import { useStoredState } from "../../lib/useStoredState";
import { DatasetSwitcher } from "../../features/projects/DatasetSwitcher";
import { CyfouriLogo } from "../../ui/CyfouriLogo";
import { Logo } from "../../ui/Logo";

/** `onMenu`: on a phone, the button that opens the navigation drawer (041). `appMark`: the sidebar
 *  is an icon rail (tablet, or collapsed) showing CYFOURI only, so the app's mark and name come
 *  here, next to it (043). */
export function TopBar({ onMenu, appMark = false }: { onMenu?: () => void; appMark?: boolean }) {
  // 098: Home and Recipes keep showing the dataset one works on (filters only on its own pages)
  const { datasetId, routed } = useCurrentDataset();
  const dataset = useDataset(datasetId);
  const ready = Boolean(routed && datasetId && dataset.data?.status === "ready");
  const [showChips, setShowChips] = useStoredState("de.filters.chips", false);

  return (
    <>
      <header className="flex h-(--topbar-h) shrink-0 items-center justify-between gap-4 border-b border-line bg-surface-0 px-6 max-sm:box-content max-sm:gap-2 max-sm:px-2 max-sm:pt-[env(safe-area-inset-top)]">
        {onMenu && (
          <button
            type="button"
            onClick={onMenu}
            aria-label={t("nav.openMenu")}
            className="inline-flex size-9 shrink-0 items-center justify-center rounded-md text-ink-2 hover:bg-surface-2 hover:text-ink-1"
          >
            <Menu size={18} strokeWidth={1.5} aria-hidden />
          </button>
        )}
        {onMenu && (
          // The sidebar (and its logos) is in the drawer on a phone: CYFOURI, then the app's mark, stay in sight here.
          <Link to="/" aria-label={t("app.name")} className="flex shrink-0 items-center gap-2 text-ink-1">
            <CyfouriLogo size={16} />
            <Logo size={20} />
          </Link>
        )}
        {appMark && (
          <Link to="/" className="flex shrink-0 items-center gap-2 text-ink-1">
            <Logo size={20} />
            <span className="text-sm font-semibold tracking-tight">{t("app.name")}</span>
            <span className="h-5 w-px bg-line" aria-hidden />
          </Link>
        )}
        {datasetId && dataset.data ? (
          <span className="flex min-w-0 flex-1 items-baseline gap-1.5">
            <span className="shrink-0 text-ink-3 max-sm:hidden">{t("topbar.dataset")}</span>
            <DatasetSwitcher datasetId={datasetId} name={dataset.data.name} />
            {ready && <CleanedVersionChip datasetId={datasetId} />}
            {!routed && (
              <button
                type="button"
                onClick={closeCurrentDataset}
                aria-label={t("topbar.closeDataset")}
                title={t("topbar.closeDataset")}
                className="inline-flex size-6 shrink-0 items-center justify-center self-center rounded-md text-ink-3 hover:bg-surface-2 hover:text-ink-1"
              >
                <X size={14} strokeWidth={1.5} aria-hidden />
              </button>
            )}
          </span>
        ) : datasetId ? (
          // Still loading (042): the name comes in a moment, "No dataset open" would be wrong.
          <span className="min-w-0 flex-1" />
        ) : (
          <span className="min-w-0 flex-1 truncate text-ink-3">{t("topbar.noDataset")}</span>
        )}
        <div className="flex shrink-0 items-center gap-2.5 max-sm:gap-1.5">
          <button
            type="button"
            onClick={() => window.dispatchEvent(new Event(OPEN_PALETTE_EVENT))}
            aria-label={t("palette.open")}
            className="inline-flex h-(--control-h) items-center gap-2 rounded-md border border-line bg-surface-1 px-2.5 text-[13px] text-ink-3 hover:bg-surface-2 hover:text-ink-1"
          >
            <Search size={14} strokeWidth={1.5} aria-hidden />
            <span className="hidden md:inline">{t("palette.open")}</span>
            <kbd className="hidden rounded border border-line px-1 text-[11px] md:inline">
              Ctrl K
            </kbd>
          </button>
          {ready && datasetId && (
            <FilterButton
              datasetId={datasetId}
              showChips={showChips}
              onShowChips={setShowChips}
            />
          )}
        </div>
      </header>
      {ready && datasetId && showChips && <FilterChips datasetId={datasetId} />}
    </>
  );
}
