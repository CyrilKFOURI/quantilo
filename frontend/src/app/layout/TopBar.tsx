import { Link, useParams } from "@tanstack/react-router";
import { Menu, Search } from "lucide-react";

import { useDataset } from "../../api/hooks";
import { OPEN_PALETTE_EVENT } from "../../features/command/Shortcuts";
import { FilterButton, FilterChips } from "../../features/filters/FilterBar";
import { t } from "../../lib/i18n";
import { useStoredState } from "../../lib/useStoredState";
import { CyfouriLogo } from "../../ui/CyfouriLogo";
import { Logo } from "../../ui/Logo";

/** `onMenu`: on a phone, the button that opens the navigation drawer (041). */
export function TopBar({ onMenu }: { onMenu?: () => void }) {
  const { datasetId } = useParams({ strict: false });
  const dataset = useDataset(datasetId);
  const ready = Boolean(datasetId && dataset.data?.status === "ready");
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
        {datasetId && dataset.data ? (
          <span className="flex min-w-0 flex-1 items-baseline gap-1.5">
            <span className="shrink-0 text-ink-3 max-sm:hidden">{t("topbar.dataset")}</span>
            <span className="truncate font-medium text-ink-1">
              {dataset.data.name}
            </span>
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
