import { Link } from "@tanstack/react-router";
import { CircleCheck, OctagonAlert, TriangleAlert, X, type LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import type { ChangeLevel, Changes, Dataset } from "../../api/client";
import { useChanges, useVersions } from "../../api/hooks";
import { t, type MessageKey } from "../../lib/i18n";
import { useStoredState } from "../../lib/useStoredState";

/** Each level has an icon and a word: the colour is never the only signal. */
export const LEVEL_META: Record<ChangeLevel, { icon: LucideIcon; tone: string; label: MessageKey }> = {
  none: { icon: CircleCheck, tone: "text-ok", label: "changes.level.none" },
  check: { icon: TriangleAlert, tone: "text-warn", label: "changes.level.check" },
  breaking: { icon: OctagonAlert, tone: "text-bad", label: "changes.level.breaking" },
};

export function LevelBadge({ level }: { level: ChangeLevel }) {
  const meta = LEVEL_META[level];
  const Icon = meta.icon;
  return (
    <span className="inline-flex shrink-0 items-center gap-1.5 rounded-md border border-line bg-surface-2 px-2 py-0.5 text-xs font-medium text-ink-1">
      <Icon size={14} strokeWidth={1.75} className={meta.tone} aria-hidden />
      {t(meta.label)}
    </span>
  );
}

/** The one-line summary; `compact` (the overview banner) leaves the reasons to the Changes page. */
export function ChangesSummary({ changes, action, compact = false }: { changes: Changes; action?: ReactNode; compact?: boolean }) {
  return (
    <div className="flex items-start gap-3 rounded-md border border-line bg-surface-1 px-3 py-2.5">
      <LevelBadge level={changes.level} />
      <div className="min-w-0 flex-1">
        <p className="text-ink-1">{changes.headline}</p>
        {!compact && changes.reasons.length > 0 && (
          <ul className="mt-1 space-y-0.5 text-xs text-ink-2">
            {changes.reasons.slice(0, 4).map((reason) => (
              <li key={reason}>{reason}</li>
            ))}
            {changes.reasons.length > 4 && <li className="text-ink-3">{t("changes.moreReasons", { count: String(changes.reasons.length - 4) })}</li>}
          </ul>
        )}
      </div>
      {action}
    </div>
  );
}

/** On the overview of a reimported dataset: what changed since the previous version, until closed. */
export function ChangesBanner({ dataset }: { dataset: Dataset }) {
  const versions = useVersions(dataset);
  const current = versions.data?.find((version) => version.current)?.version ?? 0;
  const [dismissed, setDismissed] = useStoredState(`de.changes.${dataset.id}.dismissed`, 0);
  const show = (versions.data?.length ?? 0) > 1 && current > dismissed;
  const changes = useChanges(dataset, null, show);
  if (!show || !changes.data) return null;
  return (
    <section aria-label={t("changes.bannerLabel")}>
      <ChangesSummary
        compact
        changes={changes.data}
        action={
          <div className="flex shrink-0 items-center gap-1">
            <Link
              to="/d/$datasetId/changes"
              params={{ datasetId: dataset.id }}
              search={{ vs: undefined, tab: undefined, column: undefined }}
              className="inline-flex h-7 items-center rounded-md px-2 text-xs text-accent hover:bg-surface-2 hover:underline"
            >
              {t("changes.seeDetail")}
            </Link>
            <button
              type="button"
              onClick={() => setDismissed(current)}
              aria-label={t("changes.dismiss")}
              title={t("changes.dismiss")}
              className="inline-flex size-7 items-center justify-center rounded-md text-ink-3 hover:bg-surface-2 hover:text-ink-1"
            >
              <X size={14} strokeWidth={1.75} aria-hidden />
            </button>
          </div>
        }
      />
    </section>
  );
}
