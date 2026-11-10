import { Info, Scale, TriangleAlert } from "lucide-react";

import { t } from "../../lib/i18n";
import type { KpiSignal } from "./signals";

/** 054: one line per signal, an icon and its sentence (rule included); never colour alone. */
export function SignalLines({ signals, label }: { signals: KpiSignal[]; label: string }) {
  if (!signals.length) return null;
  return (
    <ul className="space-y-1" aria-label={label}>
      {signals.map((signal, i) => {
        const Icon = signal.scope === "change" ? Scale : signal.severity === "warning" ? TriangleAlert : Info;
        return (
          <li key={`${signal.rule}-${i}`} className="flex items-start gap-1.5 text-xs text-ink-2">
            <Icon
              size={13}
              strokeWidth={1.75}
              aria-hidden
              className={`mt-px shrink-0 ${signal.severity === "warning" ? "text-warn" : "text-ink-3"}`}
            />
            <span className="min-w-0">
              {signal.severity === "warning" && <span className="sr-only">{t("kpi.signal.warning")} </span>}
              {signal.message}
            </span>
          </li>
        );
      })}
    </ul>
  );
}
