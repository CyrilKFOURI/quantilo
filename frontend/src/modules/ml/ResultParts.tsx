import type { ReactNode } from "react";

import type { Metric } from "../../features/variable/Metrics";
import { InfoTip } from "../../ui/InfoTip";

/** Key figures in one band of numbers separated by thin rules (no tiles). */
export function ScoreBand({ items }: { items: Metric[] }) {
  return (
    <dl className="flex flex-wrap border-y border-line">
      {items.map((item, i) => (
        <div key={item.label} className={`min-w-32 flex-1 px-4 py-2.5 ${i > 0 ? "border-l border-line" : "pl-0"}`}>
          <dt className="flex items-center gap-1 text-xs text-ink-3">
            {item.label}
            {item.explain && <InfoTip label={item.label} text={item.explain} />}
          </dt>
          <dd className="text-[22px] leading-tight font-semibold text-ink-1 tabular-nums">{item.value}</dd>
          {item.note && <dd className="text-xs text-ink-3">{item.note}</dd>}
        </div>
      ))}
    </dl>
  );
}

export function Panel({ title, children, wide = false }: { title: string; children: ReactNode; wide?: boolean }) {
  return (
    <figure className={`min-w-0 border-t border-line pt-2 ${wide ? "@6xl:col-span-2" : ""}`}>
      <figcaption className="mb-1 text-xs font-medium text-ink-2">{title}</figcaption>
      {children}
    </figure>
  );
}
