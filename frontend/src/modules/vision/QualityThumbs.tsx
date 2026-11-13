import { Ban } from "lucide-react";

import { imageUrl } from "../../features/import/imageSetApi";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { isExcluded, type Draft, type ImageRow } from "./model";

const THUMBS = 24;

/** Thumbnails of table rows (108): a click opens the image sheet; excluded ones are faded.
 *  `captions` (b2, R-VQ-16) replaces the file name under a thumbnail. */
export function Thumbs({
  datasetId,
  rows,
  positions,
  draft,
  onOpen,
  captions,
}: {
  datasetId: string;
  rows: readonly ImageRow[];
  positions: readonly number[];
  draft: Draft;
  onOpen: (row: ImageRow) => void;
  captions?: ReadonlyMap<number, string>;
}) {
  const members = positions
    .slice(0, THUMBS)
    .map((position) => ({ position, row: rows[position] }))
    .filter((item): item is { position: number; row: ImageRow } => item.row !== undefined);
  return (
    <ul className="flex flex-wrap gap-2 pt-1">
      {members.map(({ position, row }) => {
        const excluded = isExcluded(row, draft);
        return (
          <li key={row.key}>
            <button type="button" onClick={() => onOpen(row)} title={row.key} className="group/thumb block w-[76px] text-left">
              <span className="relative block size-[76px] overflow-hidden rounded-md bg-surface-2">
                <img
                  src={imageUrl(datasetId, row.id, "thumb")}
                  alt={[row.name, row.label].filter(Boolean).join(", ")}
                  loading="lazy"
                  decoding="async"
                  className={`size-full object-contain ${excluded ? "opacity-35" : ""}`}
                />
                <span aria-hidden className="pointer-events-none absolute inset-0 rounded-md ring-1 ring-ink-1/8 ring-inset transition-shadow group-hover/thumb:ring-2 group-hover/thumb:ring-accent" />
                {excluded && (
                  <Ban size={12} strokeWidth={1.75} aria-label={t("vision.tile.excluded")} className="absolute right-1 bottom-1 rounded-sm bg-surface-1/95 text-ink-1" />
                )}
              </span>
              <span className="mt-1 block truncate font-mono text-[10px] text-ink-2">{captions?.get(position) ?? row.name}</span>
            </button>
          </li>
        );
      })}
      {positions.length > THUMBS && (
        <li className="flex size-[76px] items-center justify-center rounded-md border border-dashed border-line text-xs text-ink-3 tabular-nums">
          +{formatCount(positions.length - THUMBS)}
        </li>
      )}
    </ul>
  );
}
