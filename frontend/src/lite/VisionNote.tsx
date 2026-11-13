import { t } from "../lib/i18n";
import { DESKTOP_ONLY, DESKTOP_ONLY_KINDS, MAX_IMAGES, MAX_MB, MEASURE_SIDE } from "./images";

/** "a, b and c". */
function listing(items: readonly string[]): string {
  if (items.length <= 1) return items.join("");
  return `${items.slice(0, -1).join(", ")} ${t("lite.and")} ${items.at(-1)}`;
}

/** 119: Vision's one line in the browser version: what runs here, what is left to the desktop
 *  app (R-VIS-WEB-01 to 03); each deep option also says "Desktop only" where it is offered. */
export function VisionBrowserNote() {
  return (
    <p className="text-xs text-pretty text-ink-3">
      {t("lite.visionNote", {
        images: MAX_IMAGES.toLocaleString("en-US"),
        mb: String(MAX_MB),
        side: String(MEASURE_SIDE),
        deep: listing([...DESKTOP_ONLY, ...DESKTOP_ONLY_KINDS.map((key) => t(key))]),
      })}
    </p>
  );
}
