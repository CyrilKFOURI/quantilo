import { locale, t } from "./i18n";

const LOCALE = "en-US";

const integer = new Intl.NumberFormat(LOCALE);
const compact = new Intl.NumberFormat(LOCALE, { notation: "compact", maximumFractionDigits: 1 });
// Relative times follow the chosen language (127); numbers keep LOCALE.
const relatives = new Map<string, Intl.RelativeTimeFormat>();
function relativeFormat(): Intl.RelativeTimeFormat {
  const language = locale() === "fr" ? "fr-FR" : LOCALE;
  let format = relatives.get(language);
  if (!format) {
    format = new Intl.RelativeTimeFormat(language, { numeric: "auto" });
    relatives.set(language, format);
  }
  return format;
}

export function formatCount(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return value >= 100_000 ? compact.format(value) : integer.format(value);
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value < 10 && unit > 0 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`;
}

export function formatRelativeTime(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return "—";
  const seconds = Math.round((new Date(iso).getTime() - now) / 1000);
  const steps: [Intl.RelativeTimeFormatUnit, number][] = [
    ["year", 31_536_000],
    ["month", 2_592_000],
    ["day", 86_400],
    ["hour", 3_600],
    ["minute", 60],
  ];
  for (const [unit, size] of steps) {
    if (Math.abs(seconds) >= size) return relativeFormat().format(Math.round(seconds / size), unit);
  }
  return t("fig.time.justNow");
}

/** Last segment of a file system path (Windows or POSIX). */
export function baseName(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).pop() ?? path;
}
