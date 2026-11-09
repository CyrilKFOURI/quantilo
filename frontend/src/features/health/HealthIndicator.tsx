import { useHealth } from "../../api/hooks";
import { t } from "../../lib/i18n";

/** Says something only when the backend is down or degraded; a working app stays silent. */
export function HealthIndicator() {
  const { data, isError } = useHealth();
  if (!isError && (!data || data.status === "ok")) return null;
  const down = isError;

  return (
    <p role="status" className="flex items-center gap-2 text-xs text-ink-2">
      <span aria-hidden className={`size-1.5 rounded-full ${down ? "bg-bad" : "bg-warn"}`} />
      {down ? t("health.down") : t("health.degraded")}
    </p>
  );
}
