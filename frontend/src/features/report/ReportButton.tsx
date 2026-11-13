import { FileDown } from "lucide-react";

import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { ReportPreview } from "./ReportPreview";
import { useReport } from "./useReport";

/** Progress of a running report, its Cancel, or why it failed (shared by Overview and Data). */
export function ReportProgress({
  report,
}: {
  report: ReturnType<typeof useReport>;
}) {
  if (report.running) {
    const progress = report.job?.progress ?? 0;
    return (
      <span
        className="inline-flex items-center gap-2 text-xs text-ink-2"
        role="status"
      >
        <span className="tabular-nums">
          {report.job?.message
            ? t("report.running", { message: report.job.message })
            : t("report.starting")}
        </span>
        <span
          className="h-1 w-20 overflow-hidden rounded-full bg-surface-2"
          aria-hidden
        >
          <span
            className="block h-full bg-accent transition-[width]"
            style={{ width: `${Math.round(progress * 100)}%` }}
          />
        </span>
        <button
          type="button"
          onClick={report.cancel}
          className="text-accent hover:underline"
        >
          {t("report.cancel")}
        </button>
      </span>
    );
  }
  if (report.error) {
    return (
      <span className="text-xs text-bad" role="alert">
        {t("report.failed", { error: report.error })}
      </span>
    );
  }
  return null;
}

/** "Export report" on the Overview: the whole dataset in one HTML file (025). */
export function ReportButton({ datasetId }: { datasetId: string }) {
  const report = useReport(datasetId);
  return (
    <span className="flex flex-col items-end gap-1">
      <span className="flex flex-wrap items-center justify-end gap-1">
        {/* 025, R-REP-17: the print preview, then the browser's print dialog (PDF) */}
        <ReportPreview report={report} />
        <Button
          icon={FileDown}
          onClick={report.start}
          disabled={report.running}
          title={t("report.hint")}
        >
          {t("report.export")}
        </Button>
      </span>
      <ReportProgress report={report} />
    </span>
  );
}
