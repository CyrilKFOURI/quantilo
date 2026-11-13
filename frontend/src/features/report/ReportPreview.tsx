import { Printer } from "lucide-react";
import { useRef, useState } from "react";

import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { ErrorNote } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { useReport, useReportHtml } from "./useReport";

/**
 * 025, R-REP-17: the report's print preview in the app, then the browser's own print dialog
 * ("Save as PDF" is one of its printers). The report stays a page without any script (R-REP-01):
 * it is shown in a sandboxed frame (`srcdoc`, no `allow-scripts`); the app asks that frame to
 * print. Its print style sheet (R-REP-16) lays out the pages.
 */
export function ReportPreview({ report }: { report: ReturnType<typeof useReport> }) {
  const [open, setOpen] = useState(false);
  const frame = useRef<HTMLIFrameElement>(null);
  // the report asked by this click, not one finished before it
  const done = open && !report.running && report.job?.status === "done" ? report.job.id : null;
  const html = useReportHtml(done);
  return (
    <>
      <Button
        variant="ghost"
        icon={Printer}
        disabled={report.running}
        title={t("report.printHint")}
        onClick={() => {
          setOpen(true);
          report.startPreview();
        }}
      >
        {t("report.print")}
      </Button>
      {open && (
        <Dialog
          open
          onOpenChange={(next) => {
            if (!next) setOpen(false);
          }}
          size="lg"
          title={t("report.previewTitle")}
          description={t("report.previewHint")}
          footer={
            <>
              <Button variant="ghost" onClick={() => setOpen(false)}>
                {t("report.close")}
              </Button>
              <Button variant="primary" icon={Printer} disabled={!html.data} onClick={() => frame.current?.contentWindow?.print()}>
                {t("report.printNow")}
              </Button>
            </>
          }
        >
          {report.error ? (
            <ErrorNote>{t("report.failed", { error: report.error })}</ErrorNote>
          ) : html.isError ? (
            <ErrorNote>{(html.error as Error).message}</ErrorNote>
          ) : !html.data ? (
            <Loading label={report.job?.message ? t("report.running", { message: report.job.message }) : t("report.starting")} className="h-[60dvh] rounded-md" />
          ) : (
            <iframe
              ref={frame}
              title={t("report.previewTitle")}
              srcDoc={html.data}
              sandbox="allow-same-origin allow-modals"
              className="h-[65dvh] w-full rounded-md border border-line bg-white max-sm:h-[75dvh]"
            />
          )}
        </Dialog>
      )}
    </>
  );
}
