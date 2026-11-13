import { Download } from "lucide-react";

import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { ErrorNote } from "../../ui/fields";
import { aiTitle, type AiModel } from "./forecastAi";

/** 028 B1: nothing is downloaded before this confirmation, which shows the size, the licence and
 *  the Hugging Face source at its pinned revision, and says that the data stays here. */
export function DownloadDialog({ model, onClose, onConfirm }: { model: AiModel; onClose: () => void; onConfirm: (key: string) => void }) {
  const title = aiTitle(model.key);
  return (
    <Dialog
      open
      onOpenChange={(open) => !open && onClose()}
      size="sm"
      title={t("fcai.download.title", { title })}
      description={t("fcai.download.body", { title, size: model.size_text, licence: model.licence })}
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            {t("fcai.download.cancel")}
          </Button>
          <Button
            variant="primary"
            icon={Download}
            onClick={() => {
              onConfirm(model.key);
              onClose();
            }}
          >
            {t("fcai.download.confirm", { size: model.size_text })}
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-2 text-[13px] text-ink-2">
        <p className="font-medium text-ink-1">{t("fcai.download.safety")}</p>
        <p className="break-all text-xs text-ink-3">{t("fcai.download.source", { source: model.source })}</p>
        <p className="break-all text-xs text-ink-3">{t("fcai.download.where", { folder: model.folder })}</p>
      </div>
    </Dialog>
  );
}

/** The downloads of the chosen models: progress and Stop, or why one failed. */
export function DownloadProgress({ models, onStop }: { models: AiModel[]; onStop: (key: string) => void }) {
  const shown = models.filter((m) => m.download === "running" || m.download === "failed");
  if (!shown.length) return null;
  return (
    <div className="space-y-1">
      {shown.map((m) =>
        m.download === "failed" ? (
          <ErrorNote key={m.key}>{t("fcai.download.failed", { title: aiTitle(m.key), error: m.error ?? "" })}</ErrorNote>
        ) : (
          <p key={m.key} className="flex flex-wrap items-center gap-2 text-xs text-ink-2" role="status">
            <span className="tabular-nums">{t("fcai.download.progress", { title: aiTitle(m.key), pct: String(Math.round(m.progress * 100)) })}</span>
            <Button variant="ghost" onClick={() => onStop(m.key)}>
              {t("fcai.download.stop")}
            </Button>
          </p>
        ),
      )}
    </div>
  );
}
