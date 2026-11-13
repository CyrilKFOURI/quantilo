import { Lightbulb } from "lucide-react";
import { useState } from "react";

import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { ErrorNote, TextInput } from "../../ui/fields";
import { useSaveConcept } from "./moreApi";

/**
 * 118: the gallery's selection saved as a named concept (R-VCB-01, R-VTC-01): the concept
 * bottleneck can learn it (Model › Advanced › More) and TCAV can test it (Explain › More). The
 * images of the group have the concept; the others are taken as without it, said in the dialog.
 */
export function SaveConcept({ datasetId, imageIds }: { datasetId: string; imageIds: readonly string[] }) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const save = useSaveConcept(datasetId);
  const distinct = [...new Set(imageIds)];
  const submit = () =>
    save.mutate(
      { name: name.trim(), image_ids: distinct },
      {
        onSuccess: () => {
          setOpen(false);
          setName("");
        },
      },
    );
  return (
    <>
      <Button variant="ghost" icon={Lightbulb} onClick={() => setOpen(true)}>
        {t("vision.more.saveConcept")}
      </Button>
      {open && (
        <Dialog
          open
          onOpenChange={(next) => !next && setOpen(false)}
          size="sm"
          title={distinct.length === 1 ? t("vision.more.saveTitleOne") : t("vision.more.saveTitle", { count: formatCount(distinct.length) })}
          description={t("vision.more.saveBody")}
          footer={
            <>
              <Button variant="ghost" onClick={() => setOpen(false)}>
                {t("vision.model.cancel")}
              </Button>
              <Button variant="primary" icon={Lightbulb} onClick={submit} disabled={!name.trim() || save.isPending}>
                {t("vision.more.saveConfirm")}
              </Button>
            </>
          }
        >
          <form
            className="flex flex-col gap-3 text-[13px]"
            onSubmit={(event) => {
              event.preventDefault();
              if (name.trim()) submit();
            }}
          >
            <label className="flex flex-col gap-1">
              <span className="text-xs text-ink-2">{t("vision.more.saveName")}</span>
              <TextInput value={name} maxLength={60} autoFocus placeholder={t("vision.more.savePlaceholder")} onChange={(event) => setName(event.target.value)} />
            </label>
            <p className="text-xs text-pretty text-ink-3">{distinct.length < 5 ? t("vision.more.saveTooFew", { count: formatCount(distinct.length) }) : distinct.length < 20 ? t("vision.more.saveFew", { count: formatCount(distinct.length) }) : t("vision.more.saveEnough")}</p>
            {save.isError && <ErrorNote>{(save.error as Error).message}</ErrorNote>}
          </form>
        </Dialog>
      )}
    </>
  );
}
