import { EyeOff, RotateCcw } from "lucide-react";
import { useId, useState } from "react";

import { formatCount, formatRelativeTime } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { ErrorNote, Field, TextInput } from "../../ui/fields";
import { useDismiss, useRestore, type CleanDismissal, type DismissScope, type HiddenSuggestion, type Suggestion } from "./api";
import { useReview } from "./context";
import { suggestionTitle } from "./titles";

/** N05: the reaches a dismissal can have, from the narrowest (the default) to the widest. A
 *  column reach needs the suggestion to be about one column. */
function scopesOf(suggestion: Suggestion): DismissScope[] {
  return suggestion.columns.length === 1 ? ["suggestion", "rule", "column", "everywhere"] : ["suggestion", "rule", "everywhere"];
}

/** "Don't suggest again" on a suggestion: a dialog to choose how far (this suggestion here, its
 *  rule here, its rule on this column name in every dataset, its rule everywhere) and say why.
 *  What it hides leaves the draft, is never applied for you and stays listed under Dismissed. */
export function DismissButton({ suggestion }: { suggestion: Suggestion }) {
  const context = useReview();
  const [open, setOpen] = useState(false);
  if (!context) return null;
  const title = suggestionTitle(suggestion);
  return (
    <>
      <Button variant="ghost" icon={EyeOff} iconOnly aria-label={`${t("clean.dismiss.open")}: ${title}`} title={t("clean.dismiss.open")} onClick={() => setOpen(true)} />
      {open && <DismissDialog suggestion={suggestion} datasetId={context.datasetId} onHide={context.onHide} onClose={() => setOpen(false)} />}
    </>
  );
}

function DismissDialog({ suggestion, datasetId, onHide, onClose }: { suggestion: Suggestion; datasetId: string; onHide?: (ids: string[]) => void; onClose: () => void }) {
  const [scope, setScope] = useState<DismissScope>("suggestion");
  const [note, setNote] = useState("");
  const name = useId();
  const dismiss = useDismiss(datasetId);
  const column = suggestion.columns[0] ?? "";
  const submit = () =>
    dismiss.mutate(
      { scope, rule: suggestion.rule, column: column || null, suggestion: suggestion.id, title: suggestionTitle(suggestion).slice(0, 300), note: note.trim() || null },
      {
        onSuccess: (done) => {
          onHide?.(done.hidden);
          onClose();
        },
      },
    );
  return (
    <Dialog
      open
      onOpenChange={(open) => !open && onClose()}
      title={t("clean.dismiss.title")}
      description={suggestionTitle(suggestion)}
      size="sm"
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            {t("import.cancel")}
          </Button>
          <Button variant="primary" icon={EyeOff} disabled={dismiss.isPending} onClick={submit}>
            {t("clean.dismiss.confirm")}
          </Button>
        </>
      }
    >
      <div className="space-y-3 text-[13px]">
        <p className="text-ink-2">{t("clean.dismiss.body")}</p>
        <fieldset className="space-y-1.5">
          <legend className="mb-1 text-xs text-ink-2">{t("clean.dismiss.scope")}</legend>
          {scopesOf(suggestion).map((option) => (
            <label key={option} className="flex min-h-8 cursor-pointer items-start gap-2">
              <input type="radio" name={name} className="mt-0.5 accent-(--accent)" checked={scope === option} onChange={() => setScope(option)} />
              <span className="text-ink-1">{t(`clean.dismiss.scope.${option}` as MessageKey, { rule: suggestion.rule, column })}</span>
            </label>
          ))}
        </fieldset>
        <Field label={t("clean.dismiss.note")}>
          <TextInput value={note} maxLength={300} placeholder={t("clean.dismiss.notePlaceholder")} onChange={(event) => setNote(event.target.value)} />
        </Field>
        {dismiss.error && <ErrorNote>{dismiss.error.message}</ErrorNote>}
      </div>
    </Dialog>
  );
}

/** N05, under Dismissed: each dismissal that applies here, its reach, the suggestions it hides
 *  now (or what it hid when it was made), the user's reason, and Suggest again. */
export function DismissedList({ datasetId, dismissals, hidden }: { datasetId: string; dismissals: CleanDismissal[]; hidden: HiddenSuggestion[] }) {
  const restore = useRestore(datasetId);
  return (
    <div className="space-y-2 p-3">
      <p className="max-w-prose text-xs text-ink-2">{t("clean.dismissed.hint")}</p>
      <ul className="divide-y divide-line rounded-md border border-line">
        {dismissals.map((dismissal) => {
          const hides = hidden.filter((item) => item.dismissal === dismissal.id).map((item) => suggestionTitle(item.suggestion));
          const what = hides.length ? hides.slice(0, 3).join(" · ") + (hides.length > 3 ? ` · ${t("clean.dismissed.more", { count: formatCount(hides.length - 3) })}` : "") : (dismissal.title ?? "");
          const reach = t(`clean.dismissed.reach.${dismissal.scope}` as MessageKey, { rule: dismissal.rule, column: dismissal.column ?? "" });
          return (
            <li key={dismissal.id} className="flex flex-wrap items-start gap-x-3 gap-y-1 px-3 py-2">
              <div className="min-w-0 flex-1 basis-64 space-y-0.5">
                <p className="text-[13px] text-ink-1">{what || reach}</p>
                <p className="text-xs text-ink-2">
                  <span className="font-mono text-ink-3">{dismissal.rule}</span>
                  {" · "}
                  {reach}
                  {" · "}
                  {hides.length ? t("clean.dismissed.hides", { count: formatCount(hides.length) }) : t("clean.dismissed.hidesNone")}
                </p>
                {dismissal.note && <p className="text-xs text-ink-2">{t("clean.dismissed.note", { note: dismissal.note })}</p>}
                <p className="text-[11px] text-ink-3">{t("clean.dismissed.made", { when: formatRelativeTime(dismissal.created_at) })}</p>
              </div>
              <Button variant="ghost" icon={RotateCcw} disabled={restore.isPending} aria-label={`${t("clean.dismissed.restore")}: ${what || reach}`} onClick={() => restore.mutate(dismissal.id)}>
                {t("clean.dismissed.restore")}
              </Button>
            </li>
          );
        })}
      </ul>
      {restore.error && <ErrorNote>{restore.error.message}</ErrorNote>}
    </div>
  );
}
