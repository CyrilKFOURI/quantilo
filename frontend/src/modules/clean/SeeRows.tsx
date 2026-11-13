import { Link } from "@tanstack/react-router";
import { ArrowRight, Table2 } from "lucide-react";

import { useColumns } from "../../api/hooks";
import { describe } from "../../features/filters/conditions";
import { t } from "../../lib/i18n";
import { Button, buttonClass } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { usePreview, type Step, type Suggestion } from "./api";
import { useReview, type ReviewContextValue } from "./context";
import { PreviewPanel } from "./PreviewTable";
import { filterOf } from "./rows";
import { withSuggestion } from "./steps";
import { suggestionTitle } from "./titles";

/** N41: "See the rows" of a fix: its rows before and after, as the draft would make them (the
 *  steps before it applied first), and, when its step can be written as a filter, the same rows
 *  with every column on the Data page. Nothing is computed until it is opened. */
export function SeeRows({ suggestion, open, onOpenChange }: { suggestion: Suggestion; open: boolean; onOpenChange: (open: boolean) => void }) {
  const context = useReview();
  if (!context || !suggestion.step) return null;
  const title = suggestionTitle(suggestion);
  return (
    <>
      <Button variant="ghost" icon={Table2} iconOnly aria-label={`${t("clean.rows.open")}: ${title}`} title={t("clean.rows.open")} onClick={() => onOpenChange(true)} />
      {open && (
        <Dialog open onOpenChange={onOpenChange} title={t("clean.rows.title")} description={title} size="lg">
          <RowsBody suggestion={suggestion} context={context} />
        </Dialog>
      )}
    </>
  );
}

/** The draft with this fix in it (as it is there, turned on), and its step's id (N35 too). */
export function withFix(suggestion: Suggestion, { steps, found }: ReviewContextValue): { steps: Step[]; step: Step | null; ticked: boolean } {
  const there = steps.find((step) => step.suggestion === suggestion.id);
  const list = there ? steps.map((step) => (step === there ? ({ ...step, enabled: true } as Step) : step)) : withSuggestion(steps, suggestion, found);
  return { steps: list, step: list.find((step) => step.suggestion === suggestion.id) ?? null, ticked: Boolean(there) };
}

function RowsBody({ suggestion, context }: { suggestion: Suggestion; context: ReviewContextValue }) {
  const { datasetId } = context;
  const fix = withFix(suggestion, context);
  const preview = usePreview(datasetId, fix.steps, fix.step?.id ?? null);
  const columns = useColumns(datasetId, true);
  const types = Object.fromEntries((columns.data ?? []).map((column) => [column.name, column.type]));
  const conditions = fix.step && columns.data ? filterOf(fix.step, types) : null;
  return (
    <div className="space-y-3">
      <p className="text-xs text-ink-2">{t(fix.ticked ? "clean.rows.ticked" : "clean.rows.unticked")}</p>
      <PreviewPanel title={t("clean.rows.preview")} preview={preview.data} loading={preview.isFetching} error={preview.error} />
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-t border-line pt-3">
        {conditions ? (
          <>
            <Link to="/d/$datasetId/data" params={{ datasetId }} search={{ filters: { dataset: datasetId, conditions } }} className={buttonClass("secondary")}>
              {t("clean.rows.data")}
              <ArrowRight size={14} strokeWidth={1.5} aria-hidden />
            </Link>
            <p className="min-w-0 flex-1 basis-64 text-xs text-ink-3">{t("clean.rows.dataHint", { filter: conditions.map(describe).join(" · ") })}</p>
          </>
        ) : (
          columns.data && <p className="text-xs text-ink-3">{t("clean.rows.noData")}</p>
        )}
      </div>
    </div>
  );
}
