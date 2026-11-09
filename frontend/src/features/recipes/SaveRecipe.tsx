import { Link, useNavigate } from "@tanstack/react-router";
import { BookmarkPlus, X } from "lucide-react";
import { useState } from "react";

import {
  useCompatibleRecipes,
  useCreateRecipe,
  useRecipeDraft,
  type RecipeStepView,
} from "../../api/recipes";
import { t, type MessageKey } from "../../lib/i18n";
import { useStoredState } from "../../lib/useStoredState";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { Checkbox, ErrorNote, Field, TextInput } from "../../ui/fields";
import { Loading } from "../../ui/Loading";

const GROUPS: { kind: string; label: MessageKey }[] = [
  { kind: "column", label: "recipes.kind.column" },
  { kind: "filter", label: "recipes.kind.filter" },
  { kind: "kpi", label: "recipes.kind.kpi" },
  { kind: "timeseries", label: "recipes.kind.timeseries" },
  { kind: "ml", label: "recipes.kind.ml" },
];

/** Overview › Save as recipe: the dataset's kept work, every step ticked (030). */
export function SaveRecipeButton({ datasetId, datasetName }: { datasetId: string; datasetName: string }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Button icon={BookmarkPlus} onClick={() => setOpen(true)}>
        {t("recipes.save")}
      </Button>
      {open && <SaveRecipeDialog datasetId={datasetId} datasetName={datasetName} onClose={() => setOpen(false)} />}
    </>
  );
}

function SaveRecipeDialog({
  datasetId,
  datasetName,
  onClose,
}: {
  datasetId: string;
  datasetName: string;
  onClose: () => void;
}) {
  const draft = useRecipeDraft(datasetId, true);
  const create = useCreateRecipe();
  const navigate = useNavigate();
  const [name, setName] = useState(datasetName);
  const [unticked, setUnticked] = useState<Set<string>>(new Set());
  const [active, setActive] = useState<string | null>(null);
  const [report, setReport] = useState(false);
  const [csv, setCsv] = useState(false);
  const steps = (draft.data?.steps ?? []).filter((step) => step.kind !== "output");
  const keep = steps.filter((step) => !unticked.has(step.id)).map((step) => step.id);
  const toggle = (id: string) =>
    setUnticked((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const save = () =>
    create.mutate(
      {
        name: name.trim(),
        dataset_id: datasetId,
        keep,
        active_filter: active && keep.includes(active) ? active : null,
        report,
        csv,
      },
      {
        onSuccess: (recipe) => {
          onClose();
          void navigate({ to: "/recipes/$recipeId", params: { recipeId: recipe.id } });
        },
      },
    );

  return (
    <Dialog
      open
      onOpenChange={(open) => !open && onClose()}
      title={t("recipes.save.title")}
      description={t("recipes.save.description")}
      size="md"
      footer={
        <div className="flex w-full items-center justify-between gap-3">
          <span className="text-xs text-ink-3">{t("recipes.save.count", { count: String(keep.length) })}</span>
          <div className="flex gap-2">
            <Button onClick={onClose}>{t("recipes.cancel")}</Button>
            <Button
              variant="primary"
              disabled={!name.trim() || (keep.length === 0 && !report && !csv) || create.isPending}
              onClick={save}
            >
              {t("recipes.save.confirm")}
            </Button>
          </div>
        </div>
      }
    >
      <div className="space-y-4">
        <Field label={t("recipes.save.name")}>
          <TextInput value={name} maxLength={200} onChange={(event) => setName(event.target.value)} />
        </Field>
        {draft.isError && <ErrorNote>{draft.error.message}</ErrorNote>}
        {create.isError && <ErrorNote>{create.error.message}</ErrorNote>}
        {!draft.data && !draft.isError && <Loading label={t("loading.draft")} className="h-32 rounded-md" />}
        {draft.data && (
          <>
            {GROUPS.map((group) => {
              const items = steps.filter((step) => step.kind === group.kind);
              return (
                <fieldset key={group.kind} className="space-y-1">
                  <legend className="mb-1 text-xs text-ink-3">{t(group.label)}</legend>
                  {items.length === 0 ? (
                    <p className="text-xs text-ink-3">{t(`recipes.save.none.${group.kind}` as MessageKey)}</p>
                  ) : (
                    items.map((step) => (
                      <StepChoice
                        key={step.id}
                        step={step}
                        checked={!unticked.has(step.id)}
                        onToggle={() => toggle(step.id)}
                        active={active === step.id}
                        onActive={group.kind === "filter" ? () => setActive(active === step.id ? null : step.id) : undefined}
                      />
                    ))
                  )}
                </fieldset>
              );
            })}
            <fieldset className="space-y-1">
              <legend className="mb-1 text-xs text-ink-3">{t("recipes.kind.output")}</legend>
              <Checkbox checked={report} onChange={(event) => setReport(event.target.checked)} label={t("recipes.save.report")} />
              <br />
              <Checkbox checked={csv} onChange={(event) => setCsv(event.target.checked)} label={t("recipes.save.csv")} />
            </fieldset>
          </>
        )}
      </div>
    </Dialog>
  );
}

function StepChoice({
  step,
  checked,
  onToggle,
  active,
  onActive,
}: {
  step: RecipeStepView;
  checked: boolean;
  onToggle: () => void;
  active: boolean;
  onActive?: () => void;
}) {
  return (
    <div className="flex items-start justify-between gap-3">
      <Checkbox checked={checked} onChange={onToggle} label={<span className="text-ink-1">{step.sentence}</span>} className="items-start" />
      {onActive && (
        <label className="inline-flex shrink-0 cursor-pointer items-center gap-1.5 text-xs text-ink-2">
          <input
            type="radio"
            name="active-filter"
            checked={active}
            disabled={!checked}
            onChange={onActive}
            onClick={() => active && onActive()}
            className="accent-(--accent)"
          />
          {t("recipes.save.active")}
        </label>
      )}
    </div>
  );
}

/** On the overview of a newly imported file: the recipes it fits, until dismissed. */
export function RecipeBanner({ datasetId }: { datasetId: string }) {
  const [dismissed, setDismissed] = useStoredState<string[]>(`de.recipes.${datasetId}.dismissed`, []);
  const compatible = useCompatibleRecipes(datasetId, true);
  const best = compatible.data?.find((recipe) => !dismissed.includes(recipe.recipe_id));
  if (!best) return null;
  return (
    <section
      aria-label={t("recipes.banner.label")}
      className="flex items-center gap-3 rounded-md border border-line bg-surface-1 px-3 py-2.5"
    >
      <BookmarkPlus size={15} strokeWidth={1.5} className="shrink-0 text-ink-2" aria-hidden />
      <p className="min-w-0 flex-1 text-ink-1">
        {t("recipes.banner.text", {
          name: best.name,
          matched: String(best.matched),
          total: String(best.total),
        })}
      </p>
      <Link
        to="/recipes/$recipeId"
        params={{ recipeId: best.recipe_id }}
        search={{ run: datasetId, report: undefined }}
        className="inline-flex h-7 shrink-0 items-center rounded-md px-2 text-xs text-accent hover:bg-surface-2 hover:underline"
      >
        {t("recipes.banner.run")}
      </Link>
      <button
        type="button"
        onClick={() => setDismissed([...dismissed, best.recipe_id])}
        aria-label={t("recipes.banner.dismiss")}
        title={t("recipes.banner.dismiss")}
        className="inline-flex size-7 shrink-0 items-center justify-center rounded-md text-ink-3 hover:bg-surface-2 hover:text-ink-1"
      >
        <X size={14} strokeWidth={1.75} aria-hidden />
      </button>
    </section>
  );
}
