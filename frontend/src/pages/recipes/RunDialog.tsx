import { Play } from "lucide-react";
import { useMemo, useState } from "react";

import { useDatasets } from "../../api/hooks";
import {
  useRecipeActions,
  useRecipePlan,
  type ColumnMatch,
  type Mapping,
  type RecipeDetail,
} from "../../api/recipes";
import { ImportPanel } from "../../features/import/ImportPanel";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { Checkbox, ErrorNote, Field, Select } from "../../ui/fields";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { StatusTag } from "./parts";

const MATCH_LABEL: Record<string, MessageKey> = {
  same: "recipes.match.same",
  matched: "recipes.match.matched",
  proposed: "recipes.match.proposed",
  missing: "recipes.match.missing",
};

type Props = {
  recipe: RecipeDetail;
  initialDataset: string | null;
  onClose: () => void;
  onStarted: (runId: string) => void;
};

/** Run on…: which dataset (or a new file), how its columns pair with the recipe's, then run. */
export function RunDialog({ recipe, initialDataset, onClose, onStarted }: Props) {
  const datasets = useDatasets();
  const [source, setSource] = useState<"dataset" | "file">("dataset");
  const [datasetId, setDatasetId] = useState<string | null>(initialDataset);
  const [mapping, setMapping] = useState<Mapping | null>(null);
  const plan = useRecipePlan(recipe.id, datasetId, mapping);
  const { run } = useRecipeActions(recipe.id);
  const ready = (datasets.data ?? []).filter((dataset) => dataset.status === "ready");

  const current = plan.data?.mapping ?? {};
  const choose = (recipeColumn: string, fileColumn: string | null) =>
    setMapping({ ...current, [recipeColumn]: fileColumn });
  const skipped = plan.data?.steps.filter((step) => step.skip && step.enabled) ?? [];
  const fileColumns = useMemo(() => {
    const plain = plan.data?.matching.columns.flatMap((m) => (m.column ? [m.column] : [])) ?? [];
    return [...plain, ...(plan.data?.matching.new_columns ?? [])];
  }, [plan.data]);

  const start = () => {
    if (!datasetId) return;
    run.mutate(
      { datasetId, mapping: plan.data?.mapping ?? mapping },
      { onSuccess: (started) => onStarted(started.id) },
    );
  };

  return (
    <Dialog
      open
      onOpenChange={(open) => !open && !run.isPending && onClose()}
      title={t("recipes.run.title", { name: recipe.name })}
      description={t("recipes.run.description")}
      size="lg"
      footer={
        <div className="flex w-full items-center justify-between gap-3">
          <span className="text-xs text-ink-3">
            {plan.data
              ? skipped.length
                ? t("recipes.run.willSkip", { count: String(skipped.length) })
                : t("recipes.run.allRun")
              : ""}
          </span>
          <div className="flex gap-2">
            <Button onClick={onClose}>{t("recipes.cancel")}</Button>
            <Button variant="primary" icon={Play} disabled={!plan.data || run.isPending} onClick={start}>
              {t("recipes.run.start")}
            </Button>
          </div>
        </div>
      }
    >
      <div className="space-y-5">
        <SegmentedControl
          label={t("recipes.run.on")}
          value={source}
          onChange={setSource}
          options={[
            { value: "dataset", label: t("recipes.run.aDataset") },
            { value: "file", label: t("recipes.run.aFile") },
          ]}
        />
        {source === "dataset" ? (
          <Field label={t("recipes.run.dataset")}>
            <Select
              value={datasetId ?? ""}
              onChange={(event) => {
                setDatasetId(event.target.value || null);
                setMapping(null);
              }}
            >
              <option value="">{t("recipes.run.pick")}</option>
              {ready.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.name}
                  {dataset.id === recipe.source_dataset_id ? ` (${t("recipes.run.madeHere")})` : ""}
                </option>
              ))}
            </Select>
          </Field>
        ) : (
          <ImportPanel
            compact
            onImported={(id) => {
              setDatasetId(id);
              setMapping(null);
              setSource("dataset");
            }}
          />
        )}

        {plan.isError && <ErrorNote>{plan.error.message}</ErrorNote>}
        {run.isError && <ErrorNote>{run.error.message}</ErrorNote>}
        {plan.data && (
          <>
            <section aria-label={t("recipes.run.columns")}>
              <h3 className="mb-2 text-xs text-ink-3">{t("recipes.run.columns")}</h3>
              <div className="max-h-72 overflow-auto rounded-md border border-line">
                <table className="w-full border-collapse text-[13px]">
                  <thead className="sticky top-0 bg-surface-1">
                    <tr className="border-b border-line text-left text-xs text-ink-3">
                      <th scope="col" className="px-3 py-1.5 font-normal">{t("recipes.run.recipeColumn")}</th>
                      <th scope="col" className="px-3 py-1.5 font-normal">{t("recipes.run.fileColumn")}</th>
                      <th scope="col" className="px-3 py-1.5 font-normal">{t("recipes.run.match")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {plan.data.matching.columns.map((match) => (
                      <MatchRow
                        key={match.recipe}
                        match={match}
                        chosen={current[match.recipe] ?? null}
                        options={fileColumns}
                        onChoose={(column) => choose(match.recipe, column)}
                      />
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
            {plan.data.matching.new_columns.length > 0 && (
              <p className="text-xs text-ink-2">
                {t("recipes.run.newColumns")}{" "}
                <span className="font-mono">{plan.data.matching.new_columns.join(", ")}</span>
              </p>
            )}
            {skipped.length > 0 && (
              <section aria-label={t("recipes.run.skippedTitle")}>
                <h3 className="mb-1 text-xs text-ink-3">{t("recipes.run.skippedTitle")}</h3>
                <ul className="space-y-1">
                  {skipped.map((step) => (
                    <li key={step.id} className="flex items-start gap-2">
                      <StatusTag status="skipped" />
                      <span className="text-ink-1">
                        {step.sentence}
                        <span className="block text-xs text-ink-3">{step.skip}</span>
                      </span>
                    </li>
                  ))}
                </ul>
              </section>
            )}
          </>
        )}
      </div>
    </Dialog>
  );
}

function MatchRow({
  match,
  chosen,
  options,
  onChoose,
}: {
  match: ColumnMatch;
  chosen: string | null;
  options: string[];
  onChoose: (column: string | null) => void;
}) {
  const proposed = match.status === "proposed";
  return (
    <tr className="border-b border-line last:border-0">
      <td className="px-3 py-1.5">
        <span className="font-mono text-ink-1">{match.recipe}</span>
      </td>
      <td className="px-3 py-1.5">
        <Select
          aria-label={t("recipes.run.fileColumnFor", { column: match.recipe })}
          value={chosen ?? ""}
          onChange={(event) => onChoose(event.target.value || null)}
          className="w-full font-mono"
        >
          <option value="">{t("recipes.run.none")}</option>
          {[...new Set([...(match.column ? [match.column] : []), ...options])].map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </Select>
      </td>
      <td className="px-3 py-1.5 whitespace-nowrap">
        {proposed ? (
          <Checkbox
            checked={chosen === match.column}
            onChange={(event) => onChoose(event.target.checked ? match.column : null)}
            label={
              <span className="text-xs text-warn">
                {t("recipes.match.proposed")} · {t("recipes.run.accept")}
              </span>
            }
          />
        ) : (
          <span
            className={`text-xs ${
              match.status === "missing" && !chosen ? "text-bad" : match.status === "matched" ? "text-warn" : "text-ink-2"
            }`}
          >
            {chosen && chosen !== match.column ? t("recipes.match.chosen") : t(MATCH_LABEL[match.status] ?? "recipes.match.missing")}
          </span>
        )}
      </td>
    </tr>
  );
}
