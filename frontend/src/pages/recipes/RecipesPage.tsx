import { Link, useNavigate } from "@tanstack/react-router";
import { FileUp } from "lucide-react";
import { useRef } from "react";

import { useImportRecipe, useRecipes } from "../../api/recipes";
import { formatRelativeTime } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { ErrorNote } from "../../ui/fields";
import { PageHeader } from "../../ui/PageHeader";
import { StatusTag } from "./parts";

/** Recipes: the work kept on a dataset, to replay on next month's file (030). */
export function RecipesPage() {
  const recipes = useRecipes();
  const importRecipe = useImportRecipe();
  const input = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const readFile = async (file: File | undefined) => {
    if (!file) return;
    const content = await file.text();
    importRecipe.mutate(content, {
      onSuccess: (created) =>
        void navigate({ to: "/recipes/$recipeId", params: { recipeId: created.id } }),
    });
  };

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-6 px-(--gutter) py-8">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <PageHeader title={t("recipes.title")} description={t("recipes.description")} />
        <Button icon={FileUp} onClick={() => input.current?.click()} disabled={importRecipe.isPending}>
          {t("recipes.import")}
        </Button>
        <input
          ref={input}
          type="file"
          accept=".json,application/json"
          className="hidden"
          aria-label={t("recipes.import")}
          onChange={(event) => {
            void readFile(event.target.files?.[0]);
            event.target.value = "";
          }}
        />
      </div>
      {importRecipe.isError && <ErrorNote>{importRecipe.error.message}</ErrorNote>}
      {recipes.isError && <ErrorNote>{recipes.error.message}</ErrorNote>}
      {recipes.data && recipes.data.length === 0 && (
        <div className="max-w-prose rounded-lg border border-dashed border-line px-6 py-8 text-ink-2">
          <p className="font-medium text-ink-1">{t("recipes.empty.title")}</p>
          <p className="mt-1">{t("recipes.empty.body")}</p>
        </div>
      )}
      {recipes.data && recipes.data.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-line bg-surface-1">
          <table className="w-full border-collapse text-[13px]">
            <thead>
              <tr className="border-b border-line text-left text-xs text-ink-3">
                <th scope="col" className="w-full px-3 py-2 font-normal">{t("recipes.col.name")}</th>
                <th scope="col" className="px-3 py-2 text-right font-normal">{t("recipes.col.steps")}</th>
                <th scope="col" className="px-3 py-2 font-normal">{t("recipes.col.source")}</th>
                <th scope="col" className="px-3 py-2 font-normal whitespace-nowrap">{t("recipes.col.lastRun")}</th>
                <th scope="col" className="px-3 py-2 font-normal">{t("recipes.col.status")}</th>
              </tr>
            </thead>
            <tbody>
              {recipes.data.map((recipe) => (
                <tr key={recipe.id} className="border-b border-line last:border-0 hover:bg-surface-2/60">
                  <td className="px-3 py-2">
                    <Link
                      to="/recipes/$recipeId"
                      params={{ recipeId: recipe.id }}
                      className="font-medium text-ink-1 hover:text-accent hover:underline"
                    >
                      {recipe.name}
                    </Link>
                    <span className="ml-2 text-xs text-ink-3">v{recipe.version}</span>
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums">{recipe.steps}</td>
                  <td className="px-3 py-2 whitespace-nowrap text-ink-2">{recipe.source ?? "—"}</td>
                  <td className="px-3 py-2 whitespace-nowrap text-ink-2">
                    {recipe.last_run
                      ? `${formatRelativeTime(recipe.last_run.created_at)} · ${recipe.last_run.dataset_name}`
                      : "—"}
                  </td>
                  <td className="px-3 py-2">{recipe.last_run ? <StatusTag status={recipe.last_run.status} /> : null}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
