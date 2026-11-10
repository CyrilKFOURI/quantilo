import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { LOWER_IS_BETTER, type MlCatalogue, type MlChoicesInput, type MlClasses, type MlValidation } from "../../api/client";
import { useMlCompare, useMlComparison } from "../../api/hooks";
import { formatNumber } from "../../charts/figures";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { Checkbox, ErrorNote } from "../../ui/fields";
import { DesktopOnlyModels, offered } from "./desktopOnly";
import { blocked } from "./ModelPicker";
import { CriterionSelect, metricName, PANEL } from "./SettingsDrawer";

/** Train a selection of models on the same rows and validation, ranked in one table. */
export function CompareModels({
  open,
  onOpenChange,
  datasetId,
  target,
  catalogue,
  choices,
  classes,
  validation,
  onRun,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  datasetId: string;
  target: string;
  catalogue: MlCatalogue;
  choices: MlChoicesInput;
  classes: MlClasses | null;
  validation: MlValidation;
  onRun: (runId: string) => void;
}) {
  const client = useQueryClient();
  const [picked, setPicked] = useState<string[] | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  // on a panel, the criterion over the periods ranks the models
  const [judged, setJudged] = useState<MlValidation>(validation);
  const panel = judged.method === PANEL;
  const start = useMlCompare(datasetId);
  const job = useMlComparison(jobId);
  const chosen = picked ?? catalogue.compare;
  const toggle = (key: string, on: boolean) => setPicked(on ? [...chosen, key] : chosen.filter((k) => k !== key));
  const items = job.data?.items ?? [];
  const metric = job.data?.metric ?? validation.metric ?? catalogue.default_metric;
  const lower = LOWER_IS_BETTER.has(metric);
  const ranked = [...items].sort((a, b) => {
    if (a.score === null) return 1;
    if (b.score === null) return -1;
    return lower ? a.score - b.score : b.score - a.score;
  });
  const best = ranked.find((i) => i.score !== null);
  const running = !job.data || job.data.status === "running" || start.isPending;
  const launch = () =>
    start.mutate(
      { target, models: chosen, choices, classes, validation: judged },
      { onSuccess: (created) => setJobId(created.id) },
    );
  const finished = job.data?.status === "done";
  // the new runs appear in the run list once every model is trained
  useEffect(() => {
    if (finished) void client.invalidateQueries({ queryKey: ["datasets", datasetId, "ml", "runs", target] });
  }, [finished, client, datasetId, target]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange} size="lg" title={t("ml.compare.title")} description={t("ml.compare.description")}>
      {!jobId ? (
        <div className="space-y-3">
          {catalogue.families.map((family) => {
            const models = offered(catalogue.models).filter((m) => m.family === family);
            if (!models.length) return null;
            return (
              <fieldset key={family} className="space-y-1">
                <legend className="text-[11px] font-medium tracking-wide text-ink-3 uppercase">{family}</legend>
                <div className="grid gap-x-4 gap-y-1 sm:grid-cols-2">
                  {models.map((m) => {
                    const reason = blocked(m);
                    return (
                      <Checkbox
                        key={m.key}
                        className="text-[13px] text-ink-1"
                        label={reason ? `${m.label} — ${reason}` : m.label}
                        disabled={Boolean(reason)}
                        checked={chosen.includes(m.key)}
                        onChange={(e) => toggle(m.key, e.target.checked)}
                      />
                    );
                  })}
                </div>
              </fieldset>
            );
          })}
          <DesktopOnlyModels models={catalogue.models} inset={false} />
          <div className="flex flex-wrap items-center justify-end gap-3 border-t border-line pt-3">
            {panel && (
              <label className="mr-auto flex items-center gap-2 text-xs text-ink-2">
                {t("ml.validation.criterion")}
                <CriterionSelect validation={judged} onValidation={setJudged} />
              </label>
            )}
            <span className="text-xs text-ink-3">{t("ml.compare.selected", { count: String(chosen.length) })}</span>
            <Button variant="primary" onClick={launch} disabled={chosen.length === 0 || start.isPending}>
              {t("ml.compare.start")}
            </Button>
          </div>
          {start.isError && <ErrorNote>{start.error.message}</ErrorNote>}
        </div>
      ) : (
        <div className="space-y-3">
          <p className="text-xs text-ink-2" aria-live="polite">
            {running
              ? t("ml.compare.running", { done: String(items.filter((i) => i.status === "done" || i.status === "failed").length), total: String(items.length) })
              : t("ml.compare.finished")}
          </p>
          <table className="w-full border-collapse text-[13px] tabular-nums">
            <thead className="text-left text-xs text-ink-3">
              <tr>
                <th className="py-1.5 font-normal">{t("ml.model")}</th>
                <th className="py-1.5 text-right font-normal">
                  {panel
                    ? t("ml.compare.criterion", { metric: metricName(metric), criterion: t(`ml.validation.criterion.${judged.criterion}` as MessageKey).toLowerCase() })
                    : metricName(metric)}
                </th>
                <th className="py-1.5 text-right font-normal">{t("ml.compare.seconds")}</th>
                <th className="py-1.5 font-normal" />
              </tr>
            </thead>
            <tbody>
              {ranked.map((item) => (
                <tr key={item.key} className={`border-t border-line ${best?.key === item.key ? "font-medium text-ink-1" : "text-ink-2"}`}>
                  <td className="py-1.5">
                    {item.label}
                    {best?.key === item.key && items.length > 1 ? " ★" : ""}
                    {item.error && <span className="block text-xs font-normal text-bad">{item.error}</span>}
                  </td>
                  <td className="py-1.5 text-right">{item.status === "done" ? formatNumber(item.score) : t(`ml.compare.status.${item.status}` as MessageKey)}</td>
                  <td className="py-1.5 text-right">{item.seconds === null ? "—" : item.seconds.toFixed(1)}</td>
                  <td className="py-1.5 text-right">
                    {item.run_id && (
                      <Button
                        variant="ghost"
                        onClick={() => {
                          onRun(item.run_id as string);
                          onOpenChange(false);
                        }}
                      >
                        {t("ml.compare.open")}
                      </Button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {finished && (
            <div className="flex justify-end">
              <Button variant="ghost" onClick={() => setJobId(null)}>
                {t("ml.compare.again")}
              </Button>
            </div>
          )}
        </div>
      )}
    </Dialog>
  );
}
