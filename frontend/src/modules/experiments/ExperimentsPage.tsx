import { useNavigate, useParams, useSearch } from "@tanstack/react-router";

import { useDataset } from "../../api/hooks";
import { t } from "../../lib/i18n";
import { Tabs } from "../../ui/Tabs";
import { Loading } from "../../ui/Loading";
import { Analyze } from "./Analyze";
import { useLoggedDecisions, type ExperimentDesign } from "./api";
import { Plan } from "./Plan";
import { Policies } from "./Policies";
import type { ExperimentsSearch, ExperimentsTab } from "./search";

export function ExperimentsPage() {
  const { datasetId } = useParams({ strict: false }) as { datasetId: string };
  const search = useSearch({ strict: false }) as ExperimentsSearch;
  const navigate = useNavigate();
  const dataset = useDataset(datasetId);
  const ready = dataset.data?.status === "ready";
  // 055: the Policies tab exists only when the dataset looks like logged decisions
  const decisions = useLoggedDecisions(datasetId, ready);
  const logged = decisions.data?.detected ? decisions.data : null;
  const asked: ExperimentsTab = search.tab ?? "analyze";
  const waiting = asked === "policies" && !decisions.data && !decisions.isError;
  const tab: ExperimentsTab = asked === "policies" && !logged && !waiting ? "analyze" : asked;

  const go = (patch: Partial<ExperimentsSearch>) =>
    void navigate({ to: ".", search: (previous: Record<string, unknown>) => ({ ...previous, ...patch }), replace: true });
  // an edited experiment replaces the seeds it came from (Compare, the assistant)
  const setDesign = (design: ExperimentDesign) => go({ design, variant: undefined, metric: undefined, unit: undefined });

  return (
    <div className="flex flex-col">
      <header className="px-(--gutter) pt-6 pb-4">
        <h1 className="text-xl font-semibold tracking-tight">{t("experiments.title")}</h1>
        <p className="mt-1 max-w-prose text-ink-2">{t("experiments.description")}</p>
      </header>
      <div className="px-(--gutter) pb-10">
        <Tabs
          label={t("experiments.title")}
          value={tab}
          items={[
            { value: "analyze", label: t("experiments.tab.analyze") },
            { value: "plan", label: t("experiments.tab.plan") },
            ...(logged ? [{ value: "policies" as const, label: t("experiments.tab.policies") }] : []),
          ]}
          onChange={(next) => go({ tab: next === "analyze" ? undefined : next })}
        >
          {tab === "plan" ? (
            <Plan datasetId={datasetId} ready={ready} />
          ) : tab === "policies" ? (
            logged ? (
              <Policies key={`${logged.action}-${logged.reward}`} datasetId={datasetId} decisions={logged} />
            ) : (
              <Loading label={t("loading.columns")} className="h-[420px]" />
            )
          ) : (
            <Analyze datasetId={datasetId} ready={ready} search={search} onDesign={setDesign} />
          )}
        </Tabs>
      </div>
    </div>
  );
}
