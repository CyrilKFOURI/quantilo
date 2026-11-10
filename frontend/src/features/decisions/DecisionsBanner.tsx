import { Link } from "@tanstack/react-router";
import { Scale } from "lucide-react";

import { t } from "../../lib/i18n";
import { useLoggedDecisions } from "./api";

/** 055: on the ML page, a banner when the dataset looks like logged decisions (an action, its
 *  reward, covariates): evaluating a policy answers "what if we had decided otherwise?", which a
 *  model of the reward alone does not. Nothing is shown otherwise. */
export function DecisionsBanner({ datasetId, ready }: { datasetId: string; ready: boolean }) {
  const decisions = useLoggedDecisions(datasetId, ready);
  if (!decisions.data?.detected) return null;
  return (
    <p className="flex max-w-prose flex-wrap items-center gap-x-2 gap-y-1 border-l-2 border-accent pl-3 text-[13px] text-ink-2">
      <Scale size={14} strokeWidth={1.75} className="shrink-0 text-ink-3" aria-hidden />
      <span>{t("experiments.policies.banner", { action: decisions.data.action ?? "", reward: decisions.data.reward ?? "" })}</span>
      <Link
        to={"/d/$datasetId/experiments" as never}
        params={{ datasetId } as never}
        search={{ tab: "policies" } as never}
        className="text-accent hover:underline"
      >
        {t("experiments.policies.bannerLink")}
      </Link>
    </p>
  );
}
