import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { request } from "../../api/client";
import type { components } from "../../api/schema";
import { t, type MessageKey } from "../../lib/i18n";
import { LITE } from "../../lite/flag";

/** AI forecasting models (028 phase B): their state here, the download of their weights (only
 *  after a confirmation that shows the size and the source) and Settings › Models.
 *  Routes: backend/app/modules/timeseries/forecast_ai.py. */

type Schemas = components["schemas"];
export type AiModel = Schemas["AiModelOut"];
export type AiModels = Schemas["AiModelsOut"];
export type ModelSuggestion = Schemas["ModelSuggestion"];

/** Every AI key of the forecast catalogue (Methods.tsx), pretrained first. */
export const PRETRAINED = ["chronos_bolt_tiny", "chronos_bolt_small", "chronos_bolt_base", "chronos_2", "timesfm_2p5"];
export const TRAINED = ["nhits", "nbeats"];
export const AI_MODELS = [...PRETRAINED, ...TRAINED];
/** 028 B4: a network needs this many periods (neural.MIN_PERIODS), or many groups. */
export const MIN_PERIODS = 500;

const base = "/modules/timeseries/forecast-ai";
const key = ["timeseries", "forecast-ai"] as const;

export const aiApi = {
  list: () => request<AiModels>("GET", base),
  download: (model: string) => request<AiModel>("POST", `${base}/${encodeURIComponent(model)}/download`),
  stop: (model: string) => request<AiModel>("DELETE", `${base}/${encodeURIComponent(model)}/download`),
  remove: (model: string) => request<AiModel>("DELETE", `${base}/${encodeURIComponent(model)}`),
  offline: (offline: boolean) => request<AiModels>("PUT", `${base}/offline`, { offline }),
};

/** The models and their state; followed every 1.5 s while a download runs (also in a hidden tab). */
export function useAiModels(enabled = true) {
  return useQuery({
    queryKey: key,
    queryFn: aiApi.list,
    enabled: enabled && !LITE,
    staleTime: 30_000,
    refetchInterval: (query) => (query.state.data?.models.some((m) => m.download === "running") ? 1_500 : false),
    refetchIntervalInBackground: true,
  });
}

function useAiMutation<T>(run: (value: T) => Promise<unknown>) {
  const client = useQueryClient();
  return useMutation({ mutationFn: run, onSettled: () => void client.invalidateQueries({ queryKey: key }) });
}

/** Called only from the confirmation dialog, which showed the size, the licence and the source. */
export const useDownloadModel = () => useAiMutation(aiApi.download);
export const useStopDownload = () => useAiMutation(aiApi.stop);
export const useDeleteModel = () => useAiMutation(aiApi.remove);
export const useNeverDownload = () => useAiMutation(aiApi.offline);

/** Why a model cannot be ticked here (shown instead of its sentence), by coded rule: the extra,
 *  Never download, the browser version, a network on a short series. A model whose weights only
 *  need downloading is not listed: ticking it asks for the download. */
export function aiReasons(models: AiModel[] | undefined, periods: number | undefined): Record<string, string> {
  const reasons: Record<string, string> = {};
  if (LITE) {
    for (const model of AI_MODELS) reasons[model] = t("fcai.state.browser");
    return reasons;
  }
  for (const model of models ?? []) {
    if (model.state === "needs_extra") reasons[model.key] = t("fcai.state.needs_extra", { install: model.install ?? "" });
    else if (model.state === "offline") reasons[model.key] = t("fcai.state.offline", { folder: model.folder });
    else if (model.state === "browser") reasons[model.key] = t("fcai.state.browser");
  }
  if (periods !== undefined && periods < MIN_PERIODS) {
    for (const model of TRAINED)
      reasons[model] ??= t("fcai.tooShort", { min: String(MIN_PERIODS), periods: String(periods) });
  }
  return reasons;
}

/** "Download · 35 MB", "Downloading 40 %" or "Pretrained · zero-shot" after a pretrained model's name. */
export function aiBadge(model: AiModel | undefined): string | null {
  if (!model?.pretrained) return null;
  if (model.download === "running") return t("fcai.badge.downloading", { pct: String(Math.round(model.progress * 100)) });
  if (model.state === "needs_download") return t("fcai.badge.download", { size: model.size_text });
  if (model.state === "ready") return t("fcai.badge.ready");
  return null;
}

/** The model's name in the interface language. */
export const aiTitle = (model: string) => t(`tsm.${model}` as MessageKey);
