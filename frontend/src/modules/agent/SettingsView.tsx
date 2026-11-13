import { Check, KeyRound, RefreshCw } from "lucide-react";
import { useState } from "react";

import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { ErrorNote, Field, Select, TextInput } from "../../ui/fields";
import type { AgentSettings } from "./api";
import { privacyText } from "./RowsCard";
import { RowsSharing } from "./RowsSharing";
import { useAgentModels, useTestAgentKey, useUpdateAgentSettings } from "./useAgent";

const MODEL_LABELS: Record<string, string> = {
  "claude-opus-5-5": "Claude Opus 5.5 — most capable",
  "claude-sonnet-5-5": "Claude Sonnet 5.5 — faster, half the price",
  "claude-haiku-5-5": "Claude Haiku 5.5 — quickest, cheapest",
};
const KEY_PLACEHOLDERS: Record<string, string> = {
  anthropic: "sk-ant-…",
  gemini: "AIza…",
  openrouter: "sk-or-…",
  groq: "gsk_…",
  openai: "sk-…",
};
const EFFORTS = ["low", "medium", "high", "xhigh", "max"] as const;

/** Provider, key, model, effort, budget and row sharing of the assistant (spec 026 § 9). */
export function SettingsView({
  datasetId,
  settings,
  onDone,
}: {
  datasetId: string;
  settings: AgentSettings;
  onDone: () => void;
}) {
  const update = useUpdateAgentSettings();
  const test = useTestAgentKey();
  const [key, setKey] = useState("");
  const [budget, setBudget] = useState(String(settings.budget_usd));
  const [baseUrl, setBaseUrl] = useState(settings.base_url ?? "");
  const active = settings.providers.find((p) => p.id === settings.active);
  const custom = settings.active === "custom";

  const saveKey = () => {
    update.mutate({ api_key: key }, { onSuccess: () => setKey("") });
  };

  return (
    <div className="space-y-5 p-4 text-[13px]">
      {!settings.sdk_installed && <ErrorNote>{t("agent.sdkMissing")}</ErrorNote>}

      <section className="space-y-2">
        <Field label={t("agent.provider")}>
          <Select value={settings.provider} onChange={(event) => update.mutate({ provider: event.target.value })}>
            <option value="auto">{t("agent.providerAuto", { provider: settings.active_label })}</option>
            {settings.providers.map((p) => (
              <option key={p.id} value={p.id}>
                {p.configured ? t("agent.providerWithKey", { provider: p.label }) : p.label}
              </option>
            ))}
          </Select>
        </Field>
        <p className="text-xs text-ink-3">{t("agent.providerHint")}</p>
      </section>

      <section className="space-y-2">
        <h3 className="flex items-center gap-1.5 font-medium text-ink-1">
          <KeyRound size={14} aria-hidden />
          {t("agent.keyFor", { provider: settings.active_label })}
        </h3>
        <p className="text-xs text-ink-3">
          {settings.key.configured
            ? t(settings.key.source === "env" ? "agent.keyFromEnv" : "agent.keySaved", { hint: settings.key.hint ?? "" })
            : custom
              ? t("agent.keyOptional")
              : t("agent.keyNone", { console: active?.console ?? "" })}
        </p>
        {settings.key.source !== "env" && (
          <div className="flex gap-2">
            <TextInput
              type="password"
              autoComplete="off"
              placeholder={KEY_PLACEHOLDERS[settings.active] ?? t("agent.keyPlaceholder")}
              aria-label={t("agent.keyFor", { provider: settings.active_label })}
              value={key}
              onChange={(event) => setKey(event.target.value)}
              className="min-w-0 flex-1"
            />
            <Button onClick={saveKey} disabled={!key.trim() || update.isPending}>
              {t("agent.save")}
            </Button>
          </div>
        )}
        {custom && (
          <Field label={t("agent.baseUrl")}>
            <TextInput
              type="url"
              placeholder="http://localhost:11434/v1"
              value={baseUrl}
              onChange={(event) => setBaseUrl(event.target.value)}
              onBlur={() => {
                if (baseUrl.trim() !== (settings.base_url ?? "")) update.mutate({ base_url: baseUrl.trim() });
              }}
            />
          </Field>
        )}
        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={() => test.mutate()} disabled={!settings.ready || test.isPending}>
            {test.isPending ? t("agent.testing") : t("agent.test")}
          </Button>
          {settings.key.source === "file" && (
            <Button variant="ghost" onClick={() => update.mutate({ api_key: "" })}>
              {t("agent.removeKey")}
            </Button>
          )}
          {test.isSuccess && (
            <span className="inline-flex items-center gap-1 text-xs text-ok">
              <Check size={13} aria-hidden />
              {t("agent.keyWorks", { model: test.data.model })}
            </span>
          )}
        </div>
        {test.isError && <ErrorNote>{test.error.message}</ErrorNote>}
        <p className="text-xs text-ink-3">{privacyText(settings.rows_sharing)}</p>
      </section>

      <RowsSharing datasetId={datasetId} level={settings.rows_sharing} />

      <section className="space-y-3">
        <ModelPicker key={settings.active} settings={settings} onChange={(model) => update.mutate({ model })} />
        {settings.effort_supported && (
          <>
            <Field label={t("agent.effort")}>
              <Select
                value={settings.effort}
                onChange={(event) => update.mutate({ effort: event.target.value as (typeof EFFORTS)[number] })}
              >
                {EFFORTS.map((effort) => (
                  <option key={effort} value={effort}>
                    {effort}
                  </option>
                ))}
              </Select>
            </Field>
            <p className="-mt-1 text-xs text-ink-3">{t("agent.effortHint")}</p>
            <Field label={t("agent.budget")}>
              <TextInput
                type="number"
                min={0.05}
                max={100}
                step={0.05}
                value={budget}
                onChange={(event) => setBudget(event.target.value)}
                onBlur={() => {
                  const value = Number(budget);
                  if (value > 0 && value <= 100 && value !== settings.budget_usd) update.mutate({ budget_usd: value });
                }}
              />
            </Field>
            <p className="-mt-1 text-xs text-ink-3">{t("agent.budgetHint")}</p>
          </>
        )}
        {!settings.effort_supported && (
          <p className="text-xs text-ink-3">
            {t("agent.tokenBudget", { limit: formatTokens(settings.token_limit) })}
          </p>
        )}
      </section>
      {update.isError && <ErrorNote>{update.error.message}</ErrorNote>}

      <Button variant="primary" onClick={onDone}>
        {t("agent.backToChat")}
      </Button>
    </div>
  );
}

/** The models the provider offers to the key; a text field when they can't be listed. */
function ModelPicker({ settings, onChange }: { settings: AgentSettings; onChange: (model: string) => void }) {
  const listed = settings.known_models.length > 0;
  const models = useAgentModels(settings.ready && !listed);
  const [typed, setTyped] = useState(settings.model);
  const [refreshing, setRefreshing] = useState(false);
  const options = listed ? settings.known_models : (models.data?.models ?? []);
  const fallback = listed ? settings.known_models[0] : models.data?.default;

  if (!listed && models.isError) {
    return (
      <div className="space-y-2">
        <Field label={t("agent.model")}>
          <TextInput
            placeholder={t("agent.modelTyped")}
            value={typed}
            onChange={(event) => setTyped(event.target.value)}
            onBlur={() => typed.trim() !== settings.model && onChange(typed.trim())}
          />
        </Field>
        <ErrorNote>{models.error.message}</ErrorNote>
      </div>
    );
  }

  const refresh = async () => {
    setRefreshing(true);
    try {
      await models.refresh();
    } finally {
      setRefreshing(false);
    }
  };

  return (
    <div className="flex items-end gap-2">
      <div className="min-w-0 flex-1">
        <Field label={t("agent.model")}>
          <Select
            value={settings.model}
            disabled={!listed && !models.data}
            onChange={(event) => onChange(event.target.value)}
          >
            <option value="">
              {fallback ? t("agent.modelDefault", { model: MODEL_LABELS[fallback] ?? fallback }) : t("agent.modelLoading")}
            </option>
            {settings.model && !options.includes(settings.model) && (
              <option value={settings.model}>{settings.model}</option>
            )}
            {options.map((model) => (
              <option key={model} value={model}>
                {MODEL_LABELS[model] ?? model}
              </option>
            ))}
          </Select>
        </Field>
      </div>
      {!listed && (
        <Button
          variant="ghost"
          iconOnly
          icon={RefreshCw}
          aria-label={t("agent.modelRefresh")}
          title={t("agent.modelRefresh")}
          disabled={!settings.ready || refreshing}
          onClick={() => void refresh()}
        />
      )}
    </div>
  );
}

export function formatTokens(count: number): string {
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(count % 1_000_000 ? 1 : 0)}M`;
  if (count >= 1_000) return `${Math.round(count / 1_000)}k`;
  return String(count);
}
