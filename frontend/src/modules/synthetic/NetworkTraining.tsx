import { useState } from "react";

import type { SyntheticTraining } from "../../api/client";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { Checkbox, Field, Select, TextInput } from "../../ui/fields";

const OPTIMIZERS = ["adamw", "adam", "sgd_nesterov", "radam"] as const;
const SCHEDULES = ["constant", "cosine", "one_cycle"] as const;
/** What "Customise" starts from: AdamW ★, a constant rate, nothing else (066). */
export const DEFAULT_TRAINING: SyntheticTraining = {
  optimizer: "adamw",
  learning_rate: null,
  weight_decay: null,
  schedule: "constant",
  warmup: 0,
  clip_norm: null,
  ema: null,
};

const optional = (text: string, low: number, high: number): number | null => {
  if (text.trim() === "") return null;
  const value = Number(text);
  return Number.isFinite(value) ? Math.min(high, Math.max(low, value)) : null;
};

/** Advanced › More of the deep methods (066): how the networks are trained. Closed, nothing is
 *  sent and every network trains as its paper says. */
export function NetworkTraining({ training, onChange }: { training: SyntheticTraining | null; onChange: (next: SyntheticTraining | null) => void }) {
  const [open, setOpen] = useState(false);
  const custom = training !== null;
  const value = training ?? DEFAULT_TRAINING;
  const set = (patch: Partial<SyntheticTraining>) => onChange({ ...value, ...patch });
  return (
    <>
      <button type="button" onClick={() => setOpen(true)} className="text-ink-2 underline-offset-2 hover:text-ink-1 hover:underline" aria-haspopup="dialog">
        {custom ? t("synthetic.training.custom") : t("synthetic.training.open")}
      </button>
      <Dialog
        open={open}
        onOpenChange={setOpen}
        title={t("synthetic.training.title")}
        description={t("synthetic.training.description")}
        footer={<Button onClick={() => setOpen(false)}>{t("synthetic.training.done")}</Button>}
      >
        <div className="space-y-3 text-xs">
          <Checkbox className="text-ink-2" label={t("synthetic.training.customise")} checked={custom} onChange={(e) => onChange(e.target.checked ? DEFAULT_TRAINING : null)} />
          {!custom && <p className="text-ink-3">{t("synthetic.training.paper")}</p>}
          {custom && (
            <div className="grid gap-3 sm:grid-cols-2">
              <Field label={t("synthetic.training.optimizer")}>
                <Select aria-label={t("synthetic.training.optimizer")} value={value.optimizer} onChange={(e) => set({ optimizer: e.target.value as SyntheticTraining["optimizer"] })} className="h-7 text-xs">
                  {OPTIMIZERS.map((o) => (
                    <option key={o} value={o}>
                      {t(`synthetic.training.optimizer.${o}` as MessageKey)}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label={t("synthetic.training.rate")}>
                <TextInput
                  type="number"
                  step="any"
                  min={0}
                  aria-label={t("synthetic.training.rate")}
                  placeholder={t("synthetic.training.ratePaper")}
                  value={value.learning_rate ?? ""}
                  onChange={(e) => set({ learning_rate: optional(e.target.value, 1e-6, 1) })}
                  className="h-7 text-xs"
                />
              </Field>
              <Field label={t("synthetic.training.schedule")}>
                <Select aria-label={t("synthetic.training.schedule")} value={value.schedule} onChange={(e) => set({ schedule: e.target.value as SyntheticTraining["schedule"] })} className="h-7 text-xs">
                  {SCHEDULES.map((s) => (
                    <option key={s} value={s}>
                      {t(`synthetic.training.schedule.${s}` as MessageKey)}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label={t("synthetic.training.warmup")}>
                <TextInput
                  type="number"
                  min={0}
                  max={50}
                  aria-label={t("synthetic.training.warmup")}
                  value={Math.round(value.warmup * 100)}
                  onChange={(e) => set({ warmup: (optional(e.target.value, 0, 50) ?? 0) / 100 })}
                  className="h-7 text-xs"
                />
              </Field>
              <Field label={t("synthetic.training.clip")}>
                <TextInput
                  type="number"
                  step="any"
                  min={0}
                  aria-label={t("synthetic.training.clip")}
                  placeholder={t("synthetic.training.none")}
                  value={value.clip_norm ?? ""}
                  onChange={(e) => set({ clip_norm: optional(e.target.value, 0.001, 100) })}
                  className="h-7 text-xs"
                />
              </Field>
              <Field label={t("synthetic.training.ema")}>
                <TextInput
                  type="number"
                  step="any"
                  min={0.9}
                  max={0.9999}
                  aria-label={t("synthetic.training.ema")}
                  placeholder={t("synthetic.training.none")}
                  value={value.ema ?? ""}
                  onChange={(e) => set({ ema: optional(e.target.value, 0.9, 0.9999) })}
                  className="h-7 text-xs"
                />
              </Field>
              <p className="text-[11px] text-ink-3 sm:col-span-2">{t("synthetic.training.hint")}</p>
            </div>
          )}
        </div>
      </Dialog>
    </>
  );
}
