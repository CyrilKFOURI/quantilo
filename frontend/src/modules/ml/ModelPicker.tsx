import { ChevronDown, ChevronRight, Search } from "lucide-react";
import { useState } from "react";

import type { MlCatalogue, MlModel } from "../../api/client";
import { t } from "../../lib/i18n";
import { Dialog } from "../../ui/Dialog";
import { TextInput } from "../../ui/fields";
import { DesktopOnlyModels, offered } from "./desktopOnly";

/** Why a model cannot be picked here, or null. */
export function blocked(model: MlModel): string | null {
  if (!model.installed) return t("ml.picker.notInstalled");
  return model.unavailable;
}

function Badges({ badges }: { badges: string[] }) {
  if (!badges.length) return null;
  return (
    <span className="flex flex-wrap gap-1">
      {badges.map((badge) => (
        <span key={badge} className="rounded-full bg-surface-2 px-1.5 py-0.5 text-[10px] whitespace-nowrap text-ink-2">
          {badge}
        </span>
      ))}
    </span>
  );
}

function ModelRow({ model, why, current, onPick }: { model: MlModel; why?: string; current: boolean; onPick: (key: string) => void }) {
  const reason = blocked(model);
  return (
    <li>
      <button
        type="button"
        disabled={Boolean(reason)}
        onClick={() => onPick(model.key)}
        aria-current={current || undefined}
        className={`grid w-full grid-cols-[minmax(0,1fr)_auto] items-start gap-3 rounded-md px-3 py-2 text-left hover:bg-surface-2 disabled:cursor-not-allowed disabled:opacity-55 disabled:hover:bg-transparent ${current ? "bg-accent-soft" : ""}`}
      >
        <span className="min-w-0">
          <span className="text-[13px] font-medium text-ink-1">{model.label}</span>
          <span className="block text-xs text-ink-3">{reason ?? why ?? model.description}</span>
        </span>
        <Badges badges={model.badges} />
      </button>
    </li>
  );
}

/** A wide, searchable menu: recommendations for this data first, then every family. */
export function ModelPicker({ catalogue, current, onPick }: { catalogue: MlCatalogue; current: MlModel; onPick: (key: string) => void }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [shut, setShut] = useState<Record<string, boolean>>({});
  const needle = query.trim().toLowerCase();
  const listed = offered(catalogue.models);
  const found = listed.filter(
    (m) => !needle || `${m.label} ${m.family} ${m.description} ${m.key}`.toLowerCase().includes(needle),
  );
  const byKey = new Map(catalogue.models.map((m) => [m.key, m]));
  const pick = (key: string) => {
    onPick(key);
    setOpen(false);
    setQuery("");
  };
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-haspopup="dialog"
        className="flex w-full items-center justify-between gap-2 rounded-md border border-line bg-surface-1 px-2.5 py-1.5 text-left hover:bg-surface-2"
      >
        <span className="min-w-0">
          <span className="block truncate text-[13px] font-medium text-ink-1">{current.label}</span>
          <span className="block truncate text-xs text-ink-3">{current.family}</span>
        </span>
        <ChevronDown size={14} strokeWidth={1.5} className="shrink-0 text-ink-3" aria-hidden />
      </button>
      <Dialog open={open} onOpenChange={setOpen} title={t("ml.picker.title")} description={t("ml.picker.count", { count: String(listed.length) })}>
        <div className="space-y-3">
          <div className="relative">
            <Search size={14} strokeWidth={1.5} className="pointer-events-none absolute top-1/2 left-2 -translate-y-1/2 text-ink-3" aria-hidden />
            <TextInput
              autoFocus
              aria-label={t("ml.picker.search")}
              placeholder={t("ml.picker.search")}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="w-full pl-7"
            />
          </div>
          {!needle && catalogue.recommended.length > 0 && (
            <section aria-label={t("ml.picker.recommended")}>
              <h3 className="px-3 pb-1 text-[11px] font-medium tracking-wide text-ink-3 uppercase">{t("ml.picker.recommended")}</h3>
              <ul>
                {catalogue.recommended.flatMap((r) => {
                  const model = byKey.get(r.key);
                  return model ? [<ModelRow key={r.key} model={model} why={r.why} current={model.key === current.key} onPick={pick} />] : [];
                })}
              </ul>
            </section>
          )}
          {catalogue.families.map((family) => {
            const models = found.filter((m) => m.family === family);
            if (!models.length) return null;
            const closed = !needle && (shut[family] ?? family !== current.family);
            return (
              <section key={family}>
                <button
                  type="button"
                  aria-expanded={!closed}
                  onClick={() => setShut({ ...shut, [family]: !closed })}
                  className="flex w-full items-center gap-1 px-2 py-1 text-[11px] font-medium tracking-wide text-ink-3 uppercase hover:text-ink-1"
                >
                  {closed ? <ChevronRight size={12} aria-hidden /> : <ChevronDown size={12} aria-hidden />}
                  {family}
                  <span className="font-normal normal-case">· {models.length}</span>
                </button>
                {!closed && (
                  <ul>
                    {models.map((m) => (
                      <ModelRow key={m.key} model={m} current={m.key === current.key} onPick={pick} />
                    ))}
                  </ul>
                )}
              </section>
            );
          })}
          {found.length === 0 && <p className="px-3 text-[13px] text-ink-3">{t("ml.picker.none")}</p>}
          {!needle && <DesktopOnlyModels models={catalogue.models} />}
        </div>
      </Dialog>
    </>
  );
}
