import * as Menu from "@radix-ui/react-dropdown-menu";
import { Check, ChevronDown, RotateCcw } from "lucide-react";

import type { AnalysisType } from "../../api/client";
import { t } from "../../lib/i18n";
import { ANALYSIS_TYPES, TYPE_META } from "./columnTypes";

type TypeBadgeProps = {
  type: AnalysisType;
  /** When given, the badge opens a menu to change the type. */
  onChange?: (type: AnalysisType) => void;
  onReset?: () => void;
  detectedType?: AnalysisType;
  reasons?: string[];
};

export function TypeBadge({ type, onChange, onReset, detectedType, reasons }: TypeBadgeProps) {
  const meta = TYPE_META[type];
  const Icon = meta.icon;
  const label = (
    <>
      <Icon size={14} strokeWidth={1.5} className={meta.tone} aria-hidden />
      <span className="text-ink-1">{t(meta.label)}</span>
    </>
  );

  if (!onChange) {
    return <span className="inline-flex items-center gap-1.5 text-[13px]">{label}</span>;
  }

  return (
    <Menu.Root>
      <Menu.Trigger
        aria-label={t("columns.changeType")}
        className="inline-flex h-7 items-center gap-1.5 rounded-md border border-transparent px-1.5 text-[13px] hover:border-line hover:bg-surface-1 data-[state=open]:border-line data-[state=open]:bg-surface-1 max-sm:whitespace-nowrap"
      >
        {label}
        <ChevronDown size={13} strokeWidth={1.5} className="text-ink-3" aria-hidden />
      </Menu.Trigger>
      <Menu.Portal>
        <Menu.Content
          align="start"
          sideOffset={4}
          collisionPadding={12}
          className="z-50 max-h-(--radix-dropdown-menu-content-available-height) w-72 overflow-y-auto overscroll-contain rounded-lg border border-line bg-surface-1 p-1 shadow-[0_12px_32px_-12px_rgb(0_0_0/0.35)]"
        >
          {reasons && reasons.length > 0 && (
            <div className="border-b border-line px-2.5 pt-1.5 pb-2 text-xs text-ink-2">
              <p className="mb-0.5 text-ink-3">
                {t("columns.detectedAs", { type: t(TYPE_META[detectedType ?? type].label) })}
              </p>
              {reasons.join(" · ")}
            </div>
          )}
          <Menu.RadioGroup
            value={type}
            onValueChange={(value) => onChange(value as AnalysisType)}
            className="py-1"
          >
            {ANALYSIS_TYPES.map((option) => {
              const optionMeta = TYPE_META[option];
              const OptionIcon = optionMeta.icon;
              return (
                <Menu.RadioItem
                  key={option}
                  value={option}
                  className="flex cursor-default items-start gap-2 rounded-md px-2 py-1 outline-none data-[highlighted]:bg-surface-2"
                >
                  <OptionIcon
                    size={15}
                    strokeWidth={1.5}
                    className={`mt-0.5 shrink-0 ${optionMeta.tone}`}
                    aria-hidden
                  />
                  <span className="min-w-0 flex-1">
                    <span className="block text-[13px] text-ink-1">{t(optionMeta.label)}</span>
                    <span className="block truncate text-xs text-ink-3" title={t(optionMeta.description)}>
                      {t(optionMeta.description)}
                    </span>
                  </span>
                  <Menu.ItemIndicator>
                    <Check size={14} strokeWidth={1.5} className="mt-0.5 text-accent" aria-hidden />
                  </Menu.ItemIndicator>
                </Menu.RadioItem>
              );
            })}
          </Menu.RadioGroup>
          {onReset && (
            <>
              <Menu.Separator className="my-1 h-px bg-line" />
              <Menu.Item
                onSelect={onReset}
                className="flex cursor-default items-center gap-2.5 rounded-md px-2.5 py-1.5 text-[13px] text-ink-2 outline-none data-[highlighted]:bg-surface-2"
              >
                <RotateCcw size={14} strokeWidth={1.5} aria-hidden />
                {t("columns.resetDetection")}
              </Menu.Item>
            </>
          )}
        </Menu.Content>
      </Menu.Portal>
    </Menu.Root>
  );
}
