import {
  AlignLeft,
  ArrowUpNarrowWide,
  Calendar,
  ChartColumn,
  EyeOff,
  Fingerprint,
  Hash,
  Tags,
  ToggleLeft,
  type LucideIcon,
} from "lucide-react";

import type { AnalysisType } from "../../api/client";
import type { MessageKey } from "../../lib/i18n";

type TypeMeta = {
  label: MessageKey;
  description: MessageKey;
  icon: LucideIcon;
  /** Text color utility bound to a type token (docs/UI_UX.md). */
  tone: string;
  /** Background utility of the same token (written out: Tailwind only sees literal classes). */
  swatch: string;
};

export const TYPE_META: Record<AnalysisType, TypeMeta> = {
  numeric_continuous: {
    label: "type.numeric_continuous",
    description: "type.numeric_continuous.hint",
    icon: Hash,
    tone: "text-type-numeric",
    swatch: "bg-type-numeric",
  },
  numeric_discrete: {
    label: "type.numeric_discrete",
    description: "type.numeric_discrete.hint",
    icon: ChartColumn,
    tone: "text-type-numeric",
    swatch: "bg-type-numeric",
  },
  categorical_nominal: {
    label: "type.categorical_nominal",
    description: "type.categorical_nominal.hint",
    icon: Tags,
    tone: "text-type-categorical",
    swatch: "bg-type-categorical",
  },
  categorical_ordinal: {
    label: "type.categorical_ordinal",
    description: "type.categorical_ordinal.hint",
    icon: ArrowUpNarrowWide,
    tone: "text-type-categorical",
    swatch: "bg-type-categorical",
  },
  boolean: {
    label: "type.boolean",
    description: "type.boolean.hint",
    icon: ToggleLeft,
    tone: "text-type-bool",
    swatch: "bg-type-bool",
  },
  datetime: {
    label: "type.datetime",
    description: "type.datetime.hint",
    icon: Calendar,
    tone: "text-type-date",
    swatch: "bg-type-date",
  },
  text: { label: "type.text", description: "type.text.hint", icon: AlignLeft, tone: "text-type-text", swatch: "bg-type-text" },
  identifier: {
    label: "type.identifier",
    description: "type.identifier.hint",
    icon: Fingerprint,
    tone: "text-type-id",
    swatch: "bg-type-id",
  },
  ignored: { label: "type.ignored", description: "type.ignored.hint", icon: EyeOff, tone: "text-ink-3", swatch: "bg-ink-3" },
};

export const ANALYSIS_TYPES = Object.keys(TYPE_META) as AnalysisType[];

/** Roles that name a kind of text (050): the badge says "Short text" or "Address", the type stays text. */
const TEXT_KIND_LABELS: Record<string, MessageKey> = {
  label: "type.text.short",
  address: "type.text.address",
};

/** The badge label of a type, refined by the role for text columns. */
export function typeLabel(type: AnalysisType, role?: string | null): MessageKey {
  if (type === "text" && role && TEXT_KIND_LABELS[role]) return TEXT_KIND_LABELS[role];
  return TYPE_META[type].label;
}
