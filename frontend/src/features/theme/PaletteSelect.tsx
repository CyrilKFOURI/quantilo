import { t } from "../../lib/i18n";
import { DEFAULT_PALETTE, PALETTES, paletteById, paletteLabel } from "../../lib/palettes";
import { Select } from "../../ui/fields";

/** 047: a palette picker, "Name (n)" per palette. `inherit` adds a first choice that keeps the app's palette. */
export function PaletteSelect({
  value,
  onChange,
  inherit,
  className = "",
}: {
  value: string | null;
  onChange: (id: string | null) => void;
  /** The label of the "use the app's palette" choice (a KPI); without it, Quantilo is the first choice. */
  inherit?: string;
  className?: string;
}) {
  const quantilo = paletteLabel(undefined, t("palette.quantilo"));
  return (
    <Select
      aria-label={t("palette.label")}
      value={value ?? ""}
      onChange={(event) => onChange(event.target.value || null)}
      className={className}
    >
      {inherit !== undefined && <option value="">{inherit}</option>}
      <option value={DEFAULT_PALETTE}>{quantilo}</option>
      {PALETTES.map((palette) => (
        <option key={palette.id} value={palette.id}>
          {paletteLabel(palette, "")}
          {palette.colorBlindSafe ? ` · ${t("palette.colorBlindSafe")}` : ""}
        </option>
      ))}
    </Select>
  );
}

/** The palette's colours side by side (Quantilo: its theme tokens). */
export function PaletteSwatches({ id }: { id: string }) {
  const palette = paletteById(id);
  const colors = palette ? palette.colors : Array.from({ length: 8 }, (_, i) => `var(--chart-${i + 1})`);
  return (
    <span className="flex h-3 w-full max-w-60 overflow-hidden rounded-sm" aria-hidden>
      {colors.map((color, i) => (
        <span key={i} className="h-full flex-1" style={{ background: color }} />
      ))}
    </span>
  );
}
