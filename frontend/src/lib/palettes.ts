import type { Preference } from "./preferences";

/** 047: categorical palettes the charts can use, in their published order (sources in docs/features/047-palettes). */
export type Palette = { id: string; name: string; colors: string[]; colorBlindSafe?: boolean };

/** The app's own palette: the --chart-1…8 tokens, which follow the light/dark theme. */
export const DEFAULT_PALETTE = "quantilo";

export const PALETTES: Palette[] = [
  { id: "okabe-ito", name: "Okabe–Ito", colorBlindSafe: true, colors: ["#e69f00", "#56b4e9", "#009e73", "#f0e442", "#0072b2", "#d55e00", "#cc79a7"] },
  { id: "tol-bright", name: "Tol Bright", colorBlindSafe: true, colors: ["#4477aa", "#ee6677", "#228833", "#ccbb44", "#66ccee", "#aa3377", "#bbbbbb"] },
  { id: "tol-vibrant", name: "Tol Vibrant", colorBlindSafe: true, colors: ["#ee7733", "#0077bb", "#33bbee", "#ee3377", "#cc3311", "#009988", "#bbbbbb"] },
  { id: "set2", name: "ColorBrewer Set 2", colors: ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"] },
  { id: "dark2", name: "ColorBrewer Dark 2", colors: ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"] },
  {
    id: "tableau10",
    name: "Tableau 10",
    colors: ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"],
  },
  {
    id: "observable10",
    name: "Observable 10",
    colors: ["#4269d0", "#efb118", "#ff725c", "#6cc5b0", "#3ca951", "#ff8ab7", "#a463f2", "#97bbf5", "#9c6b4e", "#9498a0"],
  },
  {
    id: "category10",
    name: "D3 Category 10",
    colors: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
  },
  {
    id: "carto-bold",
    name: "CARTO Bold",
    colors: ["#7f3c8d", "#11a579", "#3969ac", "#f2b701", "#e73f74", "#80ba5a", "#e68310", "#008695", "#cf1c90", "#f97b72", "#4b4b8f", "#a5aa99"],
  },
  {
    id: "carto-vivid",
    name: "CARTO Vivid",
    colors: ["#e58606", "#5d69b1", "#52bca3", "#99c945", "#cc61b0", "#24796c", "#daa51b", "#2f8ac4", "#764e9f", "#ed645a", "#cc3a8e", "#a5aa99"],
  },
  {
    id: "paired",
    name: "ColorBrewer Paired",
    colors: ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a", "#ffff99", "#b15928"],
  },
  {
    id: "tableau20",
    name: "Tableau 20",
    colors: [
      "#4e79a7", "#a0cbe8", "#f28e2b", "#ffbe7d", "#59a14f", "#8cd17d", "#b6992d", "#f1ce63", "#499894", "#86bcb6",
      "#e15759", "#ff9d9a", "#79706e", "#bab0ac", "#d37295", "#fabfd2", "#b07aa1", "#d4a6c8", "#9d7660", "#d7b5a6",
    ],
  },
  {
    id: "category20",
    name: "D3 Category 20",
    colors: [
      "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
      "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
    ],
  },
];

const TOKEN_COUNT = 8;

export const palettePreference: Preference<string> = {
  storageKey: "de.palette",
  attribute: "palette",
  values: [DEFAULT_PALETTE, ...PALETTES.map((p) => p.id)],
  defaultValue: DEFAULT_PALETTE,
};

export function paletteById(id: string | null | undefined): Palette | undefined {
  return PALETTES.find((p) => p.id === id);
}

/** "Tableau 20 (20)": the name and how many colours it holds. */
export function paletteLabel(palette: Palette | undefined, defaultName: string): string {
  return palette ? `${palette.name} (${palette.colors.length})` : `${defaultName} (${TOKEN_COUNT})`;
}

/** A stylesheet giving each palette's first eight colours to --chart-1…8 under <html data-palette="…">,
 *  so what is drawn in CSS follows the choice too. More specific than the theme's rules. */
export function paletteStyles(): string {
  return PALETTES.map((palette) => {
    const vars = Array.from(
      { length: TOKEN_COUNT },
      (_, i) => `--chart-${i + 1}: ${palette.colors[i % palette.colors.length]};`,
    ).join(" ");
    return `html:root[data-palette="${palette.id}"] { ${vars} }`;
  }).join("\n");
}

export function installPaletteStyles(doc: Document = document): void {
  if (doc.getElementById("palette-styles")) return;
  const style = doc.createElement("style");
  style.id = "palette-styles";
  style.textContent = paletteStyles();
  doc.head.appendChild(style);
}
