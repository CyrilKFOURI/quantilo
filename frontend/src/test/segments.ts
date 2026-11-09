/** Server answers for the Segments tests (shapes of the generated API types). */

const decision = (column: string, type: string, use: boolean, why: string[], rules: string[], extra: Record<string, unknown> = {}) => ({
  column,
  type,
  use,
  rules,
  why,
  impute: use ? "median" : null,
  indicator: false,
  transform: null,
  encoding: null,
  scaling: use && type.startsWith("numeric") ? "robust" : null,
  order: null,
  auto: {},
  manual: [],
  options: use
    ? {
        use: [{ value: "yes", reason: null }, { value: "no", reason: null }],
        transform: [{ value: "none", reason: null }, { value: "log1p", reason: null }, { value: "yeojohnson", reason: null }],
        scaling: [{ value: "none", reason: null }, { value: "standard", reason: null }, { value: "robust", reason: null }, { value: "minmax", reason: null }],
      }
    : { use: [{ value: "yes", reason: why[0] ?? null }, { value: "no", reason: null }] },
  reactivable: !use && rules[0] !== "R-SEL-02",
  leak: false,
  missing: 0,
  ...extra,
});

export const preparation = {
  decisions: [
    decision("customer_id", "identifier", false, ["Identifier (300 distinct values)."], ["R-SEL-02"]),
    decision("signup", "datetime", false, ["Raw date: it says when, not what kind of row."], ["R-DIST-03"]),
    decision("income", "numeric_continuous", true, ["Missing values → median (skewness 0.10).", "Robust scaling (median, IQR)."], ["R-NUM-03", "R-DIST-01"]),
    decision("age", "numeric_continuous", true, ["Missing values → mean (nearly symmetric).", "Robust scaling (median, IQR)."], ["R-NUM-02", "R-DIST-01"]),
  ],
  inputs: 2,
  excluded: 2,
  choices: {
    columns: {},
    numeric_impute: "auto",
    indicator_share: 0.05,
    rare_share: 0.01,
    rare_rows: 10,
    one_hot_limit: 15,
    target_smoothing: 10,
    outliers: "none",
    imbalance: "auto",
    selection: [],
  },
  custom: false,
  warnings: [],
  selection: [],
  balance: null,
  dropped_rows: 0,
  distance: true,
};

const option = (key: string, label: string, available = true, reason: string | null = null) => ({ key, label, available, reason });

export const plan = {
  preparation,
  rows: 300,
  fit_rows: 300,
  sampled: false,
  method: "kmeans",
  method_why: "k-means: mostly numbers.",
  categorical_share: 0,
  methods: [
    option("kmeans", "k-means", true, "Auto with mostly numbers."),
    option("kprototypes", "k-prototypes", true, "Auto from 30% of category columns."),
    option("gmm", "Gaussian mixture"),
    option("ward", "Hierarchical (Ward)"),
    option("hdbscan", "HDBSCAN"),
  ],
};

const score = (k: number, silhouette: number) => ({ k, silhouette, calinski_harabasz: 100 * k, davies_bouldin: 1 / k, inertia: 1000 / k, bic: null });
const trait = (column: string, text: string, effect: number) => ({ column, kind: "number", text, effect, value: null, segment: 1, overall: 1 });
const segment = (id: number, name: string, size: number, traits: unknown[]) => ({
  id,
  name,
  size,
  share: size / 300,
  traits,
  rules: {
    rules: [{ parts: [], text: `income > ${20 + id},000`, rows: size, precision: 0.95 }],
    recall: 0.91,
    precision: 0.95,
    text: "These rules describe 91 % of the segment (95 % of the rows they match are in it).",
  },
  membership: null,
  grouped: true,
});

export const run = {
  id: "aaaaaaaaaaa1",
  dataset_id: "d1",
  created_at: "2026-11-07T10:00:00Z",
  request: { columns: null, method: "auto", k: null, preparation: null, seed: 42 },
  conditions: [],
  created_dataset_id: null,
  result: {
    method: "kmeans",
    method_label: "k-means",
    method_why: "k-means: mostly numbers.",
    k: 3,
    choice: { k: 3, forced: false, why: "3 segments: best silhouette, and the elbow is at 3.", votes: { silhouette: 3 } },
    scores: [score(2, 0.5), score(3, 0.71), score(4, 0.6)],
    chosen: score(3, 0.71),
    stability: { ari: 0.97, runs: Array.from({ length: 10 }, () => 0.97), rows: 240, stable: true, warning: null },
    segments: [
      segment(0, "High income · Low age", 120, [trait("income", "income: 2.1× the average", 1.8)]),
      segment(1, "Low income", 100, [trait("income", "income: 0.4× the average", -1.2)]),
      segment(2, "High age", 80, [trait("age", "age: 1.6× the average", 1.4)]),
    ],
    profiles: [
      {
        column: "income",
        kind: "number",
        overall: { value: 25000, share: null, effect: null, strong: false },
        cells: [
          { value: 52000, share: null, effect: 1.8, strong: true },
          { value: 10000, share: null, effect: -1.2, strong: true },
          { value: 24000, share: null, effect: -0.1, strong: false },
        ],
      },
    ],
    map: { projection: "pca", x: [0, 1, 2], y: [0, 1, 2], segments: [0, 1, 2], rows: [0, 1, 2], centres: [], explained: [0.7, 0.2], points: 3, of: 300 },
    dendrogram: null,
    preparation,
    inputs: 2,
    categorical_share: 0,
    rows: 300,
    fit_rows: 300,
    sampled: false,
    seconds: 1.2,
    warnings: [],
    tsne: null,
  },
};

export const summary = {
  id: run.id,
  created_at: run.created_at,
  method: "kmeans",
  method_label: "k-means",
  k: 3,
  forced: false,
  rows: 300,
  seconds: 1.2,
  silhouette: 0.71,
  stability: 0.97,
  columns: ["income", "age"],
  names: run.result.segments.map((s) => s.name),
  filtered: false,
  created_dataset_id: null,
};

const reason = (column: string, text: string, contribution: number) => ({ column, contribution, text });
const listed = (rank: number, score: number) => ({
  row: rank * 10,
  key: `customer_id=C${rank}`,
  rank,
  score,
  reasons: [reason("orders", `orders is in the bottom 0.1 % (${rank})`, 0.2), reason("revenue", "revenue = 3,012 (typical: 1,500)", 0.1)],
  combination: rank === 1 ? "orders = 0 while revenue = 3,012: this combination appears in 0.05 % of rows" : null,
  values: { customer_id: `C${rank}`, orders: rank, revenue: 3012 },
  segment: null,
});

export const unusualRun = {
  id: "bbbbbbbbbbb1",
  dataset_id: "d1",
  created_at: "2026-11-07T10:05:00Z",
  request: { columns: null, method: "auto", preparation: null, seed: 42, segments_run: null },
  conditions: [],
  marks: {},
  result: {
    method: "isolation_forest",
    method_label: "Isolation Forest",
    method_why: "Isolation Forest (200 trees): categories among the columns.",
    rows: 2000,
    fit_rows: 2000,
    sampled: false,
    threshold: { value: 0.7, rows: 2, rule: "break", why: "The scores drop sharply over the 2 most unusual rows, then level off." },
    curve: { ranks: [1, 2, 3, 4], scores: [0.8, 0.7, 0.5, 0.45] },
    listed: [listed(1, 0.8), listed(2, 0.7), listed(3, 0.5), listed(4, 0.45)],
    columns: ["customer_id", "orders", "revenue"],
    used: ["orders", "revenue"],
    preparation,
    hidden_expected: 1,
    seconds: 0.8,
    warnings: [],
  },
};

export const unusualSummary = {
  id: unusualRun.id,
  created_at: unusualRun.created_at,
  method_label: "Isolation Forest",
  rows: 2000,
  proposed: 2,
  seconds: 0.8,
};
