/** Variable page fixtures shaped like the API's VariableProfile, Fit and Outliers. */

const describe = {
  rows: 1200,
  count: 1190,
  missing: 10,
  infinite: 0,
  zeros: 3,
  negatives: 0,
  distinct: 900,
  distinct_approx: false,
  mean: 136.2,
  std: 208.4,
  cv: 1.53,
  min: 3.5,
  p1: 10.7,
  p5: 19.7,
  p10: 28.3,
  q1: 47.9,
  median: 90.9,
  q3: 156.1,
  p90: 290,
  p95: 400,
  p99: 900,
  max: 3348,
  iqr: 108.2,
  mad: 50.1,
  skewness: 8.86,
  kurtosis: 114,
  sum: 162_078,
};

const test = (key: string, p: number | null, note: string | null = null) => ({
  key,
  statistic: p === null ? null : 0.42,
  p_value: p,
  n: 1190,
  note,
});

export function numericProfile(pValues: (number | null)[] = [0.0001, 0.0001, 0.01, 0.0001, 0.03]) {
  const keys = ["shapiro", "dagostino", "anderson", "jarque_bera", "lilliefors"];
  return {
    name: "price",
    type: "numeric_continuous",
    numeric: {
      describe,
      normality: {
        n: 1190,
        sample: 1190,
        tests: keys.map((key, index) => test(key, pValues[index] ?? null, pValues[index] === null ? "Needs at least 20 values." : null)),
        reason: null,
      },
      kde: { x: [0, 1, 2], y: [0.1, 0.5, 0.1] },
      ecdf: { x: [0, 1, 2], y: [0, 0.5, 1] },
      qq: { theoretical: [-1, 0, 1], sample: [40, 90, 160], slope: 80, intercept: 90 },
      sample: [10, 20, 30],
    },
    categorical: null,
    date: null,
    text: null,
    identifier: null,
  };
}

export const categoricalProfile = {
  name: "city",
  type: "categorical_nominal",
  numeric: null,
  categorical: {
    rows: 100,
    missing: 1,
    distinct: 3,
    entropy: 0.52,
    normalized_entropy: 0.33,
    herfindahl: 0.82,
    top: { label: "Paris", count: 90 },
    top_share: 0.909,
    imbalanced: true,
    rare_count: 0,
    rare: [],
    chi2: 150,
    chi2_p: 0.0001,
  },
  date: null,
  text: null,
  identifier: null,
};

// 050: survey answers, mostly French, with a template answered 3 times
export const textProfile = {
  name: "survey_answer",
  type: "text",
  numeric: null,
  categorical: null,
  date: null,
  text: {
    rows: 600,
    missing: 58,
    distinct: 190,
    min_length: 24,
    max_length: 61,
    mean_length: 44.2,
    words: [
      { label: "livraison", count: 95 },
      { label: "produit", count: 90 },
    ],
    bigrams: [
      { label: "très rapide", count: 92 },
      { label: "service client", count: 88 },
    ],
    trigrams: [{ label: "produit est arrivé", count: 90 }],
    distinct_words: 61,
    median_words: 9,
    p90_words: 11,
    language: "fr",
    language_label: "French",
    language_share: 0.9,
    languages: [
      { label: "French", count: 90 },
      { label: "English", count: 10 },
    ],
    shortest: ["Delivery was fine selon moi."],
    longest: ["Le service client était assez décevant malgré quelques soucis."],
    repeated: [{ label: "La qualité est parfait et je recommande.", count: 3 }],
    repeated_count: 1,
    repeated_rows: 3,
    sampled_texts: null,
  },
  identifier: null,
};

export const fit = {
  n: 1190,
  grid: [0, 1, 2],
  sample_quantiles: [1, 2, 3],
  laws: [
    {
      key: "lognormal",
      label: "Log-normal",
      discrete: false,
      params: { s: 0.91, scale: 87.6 },
      log_likelihood: -3478,
      aic: 6961,
      bic: 6970,
      ks: 0.029,
      density: [0.1, 0.2, 0.1],
      quantiles: [1, 2, 3],
    },
    {
      key: "normal",
      label: "Normal",
      discrete: false,
      params: { loc: 136, scale: 208 },
      log_likelihood: -4053,
      aic: 8111,
      bic: 8120,
      ks: 0.264,
      density: [0.1, 0.1, 0.1],
      quantiles: [1, 2, 3],
    },
  ],
  transforms: [
    {
      key: "log",
      lmbda: null,
      skewness_before: 8.86,
      skewness_after: 0.03,
      shapiro_before: 0.0001,
      shapiro_after: 0.045,
      bins: [{ start: 0, end: 1, count: 5 }],
    },
  ],
  reason: null,
};

export const outliers = {
  method: "iqr",
  rules: [
    { method: "iqr", low: -114, high: 317, count: 41, pct: 6.83 },
    { method: "iqr3", low: -275, high: 479, count: 14, pct: 2.33 },
    { method: "zscore", low: -488, high: 761, count: 6, pct: 1 },
    { method: "robust", low: -165, high: 347, count: 35, pct: 5.83 },
  ],
  rows: [
    { row: 257, value: 3348 },
    { row: 168, value: 2165 },
  ],
};

/** 056: the right tail of a heavy column. */
export const tail = {
  n: 1190,
  sample: 1190,
  sampled: false,
  rule: { fires: true, reason: "Heavy tail: excess kurtosis 12.0 (above 3), 60 values above the threshold.", kurtosis: 12, xi: null },
  reason: null,
  threshold: { value: 300, level: 0.95, exceedances: 60, linear_r2: 0.93, why: "95th percentile: the mean excess is roughly linear above it (R² 0.93)." },
  xi: 0.37,
  xi_low: 0.2,
  xi_high: 0.58,
  sigma: 120,
  rate: 0.05,
  endpoint: null,
  stability: [
    { level: 0.9, threshold: 200, exceedances: 119, xi: 0.4 },
    { level: 0.95, threshold: 300, exceedances: 60, xi: 0.37 },
    { level: 0.975, threshold: 420, exceedances: 30, xi: null },
  ],
  return_levels: [
    { period: 100, value: 1015, low: 990, high: 1040, beyond_data: false, empirical: 1020 },
    { period: 1000, value: 2966, low: 2800, high: 3100, beyond_data: false, empirical: 2980 },
    { period: 10000, value: 7559, low: 6900, high: null, beyond_data: true, empirical: null },
  ],
  risk: [
    { level: 0.95, var: 365, es: 828.8, rows: 60 },
    { level: 0.99, var: 1018, es: 1833, rows: 12 },
  ],
  hill: { x: [10, 60], y: [0.5, 0.45], low: [0.2, 0.33], high: [0.8, 0.57], count: [] },
  mean_excess: { x: [100, 300], y: [200, 260], low: [], high: [], count: [400, 60] },
  curve: { x: [20, 1000], y: [300, 2966], low: [], high: [], count: [] },
  observed: { x: [20, 1190], y: [300, 3348], low: [], high: [], count: [] },
  conclusion: "Heavy right tail (ξ = 0.37): the 1-in-1,000 value is 2.9× the 99th percentile.",
  notes: [],
};

export const robust = {
  fires: true,
  reason: "6.8 % of values are outliers by the IQR rule (above 1 %): the mean and the standard deviation are pulled by them.",
  outliers: 81,
  outlier_share: 0.068,
  median: 90,
  mad_sigma: 60,
  iqr_sigma: 58,
  huber_center: 95,
  huber_scale: 62,
  mean_shift: 0.38,
  sample: 1190,
  note: null,
};
