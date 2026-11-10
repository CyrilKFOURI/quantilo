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
