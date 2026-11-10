"""How many real dimensions the numeric columns carry (078): standardised PCA with Horn's parallel
analysis, and, one level deeper, a varimax factor analysis with KMO and Bartlett's test.

Everything reads the numeric columns of the association matrix (same choice, filters, sample) and
runs on numpy, scipy and scikit-learn only, without threads: it works in the browser too.
"""

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from sklearn.decomposition import FactorAnalysis

from quantilo.correlations.matrix import Matrix
from quantilo.segments.projection import sample

SEED = 42
MIN_NUMERIC = 5  # R-STR-1
MIN_STRONG = 3
STRONG = 0.5  # Cohen's "large" correlation
MAX_MISSING = 0.3  # R-STR-2: columns missing more than this are left out
MIN_COMPLETE = 0.8  # complete rows kept if they are at least this share, else medians fill
MIN_COLUMNS = 3
MIN_ROWS = 50
PERMUTATIONS = 100  # R-STR-4
PARALLEL_ROWS = 5_000
PERCENTILE = 95
MAX_COMPONENTS = 20  # scree
MAX_SHOWN = 8  # loadings heatmap
BIPLOT_POINTS = 2_000
BIPLOT_ARROWS = 12
FACTOR_ROWS = 20_000  # R-STR-7
FACTOR_ITERATIONS = 300
NAMED = 3  # columns named per component in the sentence
SINGULAR = 1e-10  # smallest eigenvalue of R below this: exactly redundant columns
KMO_LABELS = (  # Kaiser (1974)
    (0.9, "marvelous"),
    (0.8, "meritorious"),
    (0.7, "middling"),
    (0.6, "mediocre"),
    (0.5, "miserable"),
)


@dataclass
class StructureRule:
    """Whether the Correlations page proposes the Structure section (R-STR-1), and why."""

    proposed: bool
    numeric: int  # numeric columns in the matrix
    strong: int  # number × number pairs with |r| ≥ 0.5
    reason: str


@dataclass
class StructureLeftOut:
    name: str
    reason: str


@dataclass
class StructureComponent:
    index: int  # 1 for PC1
    eigenvalue: float
    explained: float  # share of the total variance (0..1)
    cumulative: float
    threshold: float | None  # 95th percentile of the permuted eigenvalues at this rank
    retained: bool  # kept by the parallel analysis
    label: str  # its columns: "sqft_living, sqft_above, grade…"


@dataclass
class BiplotArrow:
    column: str
    x: float  # loading on PC1
    y: float  # loading on PC2


@dataclass
class StructureBiplot:
    x: list[float]  # PC1 score of each drawn row
    y: list[float]
    points: int
    of: int
    arrows: list[BiplotArrow]


@dataclass
class Structure:
    columns: list[str]  # the columns analysed, in matrix order
    left_out: list[StructureLeftOut]
    rows: int  # rows analysed
    total: int  # rows after the filters
    sampled: bool
    dropped: int  # incomplete rows left out
    filled: int  # cells filled with the column median
    components: list[StructureComponent]  # the first MAX_COMPONENTS
    retained: int  # components kept by the parallel analysis
    kaiser: int  # eigenvalues above 1, for comparison
    loadings: list[list[float]]  # columns × the first `shown` components
    shown: int
    biplot: StructureBiplot | None
    conclusion: str
    permutations: int = PERMUTATIONS
    parallel_rows: int = 0  # rows the permutations were drawn from
    notes: list[str] = field(default_factory=list)
    reason: str | None = None  # why nothing could be computed


@dataclass
class Factors:
    columns: list[str]
    left_out: list[StructureLeftOut]
    rows: int  # rows the factor model was fitted on
    sampled: bool
    k: int
    rotation: str | None  # "varimax" (None with a single factor)
    loadings: list[list[float]]  # columns × factors, after rotation
    communalities: list[float]  # h², share of each column the factors explain
    uniqueness: list[float]  # ψ, the rest
    variance: list[float]  # share of the total variance per factor
    labels: list[str]  # each factor's columns
    kmo: float | None
    kmo_label: str
    msa: list[float | None]  # KMO of each column
    bartlett_chi2: float | None
    bartlett_df: int
    bartlett_p: float | None
    conclusion: str
    notes: list[str] = field(default_factory=list)
    reason: str | None = None


# ---------------------------------------------------------------------------------------- rule


def structure_rule(matrix: Matrix) -> StructureRule:
    """R-STR-1: at least 5 numbers and 3 strong pairs of numbers."""
    numeric = sum(1 for member in matrix.columns if member.role == "num")
    strong = sum(1 for p in matrix.pairs if p.metric == matrix.method and abs(p.value) >= STRONG)
    symbol = "ρ" if matrix.method == "spearman" else "r"
    proposed = numeric >= MIN_NUMERIC and strong >= MIN_STRONG
    if proposed:
        reason = (
            f"{numeric} numeric columns, {strong} pairs with |{symbol}| ≥ {STRONG}: they may "
            "measure fewer underlying dimensions."
        )
    elif numeric < MIN_NUMERIC:
        reason = f"Needs at least {MIN_NUMERIC} numeric columns ({numeric} here)."
    else:
        reason = f"Needs at least {MIN_STRONG} pairs with |{symbol}| ≥ {STRONG} ({strong} here)."
    return StructureRule(proposed, numeric, strong, reason)


# ------------------------------------------------------------------------------- preparation


@dataclass
class Prepared:
    names: list[str]
    z: np.ndarray  # rows × columns, z-scores
    left_out: list[StructureLeftOut]
    dropped: int
    filled: int
    notes: list[str]
    reason: str | None = None


def prepare(arrays: dict[str, np.ndarray]) -> Prepared:
    """R-STR-2: constant or mostly missing columns out; complete rows, or medians when too few."""
    left_out: list[StructureLeftOut] = []
    kept: list[str] = []
    for name, values in arrays.items():
        n = len(values)
        present = ~np.isnan(values)
        missing = 1 - present.mean() if n else 1.0
        if missing > MAX_MISSING:
            left_out.append(StructureLeftOut(name, f"{round(100 * missing)}% missing values."))
        elif present.sum() < 2 or np.nanstd(values) == 0:
            left_out.append(StructureLeftOut(name, "Constant: it carries no variance."))
        else:
            kept.append(name)
    notes: list[str] = []
    empty = np.empty((0, len(kept)))
    if len(kept) < MIN_COLUMNS:
        reason = f"Needs at least {MIN_COLUMNS} usable numeric columns ({len(kept)} here)."
        return Prepared(kept, empty, left_out, 0, 0, notes, reason)
    data = np.column_stack([arrays[name] for name in kept])
    complete = ~np.isnan(data).any(axis=1)
    n = len(data)
    dropped = filled = 0
    if complete.mean() >= MIN_COMPLETE:
        dropped = int(n - complete.sum())
        data = data[complete]
        if dropped:
            notes.append(f"{dropped:,} rows with a missing value were left out.")
    else:
        holes = np.isnan(data)
        filled = int(holes.sum())
        medians = np.nanmedian(data, axis=0)
        data = np.where(holes, medians, data)
        notes.append(
            f"Only {complete.mean():.0%} of the rows are complete: {filled:,} missing cells were "
            "filled with the column median, which slightly weakens the links."
        )
    if len(data) < max(MIN_ROWS, len(kept) + 1):
        reason = f"Needs at least {max(MIN_ROWS, len(kept) + 1)} complete rows ({len(data)} here)."
        return Prepared(kept, empty, left_out, dropped, filled, notes, reason)
    spread = data.std(axis=0)
    still = spread > 0  # a column can turn constant once incomplete rows are gone
    for name, ok in zip(kept, still, strict=True):
        if not ok:
            left_out.append(StructureLeftOut(name, "Constant on the complete rows."))
    kept = [name for name, ok in zip(kept, still, strict=True) if ok]
    data = data[:, still]
    if len(kept) < MIN_COLUMNS:
        reason = f"Needs at least {MIN_COLUMNS} usable numeric columns ({len(kept)} here)."
        return Prepared(kept, empty, left_out, dropped, filled, notes, reason)
    z = (data - data.mean(axis=0)) / data.std(axis=0)
    return Prepared(kept, z, left_out, dropped, filled, notes)


def correlation(z: np.ndarray) -> np.ndarray:
    r = (z.T @ z) / len(z)
    np.fill_diagonal(r, 1.0)
    return np.clip((r + r.T) / 2, -1, 1)


def eigen(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending) and eigenvectors, each oriented so its largest entry is positive."""
    values, vectors = np.linalg.eigh(r)
    order = np.argsort(values)[::-1]
    values = np.clip(values[order], 0, None)
    vectors = vectors[:, order]
    biggest = np.abs(vectors).argmax(axis=0)
    signs = np.sign(vectors[biggest, np.arange(vectors.shape[1])])
    signs[signs == 0] = 1
    return values, vectors * signs


def parallel_thresholds(
    z: np.ndarray, seed: int = SEED, permutations: int = PERMUTATIONS
) -> tuple[np.ndarray, int]:
    """R-STR-4: the 95th percentile of each eigenvalue rank when every column is shuffled
    on its own (laws kept, links broken), from at most PARALLEL_ROWS rows."""
    rng = np.random.default_rng(seed)
    rows = sample(len(z), seed, PARALLEL_ROWS)
    sub = z[rows]
    found = np.empty((permutations, z.shape[1]))
    for i in range(permutations):
        shuffled = rng.permuted(sub, axis=0)
        shuffled = (shuffled - shuffled.mean(axis=0)) / shuffled.std(axis=0)
        found[i] = np.sort(np.linalg.eigvalsh(correlation(shuffled)))[::-1]
    return np.percentile(found, PERCENTILE, axis=0), len(sub)


def retained_count(values: np.ndarray, thresholds: np.ndarray) -> int:
    """The leading components above chance, stopping at the first that is not."""
    count = 0
    for value, threshold in zip(values, thresholds, strict=True):
        if value <= threshold:
            break
        count += 1
    return count


def component_label(loadings: np.ndarray, names: list[str]) -> str:
    """R-STR-5: the columns loading ≥ 0.5, strongest first; negative ones after "against"."""
    order = np.argsort(-np.abs(loadings))
    strong = [i for i in order if abs(loadings[i]) >= STRONG] or [int(order[0])]
    positive = [names[i] for i in strong if loadings[i] >= 0]
    negative = [names[i] for i in strong if loadings[i] < 0]

    def listed(items: list[str]) -> str:
        return ", ".join(items[:NAMED]) + ("…" if len(items) > NAMED else "")

    text = listed(positive) if positive else ""
    if negative:
        text = f"{text} against {listed(negative)}" if text else f"against {listed(negative)}"
    return text


def _percent(share: float) -> str:
    return f"{round(100 * share)}%"


def _conclusion(retained: int, cumulative: float, columns: int, labels: list[str]) -> str:
    if retained == 0:
        return (
            "No component stands out from chance: these numeric columns do not share a few common "
            "dimensions (parallel analysis)."
        )
    named = "; ".join(f"PC{i + 1} = {label}" for i, label in enumerate(labels[: min(retained, 4)]))
    if retained > 4:
        named += f"; and {retained - 4} more"
    head = "One component explains" if retained == 1 else f"{retained} components explain"
    return f"{head} {_percent(cumulative)} of the variance of {columns} numeric columns: {named}."


def _rounded(values: np.ndarray, digits: int = 4) -> list[float]:
    return [round(float(v), digits) for v in values]


def _refused(prepared: Prepared, rows: int, total: int) -> Structure:
    return Structure(
        prepared.names,
        prepared.left_out,
        rows,
        total,
        total > rows,
        prepared.dropped,
        prepared.filled,
        [],
        0,
        0,
        [],
        0,
        None,
        prepared.reason or "",
        notes=prepared.notes,
        reason=prepared.reason,
    )


def pca_structure(
    arrays: dict[str, np.ndarray], total: int | None = None, seed: int = SEED
) -> Structure:
    """R-STR-3 to 6: standardised PCA, parallel analysis, sentence, scree, loadings, biplot."""
    loaded = len(next(iter(arrays.values()))) if arrays else 0
    total = loaded if total is None else total
    prepared = prepare(arrays)
    if prepared.reason:
        return _refused(prepared, loaded, total)
    z, names = prepared.z, prepared.names
    values, vectors = eigen(correlation(z))
    loadings = vectors * np.sqrt(values)
    thresholds, parallel_rows = parallel_thresholds(z, seed)
    retained = retained_count(values, thresholds)
    kaiser = int((values > 1).sum())
    p = len(names)
    shares = values / values.sum()
    cumulative = np.cumsum(shares)
    labels = [component_label(loadings[:, i], names) for i in range(min(p, MAX_COMPONENTS))]
    components = [
        StructureComponent(
            i + 1,
            round(float(values[i]), 4),
            round(float(shares[i]), 4),
            round(float(cumulative[i]), 4),
            round(float(thresholds[i]), 4),
            i < retained,
            labels[i],
        )
        for i in range(min(p, MAX_COMPONENTS))
    ]
    shown = min(p, max(retained, 2), MAX_SHOWN)
    notes = list(prepared.notes)
    if kaiser != retained:
        notes.append(
            f"Kaiser's rule (eigenvalue above 1) would keep {kaiser}; the parallel analysis keeps "
            f"{retained}, since Kaiser tends to keep too many components when there are many "
            "columns."
        )
    rows = len(z)
    biplot = _biplot(z, vectors, loadings, names, seed)
    conclusion = _conclusion(
        retained, float(cumulative[retained - 1]) if retained else 0.0, p, labels
    )
    return Structure(
        names,
        prepared.left_out,
        rows,
        total,
        total > loaded,
        prepared.dropped,
        prepared.filled,
        components,
        retained,
        kaiser,
        [_rounded(row) for row in loadings[:, :shown]],
        shown,
        biplot,
        conclusion,
        PERMUTATIONS,
        parallel_rows,
        notes,
    )


def _biplot(
    z: np.ndarray, vectors: np.ndarray, loadings: np.ndarray, names: list[str], seed: int
) -> StructureBiplot | None:
    if vectors.shape[1] < 2:
        return None
    drawn = sample(len(z), seed, BIPLOT_POINTS)
    scores = z[drawn] @ vectors[:, :2]
    reach = np.hypot(loadings[:, 0], loadings[:, 1])
    order = np.argsort(-reach)[:BIPLOT_ARROWS]
    arrows = [
        BiplotArrow(names[i], round(float(loadings[i, 0]), 4), round(float(loadings[i, 1]), 4))
        for i in order
    ]
    return StructureBiplot(
        _rounded(scores[:, 0]), _rounded(scores[:, 1]), len(drawn), len(z), arrows
    )


# -------------------------------------------------------------------------- factor analysis


def ledermann(p: int) -> int:
    """The most factors an identified model allows: (p - k)^2 >= p + k."""
    k = 0
    while k + 1 < p and (p - k - 1) ** 2 >= p + k + 1:
        k += 1
    return k


def kmo(r: np.ndarray) -> tuple[float | None, list[float | None], bool]:
    """Kaiser-Meyer-Olkin: correlations against partial correlations, overall and per column.
    Returns whether R was singular (then the pseudo-inverse stands in for the inverse)."""
    singular = bool(np.linalg.eigvalsh(r).min() < SINGULAR)
    inverse = np.linalg.pinv(r) if singular else np.linalg.inv(r)
    diagonal = np.sqrt(np.clip(np.diag(inverse), 1e-12, None))
    partial = -inverse / np.outer(diagonal, diagonal)
    off = ~np.eye(len(r), dtype=bool)
    r2 = np.where(off, r**2, 0.0)
    q2 = np.where(off, partial**2, 0.0)
    total = r2.sum() + q2.sum()
    overall = float(r2.sum() / total) if total > 0 else None
    per_column: list[float | None] = []
    for j in range(len(r)):
        denominator = r2[j].sum() + q2[j].sum()
        per_column.append(round(float(r2[j].sum() / denominator), 4) if denominator > 0 else None)
    return (round(overall, 4) if overall is not None else None), per_column, singular


def kmo_label(value: float | None) -> str:
    if value is None:
        return "unknown"
    return next((label for low, label in KMO_LABELS if value >= low), "unacceptable")


def bartlett(r: np.ndarray, n: int) -> tuple[float | None, int, float | None]:
    """Bartlett's sphericity: are the columns uncorrelated (R = I)?"""
    p = len(r)
    df = p * (p - 1) // 2
    sign, logdet = np.linalg.slogdet(r)
    if sign <= 0 or not math.isfinite(logdet):
        return None, df, None
    chi2 = -(n - 1 - (2 * p + 5) / 6) * logdet
    return round(float(chi2), 2), df, float(stats.chi2.sf(chi2, df))


def _p_text(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def factor_analysis(
    arrays: dict[str, np.ndarray],
    k: int | None = None,
    retained: int | None = None,
    seed: int = SEED,
) -> Factors:
    """R-STR-7 to 9: varimax factor analysis on the z-scores, KMO and Bartlett's test.
    `k=None` takes the parallel analysis' count (`retained`, computed when not given)."""
    prepared = prepare(arrays)
    names = prepared.names
    if prepared.reason:
        return _no_factors(prepared)
    z = prepared.z
    p = len(names)
    notes = list(prepared.notes)
    r = correlation(z)
    overall, per_column, singular = kmo(r)
    chi2, df, p_value = bartlett(r, len(z)) if not singular else (None, p * (p - 1) // 2, None)
    if singular:
        notes.append(
            "Some columns are exactly redundant (one is a combination of others): KMO uses the "
            "pseudo-inverse and Bartlett's test cannot be computed."
        )
    if retained is None and k is None:
        values, _ = eigen(r)
        retained = retained_count(values, parallel_thresholds(z, seed)[0])
    wanted = k if k is not None else max(1, retained or 0)
    most = max(1, ledermann(p))
    chosen = max(1, min(wanted, most))
    if chosen < wanted:
        notes.append(
            f"{wanted} factors cannot be identified from {p} columns: {chosen} at most "
            "(Ledermann's bound)."
        )
    rows = sample(len(z), seed, FACTOR_ROWS)
    fitted_on = z[rows]
    rotation = "varimax" if chosen >= 2 else None
    model = FactorAnalysis(
        n_components=chosen, rotation=rotation, max_iter=FACTOR_ITERATIONS, random_state=seed
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(fitted_on)
    if any("converge" in str(w.message).lower() for w in caught):
        notes.append("The factor model did not fully converge: read the loadings as approximate.")
    loadings = np.asarray(model.components_).T  # columns × factors
    biggest = np.abs(loadings).argmax(axis=0)
    signs = np.sign(loadings[biggest, np.arange(chosen)])
    signs[signs == 0] = 1
    loadings = loadings * signs
    variance = (loadings**2).sum(axis=0) / p
    order = np.argsort(-variance)
    loadings, variance = loadings[:, order], variance[order]
    communalities = (loadings**2).sum(axis=1)
    uniqueness = np.asarray(model.noise_variance_)
    labels = [component_label(loadings[:, i], names) for i in range(chosen)]
    weak = [n for n, m in zip(names, per_column, strict=True) if m is not None and m < 0.5]
    if weak and overall is not None and overall >= 0.5:  # below, the sentence says it all
        one = len(weak) == 1
        notes.append(
            f"KMO below 0.5 for {', '.join(weak)}: {'it shares' if one else 'they share'} little "
            f"with the others; consider leaving {'it' if one else 'them'} out."
        )
    if p_value is not None and len(z) > 5_000:
        notes.append(
            f"With {len(z):,} rows, Bartlett's test rejects almost any set of columns: read KMO "
            "first."
        )
    return Factors(
        names,
        prepared.left_out,
        len(fitted_on),
        len(fitted_on) < len(z),
        chosen,
        rotation,
        [_rounded(row) for row in loadings],
        _rounded(communalities),
        _rounded(uniqueness),
        _rounded(variance),
        labels,
        overall,
        kmo_label(overall),
        per_column,
        chi2,
        df,
        p_value,
        _factor_conclusion(overall, p_value, chosen, rotation, labels),
        notes,
    )


def _no_factors(prepared: Prepared) -> Factors:
    return Factors(
        columns=prepared.names,
        left_out=prepared.left_out,
        rows=0,
        sampled=False,
        k=0,
        rotation=None,
        loadings=[],
        communalities=[],
        uniqueness=[],
        variance=[],
        labels=[],
        kmo=None,
        kmo_label="unknown",
        msa=[],
        bartlett_chi2=None,
        bartlett_df=0,
        bartlett_p=None,
        conclusion=prepared.reason or "",
        notes=prepared.notes,
        reason=prepared.reason,
    )


def _factor_conclusion(
    value: float | None, p_value: float | None, k: int, rotation: str | None, labels: list[str]
) -> str:
    label = kmo_label(value)
    if value is None:
        head = "KMO could not be computed."
    elif value < 0.5:
        head = f"KMO = {value:.2f} ({label}): these columns share too little for a factor analysis."
    elif value < 0.6:
        head = f"KMO = {value:.2f} ({label}): a factor analysis is possible but weak."
    else:
        head = f"KMO = {value:.2f} ({label}): the columns share enough to be factored."
    if p_value is not None:
        verdict = "rejects" if p_value < 0.05 else "does not reject"
        head += f" Bartlett's test {verdict} independent columns ({_p_text(p_value)})."
    named = "; ".join(f"F{i + 1} = {text}" for i, text in enumerate(labels[:4]))
    if k > 4:
        named += f"; and {k - 4} more"
    tail = f"{k} factors after {rotation} rotation" if rotation else "One factor"
    return f"{head} {tail}: {named}."
