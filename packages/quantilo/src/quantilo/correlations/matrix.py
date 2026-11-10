"""Mixed association matrix: Pearson / Spearman, bias-corrected Cramér's V, correlation ratio η.

The rows are loaded once (a seeded sample beyond MAX_ROWS) into numpy: numbers as floats with
NaN for missing values, categories as integer codes with -1 for missing.
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import duckdb
import numpy as np
import pyarrow as pa
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from quantilo.compare.analyze import label_sql, number_sql
from quantilo.compare.multiple import Correction, MultipleTesting, correct
from quantilo.correlations.evidence import Evidence, effect_size, grade
from quantilo.typing.types import AnalysisType, Parser
from quantilo.variable.source import SEED

Role = Literal["num", "cat"]
Method = Literal["pearson", "spearman"]
Metric = Literal["pearson", "spearman", "cramers_v", "eta"]

ROLES: dict[str, Role] = {
    "numeric_continuous": "num",
    "numeric_discrete": "num",
    "categorical_nominal": "cat",
    "categorical_ordinal": "cat",
    "boolean": "cat",
}
EXCLUDED_REASONS = {
    "datetime": "Dates: compare them with a numeric or category on the Compare page.",
    "text": "Free text has no association measure.",
    "identifier": "Identifiers carry no association.",
    "ignored": "Ignored column.",
}
MAX_ROWS = 200_000
MAX_COLUMNS = 60
MAX_CATEGORIES = 30
MOSTLY_UNIQUE = 0.5  # categories with more distinct labels than this share of rows look like ids
MIN_PAIR_ROWS = 3


@dataclass
class MatrixColumn:
    name: str
    type: AnalysisType
    parser: Parser


@dataclass
class MatrixMember:
    name: str
    type: str
    role: Role


@dataclass
class Excluded:
    name: str
    reason: str


@dataclass
class Pair:
    a: str
    b: str
    metric: Metric
    value: float  # signed for num × num, 0..1 otherwise
    p_value: float | None
    n: int
    q_value: float | None = None  # p adjusted over every pair, by `Matrix.multiple.method`
    evidence: Evidence = "none"


@dataclass
class Matrix:
    method: Method
    columns: list[MatrixMember]
    values: list[list[float | None]]  # symmetric, 1 on the diagonal
    metrics: list[list[Metric | None]]
    n: list[list[int]]
    clustered: list[int]  # column order grouping related variables
    rows: int  # rows the matrix was computed on
    total: int  # rows after the global filters
    sampled: bool
    excluded: list[Excluded] = field(default_factory=list)
    truncated: int = 0  # eligible columns left out beyond MAX_COLUMNS
    pairs: list[Pair] = field(default_factory=list)  # every computed pair, strongest first
    multiple: MultipleTesting | None = None  # the correction over the pairs (051)


def eligible(
    columns: list[MatrixColumn], chosen: list[str] | None
) -> tuple[list[MatrixColumn], list[Excluded], int]:
    """Columns that can enter the matrix (in file order), those that can't, and the overflow."""
    excluded: list[Excluded] = []
    usable: list[MatrixColumn] = []
    for column in columns:
        if chosen is not None and column.name not in chosen:
            continue
        if column.type in ROLES:
            usable.append(column)
        else:
            excluded.append(
                Excluded(column.name, EXCLUDED_REASONS.get(column.type, "Not supported."))
            )
    return usable[:MAX_COLUMNS], excluded, max(0, len(usable) - MAX_COLUMNS)


@dataclass
class Loaded:
    arrays: dict[str, np.ndarray]  # floats (NaN = missing) or category codes (-1 = missing)
    labels: dict[str, list[str]]  # category labels by code (most frequent first)
    rows: int  # rows loaded
    total: int  # rows after filters


def load(connection: duckdb.DuckDBPyConnection, source: str, columns: list[MatrixColumn]) -> Loaded:
    (total,) = connection.execute(f"SELECT count(*) FROM {source}").fetchone() or (0,)
    expressions = [
        f"{number_sql(c)} AS c{i}" if ROLES[c.type] == "num" else f"{label_sql(c)} AS c{i}"
        for i, c in enumerate(columns)
    ]
    query = f"SELECT {', '.join(expressions)} FROM {source}"
    if total > MAX_ROWS:
        # Sort the row numbers only, then fetch those rows: sorting 60 wide columns is slow.
        picked = (
            f"SELECT file_row_number FROM {source} "
            f"ORDER BY hash(file_row_number + {SEED}) LIMIT {MAX_ROWS}"
        )
        query += f" WHERE file_row_number IN ({picked})"
    # execute(): the connection may hold an unfinished result, which DuckDB 2 refuses for sql().
    table = connection.execute(query).to_arrow_table()
    assert table is not None
    arrays: dict[str, np.ndarray] = {}
    labels: dict[str, list[str]] = {}
    for i, column in enumerate(columns):
        values = table.column(f"c{i}").combine_chunks()
        if ROLES[column.type] == "num":
            data = np.asarray(values.to_numpy(zero_copy_only=False), dtype=float)
            arrays[column.name] = np.where(np.isfinite(data), data, np.nan)
        else:
            arrays[column.name], labels[column.name] = codes(values)
    return Loaded(arrays, labels, int(table.num_rows), int(total))


def codes(values: pa.Array) -> tuple[np.ndarray, list[str]]:
    """Category labels → codes by frequency (most frequent first), the tail merged, -1 = missing."""
    encoded = values.dictionary_encode()
    indices = np.asarray(encoded.indices.fill_null(-1).to_numpy(zero_copy_only=False), dtype=int)
    dictionary = [str(label) for label in encoded.dictionary.to_pylist()]
    present = indices >= 0
    counts = np.bincount(indices[present], minlength=len(dictionary))
    order = np.argsort(-counts, kind="stable")
    rank = np.empty(len(dictionary), dtype=int)
    rank[order] = np.arange(len(dictionary))
    result = np.full(len(indices), -1, dtype=int)
    result[present] = np.minimum(rank[indices[present]], MAX_CATEGORIES)
    return result, [dictionary[i] for i in order]


def _quiet(compute):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute()


def ranks(values: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan)
    present = ~np.isnan(values)
    result[present] = stats.rankdata(values[present])
    return result


def correlation(x: np.ndarray, y: np.ndarray) -> tuple[float | None, int]:
    both = ~np.isnan(x) & ~np.isnan(y)
    n = int(both.sum())
    if n < MIN_PAIR_ROWS:
        return None, n
    a, b = x[both], y[both]
    if a.std() == 0 or b.std() == 0:
        return None, n
    return float(np.corrcoef(a, b)[0, 1]), n


def correlation_matrix(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise-complete Pearson r of every column pair at once, and the pair counts.

    Missing values are zeroed and masked: every sum is a matrix product over rows where both
    columns are present.
    """
    mask = (~np.isnan(data)).astype(float)
    x = np.nan_to_num(data)
    n = mask.T @ mask
    sum_x = x.T @ mask  # [i, j]: sum of column i over rows where j is present too
    sum_xx = (x * x).T @ mask
    sum_xy = x.T @ x
    with np.errstate(invalid="ignore", divide="ignore"):
        covariance = n * sum_xy - sum_x * sum_x.T
        spread = (n * sum_xx - sum_x**2) * (n * sum_xx - sum_x**2).T
        r = covariance / np.sqrt(spread)
    r[(n < MIN_PAIR_ROWS) | ~np.isfinite(r)] = np.nan
    return np.clip(r, -1, 1), n.astype(int)


def correlation_p(r: float, n: int) -> float | None:
    if n < MIN_PAIR_ROWS:
        return None
    if abs(r) >= 1:
        return 0.0
    t = r * math.sqrt((n - 2) / (1 - r * r))
    return float(2 * stats.t.sf(abs(t), n - 2))


def cramers_v(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None, int]:
    """Bias-corrected Cramér's V (Bergsma 2013) and the χ² p-value."""
    both = (x >= 0) & (y >= 0)
    n = int(both.sum())
    if n < MIN_PAIR_ROWS:
        return None, None, n
    a, b = x[both], y[both]
    rows, columns = a.max() + 1, b.max() + 1
    table = np.bincount(a * columns + b, minlength=rows * columns).reshape(rows, columns)
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    r, k = table.shape
    if r < 2 or k < 2:
        return None, None, n
    result: Any = _quiet(lambda: stats.chi2_contingency(table, correction=False))
    phi2 = result.statistic / n
    phi2_corrected = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corrected = r - (r - 1) ** 2 / (n - 1)
    k_corrected = k - (k - 1) ** 2 / (n - 1)
    denominator = min(k_corrected - 1, r_corrected - 1)
    v = math.sqrt(phi2_corrected / denominator) if denominator > 0 else None
    return v, float(result.pvalue), n


def eta(values: np.ndarray, groups: np.ndarray) -> tuple[float | None, float | None, int]:
    """Correlation ratio η = sqrt(between-group / total sum of squares), with the ANOVA p-value."""
    both = ~np.isnan(values) & (groups >= 0)
    n = int(both.sum())
    if n < MIN_PAIR_ROWS:
        return None, None, n
    v, g = values[both], groups[both]
    counts = np.bincount(g)
    sums = np.bincount(g, weights=v)
    used = counts > 0
    k = int(used.sum())
    mean = v.mean()
    total = float(((v - mean) ** 2).sum())
    if k < 2 or total == 0:
        return None, None, n
    between = float((sums[used] ** 2 / counts[used]).sum() - n * mean**2)
    between = min(max(between, 0.0), total)
    within = total - between
    p = None
    if n > k and within > 0:
        f = (between / (k - 1)) / (within / (n - k))
        p = float(stats.f.sf(f, k - 1, n - k))
    elif within == 0:
        p = 0.0
    return math.sqrt(between / total), p, n


def cluster_order(values: list[list[float | None]]) -> list[int]:
    size = len(values)
    if size < 3:
        return list(range(size))
    distance = np.array([[1 - abs(v) if v is not None else 1.0 for v in row] for row in values])
    np.fill_diagonal(distance, 0)
    distance = np.clip((distance + distance.T) / 2, 0, 1)
    tree = linkage(squareform(distance, checks=False), method="average")
    return [int(i) for i in leaves_list(tree)]


def association_matrix(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    columns: list[MatrixColumn],
    method: Method = "pearson",
    chosen: list[str] | None = None,
    correction: Correction = "auto",
) -> tuple[Matrix, Loaded]:
    usable, excluded, truncated = eligible(columns, chosen)
    loaded = load(connection, source, usable) if usable else Loaded({}, {}, 0, 0)
    for column in list(usable):
        labels = loaded.labels.get(column.name)
        present = int((loaded.arrays[column.name] >= 0).sum()) if labels is not None else 0
        if labels is not None and present >= 20 and len(labels) > MOSTLY_UNIQUE * present:
            usable.remove(column)
            excluded.append(
                Excluded(column.name, "Mostly unique values: it behaves like an identifier.")
            )
    roles: dict[str, Role] = {c.name: ROLES[c.type] for c in usable}
    # Columns left out above were loaded already: drop them from the arrays too.
    arrays = {name: data for name, data in loaded.arrays.items() if name in roles}
    rows, total = loaded.rows, loaded.total
    if method == "spearman":
        arrays = {
            name: ranks(data) if roles[name] == "num" else data for name, data in arrays.items()
        }
    numeric = [c.name for c in usable if roles[c.name] == "num"]
    position = {name: k for k, name in enumerate(numeric)}
    r_all, n_all = (
        correlation_matrix(np.column_stack([arrays[name] for name in numeric]))
        if numeric
        else (np.empty((0, 0)), np.empty((0, 0), dtype=int))
    )
    size = len(usable)
    values: list[list[float | None]] = [[None] * size for _ in range(size)]
    metrics: list[list[Metric | None]] = [[None] * size for _ in range(size)]
    counts = [[0] * size for _ in range(size)]
    pairs: list[Pair] = []
    for i, first in enumerate(usable):
        values[i][i] = 1.0
        counts[i][i] = int(
            (~np.isnan(arrays[first.name])).sum()
            if roles[first.name] == "num"
            else (arrays[first.name] >= 0).sum()
        )
        for j in range(i + 1, size):
            second = usable[j]
            a, b = arrays[first.name], arrays[second.name]
            kinds = (roles[first.name], roles[second.name])
            if kinds == ("num", "num"):
                metric: Metric = method
                k, m = position[first.name], position[second.name]
                r, n = r_all[k, m], int(n_all[k, m])
                value = None if np.isnan(r) else float(r)
                p = correlation_p(value, n) if value is not None else None
            elif kinds == ("cat", "cat"):
                metric = "cramers_v"
                value, p, n = cramers_v(a, b)
            else:
                metric = "eta"
                value, p, n = eta(a, b) if kinds[0] == "num" else eta(b, a)
            values[i][j] = values[j][i] = value
            metrics[i][j] = metrics[j][i] = metric
            counts[i][j] = counts[j][i] = n
            if value is not None:
                pairs.append(Pair(first.name, second.name, metric, value, p, n))
    pairs.sort(key=lambda pair: -abs(pair.value))
    adjusted, multiple = correct([pair.p_value for pair in pairs], correction, noun="links")
    for pair, q in zip(pairs, adjusted, strict=True):
        pair.q_value = q
        pair.evidence = grade(effect_size(pair.metric, pair.value), q)
    matrix = Matrix(
        method,
        [MatrixMember(c.name, c.type, roles[c.name]) for c in usable],
        values,
        metrics,
        counts,
        cluster_order(values),
        rows,
        total,
        total > rows,
        excluded,
        truncated,
        pairs,
        multiple,
    )
    return matrix, loaded
