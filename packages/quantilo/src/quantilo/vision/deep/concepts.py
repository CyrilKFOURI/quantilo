"""Concept bottleneck: image -> concepts the user named -> class (118; study §§ 6.4, 13.2).

Koh, Nguyen, Tang, Mussmann, Pierson, Kim and Liang (2020), *Concept Bottleneck Models*: the
class is predicted **only** from concepts a person can read and correct; correcting a concept
changes the prediction, and the change is exact (a linear model). numpy and scikit-learn only:
it runs on the frozen DINOv2 vectors when they are here, on the classic descriptors otherwise
(and in the browser version).

- R-VCB-01 concepts: a column of the table that the import did not make (a metadata column of
  the CSV or of the Parquet file), true/false or with two values (binary), or numeric with at
  least 5 distinct values (numeric); or a group of images the user saved as a concept (the
  images of the group have it, the others are taken as without it, said). At least 2 concepts.
- R-VCB-02 stage 1, one probe per concept on the image vectors (standardised): a binary concept
  by a logistic regression (C by a cross-validation grouped by copies, on ROC AUC), a numeric
  one by a ridge regression (alpha by leave-one-out). Rows where the concept is unknown are left
  out of its probe.
- R-VCB-03 stage 2, the class from the predicted concepts only (a binary concept as its
  probability, a numeric one in its own units), standardised: an L1 logistic regression, C by a
  grouped cross-validation on balanced accuracy. It learns on out-of-fold predictions of stage 1
  (5 grouped folds): learning on the probes' own training predictions would trust concepts more
  than they deserve on new images ("sequential" bottleneck, Koh et al.).
- R-VCB-04 quality: each concept's own score on the scored images (ROC AUC for a binary
  concept, R² for a numeric one); a concept whose probe is no better than chance (AUC < 0.6,
  R² < 0.1) is flagged: the class leans on a concept the images do not show.
- R-VCB-05 intervention: the page changes a concept's value and recomputes the class
  probabilities from stage 2's weights (kept with the run), with no call to the server.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pyarrow as pa

MIN_CONCEPTS = 2  # R-VCB-01
MIN_DISTINCT_NUMERIC = 5
MIN_GROUP = 5  # images of a saved group
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
STAGE2_C = (0.01, 0.1, 1.0, 10.0)
OOF_FOLDS = 5  # R-VCB-03
WEAK_AUC = 0.6  # R-VCB-04
WEAK_R2 = 0.1

Kind = Literal["binary", "numeric"]


@dataclass
class Concept:
    key: str  # "column:<name>" or "group:<id>"
    name: str
    kind: Kind
    values: np.ndarray  # one per table row: 0/1 (binary) or the value (numeric); NaN = unknown
    source: Literal["column", "group"]
    note: str = ""
    positive: str | None = None  # binary column: the value read as 1


def _import_columns() -> set[str]:
    from quantilo.vision.edits import DERIVED
    from quantilo.vision.ingest import COLUMNS

    return set(COLUMNS) | {"label", "split", "is_normal", "mask_path", *DERIVED}


def column_concepts(table: pa.Table) -> list[Concept]:
    """R-VCB-01: the metadata columns that can be concepts."""
    made = _import_columns()
    out: list[Concept] = []
    for name in table.column_names:
        if name in made or name.startswith("_"):
            continue
        values = table[name].to_pylist()
        found = _column_concept(name, values)
        if found is not None:
            out.append(found)
    return out


def _column_concept(name: str, values: list[Any]) -> Concept | None:
    known = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not known:
        return None
    distinct = sorted({str(v) for v in known})
    if all(isinstance(v, bool) for v in known) or len(distinct) == 2:
        if all(isinstance(v, bool) for v in known):
            positive = "True"
        else:
            # the rarer value is the concept ("has spots" rather than "has no spots")
            counts = {d: sum(str(v) == d for v in known) for d in distinct}
            positive = min(distinct, key=lambda d: (counts[d], d))
        coded = np.array(
            [np.nan if v is None else float(str(v) == positive) for v in values], dtype=float
        )
        return Concept(
            f"column:{name}",
            name,
            "binary",
            coded,
            "column",
            f"Column {name!r}: {positive!r} read as having the concept (R-VCB-01).",
            positive,
        )
    if all(isinstance(v, int | float) and not isinstance(v, bool) for v in known):
        if len(distinct) < MIN_DISTINCT_NUMERIC:
            return None
        coded = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
        return Concept(
            f"column:{name}", name, "numeric", coded, "column", f"Column {name!r}, a number."
        )
    return None


def group_concept(key: str, name: str, rows: Sequence[int] | np.ndarray, n_rows: int) -> Concept:
    """R-VCB-01: a saved group of images: its images have the concept, the others not."""
    values = np.zeros(n_rows, dtype=float)
    values[np.asarray(list(rows), dtype=int)] = 1.0
    return Concept(
        key,
        name,
        "binary",
        values,
        "group",
        f"Group {name!r}: its {len(rows):,} images have the concept, the others are taken as "
        "without it (R-VCB-01).",
        "in the group",
    )


# ---------------------------------------------------------------------------------------------
# The model


@dataclass
class ConceptProbe:
    key: str
    name: str
    kind: Kind
    weight: np.ndarray  # (d,) on standardised vectors
    bias: float
    target_mean: float = 0.0  # numeric: the probe predicts (value - mean) / scale
    target_scale: float = 1.0
    setting: str = ""

    def predict(self, z: np.ndarray) -> np.ndarray:
        raw = z @ self.weight + self.bias
        if self.kind == "binary":
            return 1.0 / (1.0 + np.exp(-np.clip(raw, -50, 50)))
        return raw * self.target_scale + self.target_mean


@dataclass
class ConceptHead:
    """Stage 2: class probabilities from concept values (R-VCB-03, R-VCB-05)."""

    classes: list[str]
    mean: np.ndarray  # (c,)
    scale: np.ndarray  # (c,)
    weight: np.ndarray  # (k, c), or (1, c) for two classes
    bias: np.ndarray  # (k,) or (1,)
    setting: str = ""

    def proba(self, concepts: np.ndarray) -> np.ndarray:
        z = (np.asarray(concepts, dtype=float) - self.mean) / self.scale
        logits = z @ self.weight.T + self.bias
        if self.weight.shape[0] == 1:
            p = 1.0 / (1.0 + np.exp(-np.clip(logits[:, 0], -50, 50)))
            return np.stack([1.0 - p, p], axis=1)
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    def class_weights(self) -> np.ndarray:
        """(k, c): each concept's push towards each class, per unit of its standardised value
        (two classes: the second class gets +w, the first -w)."""
        if self.weight.shape[0] == 1:
            return np.vstack([-self.weight[0], self.weight[0]])
        return self.weight


@dataclass
class Bottleneck:
    x_mean: np.ndarray
    x_scale: np.ndarray
    probes: list[ConceptProbe]
    head: ConceptHead
    notes: list[str] = field(default_factory=list)

    def concepts(self, x: np.ndarray) -> np.ndarray:
        z = (np.nan_to_num(np.asarray(x, dtype=float)) - self.x_mean) / self.x_scale
        return np.stack([p.predict(z) for p in self.probes], axis=1)

    def proba(self, x: np.ndarray) -> np.ndarray:
        return self.head.proba(self.concepts(x))


def _cv(y: np.ndarray, groups: np.ndarray, folds: int, seed: int) -> Any:
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    n_groups = len(np.unique(groups))
    if y is not None:
        _, counts = np.unique(y, return_counts=True)
        k = int(min(folds, counts.min(), n_groups))
        if k >= 2:
            return StratifiedGroupKFold(k, shuffle=True, random_state=seed)
    k = int(min(folds, n_groups))
    return GroupKFold(max(2, k))


def _quiet() -> Any:
    from quantilo.vision.classic_models import _quiet as quiet

    return quiet()


def _fit_probe(
    z: np.ndarray, target: np.ndarray, groups: np.ndarray, concept: Concept, seed: int
) -> ConceptProbe:
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.model_selection import GridSearchCV

    if concept.kind == "binary":
        y = target.astype(int)
        if len(np.unique(y)) < 2:
            raise ValueError(
                f"Concept {concept.name!r} has a single value among the training images: it "
                "cannot be learnt (R-VCB-02)."
            )
        _, counts = np.unique(y, return_counts=True)
        balance = "balanced" if counts.max() / counts.min() > 3 else None
        try:
            search = GridSearchCV(
                LogisticRegression(max_iter=2000, class_weight=balance),
                {"C": list(C_GRID)},
                scoring="roc_auc",
                cv=_cv(y, groups, 3, seed),
            )
            with _quiet():
                search.fit(z, y, groups=groups)
            model, c = search.best_estimator_, float(search.best_params_["C"])
        except ValueError:
            model, c = LogisticRegression(max_iter=2000, class_weight=balance).fit(z, y), 1.0
        return ConceptProbe(
            concept.key,
            concept.name,
            "binary",
            np.asarray(model.coef_[0], dtype=float),
            float(model.intercept_[0]),
            setting=f"logistic, C = {c:g}",
        )
    mean = float(np.mean(target))
    scale = float(np.std(target)) or 1.0
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 13))
    with _quiet():
        ridge.fit(z, (target - mean) / scale)
    return ConceptProbe(
        concept.key,
        concept.name,
        "numeric",
        np.asarray(ridge.coef_, dtype=float),
        float(ridge.intercept_),
        mean,
        scale,
        setting=f"ridge, alpha = {float(np.asarray(ridge.alpha_)):g}",
    )


def fit_bottleneck(
    x: np.ndarray,
    concepts: Sequence[Concept],
    rows: Sequence[int] | np.ndarray,
    labels: Sequence[str] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    classes: Sequence[str],
    *,
    seed: int = 0,
) -> Bottleneck:
    """R-VCB-02 and 03 on the training `rows` (`x` and `labels` aligned with them; each
    concept's `values` are indexed by table row)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    if len(concepts) < MIN_CONCEPTS:
        raise ValueError(f"A concept bottleneck needs at least {MIN_CONCEPTS} concepts (R-VCB-01).")
    rows_arr = np.asarray(rows, dtype=int)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x).all(axis=1)
    x, rows_arr = x[ok], rows_arr[ok]
    y = np.asarray([str(v) for v in labels], dtype=object)[ok]
    g = np.asarray([str(v) for v in groups], dtype=object)[ok]
    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0
    z = (x - x_mean) / x_scale
    notes: list[str] = []

    def probes_on(train: np.ndarray) -> list[ConceptProbe]:
        out = []
        for concept in concepts:
            values = concept.values[rows_arr[train]]
            known = np.isfinite(values)
            out.append(_fit_probe(z[train][known], values[known], g[train][known], concept, seed))
        return out

    probes = probes_on(np.arange(len(z)))
    # R-VCB-03: stage 2 learns on out-of-fold concept predictions
    oof = np.zeros((len(z), len(concepts)))
    splitter = _cv(y, g, OOF_FOLDS, seed)
    try:
        for train, test in splitter.split(z, y, g):
            fold_probes = probes_on(train)
            oof[test] = np.stack([p.predict(z[test]) for p in fold_probes], axis=1)
    except ValueError as error:
        notes.append(
            f"Out-of-fold concepts not possible ({error}); stage 2 learnt on the probes' own "
            "training predictions (R-VCB-03)."
        )
        oof = np.stack([p.predict(z) for p in probes], axis=1)
    c_mean = oof.mean(axis=0)
    c_scale = oof.std(axis=0)
    c_scale[c_scale < 1e-8] = 1.0
    cz = (oof - c_mean) / c_scale
    classes = [str(c) for c in classes]
    _, counts = np.unique(y, return_counts=True)
    balance = "balanced" if counts.max() / counts.min() > 3 else None
    base = LogisticRegression(
        penalty="l1", solver="saga", max_iter=5000, class_weight=balance, random_state=seed
    )
    try:
        search = GridSearchCV(
            base, {"C": list(STAGE2_C)}, scoring="balanced_accuracy", cv=_cv(y, g, 5, seed)
        )
        with _quiet():
            search.fit(cz, y, groups=g)
        model, c = search.best_estimator_, float(search.best_params_["C"])
    except ValueError:
        with _quiet():
            model, c = base.set_params(C=1.0).fit(cz, y), 1.0
    seen = [str(v) for v in model.classes_]
    coef = np.asarray(model.coef_, dtype=float)
    intercept = np.asarray(model.intercept_, dtype=float)
    if coef.shape[0] == 1:
        if seen != classes:
            raise ValueError("Two classes must come in the model's order.")
        weight, bias = coef, intercept
    else:
        weight = np.zeros((len(classes), coef.shape[1]))
        bias = np.full(len(classes), -1e4)
        for j, label in enumerate(seen):
            weight[classes.index(label)], bias[classes.index(label)] = coef[j], intercept[j]
    head = ConceptHead(classes, c_mean, c_scale, weight, bias, f"L1 logistic, C = {c:g}")
    unused = [concepts[j].name for j in range(len(concepts)) if np.allclose(weight[:, j], 0)]
    if unused:
        notes.append(
            f"Not used by the class model (L1 set their weights to 0): {', '.join(unused)} "
            "(R-VCB-03)."
        )
    return Bottleneck(x_mean, x_scale, probes, head, notes)


@dataclass
class ConceptQuality:
    key: str
    name: str
    kind: Kind
    metric: str  # "roc_auc" or "r2"
    value: float | None
    weak: bool
    known: int  # scored images whose value is known
    setting: str
    weights: dict[str, float]  # class -> push of one standard deviation of the concept
    note: str = ""


def quality(
    bottleneck: Bottleneck,
    concepts: Sequence[Concept],
    predicted: np.ndarray,
    rows: Sequence[int] | np.ndarray,
) -> list[ConceptQuality]:
    """R-VCB-04 on the scored `rows` (`predicted` aligned with them)."""
    from quantilo.vision import metrics as vm

    rows_arr = np.asarray(rows, dtype=int)
    weights = bottleneck.head.class_weights()
    out = []
    for j, concept in enumerate(concepts):
        truth = concept.values[rows_arr]
        known = np.isfinite(truth)
        value: float | None = None
        if concept.kind == "binary":
            metric = "roc_auc"
            if known.sum() >= 2 and len(np.unique(truth[known])) == 2:
                value = float(vm.roc_auc(truth[known].astype(int), predicted[known, j]))
            weak = value is not None and value < WEAK_AUC
        else:
            metric = "r2"
            if known.sum() >= 3:
                t = truth[known]
                ss = float(np.sum((t - t.mean()) ** 2))
                value = 1.0 - float(np.sum((t - predicted[known, j]) ** 2)) / ss if ss > 0 else None
            weak = value is not None and value < WEAK_R2
        out.append(
            ConceptQuality(
                concept.key,
                concept.name,
                concept.kind,
                metric,
                None if value is None or not math.isfinite(value) else round(value, 4),
                bool(weak),
                int(known.sum()),
                bottleneck.probes[j].setting,
                {c: round(float(weights[i, j]), 4) for i, c in enumerate(bottleneck.head.classes)},
                concept.note,
            )
        )
    return out


__all__ = [
    "MIN_CONCEPTS",
    "Bottleneck",
    "Concept",
    "ConceptHead",
    "ConceptProbe",
    "ConceptQuality",
    "column_concepts",
    "fit_bottleneck",
    "group_concept",
    "quality",
]
