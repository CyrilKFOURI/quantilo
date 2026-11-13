"""Regression on images and multi-label models (c3; study § 6.2, R-VML-05 and R-VML-06).

Both read the same frozen vectors as the classifiers (classic descriptors of 103, and DINOv2 when
the `vision` extra and its weights are here, R-VCL-06) and the same grouped folds (R-VSP-06):

- R-VRG-01, regression: descriptors standardised then a PCA keeping 95 % of the variance (fitted on
  the training images, R-VCL-01), then a **ridge**, alpha by grouped inner cross-validation; with
  DINOv2, a ridge on the standardised vectors too, the desktop default.
- R-VRG-02: R², MAE, RMSE, median absolute error, Spearman; the baseline is the training mean; 95 %
  bootstrap intervals by groups of copies on R² and MAE; the largest errors both ways.
- R-VMLB-01, multi-label: one logistic regression per label (one-vs-rest), C by grouped inner
  cross-validation, `balanced` weights when a label is rare (R-VQ-13 sense), threshold 0.5.
- R-VMLB-02: exact match, Hamming loss, micro and macro F1, then F1, precision, recall, AUROC and
  support per label; the baseline gives each label its training frequency.

numpy, scipy and scikit-learn only, no thread: the browser version runs the classic part.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from pydantic import BaseModel

from quantilo.vision import metrics as vm
from quantilo.vision.split import SplitPlan

ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0)  # R-VRG-01
C_GRID = (0.01, 0.1, 1.0, 10.0)  # R-VMLB-01
INNER_FOLDS = 3
RARE_LABEL = 3.0  # R-VMLB-01: with / without (or the reverse) above this: balanced weights
THRESHOLD = 0.5
GALLERY = 24
SEED = 0


class ImageRegressionReport(BaseModel):
    """R-VRG-02, on the scored images."""

    n: int
    r2: float
    mae: float
    rmse: float
    median_ae: float
    spearman: float | None
    baseline_mae: float  # the training mean predicted for every image
    baseline_rmse: float
    baseline_r2: float
    train_mean: float
    intervals: dict[str, vm.MetricInterval] = {}
    sentence: str


class ImageLabelScores(BaseModel):
    label: str
    support: int  # scored images with the label
    prevalence: float  # share of training images with it
    precision: float
    recall: float
    f1: float
    auroc: float | None


class ImageMultilabelReport(BaseModel):
    """R-VMLB-02, on the scored images."""

    n: int
    labels: list[str]
    exact_match: float
    hamming_loss: float
    micro_f1: float
    macro_f1: float
    per_label: list[ImageLabelScores]
    baseline_exact_match: float
    baseline_hamming_loss: float
    baseline_macro_f1: float
    intervals: dict[str, vm.MetricInterval] = {}
    threshold: float = THRESHOLD
    sentence: str


# --- metrics -------------------------------------------------------------------------------------


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    total = float(np.sum((y - y.mean()) ** 2))
    if total <= 0:
        return math.nan
    return 1.0 - float(np.sum((y - pred) ** 2)) / total


def regression_report(
    y: np.ndarray,
    pred: np.ndarray,
    train_mean: float,
    *,
    groups: np.ndarray | None = None,
    bootstrap: bool = True,
    seed: int = SEED,
) -> ImageRegressionReport:
    from scipy.stats import spearmanr

    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    errors = np.abs(y - pred)
    base = np.full_like(y, train_mean)
    rho = float(np.asarray(spearmanr(y, pred))[0]) if len(y) > 2 and np.ptp(pred) > 0 else math.nan
    intervals: dict[str, vm.MetricInterval] = {}
    if bootstrap and len(y) >= 10:
        for name, statistic in (
            ("r2", lambda rows: _r2(y[rows], pred[rows])),
            ("mae", lambda rows: float(np.mean(errors[rows]))),
        ):
            found = vm.bootstrap_interval(statistic, len(y), groups=groups, seed=seed)
            if found is not None:
                intervals[name] = found
    r2 = _r2(y, pred)
    mae = float(errors.mean())
    base_mae = float(np.mean(np.abs(y - base)))
    gain = 1 - mae / base_mae if base_mae > 0 else 0.0
    sentence = (
        f"R² {r2:.3f}, MAE {mae:.4g} against {base_mae:.4g} for the training mean "
        f"({100 * gain:+.0f} %)."
    )
    return ImageRegressionReport(
        n=len(y),
        r2=r2,
        mae=mae,
        rmse=float(np.sqrt(np.mean((y - pred) ** 2))),
        median_ae=float(np.median(errors)),
        spearman=None if math.isnan(float(rho)) else float(rho),
        baseline_mae=base_mae,
        baseline_rmse=float(np.sqrt(np.mean((y - base) ** 2))),
        baseline_r2=_r2(y, base),
        train_mean=float(train_mean),
        intervals=intervals,
        sentence=sentence,
    )


def _f1(tp: float, fp: float, fn: float) -> float:
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0


def _macro_f1(truth: np.ndarray, guess: np.ndarray) -> float:
    tp = (truth & guess).sum(axis=0)
    fp = (~truth & guess).sum(axis=0)
    fn = (truth & ~guess).sum(axis=0)
    return float(np.mean([_f1(a, b, c) for a, b, c in zip(tp, fp, fn, strict=True)]))


def multilabel_report(
    truth: np.ndarray,
    proba: np.ndarray,
    labels: list[str],
    prevalence: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    bootstrap: bool = True,
    seed: int = SEED,
    threshold: float = THRESHOLD,
) -> ImageMultilabelReport:
    truth = np.asarray(truth, dtype=bool)
    guess = np.asarray(proba) >= threshold
    tp = (truth & guess).sum(axis=0)
    fp = (~truth & guess).sum(axis=0)
    fn = (truth & ~guess).sum(axis=0)
    per_label = []
    for j, name in enumerate(labels):
        both = truth[:, j].any() and (~truth[:, j]).any()
        per_label.append(
            ImageLabelScores(
                label=name,
                support=int(truth[:, j].sum()),
                prevalence=float(prevalence[j]),
                precision=float(tp[j] / (tp[j] + fp[j])) if tp[j] + fp[j] else 0.0,
                recall=float(tp[j] / (tp[j] + fn[j])) if tp[j] + fn[j] else 0.0,
                f1=_f1(tp[j], fp[j], fn[j]),
                auroc=vm.roc_auc(truth[:, j], proba[:, j]) if both else None,
            )
        )
    base = np.tile(np.asarray(prevalence) >= threshold, (len(truth), 1))
    intervals: dict[str, vm.MetricInterval] = {}
    if bootstrap and len(truth) >= 10:
        found = vm.bootstrap_interval(
            lambda rows: _macro_f1(truth[rows], guess[rows]), len(truth), groups=groups, seed=seed
        )
        if found is not None:
            intervals["macro_f1"] = found
    macro = _macro_f1(truth, guess)
    base_macro = _macro_f1(truth, base)
    exact = float(np.mean((truth == guess).all(axis=1)))
    sentence = (
        f"Macro F1 {macro:.3f} over {len(labels)} labels (each label at its training frequency: "
        f"{base_macro:.3f}); exact match {exact:.3f}."
    )
    return ImageMultilabelReport(
        n=len(truth),
        labels=labels,
        exact_match=exact,
        hamming_loss=float(np.mean(truth != guess)),
        micro_f1=_f1(float(tp.sum()), float(fp.sum()), float(fn.sum())),
        macro_f1=macro,
        per_label=per_label,
        baseline_exact_match=float(np.mean((truth == base).all(axis=1))),
        baseline_hamming_loss=float(np.mean(truth != base)),
        baseline_macro_f1=base_macro,
        intervals=intervals,
        threshold=threshold,
        sentence=sentence,
    )


# --- models --------------------------------------------------------------------------------------


def _inner_groups(groups: np.ndarray) -> Any:
    from sklearn.model_selection import GroupKFold

    k = int(min(INNER_FOLDS, len(np.unique(groups))))
    return GroupKFold(max(2, k), shuffle=True, random_state=SEED)


def fit_ridge(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, *, scale: bool
) -> tuple[Any, float]:
    """R-VRG-01: (fitted model, alpha) by grouped inner cross-validation on the squared error."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    steps: list[Any] = [StandardScaler()] if scale else []
    steps.append(Ridge())
    if len(np.unique(groups)) < 2:
        model = make_pipeline(*steps[:-1], Ridge(alpha=1.0)).fit(x, y)
        return model, 1.0
    search = GridSearchCV(
        make_pipeline(*steps),
        {"ridge__alpha": list(ALPHAS)},
        scoring="neg_mean_squared_error",
        cv=_inner_groups(groups),
    )
    search.fit(x, y, groups=groups)
    return search.best_estimator_, float(search.best_params_["ridge__alpha"])


def _weights(y: np.ndarray) -> str | None:
    pos = int(y.sum())
    neg = len(y) - pos
    if min(pos, neg) == 0:
        return None
    return "balanced" if max(pos, neg) / min(pos, neg) > RARE_LABEL else None


def fit_label(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, *, scale: bool, seed: int = SEED
) -> tuple[Any, float]:
    """R-VMLB-01: one label's logistic regression, C by grouped inner cross-validation (AUROC);
    a label the folds cannot score takes C = 1."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def build(c: float = 1.0) -> Any:
        steps: list[Any] = [StandardScaler()] if scale else []
        steps.append(LogisticRegression(C=c, max_iter=3000, class_weight=_weights(y)))
        return make_pipeline(*steps)

    if y.all() or not y.any():
        raise ValueError("A label is present on every training image, or on none (R-VMLB-02).")
    k = int(min(INNER_FOLDS, y.sum(), (~y).sum(), len(np.unique(groups))))
    if k < 2:
        return build().fit(x, y), 1.0
    search = GridSearchCV(
        build(),
        {"logisticregression__C": list(C_GRID)},
        scoring="roc_auc",
        cv=StratifiedGroupKFold(k, shuffle=True, random_state=seed),
        error_score=0.0,
    )
    search.fit(x, y, groups=groups)
    return search.best_estimator_, float(search.best_params_["logisticregression__C"])


def _space(classic: np.ndarray, train: np.ndarray, test: np.ndarray, seed: int):
    from quantilo.vision.classic_models import descriptor_space

    space = descriptor_space(classic[train], seed)
    return space.transform(classic[train]), space.transform(np.nan_to_num(classic[test]))


def regress(
    plan: SplitPlan,
    y: np.ndarray,
    groups: np.ndarray,
    classic: np.ndarray,
    deep: np.ndarray | None,
    steps: Any,
    *,
    seed: int = SEED,
    bootstrap: bool = True,
) -> tuple[list[Any], dict[str, Any]]:
    """Every regression model on every fold, out-of-fold predictions, one report per model."""
    from quantilo.vision.classic_models import VisionModelResult

    scored = np.asarray(plan.scored, dtype=int)
    position = {int(r): i for i, r in enumerate(scored)}
    keys = ["ridge"] + (["deep_ridge"] if deep is not None else [])
    pred = {k: np.full(len(scored), np.nan) for k in keys}
    alphas: dict[str, list[float]] = {k: [] for k in keys}
    seconds = dict.fromkeys(keys, 0.0)
    fold_of = np.zeros(len(scored), dtype=int)
    train_means = []
    for f, fold in enumerate(plan.folds):
        train = np.asarray(fold.train, dtype=int)
        test = np.asarray(fold.test, dtype=int)
        at = np.array([position[int(r)] for r in test], dtype=int)
        fold_of[at] = f
        ok = train[np.isfinite(classic[train]).all(axis=1)]
        train_means.append(float(np.mean(y[train])))
        for key in keys:
            steps(
                f / len(plan.folds),
                f"Training {key.replace('_', ' ')} (fold {f + 1} / {len(plan.folds)})",
            )
            started = time.perf_counter()
            if key == "ridge":
                xtr, xte = _space(classic, ok, test, seed)
                model, alpha = fit_ridge(xtr, y[ok], groups[ok].astype(str), scale=False)
            else:
                assert deep is not None
                good = train[np.isfinite(deep[train]).all(axis=1)]
                xtr, xte = deep[good], np.nan_to_num(deep[test])
                model, alpha = fit_ridge(xtr, y[good], groups[good].astype(str), scale=True)
            pred[key][at] = model.predict(xte)
            alphas[key].append(alpha)
            seconds[key] += time.perf_counter() - started
    truth = y[scored]
    results = []
    for key in keys:
        report = regression_report(
            truth, pred[key], float(np.mean(train_means)), groups=groups[scored].astype(str),
            bootstrap=bootstrap, seed=seed,
        )  # fmt: skip
        errors = pred[key] - truth
        order = np.argsort(-np.abs(errors), kind="mergesort")
        results.append(
            VisionModelResult(
                key=key,
                title=_TITLES[key],
                family="deep" if key.startswith("deep") else "classic",
                setting=_alphas(alphas[key]),
                regression=report,
                galleries={
                    "over": [int(scored[i]) for i in order if errors[i] > 0][:GALLERY],
                    "under": [int(scored[i]) for i in order if errors[i] < 0][:GALLERY],
                },
                seconds=round(seconds[key], 2),
                notes=_notes(key),
            )
        )
    _default(results)
    columns: dict[str, Any] = {"fold": fold_of.tolist(), "truth": truth.round(8).tolist()}
    for key in keys:
        columns[f"{key}_pred"] = [
            None if not math.isfinite(v) else round(float(v), 8) for v in pred[key]
        ]
    return results, columns


def multilabel(
    plan: SplitPlan,
    y: np.ndarray,
    names: list[str],
    groups: np.ndarray,
    classic: np.ndarray,
    deep: np.ndarray | None,
    steps: Any,
    *,
    seed: int = SEED,
    bootstrap: bool = True,
) -> tuple[list[Any], dict[str, Any]]:
    """One-vs-rest models on every fold; out-of-fold probabilities, one report per model."""
    from quantilo.vision.classic_models import VisionModelResult

    scored = np.asarray(plan.scored, dtype=int)
    position = {int(r): i for i, r in enumerate(scored)}
    keys = ["ovr_logistic"] + (["deep_ovr"] if deep is not None else [])
    proba = {k: np.zeros((len(scored), len(names))) for k in keys}
    cs: dict[str, list[float]] = {k: [] for k in keys}
    seconds = dict.fromkeys(keys, 0.0)
    fold_of = np.zeros(len(scored), dtype=int)
    train_rows: set[int] = set()
    notes: list[str] = []
    total = len(plan.folds) * len(keys) * len(names)
    done = 0
    for f, fold in enumerate(plan.folds):
        train = np.asarray(fold.train, dtype=int)
        test = np.asarray(fold.test, dtype=int)
        train_rows |= {int(r) for r in train}
        at = np.array([position[int(r)] for r in test], dtype=int)
        fold_of[at] = f
        ok = train[np.isfinite(classic[train]).all(axis=1)]
        spaces = {"ovr_logistic": (*_space(classic, ok, test, seed), ok, False)}
        if deep is not None:
            good = train[np.isfinite(deep[train]).all(axis=1)]
            spaces["deep_ovr"] = (deep[good], np.nan_to_num(deep[test]), good, True)
        for key in keys:
            xtr, xte, rows, scale = spaces[key]
            started = time.perf_counter()
            for j, name in enumerate(names):
                steps(done / max(1, total), f"Training {name!r} (fold {f + 1} / {len(plan.folds)})")
                done += 1
                target = y[rows, j]
                if target.all() or not target.any():  # this fold's training holds one side only
                    proba[key][at, j] = float(target.mean())
                    note = f"{name!r}: one side only in a fold's training: its frequency is used."
                    if note not in notes:
                        notes.append(note)
                    continue
                model, c = fit_label(xtr, target, groups[rows].astype(str), scale=scale, seed=seed)
                proba[key][at, j] = model.predict_proba(xte)[:, 1]
                cs[key].append(c)
            seconds[key] += time.perf_counter() - started
    truth = y[scored]
    prevalence = y[np.asarray(sorted(train_rows), dtype=int)].mean(axis=0)
    results = []
    for key in keys:
        report = multilabel_report(
            truth, proba[key], names, prevalence, groups=groups[scored].astype(str),
            bootstrap=bootstrap, seed=seed,
        )  # fmt: skip
        wrong = (truth != (proba[key] >= THRESHOLD)).sum(axis=1)
        order = np.argsort(-wrong, kind="mergesort")
        results.append(
            VisionModelResult(
                key=key,
                title=_TITLES[key],
                family="deep" if key.startswith("deep") else "classic",
                setting=_cs(cs[key]),
                multilabel=report,
                galleries={"most_wrong": [int(scored[i]) for i in order if wrong[i] > 0][:GALLERY]},
                seconds=round(seconds[key], 2),
                notes=[*_notes(key), *notes],
            )
        )
    _default(results)
    columns: dict[str, Any] = {
        "fold": fold_of.tolist(),
        "truth": [[names[j] for j in np.flatnonzero(row)] for row in truth],
    }
    for key in keys:
        columns[f"{key}_probs"] = proba[key].round(6).tolist()
        columns[f"{key}_pred"] = [
            [names[j] for j in np.flatnonzero(row >= THRESHOLD)] for row in proba[key]
        ]
    return results, columns


_TITLES = {
    "ridge": "Ridge regression on classic descriptors",
    "deep_ridge": "Ridge regression on DINOv2 ViT-S/14 (frozen)",
    "ovr_logistic": "One logistic regression per label (classic descriptors)",
    "deep_ovr": "One logistic regression per label (DINOv2 ViT-S/14, frozen)",
}


def _notes(key: str) -> list[str]:
    return {
        "ridge": [
            "Classic descriptors standardised, then a PCA fitted on the training images keeps 95 % "
            "of their variance; a ridge regression reads them, alpha by a cross-validation grouped "
            "by copies (R-VRG-01)."
        ],
        "deep_ridge": [
            "The frozen DINOv2 vectors, standardised, read by a ridge regression; the desktop "
            "default when the network is here (R-VRG-01, R-VCL-06)."
        ],
        "ovr_logistic": [
            "Each label is learnt on its own, against every image without it (one-vs-rest), on the "
            "classic descriptors' PCA; an image gets a label at a probability of 0.5 or more "
            "(R-VMLB-01)."
        ],
        "deep_ovr": [
            "The same one-vs-rest models on the frozen DINOv2 vectors; the desktop default when "
            "the network is here (R-VMLB-01, R-VCL-06)."
        ],
    }[key]


def _alphas(values: list[float]) -> str:
    if not values:
        return "default settings"
    shown = sorted(set(values))
    return f"alpha = {shown[0]:g}" if len(shown) == 1 else f"alpha {shown[0]:g} to {shown[-1]:g}"


def _cs(values: list[float]) -> str:
    if not values:
        return "default settings"
    shown = sorted(set(values))
    return f"C = {shown[0]:g}" if len(shown) == 1 else f"C {shown[0]:g} to {shown[-1]:g} by label"


def _default(results: list[Any]) -> None:
    """The deep model when it ran (R-VCL-06), else the classic one."""
    chosen = next((m for m in results if m.family == "deep"), results[0])
    chosen.default = True


__all__ = [
    "ALPHAS",
    "ImageLabelScores",
    "ImageMultilabelReport",
    "ImageRegressionReport",
    "fit_label",
    "fit_ridge",
    "multilabel",
    "multilabel_report",
    "regress",
    "regression_report",
]
