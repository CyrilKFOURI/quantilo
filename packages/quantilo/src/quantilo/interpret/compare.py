"""The readable model against the run's model, on the same held-out rows (121, 122).

R-XAI-08: a card or a tree learns on rows the run's model was trained on (the training rows of a
split) or on 70 % of the out-of-fold rows (cross-validation, panel walk-forward), and both are
judged on held-out rows where the model's probability was honest, never on a training row.

R-XAI-07: the gap is read through a paired bootstrap on those rows (200 draws): an interval that
holds 0 is noise, and the readable model is then the one to prefer when decisions must be
justified (Rudin, 2019)."""

from dataclasses import dataclass

import numpy as np

SEED = 42
DRAWS = 200
CHECK_SHARE = 0.3  # cross-validation: share of the out-of-fold rows kept to judge on
LARGE_GAP = 0.05  # R-XAI-07: an AUC gap beyond the noise and above this says the model finds more
ECE_BINS = 10


def auc(score: np.ndarray, truth: np.ndarray) -> float:
    """ROC AUC with ties counted half (Mann-Whitney), `truth` in {0, 1}."""
    truth = np.asarray(truth, dtype=bool)
    positives = int(truth.sum())
    negatives = len(truth) - positives
    if not positives or not negatives:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    sorted_scores = np.asarray(score)[order]
    # average ranks of tied scores
    _, start, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    average = start + (counts - 1) / 2 + 1
    ranks[order] = np.repeat(average, counts)
    return float((ranks[truth].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def ece(probability: np.ndarray, truth: np.ndarray, bins: int = ECE_BINS) -> float:
    """Expected calibration error: |mean probability − positive share| over 10 equal bins."""
    probability = np.clip(np.asarray(probability, dtype=float), 0, 1)
    where = np.minimum((probability * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        inside = where == b
        if inside.any():
            total += inside.mean() * abs(probability[inside].mean() - truth[inside].mean())
    return float(total)


def brier(probability: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((np.asarray(probability, dtype=float) - truth) ** 2))


def log_loss(probability: np.ndarray, truth: np.ndarray) -> float:
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    return float(-np.mean(truth * np.log(p) + (1 - truth) * np.log(1 - p)))


@dataclass
class Gap:
    """The run's model against the readable one, on the same rows."""

    metric: str  # "ROC AUC"
    model: float
    readable: float
    gap: float  # model − readable
    low: float  # 95 % paired bootstrap interval of the gap
    high: float
    within_noise: bool
    model_ece: float | None
    readable_ece: float | None
    model_brier: float | None
    readable_brier: float | None
    rows: int
    sentence: str


def gap(
    model_score: np.ndarray,
    readable_score: np.ndarray,
    truth: np.ndarray,
    what: str,
    model_label: str,
    model_probability: np.ndarray | None = None,
    readable_probability: np.ndarray | None = None,
    draws: int = DRAWS,
    seed: int = SEED,
) -> Gap:
    """R-XAI-07: the AUC gap with its paired bootstrap interval, and the sentence that concludes.
    `what`: the readable model, as the sentence names it ("A 5-condition scorecard")."""
    truth = np.asarray(truth, dtype=float)
    model_auc, readable_auc = auc(model_score, truth), auc(readable_score, truth)
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(draws):
        rows = rng.integers(0, len(truth), len(truth))
        found = auc(model_score[rows], truth[rows]) - auc(readable_score[rows], truth[rows])
        if np.isfinite(found):
            gaps.append(found)
    difference = model_auc - readable_auc
    low, high = (
        (float(np.quantile(gaps, 0.025)), float(np.quantile(gaps, 0.975)))
        if gaps
        else (float("nan"), float("nan"))
    )
    within = bool(gaps) and low <= 0 <= high
    points = abs(difference) * 100
    if within:
        side = "below" if difference > 0 else "above"
        sentence = (
            f"{what} is {points:.1f} pts of ROC AUC {side} the {model_label}, 95 % interval "
            f"{low * 100:+.1f} to {high * 100:+.1f}: within noise, so prefer it if you must "
            "justify decisions (R-XAI-07)."
        )
    elif difference > 0:
        sentence = (
            f"{what} is {points:.1f} pts of ROC AUC below the {model_label}, 95 % interval "
            f"{low * 100:+.1f} to {high * 100:+.1f}: beyond the noise."
        )
        if difference > LARGE_GAP:
            sentence += " The model finds what a short readable model cannot (R-XAI-07)."
        else:
            sentence += " A small price for a model anyone can apply by hand (R-XAI-07)."
    else:
        sentence = (
            f"{what} beats the {model_label} by {points:.1f} pts of ROC AUC, beyond the noise "
            "(R-XAI-07): the readable model is the better one here."
        )
    return Gap(
        "ROC AUC",
        model_auc,
        readable_auc,
        float(difference),
        low,
        high,
        within,
        ece(model_probability, truth) if model_probability is not None else None,
        ece(readable_probability, truth) if readable_probability is not None else None,
        brier(model_probability, truth) if model_probability is not None else None,
        brier(readable_probability, truth) if readable_probability is not None else None,
        len(truth),
        sentence,
    )


def r2(prediction: np.ndarray, truth: np.ndarray) -> float:
    spread = float(((truth - truth.mean()) ** 2).sum())
    return 1 - float(((truth - prediction) ** 2).sum()) / spread if spread > 0 else 0.0


def r2_gap(
    model_prediction: np.ndarray,
    readable_prediction: np.ndarray,
    truth: np.ndarray,
    what: str,
    model_label: str,
    draws: int = DRAWS,
    seed: int = SEED,
) -> Gap:
    """R-XAI-19 (122 bis): for a number, the R² gap with its paired bootstrap interval and the
    sentence, as R-XAI-07 does with the ROC AUC."""
    truth = np.asarray(truth, dtype=float)
    model_prediction = np.asarray(model_prediction, dtype=float)
    readable_prediction = np.asarray(readable_prediction, dtype=float)
    model_r2, readable_r2 = r2(model_prediction, truth), r2(readable_prediction, truth)
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(draws):
        rows = rng.integers(0, len(truth), len(truth))
        found = r2(model_prediction[rows], truth[rows]) - r2(readable_prediction[rows], truth[rows])
        if np.isfinite(found):
            gaps.append(found)
    difference = model_r2 - readable_r2
    low, high = (
        (float(np.quantile(gaps, 0.025)), float(np.quantile(gaps, 0.975)))
        if gaps
        else (float("nan"), float("nan"))
    )
    within = bool(gaps) and low <= 0 <= high
    points = abs(difference) * 100
    interval = f"95 % interval {low * 100:+.1f} to {high * 100:+.1f}"
    if within:
        side = "below" if difference > 0 else "above"
        sentence = (
            f"{what} is {points:.1f} pts of R² {side} the {model_label}, {interval}: within noise, "
            "so prefer it if you must justify decisions (R-XAI-19)."
        )
    elif difference > 0:
        sentence = (
            f"{what} is {points:.1f} pts of R² below the {model_label}, {interval}: beyond the "
            "noise."
        )
        if difference > LARGE_GAP:
            sentence += " The model finds what a short readable model cannot (R-XAI-19)."
        else:
            sentence += " A small price for a model anyone can follow by hand (R-XAI-19)."
    else:
        sentence = (
            f"{what} beats the {model_label} by {points:.1f} pts of R², beyond the noise "
            "(R-XAI-19): the readable model is the better one here."
        )
    return Gap(
        "R²",
        model_r2,
        readable_r2,
        float(difference),
        low,
        high,
        within,
        None,
        None,
        None,
        None,
        len(truth),
        sentence,
    )


@dataclass
class Rows:
    """R-XAI-08: which rows the readable model learns on, and which ones judge both models."""

    learn: np.ndarray  # positions in the loaded frame
    check: np.ndarray  # positions in the loaded frame
    held: np.ndarray  # for each checked row, its position among the held-out predictions
    source: str  # said under the comparison


def split(
    frame_rows: np.ndarray,
    held_rows: np.ndarray,
    limit: int,
    seed: int = SEED,
) -> Rows:
    """`frame_rows`: file row numbers of the loaded rows; `held_rows`: those of the held-out
    predictions. With training rows (a split), learn on them and check on the held-out rows;
    without (every row held out once: cross-validation, panel), cut the held-out rows 70 / 30."""
    rng = np.random.default_rng(seed)
    position = {int(r): i for i, r in enumerate(frame_rows)}
    held_at = {int(r): i for i, r in enumerate(held_rows)}
    held_positions = np.array([position[int(r)] for r in held_rows if int(r) in position], int)
    rest = np.setdiff1d(np.arange(len(frame_rows)), held_positions)
    if len(rest) >= max(50, int(0.2 * len(frame_rows))):
        learn, check = rest, held_positions
        source = "learnt on the training rows, judged on the test rows"
    else:
        shuffled = rng.permutation(held_positions)
        cut = round(len(shuffled) * (1 - CHECK_SHARE))
        learn, check = np.sort(shuffled[:cut]), np.sort(shuffled[cut:])
        source = (
            "learnt on 70 % of the out-of-fold rows, judged on the other 30 %, where the "
            "model's probability is out-of-fold too"
        )
    if len(learn) > limit:
        learn = np.sort(rng.choice(learn, limit, replace=False))
    if len(check) > limit:
        check = np.sort(rng.choice(check, limit, replace=False))
    held = np.array([held_at[int(frame_rows[i])] for i in check], dtype=int)
    return Rows(learn, check, held, f"{source} (R-XAI-08)")
