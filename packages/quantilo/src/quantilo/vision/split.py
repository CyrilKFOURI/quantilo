"""Grouped splits for image models (112; study § 6.1): no copy of an image on both sides.

Groups are the `duplicate_group` of 105 (exact copies and near duplicates); an image without a
copy is its own group; without the column, the `pixel_sha` (exact copies only).

- R-VSP-01: the producer's split (`train`, `val`... to learn, `test` to score) is kept; a group
  that straddles it is counted and said (R-VQ-11).
- R-VSP-02: no producer's split, at most 5,000 images: 5 stratified folds grouped by copies
  (`StratifiedGroupKFold`, seed 0); every image is scored by a model that did not see it.
- R-VSP-03: no producer's split, more than 5,000 images: 20 % of test, stratified and grouped.
- R-VSP-04: anomalies: learn on the normal training images, score the test (without a split:
  20 % of the normal images by group, and every defect). 10 % of the training normals (grouped)
  are held out only when the test cannot set the threshold itself (R-VTH-01 needs 2 defects and 2
  normals): R-VTH-02 needs normal images never seen.
- R-VSP-05: anomalies without any defect: 5 grouped folds on the normal images, each scored out of
  fold; the threshold is the 99 % quantile of those scores (R-VTH-02).
- R-VSP-06 (c3): regression and multi-label: the producer's split when there is one (R-VSP-01),
  else 5 folds grouped by copies (`GroupKFold`, shuffled, seed 0) up to 5,000 images, 20 % of test
  by groups beyond. No strata: a number or a set of labels has none.
"""

import warnings
from typing import Literal

import numpy as np
from pydantic import BaseModel

from quantilo.vision.task import Table, TaskChoice, column, labels_of, split_side, usable_rows

FOLDS = 5
FOLDS_UP_TO = 5_000  # R-VSP-02 / R-VSP-03
TEST_SHARE = 0.2
HOLDOUT_SHARE = 0.1  # R-VSP-04
SEED = 0


class SplitFold(BaseModel):
    train: list[int]
    test: list[int]


class SplitPlan(BaseModel):
    kind: Literal["producer", "folds", "holdout", "none"]
    code: str
    reason: str
    folds: list[SplitFold]
    holdout: list[int] = []  # anomalies: training normals held out for R-VTH-02
    leaks: int = 0  # test images with a copy in train (R-VQ-11)
    seed: int = SEED
    notes: list[str] = []

    @property
    def scored(self) -> list[int]:
        """Every row scored, in order (the union of the folds' test rows)."""
        return sorted({r for f in self.folds for r in f.test})

    @property
    def sentence(self) -> str:
        return self.reason + (" " + " ".join(self.notes) if self.notes else "")


def groups_of(data: Table) -> np.ndarray:
    """One group name per row: `duplicate_group`, else `pixel_sha`, else the row itself."""
    dup = column(data, "duplicate_group")
    sha = column(data, "pixel_sha")
    out = []
    for i, (g, s) in enumerate(zip(dup, sha, strict=True)):
        if g is not None:
            out.append(f"g{g}")
        elif s:
            out.append(f"s{s}")
        else:
            out.append(f"r{i}")
    return np.asarray(out, dtype=object)


def _grouped_share(
    rows: np.ndarray, groups: np.ndarray, share: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """(rest, part): whole groups, in a random order, go to `part` until it holds `share` of the
    rows (at least one group when there are two or more)."""
    if len(rows) == 0:
        return rows, rows
    names = np.unique(groups[rows])
    if len(names) < 2:
        return rows, rows[:0]
    order = rng.permutation(names)
    target = share * len(rows)
    part_groups: set[str] = set()
    taken = 0
    for name in order:
        if (taken >= target and part_groups) or len(part_groups) == len(names) - 1:
            break
        part_groups.add(str(name))
        taken += int(np.sum(groups[rows] == name))
    in_part = np.array([str(groups[r]) in part_groups for r in rows], dtype=bool)
    return np.sort(rows[~in_part]), np.sort(rows[in_part])


def _stratified_folds(
    rows: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedGroupKFold

    _, counts = np.unique(y[rows], return_counts=True)
    k = int(min(folds, counts.min(), len(np.unique(groups[rows]))))
    if k < 2:
        raise ValueError(
            "Copies leave fewer than 2 groups of images to split: review the near duplicates "
            "(R-VQ-10) or give a split (R-VSP-02)."
        )
    splitter = StratifiedGroupKFold(k, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # "least populated class" when groups are few
        found = list(splitter.split(rows, y[rows].astype(str), groups[rows].astype(str)))
    return [(np.sort(rows[a]), np.sort(rows[b])) for a, b in found]


def _leaks(train: np.ndarray, test: np.ndarray, groups: np.ndarray) -> int:
    seen = {str(groups[r]) for r in train}
    return int(sum(1 for r in test if str(groups[r]) in seen))


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}" + ("" if n == 1 else "s")


def plan_split(data: Table, choice: TaskChoice, seed: int = SEED) -> SplitPlan:
    """The split of the task chosen by `detect_task` (R-VSP-01 to 05)."""
    if choice.task is None:
        raise ValueError(choice.blocked or "No task can be trained on this set (R-VML-07).")
    if choice.task == "classification":
        return _classification(data, choice, seed)
    if choice.task == "anomaly":
        return _anomaly(data, seed)
    if choice.task in ("regression", "multilabel"):
        return _target_split(data, target_rows(data, choice), seed)
    usable = np.flatnonzero(usable_rows(data))
    return SplitPlan(
        kind="none",
        code="R-VCU-01",
        reason=(
            f"Clusters are drawn on all {_plural(len(usable), 'usable image')}: nothing is held "
            "out (R-VCU-01)."
        ),
        folds=[SplitFold(train=usable.tolist(), test=[])],
        seed=seed,
    )


def target_rows(data: Table, choice: TaskChoice) -> np.ndarray:
    """c3: the usable rows whose target is known (a number, R-VML-05; a set of labels, R-VML-06)."""
    from quantilo.vision import targets as tg

    usable = usable_rows(data)
    if choice.task == "regression" and choice.target:
        known = np.isfinite(tg.regression_values(data, choice.target))
    else:
        binary = choice.binary_targets if choice.label_source == "columns" else []
        known = tg.label_sets(data, usable, labels_of(data), binary).known
    return np.flatnonzero(usable & known)


def _target_split(data: Table, rows: np.ndarray, seed: int) -> SplitPlan:
    """R-VSP-06."""
    from sklearn.model_selection import GroupKFold

    groups = groups_of(data)
    sides = np.asarray([split_side(v) for v in column(data, "split")], dtype=object)
    train = rows[sides[rows] == "train"]
    test = rows[sides[rows] == "test"]
    if len(train) and len(test):
        leaks = _leaks(train, test, groups)
        notes = []
        if leaks:
            notes.append(
                f"{_plural(leaks, 'test image')} with a near-copy in train: the test score will be "
                "optimistic (R-VQ-11)."
            )
        return SplitPlan(
            kind="producer",
            code="R-VSP-01",
            reason=(
                f"The producer's split: trained on {_plural(len(train), 'image')}, scored on test "
                f"({len(test):,}) (R-VSP-01)."
            ),
            folds=[SplitFold(train=train.tolist(), test=test.tolist())],
            leaks=leaks,
            seed=seed,
            notes=notes,
        )
    n_groups = len(np.unique(groups[rows].astype(str)))
    if n_groups < 2:
        raise ValueError(
            "Copies leave fewer than 2 groups of images to split: review the near duplicates "
            "(R-VQ-10) or give a split (R-VSP-06)."
        )
    if len(rows) <= FOLDS_UP_TO:
        k = int(min(FOLDS, n_groups))
        splitter = GroupKFold(k, shuffle=True, random_state=seed)
        found = [
            (np.sort(rows[a]), np.sort(rows[b]))
            for a, b in splitter.split(rows, groups=groups[rows].astype(str))
        ]
        return SplitPlan(
            kind="folds",
            code="R-VSP-06",
            reason=(
                f"No producer's split: {k} folds grouped by copies (no strata for this target), "
                f"seed {seed}; each of the {len(rows):,} images is scored by a model that did not "
                "see it (R-VSP-06)."
            ),
            folds=[SplitFold(train=a.tolist(), test=b.tolist()) for a, b in found],
            seed=seed,
        )
    rest, part = _grouped_share(rows, groups, TEST_SHARE, np.random.default_rng(seed))
    return SplitPlan(
        kind="holdout",
        code="R-VSP-06",
        reason=(
            f"No producer's split and {len(rows):,} images: {len(part):,} test images "
            f"({100 * len(part) / len(rows):.0f} %) drawn by groups of copies, seed {seed} "
            "(R-VSP-06)."
        ),
        folds=[SplitFold(train=np.sort(rest).tolist(), test=np.sort(part).tolist())],
        seed=seed,
    )


def _classification(data: Table, choice: TaskChoice, seed: int) -> SplitPlan:
    labels = np.asarray(labels_of(data), dtype=object)
    usable = usable_rows(data)
    classes = set(choice.classes)
    rows = np.flatnonzero([usable[i] and labels[i] in classes for i in range(len(labels))])
    groups = groups_of(data)
    sides = np.asarray([split_side(v) for v in column(data, "split")], dtype=object)
    train = rows[sides[rows] == "train"]
    test = rows[sides[rows] == "test"]
    if len(train) and len(test):
        notes = []
        missing = sorted(classes - {str(v) for v in labels[train]})
        if missing:
            keep = np.array([labels[r] not in missing for r in test], dtype=bool)
            notes.append(
                f"No training image of {', '.join(missing)}: {int((~keep).sum()):,} test images "
                "left out."
            )
            test = test[keep]
        leaks = _leaks(train, test, groups)
        if leaks:
            notes.append(
                f"{_plural(leaks, 'test image')} with a near-copy in train: the test score will be "
                "optimistic (R-VQ-11)."
            )
        used = sorted({str(v).lower() for v in column(data, "split") if split_side(v) == "train"})
        return SplitPlan(
            kind="producer",
            code="R-VSP-01",
            reason=(
                f"The producer's split: trained on {' + '.join(used)} "
                f"({_plural(len(train), 'image')}), scored on test ({len(test):,}) (R-VSP-01)."
            ),
            folds=[SplitFold(train=train.tolist(), test=test.tolist())],
            leaks=leaks,
            seed=seed,
            notes=notes,
        )
    if len(rows) <= FOLDS_UP_TO:
        found = _stratified_folds(rows, labels, groups, FOLDS, seed)
        return SplitPlan(
            kind="folds",
            code="R-VSP-02",
            reason=(
                f"No producer's split: {len(found)} folds stratified by label, copies kept in the "
                f"same fold (duplicate groups), seed {seed}; each of the {len(rows):,} images is "
                "scored by a model that did not see it (R-VSP-02)."
            ),
            folds=[SplitFold(train=a.tolist(), test=b.tolist()) for a, b in found],
            seed=seed,
        )
    found = _stratified_folds(rows, labels, groups, round(1 / TEST_SHARE), seed)
    a, b = found[0]
    return SplitPlan(
        kind="holdout",
        code="R-VSP-03",
        reason=(
            f"No producer's split and {len(rows):,} images: {len(b):,} test images "
            f"({100 * len(b) / len(rows):.0f} %) drawn stratified by label, copies kept together, "
            f"seed {seed} (R-VSP-03)."
        ),
        folds=[SplitFold(train=a.tolist(), test=b.tolist())],
        seed=seed,
    )


def normal_flags(data: Table) -> np.ndarray:
    """True for a normal image: `is_normal` when it was read (R-VIS-LAY-03), else every image
    (a single class, or no label: most images are assumed normal)."""
    normal = column(data, "is_normal")
    if any(v is not None for v in normal):
        return np.array([v is True for v in normal], dtype=bool)
    return np.ones(len(normal), dtype=bool)


def _anomaly(data: Table, seed: int) -> SplitPlan:
    usable = usable_rows(data)
    normal = normal_flags(data)
    groups = groups_of(data)
    rng = np.random.default_rng(seed)
    sides = np.asarray([split_side(v) for v in column(data, "split")], dtype=object)
    rows = np.flatnonzero(usable)
    normals = rows[normal[rows]]
    defects = rows[~normal[rows]]
    notes: list[str] = []
    train_side = rows[sides[rows] == "train"]
    test_side = rows[sides[rows] == "test"]
    if len(train_side) and len(test_side):
        train = train_side[normal[train_side]]
        test = test_side
        if len(train) < len(train_side):
            notes.append(
                f"{_plural(len(train_side) - len(train), 'defect')} in train left out of "
                "training: the model learns from normal images only."
            )
        reason = (
            f"The producer's split: learnt from {_plural(len(train), 'normal image')} of train, "
            f"scored on test ({len(test):,}) (R-VSP-01, R-VSP-04)."
        )
        kind: Literal["producer", "folds", "holdout"] = "producer"
        code = "R-VSP-04"
    elif len(defects):
        whole = {str(groups[r]) for r in defects}
        free = np.array([r for r in normals if str(groups[r]) not in whole], dtype=int)
        tied = np.array([r for r in normals if str(groups[r]) in whole], dtype=int)
        train, part = _grouped_share(free, groups, TEST_SHARE, rng)
        test = np.sort(np.concatenate([part, tied, defects]).astype(int))
        reason = (
            f"No producer's split: learnt from {_plural(len(train), 'normal image')}; scored on "
            f"{len(test) - len(defects):,} other normal images (grouped by copies, seed {seed}) "
            f"and the {_plural(len(defects), 'defect')} (R-VSP-04)."
        )
        kind, code = "holdout", "R-VSP-04"
    else:
        return _anomaly_folds(normals, groups, rng, seed)
    leaks = _leaks(train, test, groups)
    if leaks:
        notes.append(
            f"{_plural(leaks, 'test image')} with a near-copy in train: the scores will be "
            "optimistic (R-VQ-11)."
        )
    defect_test = int((~normal[test]).sum()) if len(test) else 0
    normal_test = len(test) - defect_test
    holdout: np.ndarray = np.array([], dtype=int)
    if defect_test < 2 or normal_test < 2:
        train, holdout = _grouped_share(train, groups, HOLDOUT_SHARE, rng)
        notes.append(
            f"{_plural(len(holdout), 'normal training image')} held out to set the threshold: the "
            "test cannot (R-VTH-02 needs normal images the model never saw) (R-VSP-04)."
        )
    return SplitPlan(
        kind=kind,
        code=code,
        reason=reason,
        folds=[SplitFold(train=np.asarray(train).tolist(), test=np.asarray(test).tolist())],
        holdout=np.asarray(holdout).tolist(),
        leaks=leaks,
        seed=seed,
        notes=notes,
    )


def _anomaly_folds(
    normals: np.ndarray, groups: np.ndarray, rng: np.random.Generator, seed: int
) -> SplitPlan:
    names = rng.permutation(np.unique(groups[normals]))
    k = int(min(FOLDS, len(names)))
    fold_of = {str(name): i % k for i, name in enumerate(names)}
    assigned = np.array([fold_of[str(groups[r])] for r in normals])
    folds = [
        SplitFold(train=normals[assigned != f].tolist(), test=normals[assigned == f].tolist())
        for f in range(k)
    ]
    return SplitPlan(
        kind="folds",
        code="R-VSP-05",
        reason=(
            f"No defect to score against: {k} folds of the {_plural(len(normals), 'image')}, "
            f"copies kept together, seed {seed}; each image is scored by a model that did not see "
            "it, and the threshold is the 99 % quantile of those scores (R-VSP-05, R-VTH-02)."
        ),
        folds=folds,
        seed=seed,
    )


__all__ = ["FOLDS", "SplitFold", "SplitPlan", "groups_of", "normal_flags", "plan_split"]
