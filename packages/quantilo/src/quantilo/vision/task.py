"""Which task an image set calls for (112; study § 6.2): rules R-VML, said at the top of *Model*.

Rows used: decoded (`decode_ok`) and not left out (`excluded`, 105: R-VQ-01, 09, 14). The rules are
read in this order, the first that holds wins:

- R-VML-07: fewer than 20 usable images: no model (« needs at least 20 images »).
- R-VML-02: normal images were read (`is_normal`, R-VIS-LAY-03) and every training image is normal,
  or a single class: anomaly detection, evaluable when the test holds defects and normals.
- R-VML-01: at least 2 classes of at least 5 images each: classification (rarer classes left out).
- R-VML-03: no label (or labels too rare for R-VML-01): exploration by clusters; anomaly detection
  is offered with the warning « assumes most images are normal ».
- R-VML-04: masks exist: pixel metrics are added to anomaly detection.
- R-VML-06 (c3): labels joined by `|` on at least 2 usable images: multi-label (read before
  R-VML-01, which would take `cat|dog` for a class). Two true/false metadata columns or more:
  multi-label offered (Advanced).
- R-VML-05 (c3): a numeric metadata column with 20 known values and 10 distinct: regression on
  that column offered (Advanced), never the rule's task.

The user may impose another possible task; the rule stays shown (`rule_task`, `rule_message`).
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pyarrow as pa
from pydantic import BaseModel

Task = Literal["classification", "anomaly", "clusters", "regression", "multilabel"]
TASKS: tuple[Task, ...] = ("classification", "anomaly", "clusters", "regression", "multilabel")
MIN_IMAGES = 20  # R-VML-07
MIN_PER_CLASS = 5  # R-VML-01 (and R-VQ-14)
TRAIN_SPLITS = frozenset({"train", "training", "val", "valid", "validation", "dev"})
TEST_SPLITS = frozenset({"test", "testing"})
NORMAL_WARNING = "assumes most images are normal"

Table = pa.Table | Mapping[str, Sequence[Any]]


class TaskChoice(BaseModel):
    task: Task | None  # None: no model can be trained (R-VML-07)
    code: str
    message: str  # the sentence at the top of the Model tab
    reason: str
    chosen_by: Literal["rule", "user"] = "rule"
    rule_task: Task | None = None  # what the rule proposed (shown even when the user changed it)
    rule_code: str = ""
    rule_message: str = ""
    n_images: int
    n_usable: int
    classes: dict[str, int] = {}  # classification: the classes learnt and their images
    left_out: dict[str, int] = {}  # classes with fewer than MIN_PER_CLASS images
    n_normal: int = 0
    n_defect: int = 0
    evaluable: bool = False  # anomalies: the test holds defects and normals
    pixel: bool = False  # R-VML-04
    alternatives: list[Task] = []  # other tasks the user may choose
    blocked: str | None = None
    notes: list[str] = []
    # c3: regression (R-VML-05) and multi-label (R-VML-06)
    target: str | None = None  # the column predicted (regression)
    n_target: int = 0  # usable images whose target is known
    label_names: list[str] = []  # multi-label: the labels learnt
    label_source: Literal["pipe", "columns"] | None = None
    numeric_targets: list[str] = []  # columns a regression can learn (R-VML-05)
    binary_targets: list[str] = []  # true/false columns a multi-label model can learn


def n_rows(data: Table) -> int:
    if isinstance(data, Mapping):
        return len(next(iter(data.values()), []))
    return data.num_rows


def column(data: Table, name: str) -> list[Any]:
    """A column as a list (None everywhere when it is missing)."""
    if isinstance(data, Mapping):
        values = data.get(name)
        return list(values) if values is not None else [None] * n_rows(data)
    if name in data.column_names:
        return data.column(name).to_pylist()
    return [None] * data.num_rows


def usable_rows(data: Table) -> np.ndarray:
    """Decoded and not left out (105)."""
    ok = column(data, "decode_ok")
    excluded = column(data, "excluded")
    return np.array(
        [bool(o) and not bool(e) for o, e in zip(ok, excluded, strict=True)], dtype=bool
    )


def split_side(value: Any) -> Literal["train", "test"] | None:
    text = str(value).strip().lower() if value is not None else ""
    if text in TRAIN_SPLITS:
        return "train"
    if text in TEST_SPLITS:
        return "test"
    return None


def labels_of(data: Table) -> list[str | None]:
    return [None if v is None or str(v).strip() == "" else str(v) for v in column(data, "label")]


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}" + ("" if n == 1 else "s")


def _anomaly_counts(data: Table, usable: np.ndarray) -> tuple[bool, int, int, bool, bool]:
    """(normals were read, normals, defects, training images all normal, test evaluable)."""
    normal = column(data, "is_normal")
    sides = [split_side(v) for v in column(data, "split")]
    read = any(v is not None for i, v in enumerate(normal) if usable[i])
    n_normal = sum(1 for i, v in enumerate(normal) if usable[i] and v is True)
    n_defect = sum(1 for i, v in enumerate(normal) if usable[i] and v is False)
    has_split = any(s == "train" for i, s in enumerate(sides) if usable[i]) and any(
        s == "test" for i, s in enumerate(sides) if usable[i]
    )
    if has_split:
        train = [normal[i] for i in range(len(normal)) if usable[i] and sides[i] == "train"]
        test = [normal[i] for i in range(len(normal)) if usable[i] and sides[i] == "test"]
        all_normal = bool(train) and all(v is True for v in train)
        evaluable = any(v is False for v in test) and any(v is True for v in test)
    else:
        all_normal = read and n_defect == 0
        evaluable = n_defect >= 1 and n_normal >= 1
    return read, n_normal, n_defect, all_normal, evaluable


def _masks(data: Table, usable: np.ndarray) -> int:
    return sum(1 for i, v in enumerate(column(data, "mask_path")) if usable[i] and v)


def detect_task(data: Table, task: Task | None = None, target: str | None = None) -> TaskChoice:
    """R-VML-01 to 07 on an image table (§ 3.4 columns); `task` imposes another task, `target`
    the column of a regression (R-VML-05; its first candidate when None)."""
    from quantilo.vision import targets as tg

    if task is not None and task not in TASKS:
        raise ValueError(f"Unknown task {task!r}: choose one of {', '.join(TASKS)}.")
    n = n_rows(data)
    usable = usable_rows(data)
    n_usable = int(usable.sum())
    labels = labels_of(data)
    counts = Counter(lab for i, lab in enumerate(labels) if usable[i] and lab is not None)
    eligible = {k: v for k, v in sorted(counts.items()) if v >= MIN_PER_CLASS}
    rare = {k: v for k, v in sorted(counts.items()) if v < MIN_PER_CLASS}
    read, n_normal, n_defect, all_normal, evaluable = _anomaly_counts(data, usable)
    masks = _masks(data, usable)
    numeric = tg.numeric_targets(data, usable)
    binary = tg.binary_targets(data, usable)
    pipes = tg.pipe_images(labels, usable)
    sets = tg.label_sets(data, usable, labels, binary) if pipes or len(binary) >= 2 else None
    multilabel = sets is not None and len(sets.names) >= 2
    base: dict[str, Any] = {
        "n_images": n,
        "n_usable": n_usable,
        "numeric_targets": numeric,
        "binary_targets": binary,
    }

    if n_usable < MIN_IMAGES:
        blocked = (
            f"Models need at least {MIN_IMAGES} images: this set has {n_usable:,} usable "
            f"(decoded and not left out) (R-VML-07)."
        )
        return TaskChoice(
            task=None,
            code="R-VML-07",
            message=blocked,
            reason="Below this nothing can be evaluated.",
            rule_code="R-VML-07",
            rule_message=blocked,
            blocked=blocked,
            **base,
        )

    if multilabel and pipes >= tg.MIN_PIPE_IMAGES and not (read and all_normal) and sets:
        rule = _multilabel(sets)  # R-VML-06 before R-VML-01: `cat|dog` is not a class
    else:
        rule = _rule(counts, eligible, rare, read, n_normal, n_defect, all_normal, evaluable, masks)
    choice = TaskChoice(**rule, **base)
    choice.rule_task, choice.rule_code, choice.rule_message = (
        choice.task,
        choice.code,
        choice.message,
    )
    possible = _possible(eligible, len(counts), n_normal, read, n_usable)
    if numeric:
        possible.append("regression")
    if multilabel:
        possible.append("multilabel")
    choice.alternatives = [t for t in possible if t != choice.task]
    if (task is None or task == choice.task) and task != "regression":
        return choice
    if task not in possible:
        raise ValueError(_impossible(task, eligible))
    if task == "regression":
        column_name = target or numeric[0]
        if column_name not in numeric:
            raise ValueError(
                f"{column_name!r} is not a numeric column with {tg.MIN_KNOWN} known values and "
                f"{tg.MIN_DISTINCT} distinct ones: choose one of {', '.join(numeric)} (R-VML-05)."
            )
        values = tg.regression_values(data, column_name)
        known = int(np.sum(usable & np.isfinite(values)))
        forced = TaskChoice(**_regression(column_name, known), **base)
    elif task == "multilabel":
        assert sets is not None
        forced = TaskChoice(**_multilabel(sets), **base)
        forced.message += " You chose this task."
    else:
        forced = TaskChoice(
            **_forced(task, eligible, rare, read, n_normal, n_defect, evaluable, masks), **base
        )
    forced.chosen_by = "user"
    forced.rule_task, forced.rule_code, forced.rule_message = (
        choice.task,
        choice.code,
        choice.message,
    )
    forced.alternatives = [t for t in possible if t != task]
    return forced


def _rule(
    counts: Counter,
    eligible: dict[str, int],
    rare: dict[str, int],
    read: bool,
    n_normal: int,
    n_defect: int,
    all_normal: bool,
    evaluable: bool,
    masks: int,
) -> dict[str, Any]:
    if (read and all_normal) or len(counts) == 1:
        return _anomaly(
            "R-VML-02", counts, read, n_normal, n_defect, evaluable, masks, single=len(counts) == 1
        )
    if len(eligible) >= 2:
        return _classification("R-VML-01", eligible, rare)
    return _clusters(counts, rare)


def _forced(
    task: Task,
    eligible: dict[str, int],
    rare: dict[str, int],
    read: bool,
    n_normal: int,
    n_defect: int,
    evaluable: bool,
    masks: int,
) -> dict[str, Any]:
    if task == "classification":
        out = _classification("R-VML-01", eligible, rare)
    elif task == "anomaly":
        out = _anomaly("R-VML-02", Counter(), read, n_normal, n_defect, evaluable, masks)
        if not read:
            out["code"] = "R-VML-03"
            out["message"] = (
                f"Anomaly detection without labels: it {NORMAL_WARNING} (R-VML-03). "
                "Each image is scored by a model that did not see it."
            )
            out["reason"] = "Without normal or defect labels the model learns what is common."
    else:
        out = _clusters(Counter(), rare)
        out["message"] = "Exploration by clusters, as you chose: labels are not used to train."
    out["message"] += " You chose this task."
    return out


def _regression(target: str, known: int) -> dict[str, Any]:
    from quantilo.vision.targets import MIN_KNOWN

    blocked = None
    if known < MIN_KNOWN:
        blocked = f"{target!r} is known for {known:,} usable images: {MIN_KNOWN} are needed."
    return {
        "task": "regression",
        "code": "R-VML-05",
        "message": (
            f"Predict a number: {target!r}, known for {_plural(known, 'usable image')}, from "
            "frozen image vectors and a ridge regression (R-VML-05). You chose this task."
        ),
        "reason": "A numeric column of the labels file is something the images may tell.",
        "target": target,
        "n_target": known,
        "blocked": blocked,
    }


def _multilabel(sets: Any) -> dict[str, Any]:
    shown = ", ".join(sets.names[:8]) + (", ..." if len(sets.names) > 8 else "")
    how = (
        "labels joined by '|'" if sets.source == "pipe" else f"{len(sets.names)} true/false columns"
    )
    message = (
        f"Multi-label: {len(sets.names)} labels ({shown}) read from {how}; one model per label "
        f"(R-VML-06)."
    )
    notes = []
    if sets.left_out:
        left = ", ".join(f"{k} ({_plural(v, 'image')})" for k, v in sets.left_out.items())
        notes.append(
            f"Left out: {left}: fewer than 5 images with the label or without it (R-VMLB-02)."
        )
        message += " " + notes[0]
    return {
        "task": "multilabel",
        "code": "R-VML-06",
        "message": message,
        "reason": "An image can carry several labels: each is learnt on its own (one-vs-rest).",
        "label_names": list(sets.names),
        "label_source": sets.source,
        "n_target": int(sets.known.sum()),
        "notes": notes,
    }


def _possible(
    eligible: dict[str, int], n_labels: int, n_normal: int, read: bool, n_usable: int
) -> list[Task]:
    """Anomaly detection needs to know which images are normal: read as such (R-VIS-LAY-03), or
    a single class, or no label at all (then it assumes most images are normal)."""
    possible: list[Task] = []
    if len(eligible) >= 2:
        possible.append("classification")
    if (read and n_normal >= MIN_PER_CLASS) or (not read and n_labels <= 1):
        possible.append("anomaly")
    if n_usable >= MIN_IMAGES:
        possible.append("clusters")
    return possible


def _impossible(task: Task, eligible: dict[str, int]) -> str:
    if task == "regression":
        return (
            "Regression needs a numeric metadata column (from a labels file) with 20 known "
            "values and 10 distinct ones (R-VML-05)."
        )
    if task == "multilabel":
        return (
            "Multi-label needs labels joined by '|' or 2 true/false metadata columns, each label "
            "with 5 images or more, with it and without it (R-VML-06)."
        )
    if task == "classification":
        return (
            f"Classification needs at least 2 classes of {MIN_PER_CLASS} images or more; "
            f"this set has {len(eligible)} (R-VML-01)."
        )
    return (
        f"Anomaly detection needs to know the normal images: at least {MIN_PER_CLASS} read as "
        "normal (R-VIS-LAY-03), a single class, or no label at all (R-VML-02)."
    )


def _classification(code: str, eligible: dict[str, int], rare: dict[str, int]) -> dict[str, Any]:
    shown = ", ".join(f"{k} {v:,}" for k, v in list(eligible.items())[:8])
    more = ", ..." if len(eligible) > 8 else ""
    message = f"Classification: {len(eligible)} classes ({shown}{more}) ({code})."
    notes = []
    if rare:
        left = ", ".join(f"{k} ({_plural(v, 'image')})" for k, v in rare.items())
        notes.append(f"Left out: {left}, fewer than {MIN_PER_CLASS} images (R-VQ-14).")
        message += " " + notes[0]
    return {
        "task": "classification",
        "code": code,
        "message": message,
        "reason": f"Each class has at least {MIN_PER_CLASS} images to learn and test on.",
        "classes": eligible,
        "left_out": rare,
        "notes": notes,
    }


def _anomaly(
    code: str,
    counts: Counter,
    read: bool,
    n_normal: int,
    n_defect: int,
    evaluable: bool,
    masks: int,
    single: bool = False,
) -> dict[str, Any]:
    if read:
        message = (
            f"Anomaly detection: the model learns from {_plural(n_normal, 'normal image')} "
            f"({code})."
        )
        reason = "The training images are all normal: there is no defect to learn from."
    else:
        label = next(iter(counts), "one class")
        message = (
            f"Anomaly detection: every labelled image is {label!r}, so the model learns what it "
            f"looks like ({code})."
        )
        reason = "A single class: there is nothing to tell apart, only the usual to learn."
        n_normal = sum(counts.values()) if single else n_normal
    if evaluable:
        message += f" The test holds {_plural(n_defect, 'defect')}: it can be scored."
    else:
        message += " No defect to score against: the images are ranked from most to least unusual."
    notes = []
    if masks and evaluable:
        notes.append(f"{_plural(masks, 'mask')}: pixel metrics are added (R-VML-04).")
        message += " " + notes[-1]
    return {
        "task": "anomaly",
        "code": code,
        "message": message,
        "reason": reason,
        "n_normal": n_normal,
        "n_defect": n_defect,
        "evaluable": evaluable,
        "pixel": bool(masks) and evaluable,
        "notes": notes,
    }


def _clusters(counts: Counter, rare: dict[str, int]) -> dict[str, Any]:
    if counts:
        message = (
            f"Exploration by clusters: the labels are too rare to learn from (at least 2 classes "
            f"of {MIN_PER_CLASS} images are needed) (R-VML-03)."
        )
    else:
        message = "Exploration by clusters: no label to predict (R-VML-03)."
    message += f" Anomaly detection can be chosen: it {NORMAL_WARNING}."
    return {
        "task": "clusters",
        "code": "R-VML-03",
        "message": message,
        "reason": "Nothing to predict: the structure of the set is what can be read.",
        "left_out": rare,
    }


__all__ = [
    "MIN_IMAGES",
    "MIN_PER_CLASS",
    "NORMAL_WARNING",
    "TASKS",
    "Task",
    "TaskChoice",
    "detect_task",
    "labels_of",
    "split_side",
    "usable_rows",
]
