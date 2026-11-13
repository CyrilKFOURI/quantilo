"""Advanced › More models of a classification run (118; study §§ 6.1, 6.3, 6.4, 10).

Never trained before the user ticks them and starts a run (the onion rule); each one is greyed
with its reason and an estimated time when it cannot run (R-VMO-01, R-VMO-02). They train on the
same folds as the first plan and are scored on the same images, so their metrics compare with
the probe's; each says its gain over the linear probe with a paired bootstrap interval
(R-VMO-03), and none becomes the run's default (R-VMO-04: the first plan stays the reference).

- `finetune`: the top of DINOv2 ViT-S/14 and a new head (`finetune.py`).
- `protopnet`: ProtoPNet-lite on the frozen patch tokens (`protopnet.py`).
- `concept_bottleneck`: concepts named by the user, then the class (`concepts.py`); on the
  DINOv2 vectors when they are here, on the classic descriptors otherwise.

What each model needs to be explained later is returned in `files` (JSON) and `arrays` (numpy)
and kept with the run: the fine-tuned vectors and head, the prototypes and each scored image's
"this looks like that", the concept predictions and the class head of the bottleneck.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pyarrow as pa

from quantilo.vision import metrics as vm
from quantilo.vision.task import column

MORE = ("finetune", "protopnet", "concept_bottleneck")
TITLES = {
    "finetune": "Fine-tuned DINOv2 ViT-S/14 (top blocks)",
    "protopnet": "ProtoPNet-lite on DINOv2 ViT-S/14 (this looks like that)",
    "concept_bottleneck": "Concept bottleneck (your concepts, then the class)",
}
BACKBONE = "dinov2_s14"
# R-VMO-02: seconds per training image and fold, measured on Beans (1,167 training images, RTX
# 4050 Laptop, AMP: fine-tuning 57 s with the decoding, ProtoPNet-lite 16 s, the bottleneck 10 s
# on the CPU); a CPU is taken as 40 times slower than the GPU for the networks
GPU_SECONDS = {"finetune": 0.045, "protopnet": 0.012, "concept_bottleneck": 0.009}
CPU_FACTOR = 40.0
DECODE_SECONDS = 0.01
LONG_SECONDS = 15 * 60

Progress = Callable[[float, str], None]


@dataclass
class MoreOption:
    """One Advanced › More model and whether it can run here (R-VMO-01, R-VMO-02)."""

    key: str
    title: str
    available: bool
    reason: str | None
    estimate: str | None  # "≈ 2 min on NVIDIA ...", from the image count
    seconds: float | None
    note: str


def _gpu_label(allow_gpu: bool) -> tuple[bool, str]:
    from quantilo.vision import deep

    if not deep.installed():
        return False, "CPU"
    from quantilo.synthetic.deep.device import pick

    device = pick("auto", allow_gpu)
    return device.device.type != "cpu", device.label


def options(
    task: str | None,
    n_train: int,
    folds: int,
    n_concepts: int,
    *,
    allow_gpu: bool = True,
    deep_ready: bool | None = None,
) -> list[MoreOption]:
    """R-VMO-01 (availability, with its reason) and R-VMO-02 (an estimated time, from the number
    of training images and folds, said as an estimate; the first epoch is timed during the run)."""
    from quantilo.vision import deep
    from quantilo.vision.deep.budget import duration_text as duration_text_of

    notes = {
        "finetune": (
            "Fine-tunes the last 2 blocks of DINOv2 and a new head, with mixed precision and "
            "early stopping. Worth it only when the probe leaves a gap (R-VFT-01 to 06)."
        ),
        "protopnet": (
            "Learns 10 prototypes per class, each pushed onto a patch of a real training image; "
            "every prediction reads 'this patch looks like that prototype' (Chen et al., 2019; "
            "R-VPP-01 to 06)."
        ),
        "concept_bottleneck": (
            "Predicts the concepts you named (columns of the table or groups of images saved "
            "from the gallery), then the class from those concepts only; correct a concept and "
            "the prediction follows (Koh et al., 2020; R-VCB-01 to 05)."
        ),
    }
    status = deep.status()
    if deep_ready is None:
        deep_ready = status.backbone(BACKBONE).available
    deep_reason = status.backbone(BACKBONE).reason
    on_gpu, label = _gpu_label(allow_gpu) if deep_ready else (False, "CPU")
    out = []
    for key in MORE:
        reason = None
        if task != "classification":
            reason = "Only for classification: these models learn classes (R-VMO-01)."
        elif key in ("finetune", "protopnet") and not deep_ready:
            reason = deep_reason or "Needs the DINOv2 ViT-S/14 weights (R-VDP-02)."
        elif key == "concept_bottleneck" and n_concepts < 2:
            reason = (
                f"Needs at least 2 concepts; {n_concepts} found. A concept is a column of the "
                "table (true/false, two values, or a number) that the import did not make, or a "
                "group of images saved as a concept from the gallery's selection (R-VCB-01)."
            )
        seconds = None
        if reason is None:
            per = GPU_SECONDS[key] * (1.0 if on_gpu or key == "concept_bottleneck" else CPU_FACTOR)
            seconds = per * max(1, n_train) * max(1, folds)
            if key != "concept_bottleneck":
                seconds += DECODE_SECONDS * n_train
        where = label if key != "concept_bottleneck" else "the CPU (numpy)"
        estimate = None
        if seconds is not None:
            estimate = f"≈ {duration_text_of(seconds)} on {where}"
            if folds > 1:
                estimate += f" ({folds} folds)"
            if seconds > LONG_SECONDS:
                estimate += "; more than 15 min: the run will ask before going on"
        out.append(
            MoreOption(key, TITLES[key], reason is None, reason, estimate, seconds, notes[key])
        )
    return out


# ---------------------------------------------------------------------------------------------
# Training


@dataclass
class MoreOutcome:
    models: list[Any] = field(default_factory=list)  # VisionModelResult
    columns: dict[str, Any] = field(default_factory=dict)
    files: dict[str, Any] = field(default_factory=dict)  # name -> JSON-ready
    arrays: dict[str, np.ndarray] = field(default_factory=dict)  # name -> array (.npy)
    notes: list[str] = field(default_factory=list)


def _main(report: vm.ClassificationReport) -> float:
    if report.main.metric == "balanced_accuracy":
        return report.balanced_accuracy
    return report.accuracy


def _gain(
    y: np.ndarray,
    proba: np.ndarray,
    reference: np.ndarray,
    classes: list[str],
    groups: np.ndarray,
    metric: str,
    seed: int,
) -> tuple[float, vm.MetricInterval | None]:
    """R-VMO-03: (gain in the main metric over the reference, paired bootstrap interval by
    groups)."""
    pred = np.array([classes[i] for i in proba.argmax(axis=1)])
    ref = np.array([classes[i] for i in reference.argmax(axis=1)])

    def value(p: np.ndarray, rows: np.ndarray) -> float:
        if metric == "balanced_accuracy":
            present = np.unique(y[rows])
            if len(present) < 2:
                return float("nan")
            return vm.balanced_accuracy(y[rows], p[rows], labels=list(present))
        return vm.accuracy(y[rows], p[rows])

    every = np.arange(len(y))
    gain = value(pred, every) - value(ref, every)
    interval = vm.bootstrap_interval(
        lambda rows: value(pred, rows) - value(ref, rows), len(y), groups=groups, seed=seed
    )
    return float(gain), interval


def _gain_note(title: str, gain: float, interval: vm.MetricInterval | None, metric: str) -> str:
    text = f"{100 * gain:+.1f} pts of {metric.replace('_', ' ')} over the linear probe"
    if interval is not None:
        text += f" [{100 * interval.low:+.1f}; {100 * interval.high:+.1f}]"
        verdict = (
            "within noise"
            if interval.low <= 0 <= interval.high
            else ("a real gain" if gain > 0 else "a real loss")
        )
        text += f", {verdict}"
    return f"{title}: {text} (paired bootstrap by copies, same images; R-VMO-03)."


def run_more(
    keys: Sequence[str],
    table: pa.Table,
    plan: Any,
    labels: np.ndarray,
    groups: np.ndarray,
    classes: list[str],
    sources: Sequence[Any],
    classic: np.ndarray,
    deep_x: np.ndarray | None,
    steps: Progress,
    *,
    concepts: Sequence[Any] = (),
    reference: np.ndarray | None = None,
    allow_gpu: bool = True,
    confirm_long: bool = False,
    seed: int = 0,
    bootstrap: bool = True,
    cancel: Callable[[], bool] | None = None,
) -> MoreOutcome:
    """Train the More models `keys` on every fold of `plan` and score them on `plan.scored`.
    `reference` = the linear probe's probabilities on the scored images (for R-VMO-03)."""
    from quantilo.vision import classic_models as cm

    keys = [k for k in MORE if k in keys]
    outcome = MoreOutcome()
    if not keys:
        return outcome
    scored = np.asarray(plan.scored, dtype=int)
    position = {int(r): i for i, r in enumerate(scored)}
    y = labels[scored].astype(str)
    g = groups[scored].astype(str)
    y_train_all = labels[np.asarray(sorted({r for f in plan.folds for r in f.train}), dtype=int)]
    used = sorted({int(r) for f in plan.folds for r in (*f.train, *f.test)})
    share = 1.0 / len(keys)
    widths = np.asarray(column(table, "width"), dtype=object)
    heights = np.asarray(column(table, "height"), dtype=object)

    def sized(row: int) -> tuple[int, int]:
        w, h = widths[row], heights[row]
        return (int(w), int(h)) if w and h else (1, 1)

    bank = None
    net_spec = None
    for slot, key in enumerate(keys):
        low = slot * share

        def tick(part: float, message: str, low: float = low) -> None:
            steps(low + share * min(1.0, max(0.0, part)), message)

        started = time.perf_counter()
        proba = np.zeros((len(scored), len(classes)))
        notes: list[str] = []
        logs: list[Any] = []
        extra_model: dict[str, Any] = {}
        try:
            if key in ("finetune", "protopnet"):
                from quantilo.vision.deep import finetune as ft
                from quantilo.vision.deep.backbones import BACKBONES

                net_spec = BACKBONES[BACKBONE]
                if bank is None:
                    bank = ft.ImageBank(
                        sources,
                        net_spec,
                        used,
                        progress=lambda p, m: tick(0.1 * p, m),
                        cancel=cancel,
                    )
            if key == "finetune":
                proba, logs, extra_model = _finetune(
                    plan, labels, groups, classes, bank, position, tick, allow_gpu,
                    confirm_long, seed, cancel, outcome,
                )  # fmt: skip
            elif key == "protopnet":
                assert net_spec is not None
                proba, logs, extra_model = _protopnet(
                    plan, labels, groups, classes, bank, position, tick, allow_gpu, seed, cancel,
                    outcome, net_spec, sized,
                )  # fmt: skip
            else:
                proba, extra_model = _bottleneck(
                    plan, labels, groups, classes, classic, deep_x, position, tick, seed,
                    concepts, outcome,
                )  # fmt: skip
        except (cm.Cancelled, KeyboardInterrupt):
            raise
        except Exception as error:
            from quantilo.vision.deep import budget as budgets
            from quantilo.vision.deep import finetune as ft

            if isinstance(error, budgets.LongRun):
                raise
            if isinstance(error, ft.Cancelled):
                raise cm.Cancelled("Stopped: nothing was saved.") from error
            outcome.notes.append(f"{TITLES[key]} not trained: {error}")
            continue
        report = vm.classification_report(
            y, proba=proba, labels=classes, y_train=y_train_all.astype(str), groups=g,
            bootstrap=bootstrap, seed=seed,
        )  # fmt: skip
        if reference is not None:
            gain, interval = _gain(y, proba, reference, classes, g, report.main.metric, seed)
            notes.insert(0, _gain_note(TITLES[key], gain, interval, report.main.metric))
        if logs:
            first = logs[0]
            notes.append(f"Trained: {first.trained}.")
            notes.append(first.stopped)
            if first.estimate:
                notes.append(first.estimate + ".")
            notes.extend(first.notes)
            if len(logs) > 1:
                notes.append(f"Each of the {len(logs)} folds trained the same way.")
        notes.extend(extra_model.pop("notes", []))
        result = cm.VisionModelResult(
            key=key,
            title=TITLES[key],
            family="classic" if key == "concept_bottleneck" and deep_x is None else "deep",
            level="more",
            setting=extra_model.pop("setting", ""),
            classification=report,
            galleries={
                "confident_errors": [
                    int(scored[i]) for i in vm.most_confident_errors(y, proba, classes, cm.GALLERY)
                ],
                "uncertain": [int(scored[i]) for i in vm.most_uncertain(proba, cm.GALLERY)],
            },
            seconds=round(time.perf_counter() - started, 2),
            notes=notes,
            training=[asdict(log) for log in logs] or None,
            concepts=extra_model.pop("concepts", None),
            prototypes=extra_model.pop("prototypes", None),
        )
        outcome.models.append(result)
        outcome.columns[f"{key}_pred"] = [classes[i] for i in proba.argmax(axis=1)]
        outcome.columns[f"{key}_proba"] = proba.max(axis=1).round(6).tolist()
        outcome.columns[f"{key}_probs"] = proba.round(6).tolist()
    return outcome


def _fold_rows(plan: Any, f: int, ok: Callable[[np.ndarray], np.ndarray]) -> tuple[Any, Any]:
    fold = plan.folds[f]
    train = np.asarray(fold.train, dtype=int)
    test = np.asarray(fold.test, dtype=int)
    return train[ok(train)], test


def _finetune(
    plan: Any,
    labels: np.ndarray,
    groups: np.ndarray,
    classes: list[str],
    bank: Any,
    position: dict[int, int],
    tick: Progress,
    allow_gpu: bool,
    confirm_long: bool,
    seed: int,
    cancel: Callable[[], bool] | None,
    outcome: MoreOutcome,
) -> tuple[np.ndarray, list[Any], dict[str, Any]]:
    from quantilo.vision.deep import finetune as ft
    from quantilo.vision.deep.backbones import load

    proba = np.zeros((len(position), len(classes)))
    index = {c: i for i, c in enumerate(classes)}
    logs = []
    folds = len(plan.folds)
    for f in range(folds):
        train, test = _fold_rows(plan, f, bank.ok)
        net = load(BACKBONE)  # a fresh copy each fold: training changes it

        def progress(p: float, m: str, f: int = f) -> None:
            tick(
                0.1 + 0.9 * (f + p) / folds, m + (f" (fold {f + 1} / {folds})" if folds > 1 else "")
            )

        model = ft.finetune(
            net,
            bank,
            train,
            [index[str(v)] for v in labels[train]],
            groups[train],
            len(classes),
            allow_gpu=allow_gpu,
            confirm_long=confirm_long,
            seed=seed,
            progress=progress,
            cancel=cancel,
        )
        try:
            keep = np.r_[train, test]
            p, vectors = model.predict(keep.tolist())
            at = np.array([position[int(r)] for r in test], dtype=int)
            proba[at] = p[len(train) :]
            weight, bias = model.head_arrays()
            outcome.arrays[f"finetune_f{f}_rows.npy"] = keep.astype(np.int64)
            outcome.arrays[f"finetune_f{f}_vectors.npy"] = vectors.astype(np.float16)
            outcome.arrays[f"finetune_f{f}_weight.npy"] = weight
            outcome.arrays[f"finetune_f{f}_bias.npy"] = bias
            logs.append(model.log)
        finally:
            model.release()
            del net
    return (
        proba,
        logs,
        {
            "setting": f"{logs[0].epochs} epochs, best {logs[0].best_epoch}, batch "
            f"{logs[0].batch}, {logs[0].precision}"
        },
    )


def _protopnet(
    plan: Any,
    labels: np.ndarray,
    groups: np.ndarray,
    classes: list[str],
    bank: Any,
    position: dict[int, int],
    tick: Progress,
    allow_gpu: bool,
    seed: int,
    cancel: Callable[[], bool] | None,
    outcome: MoreOutcome,
    spec: Any,
    sized: Callable[[int], tuple[int, int]],
) -> tuple[np.ndarray, list[Any], dict[str, Any]]:
    from quantilo.vision.deep import protopnet as pp
    from quantilo.vision.deep.backbones import load

    used = sorted({int(r) for f in plan.folds for r in (*f.train, *f.test)})
    used = [r for r, ok in zip(used, bank.ok(used), strict=True) if ok]
    net = load(BACKBONE)
    features = pp.patch_features(
        net, bank, used, allow_gpu=allow_gpu, progress=lambda p, m: tick(0.1 + 0.3 * p, m),
        cancel=cancel,
    )  # fmt: skip
    del net
    proba = np.zeros((len(position), len(classes)))
    folds = len(plan.folds)
    logs = []
    kept_folds = []
    usable = set(used)
    notes: list[str] = []
    if features.pooled:
        notes.append(
            f"Patch features pooled 2 x 2 to fit in {pp.FEATURE_BUDGET_MB:,} MB: "
            f"{features.grid[0]} x {features.grid[1]} cells (R-VPP-01)."
        )
    for f in range(folds):
        train, test = _fold_rows(
            plan, f, lambda rows: np.array([int(r) in usable for r in rows], dtype=bool)
        )

        def progress(p: float, m: str, f: int = f) -> None:
            tick(
                0.4 + 0.6 * (f + p) / folds, m + (f" (fold {f + 1} / {folds})" if folds > 1 else "")
            )

        model, found = pp.fit_protopnet(
            features, train, labels[train].astype(str), groups[train], classes,
            allow_gpu=allow_gpu, seed=seed, progress=progress, cancel=cancel,
        )  # fmt: skip
        try:
            test_ok = np.array([r for r in test.tolist() if r in usable], dtype=int)
            p, contributions = model.explain(features.take(test_ok))
            at = np.array([position[int(r)] for r in test_ok], dtype=int)
            proba[at] = p
            images = {}
            for i, r in enumerate(test_ok.tolist()):
                size = sized(r)
                images[str(r)] = [
                    {
                        **asdict(c),
                        "box": pp.cell_box(c.cell, features.grid, size, spec),
                    }
                    for c in contributions[i]
                ]
            prototypes = []
            for proto in found.prototypes:
                prototypes.append(
                    {
                        **asdict(proto),
                        "box": pp.cell_box(proto.cell, features.grid, sized(proto.row), spec),
                    }
                )
            weights = model.head.last.weight.detach().double().cpu().numpy()
            kept_folds.append(
                {
                    "fold": f,
                    "prototypes": prototypes,
                    "weights": weights.round(5).tolist(),
                    "images": images,
                    "notes": found.notes,
                }
            )
            logs.append(found.log)
        finally:
            model.release()
    outcome.files["protopnet.json"] = {
        "version": 1,
        "backbone": BACKBONE,
        "classes": classes,
        "grid": list(features.grid),
        "per_class": pp.PER_CLASS,
        "folds": kept_folds,
    }
    distinct = len({(p["row"], p["cell"]) for p in kept_folds[0]["prototypes"]})
    return (
        proba,
        logs,
        {
            "setting": f"{pp.PER_CLASS} prototypes per class, {features.grid[0]} x "
            f"{features.grid[1]} patches",
            "prototypes": distinct,
            "notes": notes,
        },
    )


def _bottleneck(
    plan: Any,
    labels: np.ndarray,
    groups: np.ndarray,
    classes: list[str],
    classic: np.ndarray,
    deep_x: np.ndarray | None,
    position: dict[int, int],
    tick: Progress,
    seed: int,
    concepts: Sequence[Any],
    outcome: MoreOutcome,
) -> tuple[np.ndarray, dict[str, Any]]:
    from quantilo.vision import classic_models as cm
    from quantilo.vision.deep import concepts as cb

    if len(concepts) < cb.MIN_CONCEPTS:
        raise ValueError(
            f"needs at least {cb.MIN_CONCEPTS} concepts, {len(concepts)} given (R-VCB-01)."
        )
    proba = np.zeros((len(position), len(classes)))
    predicted = np.zeros((len(position), len(concepts)))
    folds = len(plan.folds)
    kept_folds = []
    first: Any = None
    space = (
        "the DINOv2 ViT-S/14 vectors (frozen)"
        if deep_x is not None
        else "the classic descriptors (standardised PCA)"
    )
    for f in range(folds):
        tick(f / folds, f"Concept bottleneck (fold {f + 1} / {folds})")
        fold = plan.folds[f]
        train = np.asarray(fold.train, dtype=int)
        test = np.asarray(fold.test, dtype=int)
        if deep_x is not None:
            ok = np.isfinite(deep_x[train]).all(axis=1)
            x_train, x_test = deep_x[train][ok], np.nan_to_num(deep_x[test])
        else:
            ok = np.isfinite(classic[train]).all(axis=1)
            transform = cm.descriptor_space(classic[train][ok], seed)
            x_train = transform.transform(classic[train][ok])
            x_test = transform.transform(np.nan_to_num(classic[test]))
        kept = train[ok]
        model = cb.fit_bottleneck(
            x_train, concepts, kept, labels[kept].astype(str), groups[kept], classes, seed=seed
        )
        c_test = model.concepts(x_test)
        at = np.array([position[int(r)] for r in test], dtype=int)
        predicted[at] = c_test
        proba[at] = model.head.proba(c_test)
        if first is None:
            first = model
        kept_folds.append(
            {
                "fold": f,
                "head": {
                    "classes": model.head.classes,
                    "mean": model.head.mean.round(6).tolist(),
                    "scale": model.head.scale.round(6).tolist(),
                    "weight": model.head.weight.round(6).tolist(),
                    "bias": model.head.bias.round(6).tolist(),
                    "setting": model.head.setting,
                },
                "rows": test.tolist(),
                "predicted": c_test.round(5).tolist(),
                "notes": model.notes,
            }
        )
    scored = np.asarray(sorted(position, key=position.__getitem__), dtype=int)
    found = cb.quality(first, concepts, predicted, scored)
    outcome.files["concept_bottleneck.json"] = {
        "version": 1,
        "space": space,
        "classes": classes,
        "concepts": [
            {
                "key": c.key,
                "name": c.name,
                "kind": c.kind,
                "source": c.source,
                "positive": c.positive,
                "note": c.note,
            }
            for c in concepts
        ],
        "truth": {
            str(int(r)): [
                None if not np.isfinite(c.values[r]) else float(c.values[r]) for c in concepts
            ]
            for r in scored.tolist()
        },
        "folds": kept_folds,
    }
    weak = [q.name for q in found if q.weak]
    notes = [f"Concepts read from {space}; the class reads the concepts only (R-VCB-02, 03)."]
    if weak:
        notes.append(
            f"Weak concepts (the images hardly show them): {', '.join(weak)}; the class leans "
            "on guesses there (R-VCB-04)."
        )
    notes.extend(first.notes)
    return proba, {
        "setting": f"{len(concepts)} concepts, {first.head.setting}",
        "concepts": [asdict(q) for q in found],
        "notes": notes,
    }


__all__ = ["MORE", "TITLES", "MoreOption", "MoreOutcome", "options", "run_more"]
