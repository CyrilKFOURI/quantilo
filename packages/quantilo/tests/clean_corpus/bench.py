"""The precision bench (N01): every rule's proposals on the corpus, judged against the truth, per
catalogue entry (N02). Run by the tests (`tests/test_clean_bench.py`) and by
`scripts/clean_bench.py`, which prints the table.

Two corpora:
- the committed one (`cases.py`): synthetic files written in code, in CI;
- the real files (`--real`): the audit's 31 files in `../data-corpus`, `../kaggle-data`,
  `../excel-corpus` next to the repository, judged by `truth/*.clean-truth.json` (the audit's
  verdicts, one file per dataset); skipped when the folders are absent.

Precision = right / (right + wrong) proposals, and the same in cells (or rows removed). A
suggestion no truth judges is counted apart (`unjudged`).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import quantilo as dx
from quantilo.cleaning import catalogue
from quantilo.cleaning.suggestion import CleanSuggestion
from tests.clean_corpus.cases import ALL_CASES, Case, Truth

HERE = Path(__file__).resolve().parent
TRUTH = HERE / "truth"
ROOT = HERE.parents[4]  # the folder holding the repository and the corpora
CORPORA = {
    "data-corpus": ROOT / "data-corpus",
    "kaggle-data": ROOT / "kaggle-data",
    "excel-corpus": ROOT / "excel-corpus",
}
WORKERS = 4


@dataclass
class Judged:
    suggestion: CleanSuggestion
    verdict: str  # correct, wrong, decision, unjudged
    truth: Truth | None = None

    @property
    def weight(self) -> int:
        return self.suggestion.cells + self.suggestion.rows


@dataclass
class CaseResult:
    case: str
    found: list[CleanSuggestion]
    judged: list[Judged]
    seconds: float = 0.0
    error: str | None = None
    missing: list[Truth] = field(default_factory=list)  # truths no suggestion matched

    def matches(self, truth: Truth) -> list[CleanSuggestion]:
        return [s for s in self.found if _matches(truth, s)]


def _map_of(suggestion: CleanSuggestion) -> dict[str, Any]:
    raw = getattr(suggestion.step, "map", None)
    return dict(raw) if isinstance(raw, dict) else {}


def _merged(suggestion: CleanSuggestion, a: str, b: str) -> bool:
    mapping = _map_of(suggestion)
    target_a, target_b = mapping.get(a, a), mapping.get(b, b)
    return (a in mapping or b in mapping) and target_a == target_b


def _base(suggestion_id: str) -> str:
    """'B-04:race:maybe' -> 'B-04:race': the rule and its column, without the key."""
    parts = suggestion_id.split(":")
    return ":".join(parts[:2])


def _matches(truth: Truth, suggestion: CleanSuggestion) -> bool:
    if truth.id is not None:
        if suggestion.id != truth.id:
            return False
    else:
        if suggestion.rule != truth.rule:
            return False
        if truth.column is None:
            if suggestion.columns and truth.rule.startswith("B-"):
                return False
        elif truth.column not in suggestion.columns:
            return False
    if truth.sub is not None and suggestion.catalogue != truth.sub:
        return False
    if truth.step is not None and (suggestion.step is not None) != truth.step:
        return False
    if truth.op is not None and getattr(suggestion.step, "op", None) != truth.op:
        return False
    return all(_merged(suggestion, a, b) for a, b in truth.merges)


def judge(found: Sequence[CleanSuggestion], truths: Iterable[Truth]) -> list[Judged]:
    """Each suggestion's verdict: wrong if a `wrong` truth matches it (one wrong group makes a
    merge wrong), else right or a decision by the truth that matches, else unjudged."""
    truths = list(truths)
    judged = []
    for suggestion in found:
        matching = [t for t in truths if _matches(t, suggestion)]
        chosen = next((t for t in matching if t.verdict == "wrong"), None) or next(
            iter(matching), None
        )
        judged.append(Judged(suggestion, chosen.verdict if chosen else "unjudged", chosen))
    return judged


def run_case(case: Case, folder: Path) -> CaseResult:
    import time

    start = time.perf_counter()
    work = Path(tempfile.mkdtemp(prefix=f"{case.name}-", dir=folder))
    try:
        path = case.write(work)
        dataset = dx.open(path, cache_dir=work / "cache", **case.open_options)  # type: ignore[arg-type]
        cleaning = dataset.clean()
        found = list(cleaning.suggestions)
        if case.advanced:
            found += cleaning.advanced()
    except Exception as error:  # reported in the table, the test fails on it
        return CaseResult(case.name, [], [], error=f"{type(error).__name__}: {error}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    result = CaseResult(case.name, found, judge(found, case.truth))
    result.missing = [t for t in case.truth if t.verdict != "wrong" and not result.matches(t)]
    result.seconds = time.perf_counter() - start
    return result


def run_cases(cases: Sequence[Case] = ALL_CASES, workers: int = WORKERS) -> list[CaseResult]:
    """Every case, a few at a time (each opens its own file in its own folder)."""
    folder = Path(tempfile.mkdtemp(prefix="clean-bench-"))
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            return list(pool.map(lambda c: run_case(c, folder), cases))
    finally:
        shutil.rmtree(folder, ignore_errors=True)


# --- The tally ---------------------------------------------------------------------------------


@dataclass
class Tally:
    entry: str
    proposals: int = 0
    correct: int = 0
    wrong: int = 0
    decision: int = 0
    unjudged: int = 0
    auto: int = 0  # proposals marked automatic
    cells: int = 0  # cells (or rows) of the judged proposals
    cells_correct: int = 0
    files: set[str] = field(default_factory=set)

    @property
    def judged(self) -> int:
        return self.correct + self.wrong

    @property
    def precision(self) -> float | None:
        return self.correct / self.judged if self.judged else None

    @property
    def cell_precision(self) -> float | None:
        return self.cells_correct / self.cells if self.cells else None


def tally(results: Iterable[CaseResult]) -> dict[str, Tally]:
    totals: dict[str, Tally] = {}
    for result in results:
        for item in result.judged:
            key = item.suggestion.catalogue or item.suggestion.rule
            row = totals.setdefault(key, Tally(key))
            row.proposals += 1
            row.files.add(result.case)
            row.auto += int(item.suggestion.auto)
            if item.verdict == "correct":
                row.correct += 1
                row.cells += item.weight
                row.cells_correct += item.weight
            elif item.verdict == "wrong":
                row.wrong += 1
                row.cells += item.weight
            elif item.verdict == "decision":
                row.decision += 1
            else:
                row.unjudged += 1
    return dict(sorted(totals.items(), key=lambda i: _order(i[0])))


def _order(entry: str) -> tuple[int, str]:
    ids = list(catalogue.CATALOGUE)
    return (ids.index(entry) if entry in ids else len(ids), entry)


def gate(totals: dict[str, Tally], every_entry: bool = True) -> list[str]:
    """The automatic entries under the threshold; with `every_entry`, also those never right on
    the corpus (the committed one has a case for each; a lossless one may have none on real
    files)."""
    problems = []
    for entry in catalogue.automatic():
        row = totals.get(entry.id)
        if row is None or not row.correct:
            if every_entry or not entry.lossless:
                problems.append(f"{entry.id}: automatic but never right on this corpus")
            continue
        precision = row.precision or 0.0
        if precision < catalogue.AUTO_PRECISION:
            problems.append(f"{entry.id}: {precision:.1%} ({row.correct}/{row.judged})")
        cells = row.cell_precision
        if cells is not None and cells < catalogue.AUTO_PRECISION:
            problems.append(f"{entry.id}: {cells:.1%} of cells")
    return problems


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def table(totals: dict[str, Tally], title: str = "Corpus") -> str:
    lines = [
        f"{title}: precision per catalogue entry (right / judged; cells or rows)",
        "",
        "| Entry | Mode | Proposals | Right | Wrong | Decision | Unjudged | Precision | Cells | "
        "Precision (cells) | Auto | Files |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, row in totals.items():
        entry = catalogue.CATALOGUE.get(key)
        mode = entry.effective_mode if entry else "not in the catalogue"
        if entry is not None and entry.mode != entry.effective_mode:
            mode += f" (automatic with {entry.needs})"
        lines.append(
            f"| {key} | {mode} | {row.proposals} | {row.correct} | {row.wrong} | {row.decision} | "
            f"{row.unjudged} | {_pct(row.precision)} | {row.cells:,} | "
            f"{_pct(row.cell_precision)} | {row.auto} | {len(row.files)} |"
        )
    return "\n".join(lines)


def by_rule(totals: dict[str, Tally]) -> str:
    """The same, one line per rule (B-04 = B-04:sure + B-04:na + ...), as the audit's § 2.3."""
    rules: dict[str, Tally] = {}
    for key, row in totals.items():
        rule = key.split(":")[0]
        total = rules.setdefault(rule, Tally(rule))
        for name in ("proposals", "correct", "wrong", "decision", "unjudged", "auto", "cells"):
            setattr(total, name, getattr(total, name) + getattr(row, name))
        total.cells_correct += row.cells_correct
        total.files |= row.files
    lines = [
        "| Rule | Judged | Right | Wrong | Precision | Cells | Precision (cells) |",
        "|---|---|---|---|---|---|---|",
    ]
    for rule, row in rules.items():
        if row.judged:
            lines.append(
                f"| {rule} | {row.judged} | {row.correct} | {row.wrong} | {_pct(row.precision)} | "
                f"{row.cells:,} | {_pct(row.cell_precision)} |"
            )
    return "\n".join(lines)


def report(results: Sequence[CaseResult]) -> str:
    """What each case missed or got wrong, and its unjudged automatic suggestions."""
    lines = []
    for result in results:
        problems = []
        if result.error:
            problems.append(f"error: {result.error}")
        problems += [f"missed {t.label} ({t.after or 'now'})" for t in result.missing]
        problems += [
            f"wrong {j.suggestion.id} [{j.suggestion.catalogue}] ({j.truth.after or 'now'})"
            for j in result.judged
            if j.verdict == "wrong" and j.truth is not None
        ]
        problems += [
            f"unjudged automatic {j.suggestion.id} [{j.suggestion.catalogue}]"
            for j in result.judged
            if j.verdict == "unjudged" and j.suggestion.auto
        ]
        state = "ok" if not problems else "; ".join(problems)
        lines.append(
            f"- {result.case} ({result.seconds:.1f} s, {len(result.found)} found): {state}"
        )
    return "\n".join(lines)


# --- The real files (local only) ---------------------------------------------------------------


@dataclass(frozen=True)
class RealFile:
    label: str
    corpus: str  # data-corpus, kaggle-data, excel-corpus
    path: str  # inside the corpus folder
    sample_pct: float
    truths: tuple[Truth, ...]
    about: str = ""

    @property
    def location(self) -> Path:
        return CORPORA[self.corpus] / self.path


def real_files() -> list[RealFile]:
    files = []
    for source in sorted(TRUTH.glob("*.clean-truth.json")):
        raw = json.loads(source.read_text(encoding="utf-8"))
        truths = tuple(
            Truth(
                v["id"].split(":")[0],
                None,
                v["verdict"],
                v.get("why", ""),
                id=v["id"],
                op=v.get("op"),
            )
            for v in raw["verdicts"]
        )
        files.append(
            RealFile(
                raw["label"],
                raw["corpus"],
                raw["path"],
                float(raw.get("sample_pct", 100)),
                truths,
                raw.get("about", ""),
            )
        )
    return files


def real_available() -> bool:
    return all(folder.is_dir() for folder in CORPORA.values())


def _real_case(real: RealFile) -> Case:
    def write(folder: Path) -> Path:
        return real.location

    options: dict[str, object] = {}
    if real.sample_pct < 100:
        options["sample_pct"] = real.sample_pct
    return Case(real.label, real.about, write, real.truths, open_options=options)


def _relaxed(truths: Sequence[Truth], found: Sequence[CleanSuggestion]) -> tuple[Truth, ...]:
    """A judged id the rules now write another way ('B-04:race:maybe' -> 'B-04:race:file', after
    a milestone changes the mode) keeps its verdict when it is the only suggestion of its rule and
    column with the same op, and no truth judges that suggestion already."""
    ids = {s.id for s in found}
    judged = {t.id for t in truths}
    out = []
    for truth in truths:
        if truth.id is None or truth.id in ids:
            out.append(truth)
            continue
        same = [
            s
            for s in found
            if _base(s.id) == _base(truth.id)
            and s.id not in judged
            and (truth.op is None or getattr(s.step, "op", None) == truth.op)
        ]
        out.append(Truth(**{**truth.__dict__, "id": same[0].id}) if len(same) == 1 else truth)
    return tuple(out)


def run_real(labels: Sequence[str] | None = None, workers: int = 1) -> list[CaseResult]:
    """The audit's files, one at a time by default (some have 100,000 rows or more)."""
    chosen = [r for r in real_files() if not labels or r.label in labels]
    present = [r for r in chosen if r.location.is_file()]
    folder = Path(tempfile.mkdtemp(prefix="clean-real-"))
    results = []
    try:

        def one(real: RealFile) -> CaseResult:
            result = run_case(_real_case(real), folder)
            truths = _relaxed(real.truths, result.found)
            result.judged = judge(result.found, truths)
            result.missing = [t for t in truths if t.verdict != "wrong" and not result.matches(t)]
            return result

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            results = list(pool.map(one, present))
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return results


def main(argv: Sequence[str]) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="The cleaning precision bench (N01).")
    parser.add_argument("--real", action="store_true", help="the audit's real files instead")
    parser.add_argument("--only", nargs="*", help="these cases or real files only")
    parser.add_argument("--json", type=Path, help="write the table as JSON here")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    if args.real:
        if not real_available():
            print("The corpora folders are not next to the repository: nothing to run.")
            return 0
        results = run_real(args.only, workers=args.workers or 1)
        title = f"Real files ({len(results)})"
    else:
        cases = [c for c in ALL_CASES if not args.only or c.name in args.only]
        results = run_cases(cases, workers=args.workers or WORKERS)
        title = f"Committed corpus ({len(results)} cases)"
    totals = tally(results)
    print(table(totals, title))
    print()
    print(by_rule(totals))
    print()
    print(report(results))
    problems = gate(totals, every_entry=not args.real) if not args.only else []
    print()
    print("Automatic entries under 99 %: " + ("none" if not problems else "; ".join(problems)))
    if args.json:
        rows = [
            {
                **{k: v for k, v in row.__dict__.items() if k != "files"},
                "files": sorted(row.files),
                "precision": row.precision,
                "cell_precision": row.cell_precision,
            }
            for row in totals.values()
        ]
        args.json.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 1 if problems else 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
