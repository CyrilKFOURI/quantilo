"""Evaluate the table detection on the external corpus (study § 8). Skips when the corpus is absent.

    uv run python -I tests/layout_corpus/evaluate.py [corpus folder] [--owner file.xlsx]

The corpus (public files, untrusted data) lives outside the repository; only the ground truth
(ranges and header rows) is here. Measures: precision and recall of the table ranges at EoB-0 and
EoB-2 (every edge within 0 or 2 cells), exact header rows, overall and per family.
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from openpyxl.utils.cell import range_boundaries

from quantilo.ingestion.layout import DetectedTable, detect_layout
from quantilo.ingestion.layout.model import HIGH, REVIEW

HERE = Path(__file__).resolve().parent
DEFAULT_CORPUS = HERE.parents[3] / ".." / "excel-corpus"


@dataclass
class Score:
    detected: int = 0
    expected: int = 0
    found0: int = 0
    found2: int = 0
    matched0: int = 0
    matched2: int = 0
    headers: int = 0
    lines: list[str] = field(default_factory=list)

    def add(self, other: "Score") -> None:
        for name in ("detected", "expected", "found0", "found2", "matched0", "matched2", "headers"):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def ratios(self) -> dict[str, float]:
        def ratio(a: int, b: int) -> float:
            return a / b if b else 1.0

        return {
            "precision_eob0": ratio(self.matched0, self.detected),
            "recall_eob0": ratio(self.found0, self.expected),
            "precision_eob2": ratio(self.matched2, self.detected),
            "recall_eob2": ratio(self.found2, self.expected),
            "headers_exact": ratio(self.headers, self.expected),
        }


def bounds(cell_range: str) -> tuple[int, int, int, int]:
    left, top, right, bottom = range_boundaries(cell_range)
    return top, left, bottom, right  # type: ignore[return-value]


def close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], tolerance: int) -> bool:
    return all(abs(x - y) <= tolerance for x, y in zip(a, b, strict=True))


def score_sheet(expected: object, tables: list[DetectedTable]) -> Score:
    score = Score()
    if expected == "no_table":
        sure = [t for t in tables if t.confidence >= HIGH]
        score.detected = len(sure)
        score.lines.append(f"no table expected, {len(sure)} sure table(s) found")
        return score
    shown = [t for t in tables if t.confidence >= REVIEW]
    score.detected = len(shown)
    assert isinstance(expected, list)
    score.expected = len(expected)
    used0: set[str] = set()
    used2: set[str] = set()
    for truth in expected:
        want = bounds(truth["range"])
        exact = next(
            (t for t in shown if t.id not in used0 and close(want, bounds(t.range), 0)), None
        )
        near = next(
            (t for t in shown if t.id not in used2 and close(want, bounds(t.range), 2)), None
        )
        if exact is not None:
            used0.add(exact.id)
            score.found0 += 1
        if near is not None:
            used2.add(near.id)
            score.found2 += 1
            header_ok = near.header_rows == truth["header_rows"]
            score.headers += header_ok
            score.lines.append(
                f"{truth['range']} -> {near.range} ({near.confidence:.2f}) header "
                f"{near.header_rows}{'' if header_ok else ' expected ' + str(truth['header_rows'])}"
            )
        else:
            best = ", ".join(f"{t.range} ({t.confidence:.2f})" for t in tables[:4]) or "none"
            score.lines.append(f"{truth['range']} MISSED (found: {best})")
    score.matched0 = len(used0)
    score.matched2 = len(used2)
    extra = [t for t in shown if t.id not in used2]
    if extra:
        score.lines.append("extra: " + ", ".join(f"{t.range} ({t.confidence:.2f})" for t in extra))
    return score


def evaluate(corpus: Path) -> tuple[Score, dict[str, Score], list[str], float]:
    truth = json.loads((HERE / "ground_truth.json").read_text(encoding="utf-8"))
    total = Score()
    families: dict[str, Score] = {}
    report: list[str] = []
    started = time.perf_counter()
    for entry in truth["files"]:
        path = corpus / entry["path"]
        if not path.exists():
            report.append(f"- {entry['path']}: absent, skipped")
            continue
        sheets = [name for name in entry["sheets"] if name]
        clock = time.perf_counter()
        layout = detect_layout(path, sheets or None)
        seconds = time.perf_counter() - clock
        family = families.setdefault(entry["family"], Score())
        report.append(f"- **{entry['path']}** ({entry['family']}, {seconds:.2f} s)")
        for sheet_name, expected in entry["sheets"].items():
            sheet = next((s for s in layout.sheets if s.name == sheet_name), None)
            tables = sheet.tables if sheet else []
            score = score_sheet(expected, tables)
            total.add(score)
            family.add(score)
            for line in score.lines:
                report.append(f"  - {sheet_name or 'csv'}: {line}")
    return total, families, report, time.perf_counter() - started


def table_line(name: str, score: Score) -> str:
    r = score.ratios()
    return (
        f"| {name} | {score.expected} | {score.detected} | {r['precision_eob0']:.2f} | "
        f"{r['recall_eob0']:.2f} | {r['precision_eob2']:.2f} | {r['recall_eob2']:.2f} | "
        f"{r['headers_exact']:.2f} |"
    )


def owner_report(path: Path) -> list[str]:
    lines = [f"Owner's file {path.name}:"]
    for sheet in detect_layout(path).sheets:
        for table in sheet.tables:
            lines.append(
                f"- {sheet.name}: {table.range}, header {table.header_rows}, {table.kind}, "
                f"confidence {table.confidence:.2f} ({table.level}), title {table.title!r}"
                + (f", warnings: {'; '.join(table.warnings)}" if table.warnings else "")
            )
    return lines


def main(argv: list[str]) -> int:
    owner = None
    if "--owner" in argv:
        index = argv.index("--owner")
        owner = Path(argv[index + 1])
        argv = argv[:index] + argv[index + 2 :]
    corpus = (
        Path(argv[0]) if argv else Path(os.environ.get("QUANTILO_LAYOUT_CORPUS", DEFAULT_CORPUS))
    )
    if not corpus.is_dir():
        print(f"Corpus not found at {corpus}: evaluation skipped.")
        return 0
    total, families, report, seconds = evaluate(corpus.resolve())
    print(f"Corpus {corpus.resolve()} in {seconds:.1f} s\n")
    print("| Family | Tables | Detected | P EoB-0 | R EoB-0 | P EoB-2 | R EoB-2 | Header exact |")
    print("|---|---|---|---|---|---|---|---|")
    for name in sorted(families):
        print(table_line(name, families[name]))
    print(table_line("**All**", total))
    print()
    print("\n".join(report))
    if owner is not None and owner.exists():
        print()
        print("\n".join(owner_report(owner)))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    raise SystemExit(main(sys.argv[1:]))
