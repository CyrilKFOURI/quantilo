"""N31: declared checks (`clean_checks`, `clean_check_runs`, migration 0033). The checks of a family
live on its original; each run, on the original or a cleaned version, is kept with what triggered
it: `declared` (saved), `opened` (the data changed since the last run), `import` (a reimport),
`version` (a new cleaned version)."""

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field
from quantilo.cleaning.checks import CheckReport, CheckResult, Checks, CleanCheck

Trigger = Literal["declared", "opened", "import", "version"]
KEPT_RUNS = 20


@dataclass
class CheckRun:
    """One run, summed up (the history of a dataset's checks)."""

    ran_at: str
    trigger: Trigger
    rows: int
    passed: int
    failed: int
    errors: int
    failing_rows: int


@dataclass
class CleanChecks:
    """The checks of a dataset's family, what they find on this dataset now, and their runs."""

    dataset_id: str
    original_id: str
    checks: list[CleanCheck]
    report: CheckReport | None = None
    ran_at: str | None = None
    trigger: Trigger | None = None
    history: list[CheckRun] = field(default_factory=list)


class ChecksRequest(BaseModel):
    checks: list[CleanCheck] = Field(default_factory=list, max_length=100)


def digest(checks: list[CleanCheck]) -> str:
    body = Checks(checks=checks).model_dump_json()
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def report_of(raw: str) -> CheckReport:
    data: dict[str, Any] = json.loads(raw)
    data["results"] = [CheckResult(**result) for result in data.get("results", [])]
    return CheckReport(**data)


def declared(connection: sqlite3.Connection, root: str) -> list[CleanCheck]:
    row = connection.execute(
        "SELECT checks FROM clean_checks WHERE dataset_id = ?", (root,)
    ).fetchone()
    return Checks.model_validate_json(row["checks"]).checks if row else []


def store(connection: sqlite3.Connection, root: str, checks: list[CleanCheck]) -> None:
    if not checks:
        connection.execute("DELETE FROM clean_checks WHERE dataset_id = ?", (root,))
        return
    connection.execute(
        "INSERT INTO clean_checks (dataset_id, checks, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT (dataset_id) DO UPDATE SET checks = excluded.checks, "
        "updated_at = excluded.updated_at",
        (root, Checks(checks=checks).model_dump_json(), datetime.now(UTC).isoformat()),
    )


def record(
    connection: sqlite3.Connection,
    dataset_id: str,
    stamp: str,
    checks: list[CleanCheck],
    trigger: Trigger,
    report: CheckReport,
) -> str:
    ran_at = datetime.now(UTC).isoformat()
    connection.execute(
        "INSERT INTO clean_check_runs (dataset_id, stamp, checks_hash, trigger, report, ran_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (dataset_id, stamp, digest(checks), trigger, json.dumps(asdict(report)), ran_at),
    )
    connection.execute(
        "DELETE FROM clean_check_runs WHERE dataset_id = ? AND id NOT IN (SELECT id FROM "
        "clean_check_runs WHERE dataset_id = ? ORDER BY id DESC LIMIT ?)",
        (dataset_id, dataset_id, KEPT_RUNS),
    )
    return ran_at


def runs(connection: sqlite3.Connection, dataset_id: str) -> list[sqlite3.Row]:
    """The dataset's runs, the latest first."""
    return connection.execute(
        "SELECT * FROM clean_check_runs WHERE dataset_id = ? ORDER BY id DESC", (dataset_id,)
    ).fetchall()


def summary(row: sqlite3.Row) -> CheckRun:
    report = report_of(row["report"])
    return CheckRun(
        ran_at=row["ran_at"],
        trigger=row["trigger"],
        rows=report.rows,
        passed=report.passed,
        failed=report.failed,
        errors=report.errors,
        failing_rows=report.failing_rows,
    )
