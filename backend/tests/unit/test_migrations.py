import shutil
from pathlib import Path

import pytest

from app.core import db


def test_migrations_create_tables(tmp_path: Path) -> None:
    connection = db.connect(tmp_path / "app.db")
    applied = db.apply_migrations(connection)
    tables = {
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    connection.close()

    assert applied == [
        "0001_init",
        "0002_datasets",
        "0003_columns",
        "0004_saved_filters",
        "0005_ml",
        "0006_ml_classes",
        "0007_ml_tuning",
        "0008_kpi_definitions",
        "0009_synthetic_runs",
        "0010_synthetic_device",
        "0011_agent",
        "0012_agent_tokens",
        "0013_agent_notes",
        "0014_dataset_versions",
        "0015_segment_runs",
        "0016_unusual_marks",
        "0020_ml_predictions",
        "0022_ts_analyses",
        "0023_recipes",
        "0024_cleaning",
        "0025_projects",
    ]
    expected = {
        "schema_migrations",
        "settings",
        "datasets",
        "dataset_files",
        "jobs",
        "columns",
        "saved_filters",
        "ml_runs",
        "ml_tunings",
        "ml_trials",
        "kpi_definitions",
        "synthetic_runs",
        "agent_conversations",
        "agent_messages",
        "agent_notes",
        "dataset_versions",
        "segment_runs",
        "unusual_marks",
        "ml_predictions",
        "ts_analyses",
        "recipes",
        "recipe_versions",
        "recipe_runs",
        "clean_versions",
        "clean_drafts",
        "projects",
        "project_items",
    }
    assert expected <= tables


def test_migrations_are_idempotent(tmp_path: Path) -> None:
    connection = db.connect(tmp_path / "app.db")
    db.apply_migrations(connection)
    second_run = db.apply_migrations(connection)
    connection.close()

    assert second_run == []


def test_projects_migration_is_additive_on_a_filled_0024_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S12 (087): a base made before projects keeps every dataset and version, none in a project."""
    before = tmp_path / "before"
    before.mkdir()
    for migration in sorted(db.MIGRATIONS_DIR.glob("*.sql")):
        if migration.stem < "0025":
            shutil.copy(migration, before / migration.name)
    monkeypatch.setattr(db, "MIGRATIONS_DIR", before)
    connection = db.connect(tmp_path / "app.db")
    db.apply_migrations(connection)
    for dataset_id in ("parent", "child"):
        connection.execute(
            "INSERT INTO datasets (id, name, source_path, source_kind, format, options, "
            "fingerprint, storage_path, status, created_at) "
            "VALUES (?, ?, '/x.csv', 'file', 'csv', '{}', 'f', 's', 'ready', '2026-01-01')",
            (dataset_id, dataset_id),
        )
    connection.execute(
        "INSERT INTO clean_versions (dataset_id, parent_id, seq, steps, origin, parent_stamp, "
        "report, created_at) VALUES ('child', 'parent', 1, '{}', 'user', '', '{}', '')"
    )
    connection.commit()

    monkeypatch.undo()
    assert db.apply_migrations(connection) == ["0025_projects"]
    names = [row[0] for row in connection.execute("SELECT id FROM datasets ORDER BY id")]
    versions = connection.execute("SELECT count(*) FROM clean_versions").fetchone()[0]
    items = connection.execute("SELECT count(*) FROM project_items").fetchone()[0]
    connection.close()
    assert (names, versions, items) == (["child", "parent"], 1, 0)
