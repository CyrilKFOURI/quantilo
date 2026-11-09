from pathlib import Path

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
    }
    assert expected <= tables


def test_migrations_are_idempotent(tmp_path: Path) -> None:
    connection = db.connect(tmp_path / "app.db")
    db.apply_migrations(connection)
    second_run = db.apply_migrations(connection)
    connection.close()

    assert second_run == []
