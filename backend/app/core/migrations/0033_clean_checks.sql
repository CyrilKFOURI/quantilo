-- N31 (docs/etudes/2026-11-12-audit-nettoyage.md): declared checks, run again at each import. The
-- checks of a family are kept on its original (`clean_checks`, one JSON list validated by
-- quantilo.cleaning.checks.Checks); each run on a dataset of the family (the original, a cleaned
-- version) is kept with what triggered it (`clean_check_runs`: declared, opened, import, version),
-- the last 20 per dataset. A check never changes the data. New tables only.
CREATE TABLE clean_checks (
    dataset_id  TEXT PRIMARY KEY REFERENCES datasets (id) ON DELETE CASCADE,
    checks      TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE clean_check_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id   TEXT NOT NULL REFERENCES datasets (id) ON DELETE CASCADE,
    stamp        TEXT NOT NULL,  -- the dataset's import, size and types when run
    checks_hash  TEXT NOT NULL,  -- the checks run (a change runs them again)
    trigger      TEXT NOT NULL
                 CHECK (trigger IN ('declared', 'opened', 'import', 'version')),
    report       TEXT NOT NULL,  -- JSON (quantilo.cleaning.checks.CheckReport)
    ran_at       TEXT NOT NULL
);
CREATE INDEX clean_check_runs_dataset ON clean_check_runs (dataset_id, id);
