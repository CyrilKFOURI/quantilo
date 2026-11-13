-- N46 (docs/etudes/2026-11-12-audit-nettoyage.md): the cleaning journal. One row per change of
-- a dataset's Clean draft (a step added, removed, changed, turned off or on, the steps moved, the
-- reliable fixes applied for the user, a reset), per saved version, per suggestion dismissed or
-- restored and per change of the checks: who (`actor`: user, auto, agent, replay), what, when.
-- `dataset_id` is the dataset cleaned (CASCADE); `version_id` the cleaned version the change went
-- into, set when it is saved (NULL: still the draft's). No foreign key on `version_id`: a deleted
-- version's rows are no longer read. New table only.
CREATE TABLE clean_journal (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id  TEXT NOT NULL REFERENCES datasets (id) ON DELETE CASCADE,
    version_id  TEXT,
    at          TEXT NOT NULL,
    actor       TEXT NOT NULL CHECK (actor IN ('user', 'auto', 'agent', 'replay')),
    action      TEXT NOT NULL,
    step_id     TEXT,
    op          TEXT,
    rule        TEXT,
    origin      TEXT,
    columns     TEXT NOT NULL DEFAULT '[]',  -- JSON list of the columns the step works on
    cells       INTEGER,
    rows        INTEGER,
    detail      TEXT
);
CREATE INDEX clean_journal_dataset ON clean_journal (dataset_id, id);
CREATE INDEX clean_journal_version ON clean_journal (version_id);
