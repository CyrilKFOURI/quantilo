-- Cleaning (065): a cleaned version is an ordinary dataset (the child) tied to the dataset it
-- was made from (the parent), which is never written. New tables only: `datasets` is unchanged.
-- Deleting the child removes its row; deleting the parent removes the link, and the child
-- stays as an ordinary dataset. The child's rowmap is datasets/<child>/clean/rowmap.parquet.
CREATE TABLE clean_versions (
    dataset_id    TEXT PRIMARY KEY REFERENCES datasets (id) ON DELETE CASCADE,
    parent_id     TEXT NOT NULL REFERENCES datasets (id) ON DELETE CASCADE,
    seq           INTEGER NOT NULL,  -- "Cleaned v<seq>", counted per parent
    steps         TEXT NOT NULL,     -- JSON (quantilo.cleaning.model.Steps), never rewritten
    origin        TEXT NOT NULL,     -- user, or auto (made after the import, Settings > Cleaning)
    parent_stamp  TEXT NOT NULL,     -- the parent's import and types when it was made
    report        TEXT NOT NULL,     -- JSON (quantilo.cleaning.report.Report)
    created_at    TEXT NOT NULL,
    UNIQUE (parent_id, seq)
);
CREATE INDEX clean_versions_parent ON clean_versions (parent_id, seq);

-- The draft of the Clean page of a dataset, saved at every change (C-11).
CREATE TABLE clean_drafts (
    dataset_id  TEXT PRIMARY KEY REFERENCES datasets (id) ON DELETE CASCADE,
    steps       TEXT NOT NULL,  -- JSON (quantilo.cleaning.model.Steps)
    updated_at  TEXT NOT NULL
);
