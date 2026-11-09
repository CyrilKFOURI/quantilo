-- What changed between two versions (031): the statistical photo of every imported version, and
-- the Parquet copy of the previous one when kept (Settings › Storage). They go with the dataset.
CREATE TABLE dataset_versions (
    dataset_id     TEXT NOT NULL REFERENCES datasets (id) ON DELETE CASCADE,
    version        INTEGER NOT NULL,  -- 1, 2… in import order
    imported_at    TEXT NOT NULL,
    storage_stamp  TEXT NOT NULL,     -- size:mtime of the Parquet copy the photo was taken of
    row_count      INTEGER NOT NULL,
    snapshot       TEXT NOT NULL,     -- JSON (quantilo.changes.snapshot)
    parquet_path   TEXT,              -- kept copy of this version's rows; NULL when not kept
    kept_bytes     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (dataset_id, version)
);
