-- Segments runs (feature 032): one segmentation per row. The full result, the segment of every row
-- and the map's prepared points are files in the dataset's folder (segments/<id>/run.json,
-- labels.parquet, drawn.npy), deleted with it.
CREATE TABLE segment_runs (
    id                  TEXT PRIMARY KEY,
    dataset_id          TEXT NOT NULL REFERENCES datasets (id) ON DELETE CASCADE,
    method              TEXT NOT NULL,
    k                   INTEGER NOT NULL,
    request             TEXT NOT NULL,
    conditions          TEXT NOT NULL,
    scores              TEXT NOT NULL,
    rows                INTEGER NOT NULL,
    seconds             REAL NOT NULL,
    created_dataset_id  TEXT REFERENCES datasets (id) ON DELETE SET NULL,
    created_at          TEXT NOT NULL
);
CREATE INDEX segment_runs_dataset ON segment_runs (dataset_id, created_at);
