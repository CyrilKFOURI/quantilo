-- Sources (085, docs/features/085-sources/spec.md): a file imported as several tables. One concept
-- for the sections of a workbook (080b, several tables of a sheet) and the linked tables of a JSON,
-- JSON Lines or XML document (083). A plain one-table import has no source: `source_id` stays NULL
-- and the dataset is standalone, as every dataset imported before this migration.
--
-- Additive only (the 0025 study's `0025_sources`, renumbered: 0025 went to projects, 0027 to 088).
-- Deleting a source deletes its tables (CASCADE); their cleaned versions become ordinary datasets
-- (0024). `DatasetOut` is unchanged: the origin of a dataset is read at /datasets/{id}/origin.
CREATE TABLE sources (
    id           TEXT PRIMARY KEY,
    kind         TEXT NOT NULL CHECK (kind IN ('workbook', 'csv', 'json', 'jsonl', 'xml', 'image')),
    name         TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    fingerprint  TEXT NOT NULL,   -- path, size and date of the file when it was imported
    size_bytes   INTEGER NOT NULL,
    meta         TEXT NOT NULL DEFAULT '{}',   -- JSON: what the first bytes said (082), sheets
    plan         TEXT,                         -- JSON: the tables chosen, their options
    created_at   TEXT NOT NULL
);

ALTER TABLE datasets ADD COLUMN source_id TEXT REFERENCES sources (id) ON DELETE CASCADE;
ALTER TABLE datasets ADD COLUMN source_locator TEXT;  -- 'Sheet1!A4:L14' | '$.orders[*].items[*]'
ALTER TABLE datasets ADD COLUMN layout TEXT;          -- JSON: rules, confidence, rows set aside
CREATE INDEX datasets_source ON datasets (source_id);

-- Keys between the tables of a document (R-KEY-03): orders._id = orders.items._parent_id.
CREATE TABLE dataset_links (
    parent_id   TEXT NOT NULL REFERENCES datasets (id) ON DELETE CASCADE,
    child_id    TEXT NOT NULL REFERENCES datasets (id) ON DELETE CASCADE,
    parent_key  TEXT NOT NULL,
    child_key   TEXT NOT NULL,
    rule        TEXT NOT NULL,
    PRIMARY KEY (parent_id, child_id, parent_key)
);
CREATE INDEX dataset_links_child ON dataset_links (child_id);
