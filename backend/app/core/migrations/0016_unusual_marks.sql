-- Unusual rows (feature 033): what the user said of a row. The key is the identifier
-- ("customer_id=C00042") when the dataset has one, else a fingerprint of the row's stored values
-- ("row:<md5>"), so a mark follows the row, not its position. "expected" rows leave the next list.
CREATE TABLE unusual_marks (
    dataset_id  TEXT NOT NULL REFERENCES datasets (id) ON DELETE CASCADE,
    row_key     TEXT NOT NULL,
    mark        TEXT NOT NULL CHECK (mark IN ('expected', 'error')),
    created_at  TEXT NOT NULL,
    PRIMARY KEY (dataset_id, row_key)
);
