-- Pinned rows (099): rows a user pinned with a label and a colour. The key is the identifier
-- ("order_id=O00042") when the dataset has one, else a fingerprint of the row's stored values
-- ("row:<md5>"), as for the unusual-row marks (0016); `row` is the row's number in the file
-- (from 0) when it was last found (R-PIN-01, R-PIN-02). All the rows of a label share its colour,
-- a slot of the theme's categorical palette (1 to 8, R-PIN-03). New table only.
CREATE TABLE pinned_rows (
    dataset_id  TEXT NOT NULL REFERENCES datasets (id) ON DELETE CASCADE,
    row_key     TEXT NOT NULL,
    row_number  INTEGER NOT NULL,
    label       TEXT NOT NULL,
    color       INTEGER NOT NULL CHECK (color BETWEEN 1 AND 8),
    created_at  TEXT NOT NULL,
    PRIMARY KEY (dataset_id, row_key)
);
CREATE INDEX pinned_rows_label ON pinned_rows (dataset_id, label);
