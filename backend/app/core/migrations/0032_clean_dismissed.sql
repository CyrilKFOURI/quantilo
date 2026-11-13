-- N05 (docs/etudes/2026-11-12-audit-nettoyage.md): "Don't suggest again". A dismissal hides the
-- cleaning suggestions it matches (quantilo.cleaning.dismiss): `suggestion` (one suggestion) and
-- `rule` (a rule on every column) belong to a dataset, the original of a family (`dataset_id`,
-- CASCADE: they go with it); `column` (a rule on the columns of a name) and `everywhere` (a rule)
-- reach every dataset (`dataset_id` NULL). Hidden suggestions are never applied for the user and
-- stay listed under Dismissed until restored (the row is then deleted). New table only.
CREATE TABLE clean_dismissed (
    id           TEXT PRIMARY KEY,
    dataset_id   TEXT REFERENCES datasets (id) ON DELETE CASCADE,
    scope        TEXT NOT NULL CHECK (scope IN ('suggestion', 'rule', 'column', 'everywhere')),
    rule         TEXT NOT NULL,
    column_name  TEXT,
    suggestion   TEXT,
    title        TEXT,  -- what it hid, as written when dismissed
    note         TEXT,  -- the user's own reason, if any
    created_at   TEXT NOT NULL
);
CREATE INDEX clean_dismissed_dataset ON clean_dismissed (dataset_id);
