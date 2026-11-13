-- C1 (docs/features/c1-nettoyage-import-projets/spec.md, part 2): projects and their model.
--
-- 1. R-PRJ-02 lifted: an item may be in several projects (087 said dropping the index was enough).
--    The unique index now holds per project: a dataset is listed once in each project.
-- 2. 088: a relation can be inactive (kept, drawn, not used by a build: R-REL-13) and compare its
--    keys with case, accents and punctuation ignored (R-REL-14).
-- 3. 089 and 088: a join of two datasets and a generated date table keep their lineage in
--    model_builds (kinds 'join' and 'dates'), to say when they are out of date and rebuild them.
--    SQLite cannot change a CHECK: the table is copied into a new one, its rows kept.
DROP INDEX project_items_dataset;
CREATE UNIQUE INDEX project_items_dataset ON project_items (project_id, dataset_id)
    WHERE dataset_id IS NOT NULL;
CREATE INDEX project_items_by_dataset ON project_items (dataset_id);

ALTER TABLE dataset_relations ADD COLUMN active INTEGER NOT NULL DEFAULT 1;
ALTER TABLE dataset_relations ADD COLUMN match TEXT NOT NULL DEFAULT 'exact'
    CHECK (match IN ('exact', 'normalized'));

CREATE TABLE model_builds_c1 (
    dataset_id  TEXT PRIMARY KEY REFERENCES datasets (id) ON DELETE CASCADE,
    project_id  TEXT REFERENCES projects (id) ON DELETE SET NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('flat', 'aggregate', 'join', 'dates')),
    spec        TEXT NOT NULL,
    sources     TEXT NOT NULL,
    report      TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
INSERT INTO model_builds_c1 (dataset_id, project_id, kind, spec, sources, report, created_at,
                             updated_at)
SELECT dataset_id, project_id, kind, spec, sources, report, created_at, updated_at
FROM model_builds;
DROP TABLE model_builds;
ALTER TABLE model_builds_c1 RENAME TO model_builds;
CREATE INDEX model_builds_project ON model_builds (project_id);
