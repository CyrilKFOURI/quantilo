-- The model of a project (088, docs/features/088-modele-projet/spec.md): relations between its
-- datasets, the layout of the Model view, and the lineage of the analysis datasets built from it.
-- New tables only. 0026 is kept for 085 (sources, decision 3 of 087): 088 takes 0027.
--
-- Relations are filed by family root (the original of a cleaned version, 065), like projects
-- (R-PRJ-04): switching a card to a cleaned version keeps them. Only the user's decisions are
-- stored (accepted or rejected); proposals are computed again by the detection (decision 3).
CREATE TABLE dataset_relations (
    id             TEXT PRIMARY KEY,
    project_id     TEXT NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    child_root     TEXT NOT NULL REFERENCES datasets (id) ON DELETE CASCADE,   -- many side
    child_columns  TEXT NOT NULL,   -- JSON list, 1 or 2 names
    parent_root    TEXT NOT NULL REFERENCES datasets (id) ON DELETE CASCADE,  -- one side (key)
    parent_columns TEXT NOT NULL,   -- JSON list, same length
    join_type      TEXT NOT NULL DEFAULT 'left'
                   CHECK (join_type IN ('left', 'inner', 'right', 'full')),
    status         TEXT NOT NULL CHECK (status IN ('accepted', 'rejected')),
    origin         TEXT NOT NULL CHECK (origin IN ('detected', 'user')),
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
CREATE UNIQUE INDEX dataset_relations_pair
    ON dataset_relations (project_id, child_root, child_columns, parent_root, parent_columns);
CREATE INDEX dataset_relations_project ON dataset_relations (project_id);

-- One row per project: where the cards are and which version of each family the model uses.
CREATE TABLE project_models (
    project_id  TEXT PRIMARY KEY REFERENCES projects (id) ON DELETE CASCADE,
    positions   TEXT NOT NULL DEFAULT '{}',   -- {root id: {"x": 0, "y": 0}}
    versions    TEXT NOT NULL DEFAULT '{}',   -- {root id: dataset id used}; absent = original
    updated_at  TEXT NOT NULL
);

-- An analysis dataset (R-BLD-04): what it was built from, to rebuild it and to say when it is out
-- of date (R-BLD-05). Deleting the dataset deletes its lineage; deleting the project keeps both.
CREATE TABLE model_builds (
    dataset_id  TEXT PRIMARY KEY REFERENCES datasets (id) ON DELETE CASCADE,
    project_id  TEXT REFERENCES projects (id) ON DELETE SET NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('flat', 'aggregate')),
    spec        TEXT NOT NULL,   -- JSON: fact or parent/child, relations, picked columns, measures
    sources     TEXT NOT NULL,   -- JSON: [{id, name, imported_at, types_version}]
    report      TEXT NOT NULL DEFAULT '{}',   -- JSON: rows before/after, per relation counts
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX model_builds_project ON model_builds (project_id);
