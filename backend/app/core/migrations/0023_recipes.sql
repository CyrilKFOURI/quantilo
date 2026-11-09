-- Recipes (030): the work kept on a dataset, replayed on another file. A recipe is versioned:
-- every change adds a row to recipe_versions (never rewritten); a run keeps the version it used.
-- A recipe outlives the dataset it was made on, and its runs outlive their datasets.
CREATE TABLE recipes (
    id                 TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    version            INTEGER NOT NULL,  -- the current version
    source_dataset_id  TEXT REFERENCES datasets (id) ON DELETE SET NULL,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);

CREATE TABLE recipe_versions (
    recipe_id   TEXT NOT NULL REFERENCES recipes (id) ON DELETE CASCADE,
    version     INTEGER NOT NULL,
    body        TEXT NOT NULL,  -- JSON (quantilo.recipes.model.Recipe)
    created_at  TEXT NOT NULL,
    PRIMARY KEY (recipe_id, version)
);

CREATE TABLE recipe_runs (
    id            TEXT PRIMARY KEY,
    recipe_id     TEXT NOT NULL REFERENCES recipes (id) ON DELETE CASCADE,
    version       INTEGER NOT NULL,
    kind          TEXT NOT NULL DEFAULT 'run',  -- run, or baseline: the figures of the dataset
                                                -- the recipe was made on, computed at creation
    dataset_id    TEXT REFERENCES datasets (id) ON DELETE SET NULL,
    dataset_name  TEXT NOT NULL,
    status        TEXT NOT NULL,  -- running, done, warning, failed, cancelled
    report        TEXT NOT NULL,  -- JSON: steps, statuses, messages, links, matching, changes
    figures       TEXT NOT NULL,  -- JSON: the key figures
    created_at    TEXT NOT NULL,
    finished_at   TEXT
);

CREATE INDEX recipe_runs_recipe ON recipe_runs (recipe_id, created_at);
