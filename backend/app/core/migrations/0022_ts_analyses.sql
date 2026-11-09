-- Time series analyses kept with "Save analysis" (030): the page's configuration as JSON
-- (time, value, group, period, aggregate, gaps, season, forecast models and horizon). `origin`
-- names the recipe that added it, NULL when the user saved it. They go with the dataset.
CREATE TABLE ts_analyses (
    id          TEXT PRIMARY KEY,
    dataset_id  TEXT NOT NULL REFERENCES datasets (id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    config      TEXT NOT NULL,
    origin      TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX ts_analyses_dataset ON ts_analyses (dataset_id, created_at);
