-- Predictions of a trained model on new data (feature 037): one per row. The predicted rows and
-- the summary are files in the model's dataset folder (ml/predictions/<id>/part-*.parquet and
-- ml/predictions/<id>.json), deleted with it. The predicted dataset may be deleted on its own:
-- its name stays, its id becomes NULL.
CREATE TABLE ml_predictions (
    id                  TEXT PRIMARY KEY,
    dataset_id          TEXT NOT NULL REFERENCES datasets (id) ON DELETE CASCADE,
    run_id              TEXT NOT NULL REFERENCES ml_runs (id) ON DELETE CASCADE,
    source_id           TEXT REFERENCES datasets (id) ON DELETE SET NULL,
    source_name         TEXT NOT NULL,
    rows                INTEGER NOT NULL,
    less_reliable       INTEGER NOT NULL,
    level               TEXT NOT NULL,  -- ready | check (a blocked check is never predicted)
    path                TEXT NOT NULL,  -- the result folder, relative to the dataset's folder
    created_dataset_id  TEXT REFERENCES datasets (id) ON DELETE SET NULL,
    created_at          TEXT NOT NULL
);
CREATE INDEX ml_predictions_run ON ml_predictions (run_id, created_at);
CREATE INDEX ml_predictions_source ON ml_predictions (source_id);
