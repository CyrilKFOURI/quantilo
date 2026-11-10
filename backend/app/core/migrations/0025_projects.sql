-- Projects (087): a folder the user makes to file datasets in. Optional: an item is in zero or
-- one project (R-PRJ-02, the partial unique index; dropping it would allow several). New tables
-- only: `datasets` and `clean_versions` are unchanged, so every existing dataset is "Not in a
-- project". Deleting a project removes its rows here, never a dataset (R-PRJ-03).
--
-- Only roots are filed: a cleaned version (065) follows its original, found through
-- clean_versions.parent_id (R-PRJ-04). Feature 085 (sources: workbooks, JSON documents, images)
-- will add `source_id TEXT REFERENCES sources (id) ON DELETE CASCADE` with its own partial unique
-- index; `item_kind` already accepts 'source', so the table is not rebuilt.
CREATE TABLE projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL COLLATE NOCASE UNIQUE,  -- "Sales" and "sales" clash (R-PRJ-08)
    description TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE project_items (
    project_id  TEXT NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    item_kind   TEXT NOT NULL DEFAULT 'dataset' CHECK (item_kind IN ('dataset', 'source')),
    dataset_id  TEXT REFERENCES datasets (id) ON DELETE CASCADE,
    added_at    TEXT NOT NULL,
    CHECK ((item_kind = 'dataset') = (dataset_id IS NOT NULL))
);
CREATE UNIQUE INDEX project_items_dataset ON project_items (dataset_id)
    WHERE dataset_id IS NOT NULL;
CREATE INDEX project_items_project ON project_items (project_id);

-- R-PRJ-06: deleting an original makes its versions ordinary datasets (0024); they stay in its
-- project. Runs before the row and its clean_versions links are deleted.
CREATE TRIGGER project_items_keep_versions BEFORE DELETE ON datasets
BEGIN
    INSERT OR IGNORE INTO project_items (project_id, item_kind, dataset_id, added_at)
    SELECT item.project_id, 'dataset', version.dataset_id,
           strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    FROM clean_versions AS version
    JOIN project_items AS item ON item.dataset_id = old.id
    WHERE version.parent_id = old.id;
END;

-- R-PRJ-07: a dataset that becomes a cleaned version follows its original from now on.
CREATE TRIGGER project_items_versions_follow AFTER INSERT ON clean_versions
BEGIN
    DELETE FROM project_items WHERE dataset_id = new.dataset_id;
END;
