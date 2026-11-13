# First look at a dataset

Work in this order; each step decides what the next one is worth.

1. **Shape and quality** (`get_dataset_overview`). Rows, columns, missing values, duplicates, the app's alerts. A column more than ~40 % empty, a constant, or an identifier is not analysed like the others: say so and set it aside.
   When the data looks dirty (extra spaces, case variants, 'N/A' words, numbers or dates stored as text, -999 codes, duplicate rows), call `cleaning_suggestions`: each fix comes with its reason and figures. Point the user to the Clean page; save a cleaned version (`save_cleaned_version`) only if they ask. The original is never changed.
   When the dataset comes from a laid-out workbook or a JSON / XML document, or the user doubts the import (an odd header, rows missing, a total among the data), `describe_layout` says how the file was read: the table's range, header, rows set aside and the rules with their confidence. Quote it before suspecting the data itself.
2. **Types** (`list_columns`). Check that each column's type makes sense (a postcode is a category, not a number; a 0/1 column is a boolean). If one looks wrong, tell the user to fix it on the Columns page: every analysis depends on it.
3. **Distributions** (`profile_column`) for the few columns that matter to the question, not all of them. Note skew, outliers, impossible values (negative ages, future dates), heavy concentration on one category. For amounts, costs or delays with a heavy right tail, quote the `tail` block ("1 in 1,000 rows exceed X", the average of the worst 5 %) rather than the mean; when `robust` is present, prefer the median or the Huber center to the mean.
4. **Relationships** (`get_correlations`, then `compare_columns` on the strongest pairs). Read effect sizes first; with many rows everything is "significant".
   When many numeric columns are strongly linked (the page proposes its Structure section), `get_structure` says how many real dimensions they carry and which columns form each; quote its sentence. Ask for `factors=true` only for a question about latent factors.
5. **Target**, if there is one (the column with the target role, or the one the user cares about): `rank_associations_with` it, then compare it with the two or three strongest.

Answer with three to five findings, most useful first, each with its figure. End with the one or two analyses worth doing next (often: fix a type, handle missing values, train a model on the target).

Do not profile every column one by one: it costs time and tokens and buries the findings.
