"""Distinctive words and topics of a dataset's text columns (059), read through the catalog.

Compare reads its words through `group_words` (one text column against a category); Segments
through `TextService.words_by_rows` (a segment per row); Changes through `words_between` (two
versions); the variable page through `TextService.topics` (Advanced › More, R-TOP-01).
"""

# ruff: noqa: RUF002  (the docstring names the "Advanced, More" layer as the app shows it)
from collections import Counter, OrderedDict
from collections.abc import Sequence
from typing import Any

import duckdb
import pandas as pd

from quantilo.catalog import Catalog
from quantilo.errors import ExplorerError, NotFoundError
from quantilo.filters.model import Condition, filters_key
from quantilo.filters.scope import scoped_source, where_for
from quantilo.ingestion.sql import lit
from quantilo.text import distinctive as dw
from quantilo.text import topics as tp
from quantilo.text.tokens import language_of, stop_words
from quantilo.typing.derived import kept_topics, with_derived
from quantilo.typing.parsers import typed_sql
from quantilo.variable.source import SEED

CACHE_SIZE = 16
TOPIC_ROLES = (None, "free_text")  # R-TOP-01


class TextError(ExplorerError):
    def __init__(self, message: str, code: str = "text", status_code: int = 422) -> None:
        super().__init__(message, code=code, status_code=status_code)


def is_words_column(column: Any) -> bool:
    """A text whose words can be read (R-WRD-06): not a list of choices."""
    return column.type == "text" and column.role != "choices"


def text_sql(column: Any) -> str:
    return f"CAST({typed_sql(column.name, column.parser)} AS VARCHAR)"


def sample(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    text: str,
    group: str | None = None,
    limit: int = dw.MAX_TEXTS,
) -> tuple[list[str], list[str], int]:
    """Up to `limit` non-empty texts (and their group), drawn by a hash of the row number
    (R-WRD-05, R-TOP-02: reproducible), and how many there were. `source` keeps row numbers."""
    label = group or "''"
    relation = (
        f"(SELECT {text} AS t, {label} AS g, file_row_number AS rn FROM {source}) "
        "WHERE t IS NOT NULL AND trim(t) <> ''" + (" AND g IS NOT NULL" if group else "")
    )
    total = connection.execute(f"SELECT count(*) FROM {relation}").fetchone()
    rows = connection.execute(
        f"SELECT t, g FROM {relation} ORDER BY hash(rn + {SEED}) LIMIT {limit}"
    ).fetchall()
    return [str(t) for t, _ in rows], [str(g) for _, g in rows], int(total[0] if total else 0)


def group_words(
    column: str,
    by: str,
    texts: list[str],
    groups: list[str],
    total: int,
    stop: frozenset[str],
    order: Sequence[str] | None = None,
) -> dw.DistinctiveWords:
    """R-WRD-02 to 05 over sampled texts: the 12 largest groups, in `order` (else largest
    first); the smaller groups are left out, and say so."""
    sizes = Counter(groups)
    largest = [g for g, _ in sizes.most_common()][: dw.MAX_GROUPS]
    kept = set(largest)
    if order:
        rank = {label: i for i, label in enumerate(order)}
        largest.sort(key=lambda g: (rank.get(g, len(rank)), -sizes[g]))
    pairs = [(t, g) for t, g in zip(texts, groups, strict=True) if g in kept]
    found = dw.distinctive([t for t, _ in pairs], [g for _, g in pairs], stop)
    by_group = {g.group: g for g in found}
    shown = [by_group[g] for g in largest if g in by_group]
    rules = list(dw.RULES)
    if len(sizes) > dw.MAX_GROUPS:
        rules.append(
            f"R-WRD-05: the {dw.MAX_GROUPS} largest groups of {len(sizes)}; "
            "the others are compared but not shown"
        )
    return dw.DistinctiveWords(
        column,
        by,
        len(texts),
        total > len(texts),
        shown,
        max(0, len(sizes) - dw.MAX_GROUPS),
        dw.sentence(column, by, shown),
        rules,
    )


def words_between(
    column: str,
    before: tuple[str, Any],
    after: tuple[str, Any],
    labels: tuple[str, str] = ("previous version", "this version"),
) -> dw.DistinctiveWords:
    """Changes (R-WRD-07): the words that rise in each version against the other. `before` and
    `after` are (storage path, column view)."""
    connection = duckdb.connect()
    try:
        texts: list[str] = []
        groups: list[str] = []
        total = 0
        for (storage, view), label in zip((before, after), labels, strict=True):
            if not is_words_column(view):
                raise TextError(f"{column!r} is not free text in the {label}.")
            source = f"read_parquet({lit(storage)}, file_row_number = true)"
            found, _, count = sample(connection, source, text_sql(view), None, dw.MAX_TEXTS // 2)
            texts.extend(found)
            groups.extend([label] * len(found))
            total += count
    finally:
        connection.close()
    stop = stop_words(language_of(after[1]))
    return group_words(column, "version", texts, groups, total, stop, list(labels))


class TextService:
    def __init__(self, columns: Catalog) -> None:
        self.columns = columns
        self._cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()

    def _view(self, dataset_id: str, column: str) -> tuple[Any, list[Any]]:
        views = with_derived(self.columns.columns(dataset_id), [column])
        view = next((v for v in views if v.name == column), None)
        if view is None or view.excluded:
            raise NotFoundError(f"Column {column!r}")
        return view, views

    def _remember(self, key: tuple[Any, ...], value: Any) -> Any:
        self._cache[key] = value
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return value

    def words_by_rows(
        self,
        dataset_id: str,
        column: str,
        rows: Sequence[int],
        groups: Sequence[str],
        by: str = "segment",
        order: Sequence[str] | None = None,
    ) -> dw.DistinctiveWords:
        """Segments (R-WRD-07): each group of rows (file row numbers) against the others."""
        dataset = self.columns.ready(dataset_id)
        view, _views = self._view(dataset_id, column)
        if not is_words_column(view):
            raise TextError(f"{column!r} is not free text: no words to read.")
        labels = pd.DataFrame({"row": list(rows), "grp": list(groups)})
        connection = duckdb.connect()
        try:
            connection.register("row_groups", labels)
            source = (
                f"(SELECT p.*, l.grp FROM read_parquet({lit(dataset.storage_path)}, "
                "file_row_number = true) p JOIN row_groups l ON p.file_row_number = l.row)"
            )
            texts, found, total = sample(connection, source, text_sql(view), "grp")
        finally:
            connection.close()
        stop = stop_words(language_of(view))
        return group_words(column, by, texts, found, total, stop, order)

    def topics(
        self,
        dataset_id: str,
        column: str,
        k: int | None = None,
        conditions: list[Condition] | None = None,
    ) -> tp.Topics:
        """R-TOP-01 to 05: topics of a free text column, on at most 20,000 texts."""
        conditions = conditions or []
        dataset = self.columns.ready(dataset_id)
        view, views = self._view(dataset_id, column)
        if view.type != "text" or view.role not in TOPIC_ROLES:
            raise TextError(f"{column!r} is not free text: topics need sentences (R-TOP-01).")
        key = (
            "topics",
            dataset_id,
            dataset.imported_at,
            dataset.types_version,
            column,
            k,
            filters_key(conditions),
        )
        if key in self._cache:
            self._cache.move_to_end(key)
            found = self._cache[key]
        else:
            connection = duckdb.connect()
            try:
                source = scoped_source(
                    dataset.storage_path, where_for(views, conditions), numbered=True
                )
                texts, _, total = sample(connection, source, text_sql(view), None, tp.MAX_TEXTS)
            finally:
                connection.close()
            stop = stop_words(language_of(view))
            found = self._remember(key, tp.fit_topics(column, texts, stop, k, total > len(texts)))
        kept = [rule.model_dump() for rule in kept_topics(view)]
        found.kept = bool(kept) and kept == [rule.model_dump() for rule in found.rules]
        return found


def topics_summary(topics: tp.Topics) -> dict[str, Any]:
    """What Keep as column stores in the column's summary (R-TOP-06)."""
    return {
        "rules": [rule.model_dump() for rule in topics.rules],
        "k": topics.k,
        "texts": topics.texts,
        "agreement": topics.agreement,
    }
