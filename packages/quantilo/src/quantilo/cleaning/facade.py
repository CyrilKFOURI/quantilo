"""`ds.clean()`: the same suggestions, preview and cleaned version as the app's Clean page."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
from pydantic import TypeAdapter

from quantilo.catalog import ColumnChange
from quantilo.cleaning import model as m
from quantilo.cleaning.apply import apply_steps
from quantilo.cleaning.compile import compile_steps
from quantilo.cleaning.preview import CleanPreview, preview
from quantilo.cleaning.report import CleanReport, TypeChange
from quantilo.cleaning.suggest import default_steps, suggest, with_suggestion
from quantilo.cleaning.suggestion import CleanSuggestion
from quantilo.typing.engine import DEFAULT_THRESHOLDS

if TYPE_CHECKING:
    from quantilo.dataset import Dataset

STEP = TypeAdapter(m.Step)
CONVERTING = ("change_type", "pad_zeros")


def renamed(steps: list[Any]) -> dict[str, str]:
    """Parent column → its name at the end (renames followed)."""
    names: dict[str, str] = {}
    for step in steps:
        if step.enabled and step.op == "rename_column":
            for parent, current in list(names.items()):
                if current == step.column:
                    names[parent] = step.to
            if step.column not in names.values():
                names.setdefault(step.column, step.to)
    return names


def corrections(steps: list[Any], records: list[Any]) -> list[ColumnChange]:
    """The user's corrections of the parent, for the child (C-09): type, role, order, exclusion;
    not the type of a column a step converted (the detection must see its new storage)."""
    names = renamed(steps)
    converted = {s.column for s in steps if s.enabled and s.op in CONVERTING}
    changes = []
    for record in records:
        fields: dict[str, Any] = {}
        if record.user_type is not None and record.name not in converted:
            fields["type"] = record.user_type
        if record.user_role is not None:
            fields["role"] = record.user_role
        if record.user_order is not None:
            fields["order"] = json.loads(record.user_order)
        if record.excluded:
            fields["excluded"] = True
        if fields:
            name: str = names.get(record.name) or record.name
            changes.append(ColumnChange(name=name, **fields))
    return changes


def type_changes(before: list[Any], after: list[Any], steps: list[Any]) -> list[TypeChange]:
    names = renamed(steps)
    old = {names.get(v.name, v.name): v.type for v in before}
    return [
        TypeChange(column=str(v.name), before=str(old[v.name]), after=str(v.type))
        for v in after
        if v.name in old and old[v.name] != v.type
    ]


class Cleaning:
    """Suggestions computed on opening; `steps` starts with the automatic ones. `use()` /
    `skip()` a suggestion, `add()` a step, `preview()`, then `apply()` for the cleaned dataset."""

    def __init__(self, dataset: Dataset, steps: list[Any] | None = None) -> None:
        self.dataset = dataset.unfiltered()
        connection = duckdb.connect()
        try:
            self.suggestions: list[CleanSuggestion] = suggest(
                connection, str(self.dataset.path), self.dataset.types
            )
        finally:
            connection.close()
        self.steps: list[Any] = (
            default_steps(self.suggestions) if steps is None else [_step(s) for s in steps]
        )
        self.report: CleanReport | None = None
        self.rowmap: Path | None = None  # child row → parent row, after apply()

    def __repr__(self) -> str:
        return (
            f"<Cleaning {self.dataset.name!r}: {len(self.suggestions)} suggestions, "
            f"{len(self.steps)} steps>"
        )

    def suggestion(self, suggestion_id: str) -> CleanSuggestion:
        for found in self.suggestions:
            if found.id == suggestion_id:
                return found
        raise m.CleanError(f"No suggestion {suggestion_id!r}.", status_code=404)

    def use(self, *suggestion_ids: str) -> Cleaning:
        for suggestion_id in suggestion_ids:
            self.steps = with_suggestion(
                self.steps, self.suggestion(suggestion_id), self.suggestions
            )
        return self

    def skip(self, *suggestion_ids: str) -> Cleaning:
        self.steps = [s for s in self.steps if s.suggestion not in suggestion_ids]
        return self

    def add(self, step: Any = None, **fields: Any) -> Cleaning:
        """A step: a model, a dict or fields (`add(op="rename_column", column="a", to="b")`)."""
        raw = step if step is not None else fields
        if isinstance(raw, dict):
            raw = {"id": m.next_id(self.steps), **raw}
        self.steps = [*self.steps, _step(raw)]
        m.Steps(steps=self.steps)  # two steps with one id fail here
        return self

    def to_json(self) -> str:
        return m.Steps(steps=self.steps).to_json()

    def preview(self, step_id: str | None = None) -> CleanPreview:
        connection = duckdb.connect()
        try:
            compiled = compile_steps(connection, str(self.dataset.path), self.steps)
            return preview(connection, compiled, step_id)
        finally:
            connection.close()

    def apply(self, name: str | None = None, cache_dir: str | Path | None = None) -> Dataset:
        """The cleaned dataset (a new one: this one is not changed). Its columns are detected
        again; your corrections of this one's columns are kept (C-09). `self.report` says what
        changed."""
        from quantilo.dataset import _load, default_cache_dir

        parent = self.dataset
        body = m.Steps(steps=self.steps).to_json()
        key = hashlib.sha256(f"{parent.path}|{body}".encode()).hexdigest()[:16]
        folder = Path(cache_dir) if cache_dir else Path(parent.path).parent
        if cache_dir is None and not folder.is_dir():
            folder = default_cache_dir()
        target = folder / f"{Path(parent.path).stem}-clean-{key}.parquet"
        rowmap = target.with_suffix(".rowmap.parquet")
        connection = duckdb.connect()
        try:
            report = apply_steps(connection, str(parent.path), self.steps, target, rowmap)
        finally:
            connection.close()
        child = _load(
            target,
            name or f"{parent.name} · cleaned",
            parent._state.info.source_path,
            DEFAULT_THRESHOLDS,
        )
        for change in corrections(self.steps, list(parent._state.records.values())):
            if change.name in child.columns:
                child._change(change)
        report.type_changes = type_changes(parent.types, child.types, self.steps)
        self.report = report
        self.rowmap = rowmap
        return child


def _step(raw: Any) -> Any:
    if isinstance(raw, dict):
        try:
            return STEP.validate_python(raw)
        except Exception as error:  # pydantic's ValidationError, said plainly
            raise m.CleanError(f"Invalid step: {error}") from None
    return raw.model_copy()
