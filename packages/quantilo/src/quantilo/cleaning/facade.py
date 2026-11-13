"""`ds.clean()`: the same suggestions, preview and cleaned version as the app's Clean page."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import duckdb
from pydantic import TypeAdapter

from quantilo.catalog import ColumnChange
from quantilo.cleaning import checks as ck
from quantilo.cleaning import dismiss as dm
from quantilo.cleaning import model as m
from quantilo.cleaning.apply import apply_steps
from quantilo.cleaning.compile import compile_steps
from quantilo.cleaning.preview import CleanPreview, preview
from quantilo.cleaning.replay import CleanRecipe, ReplayReport, parse_recipe, recipe_of, replay
from quantilo.cleaning.report import CleanReport, TypeChange
from quantilo.cleaning.suggest import default_steps, suggest, suggest_advanced, with_suggestion
from quantilo.cleaning.suggestion import CleanSuggestion
from quantilo.typing.engine import DEFAULT_THRESHOLDS

if TYPE_CHECKING:
    from quantilo.dataset import Dataset

STEP = TypeAdapter(m.Step)
# steps that change a column's storage in place: the user's type of the parent no longer applies
CONVERTING = ("change_type", "pad_zeros", "transform_number", "format_text", "extract_text")


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
    converted = {
        s.column
        for s in steps
        if s.enabled and s.op in CONVERTING and not (s.op == "extract_text" and s.to)
    }
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
    """Basic suggestions computed on opening; `advanced()` adds the Advanced ones on demand.
    `steps` starts with the automatic ones. `use()` / `skip()` a suggestion, `add()` a step,
    `preview()`, then `apply()` for the cleaned dataset."""

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
        # C-57, C-58: the other datasets the steps append or merge, by the name the steps use
        self.sources: dict[str, str] = {}
        self._advanced: list[CleanSuggestion] | None = None
        # N05: suggestions not to propose again (the recipe carries them; `hidden()` lists them)
        self.dismissed: list[dm.Dismissal] = []
        # N31: the checks declared on this dataset (the recipe carries them); `verify()` runs them
        self.checks: list[ck.CleanCheck] = []

    def advanced(self) -> list[CleanSuggestion]:
        """The Advanced suggestions (097), computed on the first call only (C-16) and added to
        `suggestions`; none is ticked (mode S or I, C-17): `use()` one to add its step."""
        if self._advanced is None:
            basic = [s for s in self.suggestions if s.layer == "basic"]
            connection = duckdb.connect()
            try:
                self._advanced = suggest_advanced(
                    connection, str(self.dataset.path), self.dataset.types, basic
                )
            finally:
                connection.close()
            self.suggestions = [*self.suggestions, *self._advanced]
        return self._advanced

    def methods(self) -> list[Any]:
        """101: the *More checks* (Expert layer) and whether each applies here; nothing is run."""
        from quantilo.cleaning import expert

        return expert.methods(self.dataset.types)

    def expert(self, method: str, **options: Any) -> Any:
        """101: run one Expert check (`unusual_rows`, `suspicious_labels` with `column=`,
        `fill_empty` with `imputer="knn"|"iterative"` and `columns=`, `cell_errors` with
        `column=` then `marks=[{"row": 12, "error": True}, …]`). Its suggestions (mode S, never
        ticked, C-30) replace those of an earlier run of the same check: `use()` one to add it."""
        from quantilo.cleaning import expert

        chosen = expert.ExpertOptions.model_validate(options)
        connection = duckdb.connect()
        try:
            result = expert.run(
                connection,
                str(self.dataset.path),
                self.dataset.types,
                cast(expert.MethodId, method),
                chosen,
            )
        finally:
            connection.close()
        rule = expert.RULES[result.method]
        kept = [s for s in self.suggestions if s.rule != rule]
        self.suggestions = [*kept, *result.suggestions]
        return result

    def dismiss(
        self, suggestion_id: str, scope: dm.Scope = "suggestion", note: str | None = None
    ) -> Cleaning:
        """N05: don't suggest this again: the suggestion (`scope="suggestion"`), its rule on every
        column (`"rule"`), on columns of this name (`"column"`) or everywhere (`"everywhere"`).
        The steps of what it hides leave `steps`; `restore()` takes it back."""
        found = self.suggestion(suggestion_id)
        dismissal = dm.Dismissal(
            scope=scope,
            rule=found.rule,
            column=found.columns[0] if found.columns else None,
            suggestion=found.id,
            title=found.title,
            note=note,
        )
        if dismissal.key() not in {d.key() for d in self.dismissed}:
            self.dismissed.append(dismissal)
        gone = {s.id for s, _ in self.hidden()}
        self.steps = [s for s in self.steps if s.suggestion not in gone]
        return self

    def restore(self, suggestion_id: str) -> Cleaning:
        """N05: show again what hides this suggestion (its step is not ticked again)."""
        found = self.suggestion(suggestion_id)
        self.dismissed = [d for d in self.dismissed if not dm.matches(d, found)]
        return self

    def hidden(self) -> list[tuple[CleanSuggestion, dm.Dismissal]]:
        """N05: the suggestions dismissed, each with what hides it."""
        _, hidden = dm.split(self.suggestions, self.dismissed)
        return [(s, self.dismissed[index]) for s, index in hidden]

    def check(self, kind: ck.Kind, *columns: str, **fields: Any) -> Cleaning:
        """N31: declare a check (`check("unique", "order_id")`, `check("compare", "start", "end",
        op="le")`, `check("in_list", "status", values=[...])`); nothing is fixed."""
        number = max((int(c.id[1:]) for c in self.checks), default=0) + 1
        found = ck.CleanCheck.model_validate(
            {"id": f"c{number}", "kind": kind, "columns": list(columns), **fields}
        )
        self.checks = [*self.checks, found]
        return self

    def verify(self) -> ck.CheckReport:
        """N31: every check on this dataset: what passes, what fails (rows), what cannot run."""
        connection = duckdb.connect()
        try:
            return ck.run_checks(
                connection, str(self.dataset.path), self.dataset.types, self.checks
            )
        finally:
            connection.close()

    def recipe(self, name: str | None = None) -> CleanRecipe:
        """101 (C-39): these steps as a cleaning recipe, to replay on another file
        (`other.clean().replay(recipe)`); `recipe.to_json()` writes it."""
        connection = duckdb.connect()
        try:
            found = recipe_of(
                connection,
                str(self.dataset.path),
                self.dataset.types,
                self.steps,
                name or f"{self.dataset.name} · cleaning",
                self.dataset.name,
            )
        finally:
            connection.close()
        here = [d for d in self.dismissed if d.scope in dm.HERE]
        schema = 2 if here or self.checks else 1
        return found.model_copy(
            update={"dismissed": here, "checks": list(self.checks), "schema_version": schema}
        )

    def replay(
        self, recipe: CleanRecipe | str | dict[str, Any], accept_proposals: bool = False
    ) -> ReplayReport:
        """101 (M5): what a cleaning made elsewhere does on this dataset, step by step; the steps
        that still apply become `steps` (those that no longer apply are kept, turned off)."""
        found = recipe if isinstance(recipe, CleanRecipe) else parse_recipe(recipe)
        connection = duckdb.connect()
        try:
            report = replay(
                connection,
                str(self.dataset.path),
                self.dataset.types,
                found,
                accept_proposals,
                dismissed=self.dismissed,
            )
        finally:
            connection.close()
        self.steps = list(report.adapted)
        known = {d.key() for d in self.dismissed}
        self.dismissed += [d for d in report.dismissed if d.key() not in known]
        if report.checks and not self.checks:
            self.checks = list(report.checks)
        return report

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

    def attach(self, name: str, dataset: Dataset) -> Cleaning:
        """Another dataset the steps can append or merge (`dataset=name` in the step)."""
        self.sources[name] = str(dataset.unfiltered().path)
        return self

    def preview(self, step_id: str | None = None) -> CleanPreview:
        connection = duckdb.connect()
        try:
            compiled = compile_steps(
                connection, str(self.dataset.path), self.steps, sources=self.sources
            )
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
            report = apply_steps(
                connection, str(parent.path), self.steps, target, rowmap, sources=self.sources
            )
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
        self.rowmap = rowmap if rowmap.is_file() else None  # C-59: none after a reshape
        return child


def _step(raw: Any) -> Any:
    if isinstance(raw, dict):
        try:
            return STEP.validate_python(raw)
        except Exception as error:  # pydantic's ValidationError, said plainly
            raise m.CleanError(f"Invalid step: {error}") from None
    return raw.model_copy()
