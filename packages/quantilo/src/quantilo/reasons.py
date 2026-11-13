"""Reasons the interface can write in its own language (M02).

A rule's reason stays the English sentence it has always been: `say` writes it from an English
template and keeps, beside it, the template's `code` and the `vars` it cites (the numbers and names,
as strings). `Reason` is that sentence as a `str`, so everything that reads reasons as text goes on
as before; the API sends `{text, code, vars}` next to the text, and the interface writes
`reason.<code>` with the vars when it has that text, the English otherwise.

`code` is the rule's code (`R-TYP-14`, `B-16`), with a suffix when a rule says several sentences
(`R-TYP-05.length`); a reason no rule code names has a stable id of its own (`TYP.distinct`).
Sentences said one after the other (`say(...) + say(...)`) join their codes with `+`: the
interface writes each part and puts them end to end, as the English does.
Templates are string literals so that `app.engine_texts_dump` can list them for the interface,
which checks its English copy against them.
"""

import string
from typing import Any

from pydantic import BaseModel, Field


class Reason(str):
    """An English reason that knows its template's code and the values it cites."""

    code: str
    vars: dict[str, str]

    def __new__(cls, text: str, code: str = "", vars: dict[str, str] | None = None) -> "Reason":
        made = super().__new__(cls, text)
        made.code = code
        made.vars = dict(vars or {})
        return made

    def __add__(self, other: str) -> str:
        """Two reasons said one after the other keep both codes (`B-04+B-04.check`), the
        interface writes each part; a plain text added loses them (English only). A var the new
        part cites with another value than an earlier part is kept as `name@index` (the part's
        position, from 0), which that part reads first."""
        joined = str.__add__(self, other)
        if not (isinstance(other, Reason) and self.code and other.code) or "+" in other.code:
            return joined
        index = self.code.count("+") + 1
        merged = dict(self.vars)
        for name, value in other.vars.items():
            merged[name if merged.get(name, value) == value else f"{name}@{index}"] = value
        return Reason(joined, f"{self.code}+{other.code}", merged)


def placeholders(template: str) -> list[str]:
    """The `{name}` fields of a template, in order."""
    return [name for _, name, _, _ in string.Formatter().parse(template) if name]


def say(code: str, template: str, **values: Any) -> Reason:
    """The reason `template` says with `values`: each `{name}` replaced by `str(values[name])`.
    A value the template does not cite, or a field without a value, is an error: the interface
    translates with exactly these vars."""
    names = placeholders(template)
    if set(names) != set(values):
        raise ValueError(f"{code}: the template cites {sorted(set(names))}, given {sorted(values)}")
    cited = {name: str(value) for name, value in values.items()}
    return Reason(template.format(**cited), code, cited)


def say_count(code: str, count: int, one: str, many: str, **values: Any) -> Reason:
    """`say` with `{count}` (written 1,234): the singular template when it is 1. The interface
    has the two forms too, `reason.<code>#one` and `reason.<code>`."""
    return say(code, one if count == 1 else many, count=f"{count:,}", **values)


class ReasonText(BaseModel):
    """A reason as the API sends it: the English text, its code and vars (none: English only)."""

    text: str
    code: str | None = None
    vars: dict[str, str] = Field(default_factory=dict)


def detail(text: str) -> ReasonText:
    if isinstance(text, Reason) and text.code:
        return ReasonText(text=str(text), code=text.code, vars=dict(text.vars))
    return ReasonText(text=str(text))


def details(texts: list[str]) -> list[ReasonText]:
    return [detail(text) for text in texts]


def stored(texts: list[str]) -> list[Any]:
    """Reasons as JSON keeps them: a plain text stays a string, a coded one is an object."""
    return [
        {"text": str(t), "code": t.code, "vars": t.vars} if isinstance(t, Reason) and t.code
        else str(t)
        for t in texts
    ]  # fmt: skip


def restored(items: list[Any]) -> list[str]:
    """`stored` read back (older records hold strings only)."""
    return [
        Reason(item["text"], item.get("code") or "", item.get("vars"))
        if isinstance(item, dict)
        else item
        for item in items
    ]
