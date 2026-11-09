"""The contract of a tool the AI agent can call (spec 026, § 5.1).

It lives in `core` so that any module can offer tools to the agent (the agent never imports a
module, and a module never imports the agent): a module lists its tools through
`Module.agent_tools` in `modules/registry.py`, the agent collects them at start-up.

A tool wraps one capability of the app: typed input (a Pydantic model, so the model's arguments are
validated before anything runs), a compact result for the model, a one-line summary and a link
that reopens the same analysis in the app. Its `kind` decides what the agent may do on its own."""

import dataclasses
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ValidationError
from quantilo.filters.model import Condition

ToolKind = Literal["read", "compute", "write", "rows"]
# read: no effect · compute: may take long, no lasting effect the user must approve (a trained
# run, a forecast) · write: changes what the user keeps (always confirmed first) · rows: reads raw
# rows (the data-sharing level applies)


@dataclass
class ToolContext:
    dataset_id: str
    conditions: list[Condition] = field(default_factory=list)  # the user's active filters
    cancelled: Callable[[], bool] = lambda: False
    progress: Callable[[str, float | None], None] = lambda message, share: None


@dataclass(frozen=True)
class AppLink:
    """A page of the app in a given state (R-AG-T5), e.g. compare?x=Sex&y=Survived."""

    page: str
    params: dict[str, str] = field(default_factory=dict)
    label: str = ""


@dataclass
class ToolOutput:
    summary: str  # one line for the conversation: "Sex × Survived: χ², V = 0.54 (strong)"
    content: dict[str, Any] | str  # what the model reads
    link: AppLink | None = None


class ToolError(Exception):
    """A failure the model can fix on its next step: say what was wrong and what to do."""


class Tool[I: BaseModel](ABC):
    name: ClassVar[str]
    description: ClassVar[str]  # when to use it, and when not to
    kind: ClassVar[ToolKind] = "read"
    Input: ClassVar[type[BaseModel]]

    def label(self, args: I) -> str:
        """What the step shows while it runs: "Comparing Sex × Survived"."""
        return self.name.replace("_", " ").capitalize()

    def confirmation(self, args: I) -> str:
        """For `write` tools: what will change, in one sentence the user approves."""
        return self.label(args)

    @abstractmethod
    def run(self, ctx: ToolContext, args: I) -> ToolOutput: ...

    def parse(self, raw: dict[str, Any] | None) -> I:
        try:
            return self.Input.model_validate(raw or {})  # type: ignore[return-value]
        except ValidationError as error:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'input'}: {e['msg']}"
                for e in error.errors()
            )
            raise ToolError(f"Invalid arguments for {self.name}: {problems}.") from error


def wait_for[T](
    poll: Callable[[], T],
    finished: Callable[[T], bool],
    ctx: ToolContext,
    describe: Callable[[T], tuple[str, float | None]] = lambda job: ("Working", None),
    cancel: Callable[[], Any] = lambda: None,
    timeout: float = 600.0,
    interval: float = 0.5,
) -> T:
    """Follows a background job of the app until it ends, relaying its progress; cancels it when
    the user stops the agent or the timeout passes (R-AG-B3)."""
    started = time.monotonic()
    last: tuple[str, float | None] | None = None
    while True:
        job = poll()
        if finished(job):
            return job
        if ctx.cancelled():
            cancel()
            raise ToolError("Stopped by the user.")
        if time.monotonic() - started > timeout:
            cancel()
            raise ToolError(f"Still running after {timeout:.0f} s: stopped. Try a lighter setting.")
        now = describe(job)
        if now != last:
            ctx.progress(*now)
            last = now
        time.sleep(interval)


def compact(value: Any, max_items: int = 20, digits: int = 4) -> Any:
    """A result made light for the model: dataclasses and models as dicts, numbers rounded,
    long lists of numbers summarised, long lists of records cut (R-AG-T2)."""
    if isinstance(value, BaseModel):
        value = value.model_dump()
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {
            str(k): compact(v, max_items, digits)
            for k, v in value.items()
            if v is not None and v != [] and v != {}
        }
    if isinstance(value, list | tuple):
        items = list(value)
        if items and all(isinstance(v, int | float) or v is None for v in items):
            if len(items) <= max_items:
                return [compact(v, max_items, digits) for v in items]
            numbers = [v for v in items if isinstance(v, int | float) and math.isfinite(v)]
            if not numbers:
                return f"[{len(items)} empty values]"
            return {
                "values": len(items),
                "first": compact(items[0], max_items, digits),
                "last": compact(items[-1], max_items, digits),
                "min": compact(min(numbers), max_items, digits),
                "max": compact(max(numbers), max_items, digits),
            }
        kept = [compact(v, max_items, digits) for v in items[:max_items]]
        if len(items) > max_items:
            kept.append(f"… {len(items) - max_items} more")
        return kept
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(f"{value:.{digits}g}")
    return value
