"""One turn of the conversation: the ReAct loop (spec 026, § 4.1).

The model answers or calls tools; tools run (in parallel when they only read); their results go
back; and so on until the model answers, the step limit or the budget is reached, or the user
stops. Pure: no HTTP, no database, no SDK, so a scripted provider tests every path."""

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from app.modules.agent.engine.budget import Budget
from app.modules.agent.engine.compaction import compact
from app.modules.agent.engine.messages import Block, Message
from app.modules.agent.engine.provider import (
    Completed,
    Provider,
    ProviderError,
    Request,
    TextDelta,
    ThinkingDelta,
    Usage,
)
from app.modules.agent.engine.tools import (
    MAX_RESULT_CHARS,
    Tool,
    ToolContext,
    ToolError,
    ToolOutput,
    ToolRegistry,
)

MAX_STEPS = 25  # tool calls per turn (R-AG-B1)
MAX_FAILURES = 3  # failures of one tool before giving up (R-AG-B5)
PARALLEL = 4

Outcome = tuple[ToolOutput, bool]  # what a tool returned, and whether it succeeded
TurnEnd = Literal[
    "answered", "max_steps", "budget", "stopped", "refused", "truncated", "tool_failures", "error"
]
Decision = Literal["allow", "deny"]


@dataclass(frozen=True)
class Verdict:
    """A gate's answer when "allow" / "deny" is not enough: what the model is told on a refusal,
    and which personal columns a `rows` tool may show unmasked (R-AG-P2)."""

    allow: bool
    message: str | None = None
    unmasked: frozenset[str] = frozenset()


@dataclass
class StepStarted:
    call_id: str
    tool: str
    label: str
    input: dict[str, Any]


@dataclass
class StepProgress:
    call_id: str
    message: str
    share: float | None  # 0..1, or None when unknown


@dataclass
class StepDone:
    call_id: str
    ok: bool
    summary: str
    link: dict[str, Any] | None
    ms: int


@dataclass
class Spent:
    turn_usd: float
    conversation_usd: float
    tokens: int  # this response
    conversation_tokens: int
    priced: bool  # False: the model's price is unknown, count tokens instead of dollars


@dataclass
class Notice:
    message: str


TurnEvent = TextDelta | ThinkingDelta | StepStarted | StepProgress | StepDone | Spent | Notice


@dataclass
class TurnResult:
    end: TurnEnd
    messages: list[Message] = field(default_factory=list)  # appended to the history, in order
    usage: list[Usage] = field(default_factory=list)
    cost_usd: float = 0.0
    error: ProviderError | None = None


def _deny_side_effects(tool: Tool[Any], call: Block) -> Decision:
    """Without a way to ask the user, nothing that changes the app or reads rows runs."""
    return "allow" if tool.kind in ("read", "compute") else "deny"


@dataclass
class Turn:
    """Everything one turn needs; `run()` plays it."""

    provider: Provider
    registry: ToolRegistry
    context: ToolContext
    model: str
    system: str
    history: list[Message]
    budget: Budget
    emit: Callable[[TurnEvent], None] = lambda event: None
    should_stop: Callable[[], bool] = lambda: False
    gate: Callable[[Tool[Any], Block], Decision | Verdict] = _deny_side_effects
    effort: Any = "high"
    max_steps: int = MAX_STEPS

    def run(self) -> TurnResult:
        result = TurnResult("answered")
        steps = 0
        failures: dict[str, int] = {}
        while True:
            if self.should_stop():
                result.end = "stopped"
                return result
            try:
                completed = self._complete(result)
            except ProviderError as error:
                result.end, result.error = "error", error
                return result
            result.messages.append(completed.message)
            if completed.stop == "refusal":
                self.emit(Notice("The model declined this request."))
                result.end = "refused"
                return result
            calls = completed.message.tool_calls
            if completed.stop == "pause_turn":
                continue
            if not calls:
                result.end = "truncated" if completed.stop == "max_tokens" else "answered"
                return result
            if completed.stop == "max_tokens":  # a tool call cut off mid-arguments: never run it
                result.end = "truncated"
                result.messages.append(self._results([(c, _failed(TRUNCATED)) for c in calls]))
                return result
            if steps + len(calls) > self.max_steps:
                result.end = "max_steps"
                note = f"Step limit ({self.max_steps}) reached: answer with what you have."
                result.messages.append(self._results([(c, _failed(note)) for c in calls]))
                return result
            steps += len(calls)
            outcomes = self._run_tools(calls)
            result.messages.append(self._results(outcomes))
            for call, (_, ok) in outcomes:
                if not ok:
                    failures[call.name or ""] = failures.get(call.name or "", 0) + 1
            if any(count >= MAX_FAILURES for count in failures.values()):
                result.end = "tool_failures"
                return result
            if self.budget.exceeded:
                result.end = "budget"
                return result

    # --- one model call ----------------------------------------------------------------------

    def _complete(self, result: TurnResult) -> Completed:
        request = Request(
            model=self.model,
            system=self.system,
            messages=compact(self.history) + result.messages,  # older results summarised
            tools=self.registry.specs(),
            effort=self.effort,
        )
        completed: Completed | None = None
        for event in self.provider.stream(request):
            if isinstance(event, Completed):
                completed = event
            else:
                self.emit(event)
        if completed is None:
            raise ProviderError("other", "The provider ended without a response.")
        result.usage.append(completed.usage)
        spent = self.budget.add(completed.usage)
        result.cost_usd += spent
        self.emit(
            Spent(
                result.cost_usd,
                self.budget.spent_usd,
                completed.usage.total,
                self.budget.tokens,
                self.budget.priced,
            )
        )
        return completed

    # --- tools -------------------------------------------------------------------------------

    def _run_tools(self, calls: list[Block]) -> list[tuple[Block, Outcome]]:
        tools = [self.registry.get(call.name or "") for call in calls]
        parallel = len(calls) > 1 and all(t is not None and t.kind == "read" for t in tools)
        if parallel:
            with ThreadPoolExecutor(max_workers=min(PARALLEL, len(calls))) as pool:
                outputs = list(pool.map(self._run_one, calls))
        else:
            outputs = [self._run_one(call) for call in calls]
        return list(zip(calls, outputs, strict=True))

    def _run_one(self, call: Block) -> Outcome:
        started = time.perf_counter()
        call_id = call.call_id or ""
        tool = self.registry.get(call.name or "")
        if tool is None:
            known = ", ".join(t.name for t in self.registry)
            return self._done(
                call_id,
                started,
                _error_output(f"Unknown tool {call.name!r}. Tools: {known}."),
                False,
            )
        try:
            args = tool.parse(call.input)
        except ToolError as error:
            return self._done(call_id, started, _error_output(str(error)), False)
        self.emit(StepStarted(call_id, tool.name, tool.label(args), call.input or {}))
        verdict = Verdict(True)
        if tool.kind not in ("read", "compute"):
            verdict = _verdict(self.gate(tool, call), tool)
            if not verdict.allow:
                return self._done(call_id, started, _error_output(verdict.message or ""), False)
        try:
            context = replace(
                self.context,
                progress=lambda message, share: self.emit(StepProgress(call_id, message, share)),
                unmasked=verdict.unmasked,
            )
            output = tool.run(context, args)
            ok = True
        except ToolError as error:
            output, ok = _error_output(str(error)), False
        except Exception as error:  # a bug in a tool must not end the conversation
            output, ok = _error_output(f"{tool.name} failed: {error}"), False
        return self._done(call_id, started, output, ok)

    def _done(self, call_id: str, started: float, output: ToolOutput, ok: bool) -> Outcome:
        link = None
        if output.link is not None:
            link = {
                "page": output.link.page,
                "params": output.link.params,
                "label": output.link.label,
            }
        ms = int((time.perf_counter() - started) * 1000)
        self.emit(StepDone(call_id, ok, output.summary, link, ms))
        return output, ok

    @staticmethod
    def _results(outcomes: list[tuple[Block, Outcome]]) -> Message:
        """Every result of one model response, in a single user message."""
        blocks = []
        for call, (output, ok) in outcomes:
            link = output.link
            blocks.append(
                Block(
                    "tool_result",
                    text=_render(output.content),
                    call_id=call.call_id,
                    name=call.name,
                    is_error=not ok,
                    meta={
                        "summary": output.summary,
                        "link": None
                        if link is None
                        else {"page": link.page, "params": link.params, "label": link.label},
                    },
                )
            )
        return Message("user", blocks)


def _verdict(answer: Decision | Verdict, tool: Tool[Any]) -> Verdict:
    if isinstance(answer, Verdict):
        if answer.allow or answer.message:
            return answer
        answer = "deny"
    if answer == "allow":
        return Verdict(True)
    return Verdict(False, f"The user did not allow {tool.name} here; continue without it.")


TRUNCATED = "The response was cut off before the tool arguments were complete; call it again."


def _error_output(message: str) -> ToolOutput:
    return ToolOutput(summary=message, content=message)


def _failed(message: str) -> Outcome:
    return _error_output(message), False


def _render(content: dict[str, Any] | str) -> str:
    """The tool result as the model reads it: JSON, capped (R-AG-T2)."""
    text = (
        content
        if isinstance(content, str)
        else json.dumps(content, ensure_ascii=False, default=str)
    )
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + " …[truncated: ask for a narrower result]"
    return text
