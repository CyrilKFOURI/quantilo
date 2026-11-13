from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

anthropic = pytest.importorskip(
    "anthropic", reason="the assistant's `[agent]` extra is not installed"
)
httpx2 = pytest.importorskip("httpx2")

from anthropic.types.beta import BetaMessage  # noqa: E402

from app.modules.agent.engine.messages import Block, Message, user_text  # noqa: E402
from app.modules.agent.engine.provider import (  # noqa: E402
    Completed,
    ProviderError,
    Request,
    TextDelta,
    ThinkingDelta,
    ToolSpec,
)
from app.modules.agent.providers.anthropic import (  # noqa: E402
    FALLBACK_BETA,
    AnthropicProvider,
    build_params,
    to_api,
    translate,
)

FINAL = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5-5",
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "content": [
        {"type": "thinking", "thinking": "Check the target first.", "signature": "sig=="},
        {"type": "text", "text": "Let me look."},
        {"type": "tool_use", "id": "toolu_1", "name": "profile_column", "input": {"column": "Age"}},
    ],
    "usage": {
        "input_tokens": 120,
        "output_tokens": 40,
        "cache_read_input_tokens": 900,
        "cache_creation_input_tokens": 0,
    },
}


def delta(kind: str, **fields: Any) -> SimpleNamespace:
    return SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type=kind, **fields))


class FakeStream:
    def __init__(self, events: list[Any], final: dict[str, Any]) -> None:
        self.events, self.final = events, final

    def __iter__(self):
        yield from self.events

    def get_final_message(self) -> BetaMessage:
        return BetaMessage.model_validate(self.final)


class FakeClient:
    def __init__(self, *runs: Any) -> None:
        self.runs = list(runs)
        self.calls: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    @contextmanager
    def _stream(self, **params: Any):
        self.calls.append(params)
        run = self.runs.pop(0)
        if isinstance(run, Exception):
            raise run
        yield run


def request(model: str = "claude-opus-5-5") -> Request:
    return Request(
        model=model,
        system="system",
        messages=[user_text("Why?")],
        tools=[ToolSpec("profile_column", "Profile a column.", {"type": "object"})],
        effort="high",
    )


def test_streams_deltas_then_the_completed_turn() -> None:
    stream = FakeStream(
        [delta("thinking_delta", thinking="Check"), delta("text_delta", text="Let me look.")], FINAL
    )
    client = FakeClient(stream)
    provider = AnthropicProvider("sk-test", client_factory=lambda key: client)

    events = list(provider.stream(request()))

    assert isinstance(events[0], ThinkingDelta) and isinstance(events[1], TextDelta)
    done = events[-1]
    assert isinstance(done, Completed) and done.stop == "tool_use"
    kinds = [b.kind for b in done.message.blocks]
    assert kinds == ["thinking", "text", "tool_call"]
    call = done.message.blocks[2]
    assert (call.call_id, call.name, call.input) == ("toolu_1", "profile_column", {"column": "Age"})
    assert done.usage.cache_read_tokens == 900 and done.usage.model == "claude-opus-5-5"


def test_request_uses_cache_adaptive_thinking_effort_and_fallback() -> None:
    params = build_params(request())

    assert params["cache_control"] == {"type": "ephemeral"}
    assert params["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert params["output_config"] == {"effort": "high"}
    assert params["fallbacks"] == "default" and params["betas"] == [FALLBACK_BETA]
    assert params["tools"][0]["eager_input_streaming"] is True
    assert "tool_choice" not in params  # forced tool use is rejected by these models

    haiku = build_params(request("claude-haiku-5-5"))
    assert "fallbacks" not in haiku and "betas" not in haiku


def test_assistant_blocks_are_replayed_unchanged() -> None:
    client = FakeClient(FakeStream([], FINAL))
    provider = AnthropicProvider("sk-test", client_factory=lambda key: client)
    done = list(provider.stream(request()))[-1]
    assert isinstance(done, Completed)

    replay = to_api(done.message)

    thinking = replay["content"][0]
    assert thinking == {
        "type": "thinking",
        "thinking": "Check the target first.",
        "signature": "sig==",
    }
    results = to_api(
        Message("user", [Block("tool_result", text="{}", call_id="toolu_1", is_error=True)])
    )
    assert results["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": "{}",
        "is_error": True,
    }


def _status(cls: type, code: int) -> Exception:
    response = httpx2.Response(code, request=httpx2.Request("POST", "https://api.anthropic.com"))
    return cls("boom", response=response, body=None)


def test_sdk_errors_become_actionable_messages() -> None:
    assert translate(_status(anthropic.AuthenticationError, 401)).kind == "auth"
    assert translate(_status(anthropic.RateLimitError, 429)).kind == "rate_limit"
    assert translate(_status(anthropic.InternalServerError, 500)).kind == "overloaded"
    assert translate(_status(anthropic.NotFoundError, 404)).kind == "bad_request"
    connection = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x"))
    assert translate(connection).kind == "network"

    client = FakeClient(_status(anthropic.AuthenticationError, 401))
    provider = AnthropicProvider("sk-bad", client_factory=lambda key: client)
    with pytest.raises(ProviderError) as raised:
        list(provider.stream(request()))
    assert raised.value.kind == "auth" and "settings" in raised.value.message


def test_no_key_is_said_before_any_call() -> None:
    with pytest.raises(ProviderError) as raised:
        list(AnthropicProvider("").stream(request()))
    assert raised.value.kind == "no_key"
