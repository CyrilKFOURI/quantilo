"""HTTP routes of the agent (`/api/v1/modules/agent/…`, spec 026 § 10)."""

import json
from collections.abc import Iterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from quantilo.filters.model import Condition

from app.modules.agent.engine.budget import TOKEN_LIMIT
from app.modules.agent.engine.messages import Message
from app.modules.agent.engine.provider import Effort
from app.modules.agent.engine.tools import ToolError
from app.modules.agent.providers.vendors import VENDORS
from app.modules.agent.repository import MAX_NOTE_CHARS, Conversation, Note
from app.modules.agent.service import AgentService, strip_context

router = APIRouter(tags=["agent"])


def get_agent(request: Request) -> AgentService:
    return request.app.state.agent


Agent = Annotated[AgentService, Depends(get_agent)]


# --- settings ---------------------------------------------------------------------------------


class AgentKey(BaseModel):
    configured: bool
    source: str  # env | file | none
    hint: str | None


class AgentProvider(BaseModel):
    id: str
    label: str
    configured: bool
    console: str
    needs_key: bool


class AgentSettingsOut(BaseModel):
    provider: str  # the setting: a vendor id or "auto"
    active: str  # the vendor in use
    active_label: str
    providers: list[AgentProvider]
    model: str  # the setting; empty: the vendor's default
    known_models: list[str]  # models listed without asking the vendor (Claude)
    base_url: str | None
    effort: Effort
    effort_supported: bool  # Claude only
    budget_usd: float
    key: AgentKey
    sdk_installed: bool
    ready: bool  # a key (or none needed), an address for a custom API, and the SDK
    token_limit: int  # per conversation, when the model's price is unknown

    @classmethod
    def of(cls, agent: AgentService) -> "AgentSettingsOut":
        settings, vendor, key = agent.settings()
        store = agent.store
        return cls(
            provider=settings.provider,
            active=vendor.id,
            active_label=vendor.label,
            providers=[
                AgentProvider(
                    id=v.id,
                    label=v.label,
                    configured=store.key_status(v).configured,
                    console=v.console,
                    needs_key=v.needs_key,
                )
                for v in VENDORS
            ],
            model=settings.model,
            known_models=list(vendor.models),
            base_url=settings.base_url,
            effort=settings.effort,
            effort_supported=vendor.kind == "anthropic",
            budget_usd=settings.budget_usd,
            key=AgentKey(configured=key.configured, source=key.source, hint=key.hint),
            sdk_installed=agent.sdk_available(vendor),
            ready=(key.configured or not vendor.needs_key)
            and (vendor.id != "custom" or bool(settings.base_url))
            and agent.sdk_available(vendor),
            token_limit=TOKEN_LIMIT,
        )


class AgentSettingsIn(BaseModel):
    provider: str | None = Field(None, description="A vendor id, or auto.")
    model: str | None = Field(None, description="Empty for the vendor's default.")
    base_url: str | None = Field(None, description="Custom OpenAI-compatible API only.")
    effort: Effort | None = None
    budget_usd: float | None = None
    api_key: str | None = Field(None, description="Write-only; an empty string removes it.")


class AgentKeyTest(BaseModel):
    ok: bool
    model: str


class AgentModels(BaseModel):
    models: list[str]
    default: str | None


@router.get("/settings")
def read_settings(agent: Agent) -> AgentSettingsOut:
    return AgentSettingsOut.of(agent)


@router.put("/settings")
def write_settings(body: AgentSettingsIn, agent: Agent) -> AgentSettingsOut:
    agent.update_settings(
        body.provider, body.model, body.base_url, body.effort, body.budget_usd, body.api_key
    )
    return AgentSettingsOut.of(agent)


@router.get("/settings/models")
def models(agent: Agent, refresh: bool = False) -> AgentModels:
    """The chat models the active vendor offers to the key (listed from its API)."""
    found, default = agent.models(refresh)
    return AgentModels(models=found, default=default)


@router.post("/settings/test")
def test_key(agent: Agent) -> AgentKeyTest:
    return AgentKeyTest(ok=True, model=agent.test_key())


# --- notes ------------------------------------------------------------------------------------


class AgentNote(BaseModel):
    id: str
    text: str
    source: str  # user | agent
    created_at: str

    @classmethod
    def of(cls, note: Note) -> "AgentNote":
        return cls(id=note.id, text=note.text, source=note.source, created_at=note.created_at)


class AgentNoteIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_NOTE_CHARS)


@router.get("/datasets/{dataset_id}/notes")
def list_notes(dataset_id: str, agent: Agent) -> list[AgentNote]:
    return [AgentNote.of(n) for n in agent.dataset_notes(dataset_id)]


@router.post("/datasets/{dataset_id}/notes", status_code=201)
def add_note(dataset_id: str, body: AgentNoteIn, agent: Agent) -> AgentNote:
    return AgentNote.of(agent.add_note(dataset_id, body.text))


@router.delete("/datasets/{dataset_id}/notes/{note_id}", status_code=204)
def delete_note(dataset_id: str, note_id: str, agent: Agent) -> None:
    agent.delete_note(dataset_id, note_id)


# --- conversations ----------------------------------------------------------------------------


class AgentConversation(BaseModel):
    id: str
    dataset_id: str
    title: str
    model: str
    cost_usd: float
    tokens: int = 0
    created_at: str
    updated_at: str
    running: bool = False

    @classmethod
    def of(cls, found: Conversation, running: bool = False) -> "AgentConversation":
        return cls(**vars(found), running=running)


class AgentLink(BaseModel):
    page: str
    params: dict[str, str]
    label: str


class AgentStep(BaseModel):
    id: str
    tool: str
    label: str
    input: dict[str, Any]
    ok: bool | None = None  # None: no result recorded
    summary: str | None = None
    link: AgentLink | None = None


class AgentMessage(BaseModel):
    role: str
    text: str
    thinking: str = ""
    steps: list[AgentStep] = Field(default_factory=list)
    unverified: list[str] = Field(
        default_factory=list, description="Figures of the answer found in no analysis."
    )


class AgentConversationDetail(BaseModel):
    conversation: AgentConversation
    messages: list[AgentMessage]


@router.get("/datasets/{dataset_id}/conversations")
def list_conversations(dataset_id: str, agent: Agent) -> list[AgentConversation]:
    return [AgentConversation.of(c, agent.running(c.id)) for c in agent.conversations(dataset_id)]


@router.post("/datasets/{dataset_id}/conversations", status_code=201)
def create_conversation(dataset_id: str, agent: Agent) -> AgentConversation:
    return AgentConversation.of(agent.create(dataset_id))


@router.get("/conversations/{conversation_id}")
def read_conversation(conversation_id: str, agent: Agent) -> AgentConversationDetail:
    found, messages = agent.conversation(conversation_id)
    return AgentConversationDetail(
        conversation=AgentConversation.of(found, agent.running(found.id)),
        messages=transcript(messages, agent),
    )


@router.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, agent: Agent) -> None:
    agent.delete(conversation_id)


class AgentSend(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    filters: list[Condition] = Field(default_factory=list, max_length=50)
    page: str | None = Field(None, max_length=300, description="Where the user is, in words.")


@router.post("/conversations/{conversation_id}/messages")
def send_message(conversation_id: str, body: AgentSend, agent: Agent) -> StreamingResponse:
    """Plays a turn and streams its events as server-sent events (one JSON object per event)."""
    run = agent.send(conversation_id, body.text, body.filters, body.page)

    def stream() -> Iterator[str]:
        for event in agent.events(run):
            yield ": keep-alive\n\n" if event is None else f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class AgentDecision(BaseModel):
    approved: bool


@router.post("/conversations/{conversation_id}/confirm/{call_id}", status_code=204)
def confirm(conversation_id: str, call_id: str, body: AgentDecision, agent: Agent) -> None:
    """The user's answer to a change the assistant asked to make (Confirm or Dismiss)."""
    agent.confirm(conversation_id, call_id, body.approved)


@router.post("/conversations/{conversation_id}/stop")
def stop(conversation_id: str, agent: Agent) -> dict[str, bool]:
    return {"stopping": agent.stop(conversation_id)}


# --- transcript -------------------------------------------------------------------------------


def transcript(messages: list[Message], agent: AgentService) -> list[AgentMessage]:
    """The stored history as the panel shows it: tool results folded into their steps."""
    shown: list[AgentMessage] = []
    steps: dict[str, AgentStep] = {}
    for message in messages:
        if message.role == "user" and all(b.kind == "tool_result" for b in message.blocks):
            for block in message.blocks:
                step = steps.get(block.call_id or "")
                if step is None:
                    continue
                meta = block.meta or {}
                step.ok = not block.is_error
                step.summary = meta.get("summary")
                link = meta.get("link")
                step.link = AgentLink(**link) if link else None
            continue
        if message.role == "user":
            shown.append(AgentMessage(role="user", text=strip_context(message.text)))
            continue
        current = shown[-1] if shown and shown[-1].role == "assistant" else None
        if current is None:
            current = AgentMessage(role="assistant", text="")
            shown.append(current)
        for block in message.blocks:
            if block.kind == "text":
                current.text += ("\n\n" if current.text else "") + block.text
                if block.meta and "unverified" in block.meta:
                    current.unverified = list(block.meta["unverified"])
            elif block.kind == "thinking" and block.text:
                current.thinking += ("\n\n" if current.thinking else "") + block.text
            elif block.kind == "tool_call":
                step = AgentStep(
                    id=block.call_id or "",
                    tool=block.name or "",
                    label=_label(agent, block.name or "", block.input),
                    input=block.input or {},
                )
                steps[step.id] = step
                current.steps.append(step)
    return shown


def _label(agent: AgentService, name: str, raw: dict[str, Any] | None) -> str:
    tool = agent.registry.get(name)
    if tool is None:
        return name
    try:
        return tool.label(tool.parse(raw))
    except ToolError:
        return name
