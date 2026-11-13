import { apiError, request } from "../../api/client";
import type { Condition } from "../../api/filters";
import type { components } from "../../api/schema";

type Schemas = components["schemas"];
export type AgentSettings = Schemas["AgentSettingsOut"];
export type AgentSettingsChange = Schemas["AgentSettingsIn"];
export type AgentConversation = Schemas["AgentConversation"];
export type AgentConversationDetail = Schemas["AgentConversationDetail"];
export type AgentMessage = Schemas["AgentMessage"];
export type AgentStep = Schemas["AgentStep"];
export type AgentLink = Schemas["AgentLink"];
export type AgentNote = Schemas["AgentNote"];
export type AgentPersonalColumn = Schemas["AgentPersonalColumn"];
export type RowsSharing = AgentSettings["rows_sharing"];
export type Mask = AgentPersonalColumn["mask"];

/** What a `sample_rows` call would share, on the card that asks the user (R-AG-P2). */
export type RowsRequest = {
  count: number;
  columns: string[];
  /** Personal columns masked unless the user ticks them, this request only. */
  masked: { name: string; mask: Mask }[];
  filtered: boolean;
};

/** One event of a turn, as the backend streams it (modules/agent/service.py). */
export type TurnEvent =
  | { type: "text"; text: string }
  | { type: "thinking"; text: string }
  | { type: "step_started"; id: string; tool: string; label: string; input: Record<string, unknown> }
  | { type: "step_progress"; id: string; message: string; share: number | null }
  | { type: "confirm"; id: string; tool: string; text: string; rows?: RowsRequest }
  | { type: "step_done"; id: string; ok: boolean; summary: string; link: AgentLink | null; ms: number }
  | {
      type: "spent";
      turn_usd: number;
      conversation_usd: number;
      tokens: number;
      conversation_tokens: number;
      priced: boolean;
    }
  | { type: "notice"; message: string }
  | {
      type: "done";
      end: string;
      message: string | null;
      turn_usd: number;
      conversation_usd: number;
      conversation_tokens: number;
      priced: boolean;
    }
  | { type: "error"; kind: string; message: string };

const BASE = "/modules/agent";

export const agentApi = {
  settings: () => request<AgentSettings>("GET", `${BASE}/settings`),
  updateSettings: (change: AgentSettingsChange) => request<AgentSettings>("PUT", `${BASE}/settings`, change),
  models: (refresh: boolean) =>
    request<Schemas["AgentModels"]>("GET", `${BASE}/settings/models${refresh ? "?refresh=true" : ""}`),
  testKey: () => request<{ ok: boolean; model: string }>("POST", `${BASE}/settings/test`),
  conversations: (datasetId: string) =>
    request<AgentConversation[]>("GET", `${BASE}/datasets/${datasetId}/conversations`),
  createConversation: (datasetId: string) =>
    request<AgentConversation>("POST", `${BASE}/datasets/${datasetId}/conversations`),
  conversation: (id: string) => request<AgentConversationDetail>("GET", `${BASE}/conversations/${id}`),
  deleteConversation: (id: string) => request<void>("DELETE", `${BASE}/conversations/${id}`),
  stop: (id: string) => request<{ stopping: boolean }>("POST", `${BASE}/conversations/${id}/stop`),
  notes: (datasetId: string) => request<AgentNote[]>("GET", `${BASE}/datasets/${datasetId}/notes`),
  addNote: (datasetId: string, text: string) =>
    request<AgentNote>("POST", `${BASE}/datasets/${datasetId}/notes`, { text }),
  deleteNote: (datasetId: string, noteId: string) =>
    request<void>("DELETE", `${BASE}/datasets/${datasetId}/notes/${noteId}`),
  confirm: (id: string, callId: string, approved: boolean, unmask?: string[]) =>
    request<void>("POST", `${BASE}/conversations/${id}/confirm/${callId}`, { approved, unmask }),
  personalColumns: (datasetId: string) =>
    request<AgentPersonalColumn[]>("GET", `${BASE}/datasets/${datasetId}/personal-columns`),
  setUnmasked: (datasetId: string, unmasked: string[]) =>
    request<AgentPersonalColumn[]>("PUT", `${BASE}/datasets/${datasetId}/personal-columns`, { unmasked }),
};

/** Sends a message and calls `onEvent` for each streamed event until the turn ends. */
export async function streamTurn(
  conversationId: string,
  body: { text: string; filters: Condition[]; page: string | null },
  onEvent: (event: TurnEvent) => void,
): Promise<void> {
  const response = await fetch(`/api/v1${BASE}/conversations/${conversationId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw await apiError(response);
  if (!response.body) throw new Error("The browser did not open the answer stream.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let end = buffer.indexOf("\n\n");
    while (end >= 0) {
      const chunk = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      for (const line of chunk.split("\n")) {
        if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)) as TurnEvent);
      }
      end = buffer.indexOf("\n\n");
    }
  }
}
