import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useRef, useState } from "react";

import type { Condition } from "../../api/filters";
import { agentApi, streamTurn, type AgentLink, type AgentSettingsChange, type RowsRequest, type TurnEvent } from "./api";

const keys = {
  settings: ["agent", "settings"] as const,
  models: ["agent", "models"] as const,
  conversations: (datasetId: string) => ["agent", "conversations", datasetId] as const,
  conversation: (id: string) => ["agent", "conversation", id] as const,
  notes: (datasetId: string) => ["agent", "notes", datasetId] as const,
  personal: (datasetId: string) => ["agent", "personal", datasetId] as const,
};

export function useAgentSettings(enabled: boolean) {
  return useQuery({ queryKey: keys.settings, queryFn: agentApi.settings, enabled });
}

export function useUpdateAgentSettings() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (change: AgentSettingsChange) => agentApi.updateSettings(change),
    onSuccess: (settings) => {
      client.setQueryData(keys.settings, settings);
      void client.invalidateQueries({ queryKey: keys.models });
    },
  });
}

/** The chat models the active provider offers to the key (listed from its API). */
export function useAgentModels(enabled: boolean) {
  const client = useQueryClient();
  const query = useQuery({ queryKey: keys.models, queryFn: () => agentApi.models(false), enabled, retry: false });
  const refresh = useCallback(async () => {
    client.setQueryData(keys.models, await agentApi.models(true));
  }, [client]);
  return { ...query, refresh };
}

export function useTestAgentKey() {
  return useMutation({ mutationFn: agentApi.testKey });
}

export function useConversations(datasetId: string, enabled: boolean) {
  return useQuery({
    queryKey: keys.conversations(datasetId),
    queryFn: () => agentApi.conversations(datasetId),
    enabled,
  });
}

export function useConversation(id: string | null) {
  return useQuery({
    queryKey: keys.conversation(id ?? ""),
    queryFn: () => agentApi.conversation(id ?? ""),
    enabled: Boolean(id),
  });
}

export function useDeleteConversation(datasetId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: agentApi.deleteConversation,
    onSuccess: () => client.invalidateQueries({ queryKey: keys.conversations(datasetId) }),
  });
}

export function useNotes(datasetId: string) {
  return useQuery({ queryKey: keys.notes(datasetId), queryFn: () => agentApi.notes(datasetId) });
}

export function useAddNote(datasetId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (text: string) => agentApi.addNote(datasetId, text),
    onSuccess: () => client.invalidateQueries({ queryKey: keys.notes(datasetId) }),
  });
}

export function useDeleteNote(datasetId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (noteId: string) => agentApi.deleteNote(datasetId, noteId),
    onSuccess: () => client.invalidateQueries({ queryKey: keys.notes(datasetId) }),
  });
}

/** The dataset's personal columns, masked before rows reach the assistant unless shared. */
export function usePersonalColumns(datasetId: string, enabled: boolean) {
  return useQuery({
    queryKey: keys.personal(datasetId),
    queryFn: () => agentApi.personalColumns(datasetId),
    enabled,
  });
}

export function useSetUnmasked(datasetId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (unmasked: string[]) => agentApi.setUnmasked(datasetId, unmasked),
    onSuccess: (columns) => client.setQueryData(keys.personal(datasetId), columns),
  });
}

export type LiveStep = {
  id: string;
  tool: string;
  label: string;
  ok?: boolean;
  summary?: string;
  link?: AgentLink | null;
  ms?: number;
  progress?: { message: string; share: number | null };
  /** A change waiting for the user's Confirm or Dismiss. */
  confirm?: string;
  /** Rows the assistant asks to see (R-AG-P2): the card lists them and the masked columns. */
  rows?: RowsRequest;
};

/** The turn being played: what the assistant has said and done so far. */
export type LiveTurn = {
  question: string;
  text: string;
  thinking: string;
  steps: LiveStep[];
};

export type TurnState = {
  live: LiveTurn | null;
  running: boolean;
  /** Spent on the conversation so far (US dollars). */
  conversationUsd: number | null;
  conversationTokens: number | null;
  /** False when the model's price is unknown: the panel counts tokens instead of dollars. */
  priced: boolean | null;
  ending: string | null;
  error: { kind: string; message: string } | null;
};

const IDLE: TurnState = {
  live: null,
  running: false,
  conversationUsd: null,
  conversationTokens: null,
  priced: null,
  ending: null,
  error: null,
};

function apply(state: TurnState, event: TurnEvent): TurnState {
  const live = state.live;
  if (!live) return state;
  switch (event.type) {
    case "text":
      return { ...state, live: { ...live, text: live.text + event.text } };
    case "thinking":
      return { ...state, live: { ...live, thinking: live.thinking + event.text } };
    case "step_started":
      return { ...state, live: { ...live, steps: [...live.steps, { id: event.id, tool: event.tool, label: event.label }] } };
    case "step_progress":
      return {
        ...state,
        live: {
          ...live,
          steps: live.steps.map((s) =>
            s.id === event.id ? { ...s, progress: { message: event.message, share: event.share } } : s,
          ),
        },
      };
    case "confirm":
      return {
        ...state,
        live: {
          ...live,
          steps: live.steps.map((s) => (s.id === event.id ? { ...s, confirm: event.text, rows: event.rows } : s)),
        },
      };
    case "step_done":
      return {
        ...state,
        live: {
          ...live,
          steps: live.steps.map((s) =>
            s.id === event.id
              ? {
                  ...s,
                  ok: event.ok,
                  summary: event.summary,
                  link: event.link,
                  ms: event.ms,
                  confirm: undefined,
                  rows: undefined,
                  progress: undefined,
                }
              : s,
          ),
        },
      };
    case "spent":
      return {
        ...state,
        conversationUsd: event.conversation_usd,
        conversationTokens: event.conversation_tokens,
        priced: event.priced,
      };
    case "notice":
      return { ...state, ending: event.message };
    case "done":
      return {
        ...state,
        running: false,
        conversationUsd: event.conversation_usd,
        conversationTokens: event.conversation_tokens,
        priced: event.priced,
        ending: event.message,
      };
    case "error":
      return { ...state, running: false, error: { kind: event.kind, message: event.message } };
  }
}

/** Plays turns of a conversation: creates it on the first message, streams, then refreshes it. */
export function useTurn(datasetId: string) {
  const client = useQueryClient();
  const [state, setState] = useState<TurnState>(IDLE);
  const conversationRef = useRef<string | null>(null);

  const send = useCallback(
    async (conversationId: string | null, text: string, filters: Condition[], page: string | null) => {
      setState({ ...IDLE, running: true, live: { question: text, text: "", thinking: "", steps: [] } });
      let id = conversationId;
      try {
        if (!id) {
          id = (await agentApi.createConversation(datasetId)).id;
          void client.invalidateQueries({ queryKey: keys.conversations(datasetId) });
        }
        conversationRef.current = id;
        await streamTurn(id, { text, filters, page }, (event) => setState((s) => apply(s, event)));
      } catch (error) {
        setState((s) => ({ ...s, running: false, error: { kind: "other", message: (error as Error).message } }));
      } finally {
        setState((s) => ({ ...s, running: false }));
        if (id) {
          await client.invalidateQueries({ queryKey: keys.conversation(id) });
          void client.invalidateQueries({ queryKey: keys.notes(datasetId) });
          void client.invalidateQueries({ queryKey: keys.conversations(datasetId) });
          // the saved transcript now shows this turn; keep only an error or an ending note
          setState((s) => (s.error ? s : { ...s, live: null }));
        }
      }
      return id;
    },
    [client, datasetId],
  );

  const stop = useCallback(() => {
    if (conversationRef.current) void agentApi.stop(conversationRef.current);
  }, []);

  const reset = useCallback(() => setState(IDLE), []);

  /** The user's answer to a change the assistant asked to make, or to rows it asked to see
   *  (`unmask`: masked columns shown as they are, for this request only). */
  const decide = useCallback((callId: string, approved: boolean, unmask?: string[]) => {
    const id = conversationRef.current;
    if (!id) return;
    setState((s) =>
      s.live
        ? {
            ...s,
            live: {
              ...s.live,
              steps: s.live.steps.map((step) => (step.id === callId ? { ...step, confirm: undefined, rows: undefined } : step)),
            },
          }
        : s,
    );
    void agentApi.confirm(id, callId, approved, unmask);
  }, []);

  return { state, send, stop, reset, decide };
}
