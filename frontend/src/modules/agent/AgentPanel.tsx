import { useParams, useRouterState } from "@tanstack/react-router";
import { ArrowUp, NotebookPen, Plus, Settings2, Sparkles, Square, Trash2, X } from "lucide-react";
import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";

import { useActiveFilters } from "../../api/filters";
import { useColumns } from "../../api/hooks";
import { t } from "../../lib/i18n";
import { useStoredState } from "../../lib/useStoredState";
import { Button } from "../../ui/Button";
import { ErrorNote, Select } from "../../ui/fields";
import { describePage } from "./links";
import { AssistantMessage, UserMessage } from "./Message";
import { NotesView } from "./NotesView";
import { formatTokens, SettingsView } from "./SettingsView";
import { useAgentSettings, useConversation, useConversations, useDeleteConversation, useTurn } from "./useAgent";

const OPEN_KEY = "de.agent.open";

/** The assistant, docked on the right of every dataset page; Ctrl+J opens and closes it. */
export function AgentPanel() {
  const { datasetId } = useParams({ strict: false }) as { datasetId?: string };
  const [open, setOpen] = useStoredState(OPEN_KEY, false);

  useEffect(() => {
    if (!datasetId) return;
    const onKey = (event: globalThis.KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "j") {
        event.preventDefault();
        setOpen(!open);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [datasetId, open, setOpen]);

  if (!datasetId) return null;
  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        // 041: on a phone, a round icon button that covers less of the page.
        className="fixed right-5 bottom-5 z-30 inline-flex items-center gap-2 rounded-full border border-line bg-surface-1 px-4 py-2.5 text-[13px] font-medium text-ink-1 shadow-[0_8px_24px_-12px_rgb(0_0_0/0.35)] hover:bg-surface-2 max-sm:right-3 max-sm:bottom-[calc(12px+env(safe-area-inset-bottom))] max-sm:size-10 max-sm:justify-center max-sm:p-0"
        title={t("agent.shortcut")}
      >
        <Sparkles size={15} className="text-accent max-sm:size-[18px]" aria-hidden />
        <span className="max-sm:sr-only">{t("agent.open")}</span>
      </button>
    );
  }
  return (
    <aside
      aria-label={t("agent.title")}
      className="flex h-dvh w-[420px] shrink-0 flex-col border-l border-line bg-surface-1 max-lg:fixed max-lg:inset-y-0 max-lg:right-0 max-lg:z-40 max-lg:w-full max-lg:max-w-[420px] max-lg:shadow-xl"
    >
      <Chat key={datasetId} datasetId={datasetId} onClose={() => setOpen(false)} />
    </aside>
  );
}

function Chat({ datasetId, onClose }: { datasetId: string; onClose: () => void }) {
  const [view, setView] = useState<"chat" | "settings" | "notes">("chat");
  const [current, setCurrent] = useStoredState<string | null>(`de.agent.${datasetId}.conversation`, null);
  const settings = useAgentSettings(true);
  const conversations = useConversations(datasetId, true);
  const detail = useConversation(current);
  const remove = useDeleteConversation(datasetId);
  const { state, send, stop, reset, decide } = useTurn(datasetId);
  const filters = useActiveFilters(datasetId);
  const location = useRouterState({ select: (s) => s.location });
  const scroller = useRef<HTMLDivElement>(null);

  // A conversation deleted elsewhere (its detail answers 404) is forgotten.
  const conversationId = current && !detail.isError ? current : null;
  const messages = conversationId ? (detail.data?.messages ?? []) : [];
  const spent = state.conversationUsd ?? detail.data?.conversation.cost_usd ?? null;
  const tokens = state.conversationTokens ?? detail.data?.conversation.tokens ?? null;
  const priced = state.priced ?? settings.data?.effort_supported ?? true;

  useEffect(() => {
    const box = scroller.current;
    if (box) box.scrollTop = box.scrollHeight;
  }, [messages.length, state.live?.text, state.live?.steps.length]);

  const ready = settings.data?.ready;

  const ask = async (text: string) => {
    const page = describePage(location.pathname, location.search as Record<string, unknown>);
    const id = await send(conversationId, text, filters, page);
    if (id && id !== conversationId) setCurrent(id);
  };

  const startNew = () => {
    reset();
    setCurrent(null);
  };

  return (
    <>
      <header className="flex h-12 shrink-0 items-center gap-1.5 border-b border-line px-3">
        <Sparkles size={15} className="shrink-0 text-accent" aria-hidden />
        {view !== "chat" ? (
          <h2 className="flex-1 text-[13px] font-medium">{t(view === "settings" ? "agent.settings" : "agent.notes")}</h2>
        ) : (
          <Select
            aria-label={t("agent.conversations")}
            value={conversationId ?? ""}
            onChange={(event) => {
              reset();
              setCurrent(event.target.value || null);
            }}
            className="h-8 min-w-0 flex-1 text-[13px]"
          >
            <option value="">{t("agent.newConversation")}</option>
            {conversations.data?.map((c) => (
              <option key={c.id} value={c.id}>
                {c.title}
              </option>
            ))}
          </Select>
        )}
        {view === "chat" && (
          <>
            <Button variant="ghost" iconOnly icon={Plus} aria-label={t("agent.newConversation")} title={t("agent.newConversation")} onClick={startNew} />
            {conversationId && (
              <Button
                variant="ghost"
                iconOnly
                icon={Trash2}
                aria-label={t("agent.delete")}
                title={t("agent.delete")}
                onClick={() => remove.mutate(conversationId, { onSuccess: startNew })}
              />
            )}
          </>
        )}
        <Button
          variant="ghost"
          iconOnly
          icon={NotebookPen}
          aria-label={t("agent.notes")}
          title={t("agent.notes")}
          aria-pressed={view === "notes"}
          onClick={() => setView(view === "notes" ? "chat" : "notes")}
        />
        <Button
          variant="ghost"
          iconOnly
          icon={Settings2}
          aria-label={t("agent.settings")}
          title={t("agent.settings")}
          aria-pressed={view === "settings"}
          onClick={() => setView(view === "settings" ? "chat" : "settings")}
        />
        <Button variant="ghost" iconOnly icon={X} aria-label={t("agent.close")} title={t("agent.shortcut")} onClick={onClose} />
      </header>

      {view === "notes" ? (
        <div className="min-h-0 flex-1 overflow-y-auto">
          <NotesView datasetId={datasetId} onDone={() => setView("chat")} />
        </div>
      ) : view === "settings" && settings.data ? (
        <div className="min-h-0 flex-1 overflow-y-auto">
          <SettingsView settings={settings.data} onDone={() => setView("chat")} />
        </div>
      ) : (
        <>
          <div ref={scroller} className="min-h-0 flex-1 space-y-5 overflow-y-auto px-4 py-4" aria-live="polite">
            {settings.data && !ready && <Setup sdk={settings.data.sdk_installed} onSettings={() => setView("settings")} />}
            {ready && messages.length === 0 && !state.live && <Suggestions datasetId={datasetId} onPick={ask} />}
            {messages.map((message, index) =>
              message.role === "user" ? (
                <UserMessage key={index} text={message.text} />
              ) : (
                <AssistantMessage
                  key={index}
                  datasetId={datasetId}
                  text={message.text}
                  thinking={message.thinking ?? ""}
                  steps={message.steps ?? []}
                  unverified={message.unverified ?? []}
                />
              ),
            )}
            {state.live && (
              <>
                <UserMessage text={state.live.question} />
                <AssistantMessage
                  datasetId={datasetId}
                  text={state.live.text}
                  thinking={state.live.thinking}
                  steps={state.live.steps}
                  streaming={state.running}
                  onDecide={decide}
                />
              </>
            )}
            {state.ending && !state.running && <p className="text-xs text-ink-3">{state.ending}</p>}
            {state.error && <ErrorNote>{state.error.message}</ErrorNote>}
          </div>
          <Composer running={state.running} disabled={!ready} onSend={ask} onStop={stop} />
          <p className="flex shrink-0 justify-between border-t border-line px-4 py-1.5 text-[11px] text-ink-3">
            <span>{filters.length ? t("agent.filtered", { count: String(filters.length) }) : t("agent.allRows")}</span>
            {priced && spent !== null && (
              <span className="tabular-nums" title={t("agent.costHint")}>
                {t("agent.cost", { cost: spent < 0.01 ? "< $0.01" : `$${spent.toFixed(2)}` })}
                {settings.data && ` / $${settings.data.budget_usd.toFixed(2)}`}
              </span>
            )}
            {!priced && tokens !== null && (
              <span className="tabular-nums" title={t("agent.tokensHint")}>
                {t("agent.tokens", { count: formatTokens(tokens) })}
                {settings.data && ` / ${formatTokens(settings.data.token_limit)}`}
              </span>
            )}
          </p>
        </>
      )}
    </>
  );
}

function Setup({ sdk, onSettings }: { sdk: boolean; onSettings: () => void }) {
  return (
    <div className="space-y-3 rounded-xl border border-line bg-surface-0 p-4 text-[13px]">
      <p className="font-medium text-ink-1">{t("agent.setupTitle")}</p>
      <p className="text-ink-2">{sdk ? t("agent.setupKey") : t("agent.sdkMissing")}</p>
      <p className="text-xs text-ink-3">{t("agent.privacy")}</p>
      {sdk && (
        <Button variant="primary" onClick={onSettings}>
          {t("agent.addKey")}
        </Button>
      )}
    </div>
  );
}

function Suggestions({ datasetId, onPick }: { datasetId: string; onPick: (text: string) => void }) {
  const columns = useColumns(datasetId, true);
  const target = columns.data?.find((c) => !c.excluded && (c.type === "boolean" || c.type === "categorical_nominal"));
  const questions = [
    t("agent.suggest.overview"),
    t("agent.suggest.quality"),
    target ? t("agent.suggest.drivers", { column: target.name }) : t("agent.suggest.links"),
  ];
  return (
    <div className="space-y-3">
      <p className="text-[13px] text-ink-2">{t("agent.intro")}</p>
      <ul className="space-y-2">
        {questions.map((question) => (
          <li key={question}>
            <button
              type="button"
              onClick={() => onPick(question)}
              className="w-full rounded-lg border border-line px-3 py-2 text-left text-[13px] text-ink-1 hover:border-accent hover:bg-accent-soft"
            >
              {question}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Composer({
  running,
  disabled,
  onSend,
  onStop,
}: {
  running: boolean;
  disabled: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
}) {
  const [text, setText] = useState("");
  const submit = (event?: FormEvent) => {
    event?.preventDefault();
    const message = text.trim();
    if (!message || running || disabled) return;
    setText("");
    onSend(message);
  };
  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) submit(event);
  };
  return (
    <form onSubmit={submit} className="flex shrink-0 items-end gap-2 border-t border-line p-3">
      <textarea
        value={text}
        onChange={(event) => setText(event.target.value)}
        onKeyDown={onKeyDown}
        rows={2}
        disabled={disabled}
        placeholder={disabled ? t("agent.placeholderDisabled") : t("agent.placeholder")}
        aria-label={t("agent.message")}
        className="max-h-40 min-h-[44px] flex-1 resize-none rounded-lg border border-line bg-surface-0 px-3 py-2 text-[13.5px] text-ink-1 outline-none placeholder:text-ink-3 focus:border-accent disabled:opacity-60"
      />
      {running ? (
        <Button variant="secondary" iconOnly icon={Square} aria-label={t("agent.stop")} title={t("agent.stop")} onClick={onStop} />
      ) : (
        <Button
          type="submit"
          variant="primary"
          iconOnly
          icon={ArrowUp}
          aria-label={t("agent.send")}
          title={t("agent.send")}
          disabled={disabled || !text.trim()}
        />
      )}
    </form>
  );
}
