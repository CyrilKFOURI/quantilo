import { Link } from "@tanstack/react-router";
import { ArrowUpRight, Check, CircleAlert, Download, LoaderCircle, SearchX, ShieldQuestion } from "lucide-react";
import { useMemo } from "react";

import { Markdown, parse } from "../../features/help/markdown";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import type { AgentLink, RowsRequest } from "./api";
import { appHref, linkTarget } from "./links";
import { RowsCard } from "./RowsCard";

export type ShownStep = {
  id: string;
  label: string;
  ok?: boolean | null;
  summary?: string | null;
  link?: AgentLink | null;
  ms?: number;
  progress?: { message: string; share: number | null };
  confirm?: string;
  rows?: RowsRequest;
};

/** The user's answer to a card; `unmask` only for rows (columns shown as they are, this once). */
export type Decide = (stepId: string, approved: boolean, unmask?: string[]) => void;

export function UserMessage({ text }: { text: string }) {
  return (
    <div className="ml-10 rounded-xl rounded-br-sm bg-accent-soft px-3.5 py-2.5 text-[13.5px] whitespace-pre-wrap text-ink-1">
      {text}
    </div>
  );
}

export function AssistantMessage({
  datasetId,
  text,
  thinking,
  steps,
  streaming = false,
  unverified = [],
  onDecide,
}: {
  datasetId: string;
  text: string;
  thinking: string;
  steps: ShownStep[];
  streaming?: boolean;
  /** Figures of the answer found in no analysis (grounding check). */
  unverified?: string[];
  /** Answers a change the assistant asked to make, or rows it asked to see (live turn only). */
  onDecide?: Decide;
}) {
  const blocks = useMemo(() => parse(text), [text]);
  return (
    <div className="space-y-2.5">
      {thinking && (
        <details className="group text-xs text-ink-3">
          <summary className="cursor-pointer select-none hover:text-ink-2">{t("agent.thinking")}</summary>
          <p className="mt-1.5 border-l-2 border-line pl-2.5 whitespace-pre-wrap">{thinking}</p>
        </details>
      )}
      {steps.length > 0 && (
        <ol className="space-y-1" aria-label={t("agent.steps")}>
          {steps.map((step) => (
            <Step key={step.id} datasetId={datasetId} step={step} onDecide={onDecide} />
          ))}
        </ol>
      )}
      {text ? (
        <div className="agent-answer text-[13.5px] leading-relaxed text-ink-1 [&_li]:my-0.5 [&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5 [&_p]:my-2 [&_table]:my-2 [&_table]:text-xs [&_td]:border-b [&_td]:border-line [&_td]:px-2 [&_td]:py-1 [&_th]:px-2 [&_th]:py-1 [&_th]:text-left [&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5">
          <Markdown blocks={blocks} href={appHref} />
        </div>
      ) : (
        streaming && <p className="text-xs text-ink-3">{t("agent.working")}</p>
      )}
      {unverified.length > 0 && (
        <p className="flex items-start gap-1.5 text-xs text-ink-3" title={t("agent.unverifiedHint")}>
          <SearchX size={13} className="mt-px shrink-0" aria-hidden />
          <span>
            {t(unverified.length === 1 ? "agent.unverifiedOne" : "agent.unverifiedMany", {
              count: String(unverified.length),
              figures: unverified.join(", "),
            })}
          </span>
        </p>
      )}
    </div>
  );
}

function Step({
  datasetId,
  step,
  onDecide,
}: {
  datasetId: string;
  step: ShownStep;
  onDecide?: Decide;
}) {
  const done = step.ok !== undefined && step.ok !== null;
  const target = step.link ? linkTarget(datasetId, step.link) : null;
  if (step.rows && onDecide) {
    return <RowsCard rows={step.rows} onDecide={(approved, unmask) => onDecide(step.id, approved, unmask)} />;
  }
  if (step.confirm && onDecide) {
    return (
      <li className="space-y-2 rounded-lg border border-accent bg-accent-soft px-3 py-2.5 text-xs" role="group" aria-label={t("agent.confirmTitle")}>
        <p className="flex items-start gap-2 text-[13px] text-ink-1">
          <ShieldQuestion size={15} className="mt-px shrink-0 text-accent" aria-hidden />
          {step.confirm}
        </p>
        <div className="flex gap-2">
          <Button variant="primary" onClick={() => onDecide(step.id, true)}>
            {t("agent.confirm")}
          </Button>
          <Button variant="ghost" onClick={() => onDecide(step.id, false)}>
            {t("agent.dismiss")}
          </Button>
        </div>
      </li>
    );
  }
  return (
    <li className="rounded-lg border border-line bg-surface-0 text-xs">
      <details className="group">
        <summary className="flex cursor-pointer list-none items-center gap-2 px-2.5 py-1.5 select-none">
          {!done ? (
            <LoaderCircle size={13} className="shrink-0 animate-spin text-ink-3" aria-label={t("agent.running")} />
          ) : step.ok ? (
            <Check size={13} className="shrink-0 text-ok" aria-label={t("agent.stepOk")} />
          ) : (
            <CircleAlert size={13} className="shrink-0 text-warn" aria-label={t("agent.stepFailed")} />
          )}
          <span className="min-w-0 flex-1 truncate text-ink-2">{step.label}</span>
          {step.ms !== undefined && <span className="shrink-0 text-ink-3 tabular-nums">{(step.ms / 1000).toFixed(1)} s</span>}
        </summary>
        {(step.summary || target) && (
          <div className="space-y-1 border-t border-line px-2.5 py-2">
            {step.summary && <p className="text-ink-2">{step.summary}</p>}
            {target && step.link && "href" in target && (
              <a href={target.href} download className="inline-flex items-center gap-1 text-accent hover:underline">
                <Download size={12} aria-hidden />
                {step.link.label || t("agent.download")}
              </a>
            )}
            {target && step.link && "to" in target && (
              <Link
                to={target.to}
                params={target.params}
                search={(previous: Record<string, unknown>) => ({ filters: previous.filters, ...target.search })}
                className="inline-flex items-center gap-1 text-accent hover:underline"
              >
                {step.link.label || t("agent.open")}
                <ArrowUpRight size={12} aria-hidden />
              </Link>
            )}
          </div>
        )}
      </details>
      {!done && step.progress && (
        <div className="space-y-1 border-t border-line px-2.5 py-1.5 text-ink-3">
          <p className="truncate">{step.progress.message}</p>
          {step.progress.share !== null && (
            <span className="block h-1 overflow-hidden rounded-full bg-surface-2" aria-hidden>
              <span className="block h-full bg-accent transition-[width]" style={{ width: `${Math.round(step.progress.share * 100)}%` }} />
            </span>
          )}
        </div>
      )}
    </li>
  );
}
