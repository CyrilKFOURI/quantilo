import { Link, useParams } from "@tanstack/react-router";
import { ArrowRight } from "lucide-react";

import type { Column } from "../../api/client";
import { useColumns, useDataset } from "../../api/hooks";
import {
  availableTools,
  THEMES,
  themeLine,
  themeName,
  toolAnswer,
  toolName,
  toolWhen,
  type Tool,
} from "../../features/toolbox/tools";
import { t } from "../../lib/i18n";
import { LockBadge } from "../../lite/Locked";

const card =
  "flex h-full min-w-0 flex-col gap-1 rounded-md border border-line bg-surface-1 px-3 py-2.5 text-left";

/** 098: every manual tool, under the title of what it does; each card says what the tool
 *  answers and when to reach for it. A tool opens empty from here (its column pickers only). */
export function ToolboxPage({ modules }: { modules: { id: string; locked?: boolean }[] }) {
  const { datasetId } = useParams({ strict: false }) as { datasetId: string };
  const dataset = useDataset(datasetId);
  const columns = useColumns(datasetId, dataset.data?.status === "ready");
  const tools = availableTools(modules);

  return (
    <div className="px-(--gutter) pt-6 pb-12">
      <header className="max-w-prose">
        <h1 className="text-xl font-semibold tracking-tight">{t("toolbox.title")}</h1>
        <p className="mt-1 text-ink-2">{t("toolbox.description")}</p>
      </header>
      <div className="mt-6 border-b border-line">
        {THEMES.map((theme) => {
          const found = tools.filter((tool) => tool.theme === theme);
          if (found.length === 0) return null;
          return (
            <section
              key={theme}
              aria-labelledby={`toolbox-${theme}`}
              className="grid gap-x-8 gap-y-3 border-t border-line py-5 lg:grid-cols-[220px_minmax(0,1fr)]"
            >
              <div className="min-w-0">
                <h2 id={`toolbox-${theme}`} className="text-base font-semibold tracking-tight text-balance">
                  {t(themeName(theme))}
                </h2>
                <p className="mt-0.5 text-[13px] text-ink-2">{t(themeLine(theme))}</p>
              </div>
              <ul className="grid min-w-0 gap-3 sm:grid-cols-2 xl:grid-cols-3">
                {found.map((tool) => (
                  <li key={tool.id} className="min-w-0">
                    <ToolCard tool={tool} datasetId={datasetId} columns={columns.data} />
                  </li>
                ))}
              </ul>
            </section>
          );
        })}
      </div>
    </div>
  );
}

function ToolCard({
  tool,
  datasetId,
  columns,
}: {
  tool: Tool & { locked: boolean };
  datasetId: string;
  columns: Column[] | undefined;
}) {
  const Icon = tool.icon;
  const name = t(toolName(tool));
  const missing =
    tool.requires &&
    columns &&
    !tool.requires.every((type) => columns.some((column) => column.type === type && !column.excluded));
  const body = (
    <>
      <span className="flex items-center gap-2">
        <Icon size={16} strokeWidth={1.5} aria-hidden className="shrink-0 text-ink-3" />
        <span className="min-w-0 flex-1 truncate text-sm font-medium text-ink-1">{name}</span>
        {tool.locked && <LockBadge />}
        {!missing && (
          <ArrowRight
            size={14}
            strokeWidth={1.5}
            aria-hidden
            className="shrink-0 text-ink-3 transition-transform duration-150 group-hover:translate-x-0.5 group-hover:text-accent motion-reduce:transition-none"
          />
        )}
      </span>
      <span className="text-[13px] text-ink-1 text-pretty">{t(toolAnswer(tool))}</span>
      <span className="text-xs text-ink-3 text-pretty">
        {missing && tool.requiresKey ? t(tool.requiresKey) : t(toolWhen(tool))}
      </span>
    </>
  );
  if (missing) {
    return (
      <span aria-disabled="true" className={`${card} cursor-not-allowed opacity-60`}>
        {body}
      </span>
    );
  }
  return (
    <Link
      to={tool.to as never}
      params={{ datasetId } as never}
      search={(tool.search ?? {}) as never}
      className={`group ${card} transition-colors hover:border-ink-3 hover:bg-surface-2 focus-visible:border-accent`}
    >
      {body}
    </Link>
  );
}
