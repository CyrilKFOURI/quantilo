import { useMemo } from "react";

import guide from "../../../../docs/USER_GUIDE.md?raw";
import { inline, Markdown, parse } from "../../features/help/markdown";
import { t } from "../../lib/i18n";

/** The user guide (docs/USER_GUIDE.md, the single source), with a table of contents. */
export function HelpPage() {
  const blocks = useMemo(() => parse(guide), []);
  const sections = blocks.filter((block) => block.kind === "heading" && block.level === 2);

  return (
    <div className="flex gap-10 px-(--gutter) py-6">
      <nav aria-label={t("help.contents")} className="sticky top-6 hidden w-52 shrink-0 self-start lg:block">
        <p className="mb-2 text-xs text-ink-3">{t("help.contents")}</p>
        <ul className="space-y-0.5 text-[13px]">
          {sections.map((block) =>
            block.kind === "heading" ? (
              <li key={block.id}>
                <a href={`#${block.id}`} className="block rounded px-2 py-1 text-ink-2 hover:bg-surface-2 hover:text-ink-1">
                  {inline(block.text)}
                </a>
              </li>
            ) : null,
          )}
        </ul>
      </nav>
      <article className="max-w-3xl min-w-0 flex-1 pb-16">
        <Markdown blocks={blocks} />
      </article>
    </div>
  );
}
