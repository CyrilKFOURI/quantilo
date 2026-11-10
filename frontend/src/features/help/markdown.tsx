import type { ReactNode } from "react";

/**
 * A small Markdown reader for the user guide: headings, paragraphs, lists, tables, bold,
 * italic, inline code and links. Enough for docs/USER_GUIDE.md, without a dependency.
 */
export type Block =
  | { kind: "heading"; level: 1 | 2 | 3; text: string; id: string }
  | { kind: "paragraph"; text: string }
  | { kind: "list"; ordered: boolean; items: string[] }
  | { kind: "table"; header: string[]; rows: string[][] };

export function slug(text: string): string {
  return text
    .toLowerCase()
    .replace(/[`*_]/g, "")
    .replace(/[^\p{L}\p{N}]+/gu, "-")
    .replace(/^-+|-+$/g, "");
}

/** Cells of a table row; "\|" is a literal bar. */
function cells(line: string): string[] {
  return line
    .trim()
    .replace(/^\||\|$/g, "")
    .split(/(?<!\\)\|/)
    .map((cell) => cell.trim().replace(/\\\|/g, "|"));
}

export function parse(markdown: string): Block[] {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const blocks: Block[] = [];
  const used = new Map<string, number>();
  let index = 0;
  while (index < lines.length) {
    const line = lines[index] ?? "";
    const heading = /^(#{1,3})\s+(.*)$/.exec(line);
    if (heading) {
      const text = heading[2] ?? "";
      const base = slug(text);
      const seen = used.get(base) ?? 0;
      used.set(base, seen + 1);
      blocks.push({ kind: "heading", level: (heading[1]?.length ?? 1) as 1 | 2 | 3, text, id: seen ? `${base}-${seen}` : base });
      index += 1;
    } else if (line.trim().startsWith("|")) {
      const rows: string[][] = [];
      while (index < lines.length && (lines[index] ?? "").trim().startsWith("|")) {
        const row = cells(lines[index] ?? "");
        if (!row.every((cell) => /^:?-{3,}:?$/.test(cell))) rows.push(row);
        index += 1;
      }
      blocks.push({ kind: "table", header: rows[0] ?? [], rows: rows.slice(1) });
    } else if (/^\s*(-|\d+\.)\s+/.test(line)) {
      const ordered = /^\s*\d+\./.test(line);
      const items: string[] = [];
      while (index < lines.length && /^\s*(-|\d+\.)\s+/.test(lines[index] ?? "")) {
        items.push((lines[index] ?? "").replace(/^\s*(-|\d+\.)\s+/, ""));
        index += 1;
      }
      blocks.push({ kind: "list", ordered, items });
    } else if (line.trim()) {
      const text: string[] = [];
      while (index < lines.length && (lines[index] ?? "").trim() && !/^(#|\||\s*(-|\d+\.)\s)/.test(lines[index] ?? "")) {
        text.push((lines[index] ?? "").trim());
        index += 1;
      }
      blocks.push({ kind: "paragraph", text: text.join(" ") });
    } else {
      index += 1;
    }
  }
  return blocks;
}

/** Decides what a link may point to: the href to use, or null to show the link as plain text. */
export type HrefFilter = (href: string) => string | null;

const anyHref: HrefFilter = (href) => href;

/** **bold**, *italic*, `code`, [text](url) and \| escapes. */
export function inline(text: string, href: HrefFilter = anyHref): ReactNode[] {
  const pattern = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g;
  return text
    .replace(/\\\|/g, "|")
    .split(pattern)
    .filter(Boolean)
    .map((part, key) => {
      if (part.startsWith("**")) return <strong key={key} className="font-semibold text-ink-1">{part.slice(2, -2)}</strong>;
      if (part.startsWith("`")) return <code key={key} className="rounded bg-surface-2 px-1 py-px font-mono text-[12px] text-ink-1">{part.slice(1, -1)}</code>;
      if (part.startsWith("[")) {
        const [, label = "", target = ""] = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(part) ?? [];
        const safe = href(target);
        if (safe === null) return label;
        return (
          // Another site opens in a new tab: leaving would close the app (and restart Python in the web version).
          <a
            key={key}
            href={safe}
            {...(/^https?:\/\//.test(safe) ? { target: "_blank", rel: "noreferrer" } : {})}
            className="text-accent underline-offset-2 hover:underline"
          >
            {label}
          </a>
        );
      }
      if (part.startsWith("*") && part.endsWith("*") && part.length > 2) return <em key={key}>{part.slice(1, -1)}</em>;
      return part;
    });
}

export function Markdown({ blocks, href = anyHref }: { blocks: Block[]; href?: HrefFilter }) {
  return (
    <>
      {blocks.map((block, key) => {
        if (block.kind === "heading") {
          const className =
            block.level === 1
              ? "text-xl font-semibold tracking-tight"
              : block.level === 2
                ? "mt-10 border-b border-line pb-2 text-base font-semibold"
                : "mt-6 text-sm font-semibold";
          const Tag = `h${block.level}` as const;
          return (
            <Tag key={key} id={block.id} className={`scroll-mt-6 ${className}`}>
              {inline(block.text, href)}
            </Tag>
          );
        }
        if (block.kind === "paragraph") return <p key={key} className="mt-3 leading-relaxed text-ink-2">{inline(block.text, href)}</p>;
        if (block.kind === "list") {
          const Tag = block.ordered ? "ol" : "ul";
          return (
            <Tag key={key} className={`mt-3 space-y-1.5 pl-5 text-ink-2 ${block.ordered ? "list-decimal" : "list-disc"} marker:text-ink-3`}>
              {block.items.map((item, i) => (
                <li key={i} className="leading-relaxed">
                  {inline(item, href)}
                </li>
              ))}
            </Tag>
          );
        }
        return (
          <div key={key} className="mt-3 overflow-x-auto rounded-md border border-line">
            <table className="w-full border-collapse text-[13px]">
              <thead className="bg-surface-2 text-left text-xs text-ink-3">
                <tr>
                  {block.header.map((cell, i) => (
                    <th key={i} scope="col" className="px-3 py-1.5 font-medium">
                      {inline(cell, href)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {block.rows.map((row, i) => (
                  <tr key={i} className="border-t border-line align-top">
                    {row.map((cell, j) => (
                      <td key={j} className={`px-3 py-1.5 ${j === 0 ? "text-ink-1" : "text-ink-2"}`}>
                        {inline(cell, href)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      })}
    </>
  );
}
