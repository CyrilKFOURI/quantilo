import notice from "../../../../docs/legal/notice.md?raw";
import privacy from "../../../../docs/legal/privacy.md?raw";
import terms from "../../../../docs/legal/terms.md?raw";
import type { MessageKey } from "../../lib/i18n";

/** The three legal texts (081, L-01): docs/legal/*.md is the single source, read here by the app and
 *  turned into the web site's static pages by lite/site.py. */
export const LEGAL_DOCS = ["privacy", "terms", "notice"] as const;
export type LegalDoc = (typeof LEGAL_DOCS)[number];

export const LEGAL_TEXT: Record<LegalDoc, string> = { privacy, terms, notice };

export const LEGAL_LABEL: Record<LegalDoc, MessageKey> = {
  privacy: "legal.privacy",
  terms: "legal.terms",
  notice: "legal.notice",
};

export const isLegalDoc = (value: string): value is LegalDoc => (LEGAL_DOCS as readonly string[]).includes(value);

/** Links inside the texts: another legal page (`privacy`, `terms`, `notice`) stays in the app;
 *  a web address opens as is; anything else is shown as plain text. */
export function legalHref(createHref: (path: string) => string) {
  return (href: string): string | null => {
    if (isLegalDoc(href)) return createHref(`/legal/${href}`);
    return /^https:\/\//.test(href) ? href : null;
  };
}
