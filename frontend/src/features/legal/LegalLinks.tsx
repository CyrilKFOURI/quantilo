import { Link } from "@tanstack/react-router";

import { t } from "../../lib/i18n";
import { LITE } from "../../lite/flag";
import { LEGAL_DOCS, LEGAL_LABEL } from "./documents";

/** Where the data goes, in one sentence (L-05): the AI assistant exists only in the app on a computer. */
export function dataStatement(lite = LITE): string {
  return lite ? t("legal.statementLite") : t("legal.statementLocal");
}

/** 081: the legal footer of the Guide, the Settings and the Datasets page (never the sidebar).
 *  `statement`: also say where the data goes and what the AI assistant sends. */
export function LegalLinks({ statement = false, className = "" }: { statement?: boolean; className?: string }) {
  return (
    <footer className={`text-xs text-ink-3 ${className}`}>
      {statement && <p className="mb-2 max-w-prose leading-relaxed">{dataStatement()}</p>}
      <nav aria-label={t("legal.label")}>
        <ul className="flex flex-wrap items-center gap-x-1 gap-y-1">
          {LEGAL_DOCS.map((doc, index) => (
            <li key={doc} className="flex items-center gap-x-1">
              {index > 0 && <span aria-hidden>·</span>}
              <Link
                to="/legal/$doc"
                params={{ doc }}
                className="rounded px-0.5 text-ink-3 underline-offset-2 hover:text-ink-1 hover:underline"
              >
                {t(LEGAL_LABEL[doc])}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
    </footer>
  );
}
