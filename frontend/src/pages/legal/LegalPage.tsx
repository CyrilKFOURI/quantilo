import { useNavigate, useParams, useRouter } from "@tanstack/react-router";
import { useMemo } from "react";

import { Markdown, parse } from "../../features/help/markdown";
import { isLegalDoc, LEGAL_DOCS, LEGAL_LABEL, LEGAL_TEXT, legalHref, type LegalDoc } from "../../features/legal/documents";
import { t } from "../../lib/i18n";
import { Tabs } from "../../ui/Tabs";
import { NotFoundPage } from "../not-found/NotFoundPage";

/** `/legal/$doc` (081): Privacy, Terms of use and Legal notice, one tab each, from docs/legal/*.md. */
export function LegalPage() {
  const { doc } = useParams({ from: "/legal/$doc" });
  if (!isLegalDoc(doc)) return <NotFoundPage />;
  return <LegalDocument doc={doc} />;
}

function LegalDocument({ doc }: { doc: LegalDoc }) {
  const navigate = useNavigate();
  const router = useRouter();
  const blocks = useMemo(() => parse(LEGAL_TEXT[doc]), [doc]);
  const href = useMemo(() => legalHref((path) => router.history.createHref(path)), [router]);

  return (
    <div className="mx-auto max-w-3xl px-(--gutter) py-6 pb-16">
      <Tabs
        label={t("legal.label")}
        value={doc}
        items={LEGAL_DOCS.map((value) => ({ value, label: t(LEGAL_LABEL[value]) }))}
        onChange={(next) => void navigate({ to: "/legal/$doc", params: { doc: next } })}
      >
        <article className="min-w-0 pt-2">
          <Markdown blocks={blocks} href={href} />
        </article>
      </Tabs>
    </div>
  );
}
