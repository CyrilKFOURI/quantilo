import { Link } from "@tanstack/react-router";

import { t } from "../../lib/i18n";
import { PageHeader } from "../../ui/PageHeader";

export function NotFoundPage() {
  return (
    <div className="px-(--gutter) py-10">
      <PageHeader title={t("notFound.title")} description={t("notFound.body")} />
      <Link to="/" className="mt-4 inline-block text-accent underline-offset-4 hover:underline">
        {t("notFound.back")}
      </Link>
    </div>
  );
}
