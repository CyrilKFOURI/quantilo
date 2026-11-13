import { Link } from "@tanstack/react-router";

import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import type { Relation, RelationCheck } from "./api";

/** N32: an accepted relation with orphan keys or a repeated key says where Clean lists them
 *  (T-01 in the table that points, T-02 in the one pointed to), with a link to that page. */
export function CleanNote({ relation, check, childName, parentName }: { relation: Relation; check: RelationCheck; childName: string; parentName: string }) {
  if (relation.status !== "accepted") return null;
  const orphans = check.orphan_rows > 0;
  const repeated = check.duplicate_keys > 0;
  if (!orphans && !repeated) return null;
  const link = "text-accent underline underline-offset-4";
  return (
    <div className="space-y-1.5 text-xs text-ink-2">
      {orphans && (
        <p>
          {t("cleanMisc.model.orphans", { count: formatCount(check.orphan_rows), child: childName, parent: parentName })}{" "}
          <Link to="/d/$datasetId/clean" params={{ datasetId: relation.child_id }} className={link}>
            {t("cleanMisc.model.open", { name: childName })}
          </Link>
        </p>
      )}
      {repeated && (
        <p>
          {t("cleanMisc.model.duplicates", { parent: parentName })}{" "}
          <Link to="/d/$datasetId/clean" params={{ datasetId: relation.parent_id }} className={link}>
            {t("cleanMisc.model.open", { name: parentName })}
          </Link>
        </p>
      )}
    </div>
  );
}
