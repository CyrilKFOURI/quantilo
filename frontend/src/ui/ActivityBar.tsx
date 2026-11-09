import { useIsFetching } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { t } from "../lib/i18n";

/** A quick answer does not make the bar flicker (042). */
const DELAY_MS = 250;

/** A 2 px line under the top bar while the page waits for data it does not have yet (042): a first
 *  load, or a new filter. Repeated checks (health, a job's own progress) leave it off. No height
 *  at rest, so the page does not move. */
export function ActivityBar() {
  const waiting = useIsFetching({ predicate: (query) => query.state.data === undefined }) > 0;
  const [shown, setShown] = useState(false);
  useEffect(() => {
    const timer = setTimeout(() => setShown(waiting), waiting ? DELAY_MS : 0);
    return () => clearTimeout(timer);
  }, [waiting]);
  return (
    <div className="relative h-0">
      {shown && (
        <div role="progressbar" aria-label={t("loading.default")} className="activity-bar absolute inset-x-0 top-0 z-20 h-0.5 overflow-hidden" />
      )}
    </div>
  );
}
