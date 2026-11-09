import { useDatasets } from "../../api/hooks";
import { DatasetList } from "../../features/datasets/DatasetList";
import { HealthIndicator } from "../../features/health/HealthIndicator";
import { ImportPanel } from "../../features/import/ImportPanel";
import { t } from "../../lib/i18n";
import { ErrorNote } from "../../ui/fields";
import { PageHeader } from "../../ui/PageHeader";

export function HomePage() {
  const datasets = useDatasets();
  const hasDatasets = (datasets.data?.length ?? 0) > 0;

  return (
    <div className="mx-auto flex min-h-full max-w-5xl flex-col gap-8 px-(--gutter) py-8">
      <PageHeader title={t("home.title")} description={t("home.body")} />
      <ImportPanel compact={hasDatasets} />
      {datasets.isError && <ErrorNote>{datasets.error.message}</ErrorNote>}
      {hasDatasets && datasets.data && <DatasetList datasets={datasets.data} />}
      <div className="mt-auto">
        <HealthIndicator />
      </div>
    </div>
  );
}
