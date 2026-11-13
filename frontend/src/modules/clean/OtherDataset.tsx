import { useParams } from "@tanstack/react-router";

import { useColumns, useDatasets } from "../../api/hooks";
import { t, type MessageKey } from "../../lib/i18n";
import { Select } from "../../ui/fields";
import type { StepOf } from "./api";
import { Cell, CommitInput } from "./Cell";

type Other = StepOf<"append_rows"> | StepOf<"merge_dataset">;
const field = "h-8 w-full min-w-0 text-[13px]";
const split = (text: string) =>
  text
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);

/** C-57, C-58: the other dataset an append or a merge reads (ready ones, images left out), then
 *  the settings that need its columns: the keys there, the columns to add. */
export function OtherDataset({ step, onChange }: { step: Other; onChange: (step: Other) => void }) {
  const { datasetId } = useParams({ strict: false }) as { datasetId?: string };
  const datasets = (useDatasets().data ?? []).filter((dataset) => dataset.status === "ready" && dataset.format !== "images" && dataset.id !== datasetId);
  const theirs = (useColumns(step.dataset, Boolean(step.dataset)).data ?? []).map((column) => column.name);
  const set = (patch: Partial<Other>) => onChange({ ...step, ...patch } as Other);
  const cells = [
    <Cell key="dataset" label={t("clean.field.dataset")} why={t(`clean.field.why.${step.op}` as MessageKey)}>
      <Select
        aria-label={t("clean.field.dataset")}
        value={step.dataset}
        onChange={(event) => {
          const picked = datasets.find((dataset) => dataset.id === event.target.value);
          set({ dataset: event.target.value, label: picked?.name ?? null });
        }}
        className={field}
      >
        <option value="">{t("clean.field.pickDataset")}</option>
        {step.dataset && !datasets.some((dataset) => dataset.id === step.dataset) && <option value={step.dataset}>{step.label ?? step.dataset}</option>}
        {datasets.map((dataset) => (
          <option key={dataset.id} value={dataset.id}>
            {dataset.name}
          </option>
        ))}
      </Select>
    </Cell>,
  ];
  if (step.op === "append_rows") {
    cells.push(
      <Cell key="source_column" label={t("clean.field.sourceColumn")} why={t("clean.field.why.sourceColumn")}>
        <CommitInput label={t("clean.field.sourceColumn")} value={step.source_column ?? ""} placeholder={t("clean.field.none")} onCommit={(name) => set({ source_column: name.trim() || null })} />
      </Cell>,
    );
    if (step.source_column)
      cells.push(
        <Cell key="this_label" label={t("clean.field.thisLabel")}>
          <CommitInput label={t("clean.field.thisLabel")} value={step.this_label ?? ""} placeholder="this dataset" onCommit={(label) => set({ this_label: label.trim() || null })} />
        </Cell>,
      );
    return <>{cells}</>;
  }
  const keyPairs = step.keys.map((key, index) => [key, step.other_keys[index] ?? key] as const);
  cells.push(
    <Cell key="keys" label={t("clean.field.keysHere")} why={t("clean.field.columnsHint")}>
      <CommitInput label={t("clean.field.keysHere")} value={step.keys.join(", ")} onCommit={(value) => split(value).length && set({ keys: split(value) })} />
    </Cell>,
    <Cell key="other_keys" label={t("clean.field.keysThere")} why={theirs.length ? theirs.slice(0, 12).join(", ") : undefined}>
      <CommitInput label={t("clean.field.keysThere")} value={keyPairs.map(([, other]) => other).join(", ")} onCommit={(value) => split(value).length && set({ other_keys: split(value) })} />
    </Cell>,
    <Cell key="how" label={t("clean.field.joinHow")} why={t("clean.field.why.joinHow")}>
      <Select aria-label={t("clean.field.joinHow")} value={step.how} onChange={(event) => set({ how: event.target.value as "left" | "inner" })} className={field}>
        {(["left", "inner"] as const).map((how) => (
          <option key={how} value={how}>
            {t(`clean.joinHow.${how}` as MessageKey)}
          </option>
        ))}
      </Select>
    </Cell>,
    <Cell key="add" label={t("clean.field.add")} why={t("clean.field.why.add")} wide>
      <CommitInput label={t("clean.field.add")} value={(step.add ?? []).join(", ")} placeholder={t("clean.field.addAll")} onCommit={(value) => set({ add: split(value).length ? split(value) : null })} />
    </Cell>,
    <Cell key="prefix" label={t("clean.field.prefix")}>
      <CommitInput label={t("clean.field.prefix")} value={step.prefix} onCommit={(prefix) => set({ prefix })} />
    </Cell>,
  );
  return <>{cells}</>;
}
