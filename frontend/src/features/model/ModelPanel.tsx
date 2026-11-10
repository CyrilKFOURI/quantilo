import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { ArrowLeft, Check, KeyRound, Trash2, X } from "lucide-react";
import { useState, type ReactNode } from "react";

import { formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { ErrorNote, Field, Select } from "../../ui/fields";
import {
  columnsOf,
  modelApi,
  useAggregate,
  useDeleteRelation,
  useSaveLayout,
  useSaveRelation,
  useUpdateRelation,
  type JoinType,
  type Model,
  type ModelTable,
  type Reason,
  type Relation,
  type RelationCheck,
} from "./api";
import { cardinalityLabel, joinLabel, percent, primaryKey, relationTitle, roleLabel, tableNames } from "./labels";
import type { Selection } from "./ModelCanvas";

const JOINS: JoinType[] = ["left", "inner", "right", "full"];

type PanelProps = {
  projectId: string;
  model: Model;
  selected: Selection;
  onSelect: (selection: Selection) => void;
};

/** The side panel (088): the relations to review, or the selected relation, or the selected table. */
export function ModelPanel({ projectId, model, selected, onSelect }: PanelProps) {
  const relation = selected?.kind === "relation" ? model.relations.find((item) => item.id === selected.id) : undefined;
  const table = selected?.kind === "table" ? model.tables.find((item) => item.id === selected.id) : undefined;
  if (relation) {
    return (
      <PanelFrame onBack={() => onSelect(null)}>
        <RelationDetail key={relation.id} projectId={projectId} model={model} relation={relation} onDone={() => onSelect(null)} />
      </PanelFrame>
    );
  }
  if (table) {
    return (
      <PanelFrame onBack={() => onSelect(null)}>
        <TableDetail key={table.id} projectId={projectId} model={model} table={table} onSelect={onSelect} />
      </PanelFrame>
    );
  }
  return <RelationList projectId={projectId} model={model} onSelect={onSelect} />;
}

function PanelFrame({ onBack, children }: { onBack: () => void; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-4">
      <button
        type="button"
        onClick={onBack}
        className="inline-flex w-fit items-center gap-1.5 text-xs text-ink-2 hover:text-accent"
      >
        <ArrowLeft size={13} strokeWidth={1.5} aria-hidden />
        {t("model.relation.back")}
      </button>
      {children}
    </div>
  );
}

function SectionTitle({ children }: { children: ReactNode }) {
  return <h3 className="text-xs font-semibold text-ink-1">{children}</h3>;
}

export function ReasonList({ reasons }: { reasons: Reason[] }) {
  if (reasons.length === 0) return null;
  return (
    <ul className="space-y-1.5">
      {reasons.map((reason, index) => (
        <li key={`${reason.code}-${index}`} className="flex gap-2 text-xs leading-relaxed text-ink-2">
          <code className="shrink-0 pt-px font-mono text-[11px] text-ink-3">{reason.code}</code>
          <span className="min-w-0">
            <Ticked text={reason.text} />
          </span>
        </li>
      ))}
    </ul>
  );
}

/** A reason names columns between backticks (`orderID`): shown as code. */
function Ticked({ text }: { text: string }) {
  return text.split("`").map((part, index) =>
    index % 2 === 1 ? (
      <code key={index} className="font-mono text-[11px] text-ink-1">
        {part}
      </code>
    ) : (
      part
    ),
  );
}

// --- The list ---------------------------------------------------------------------------------

export function RelationList({ projectId, model, onSelect }: { projectId: string; model: Model; onSelect: (selection: Selection) => void }) {
  const save = useSaveRelation(projectId);
  const names = tableNames(model);
  const suggested = model.relations.filter((relation) => relation.status === "suggested");
  const accepted = model.relations.filter((relation) => relation.status === "accepted");
  return (
    <div className="flex flex-col gap-5">
      <section className="space-y-2" aria-labelledby="model-suggested">
        <h3 id="model-suggested" className="flex items-baseline gap-2 text-xs font-semibold text-ink-1">
          {t("model.list.suggested")}
          <span className="font-normal text-ink-3 tabular-nums">{suggested.length}</span>
        </h3>
        {suggested.length === 0 ? (
          <p className="text-xs text-ink-3">{t(model.relations.length ? "model.list.reviewed" : "model.list.empty")}</p>
        ) : (
          <ul className="divide-y divide-line rounded-md border border-line bg-surface-1">
            {suggested.map((relation) => (
              <li key={relation.id} className="flex items-center gap-1 py-1 pr-1 pl-2.5">
                <RelationButton relation={relation} names={names} onSelect={onSelect} />
                <Button
                  variant="ghost"
                  icon={Check}
                  iconOnly
                  disabled={save.isPending}
                  aria-label={`${t("model.relation.accept")}: ${relationTitle(relation, names)}`}
                  title={t("model.relation.accept")}
                  onClick={() => save.mutate({ ...columnsOf(relation), status: "accepted" })}
                />
                <Button
                  variant="ghost"
                  icon={X}
                  iconOnly
                  disabled={save.isPending}
                  aria-label={`${t("model.relation.reject")}: ${relationTitle(relation, names)}`}
                  title={t("model.relation.reject")}
                  onClick={() => save.mutate({ ...columnsOf(relation), status: "rejected" })}
                />
              </li>
            ))}
          </ul>
        )}
        {save.isError && <ErrorNote>{save.error.message}</ErrorNote>}
      </section>
      {accepted.length > 0 && (
        <section className="space-y-2" aria-labelledby="model-accepted">
          <h3 id="model-accepted" className="flex items-baseline gap-2 text-xs font-semibold text-ink-1">
            {t("model.list.accepted")}
            <span className="font-normal text-ink-3 tabular-nums">{accepted.length}</span>
          </h3>
          <ul className="divide-y divide-line rounded-md border border-line bg-surface-1">
            {accepted.map((relation) => (
              <li key={relation.id} className="flex items-center py-1 pr-2 pl-2.5">
                <RelationButton relation={relation} names={names} onSelect={onSelect} />
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

function RelationButton({ relation, names, onSelect }: { relation: Relation; names: Map<string, string>; onSelect: (selection: Selection) => void }) {
  const many = relation.cardinality === "n:m";
  const unmatched = (relation.check?.orphan_rows ?? 0) + (relation.check?.child_empty ?? 0);
  return (
    <button
      type="button"
      onClick={() => onSelect({ kind: "relation", id: relation.id })}
      className="flex min-w-0 flex-1 flex-col items-start gap-0.5 rounded-md py-1 text-left hover:text-accent"
    >
      <span className="w-full truncate text-[13px] text-ink-1">
        {names.get(relation.child_id)} <span className="text-ink-3">to</span> {names.get(relation.parent_id)}
      </span>
      <span className="flex w-full min-w-0 flex-wrap items-center gap-x-2 text-xs text-ink-3">
        <span className="truncate font-mono">{relation.child_columns.join(" + ")}</span>
        <span className={many ? "text-bad" : ""}>{cardinalityLabel(relation)}</span>
        <span>{joinLabel(many ? "none" : relation.join_type)}</span>
        {unmatched > 0 && <span className="text-warn tabular-nums">{t("model.list.unmatched", { count: formatCount(unmatched) })}</span>}
        {relation.problem && <span className="text-bad">!</span>}
      </span>
    </button>
  );
}

// --- One relation -----------------------------------------------------------------------------

function RelationDetail({ projectId, model, relation, onDone }: { projectId: string; model: Model; relation: Relation; onDone: () => void }) {
  const names = tableNames(model);
  const child = model.tables.find((table) => table.id === relation.child_id);
  const parent = model.tables.find((table) => table.id === relation.parent_id);
  const childName = child?.name ?? "?";
  const parentName = parent?.name ?? "?";
  const many = relation.cardinality === "n:m";
  const [join, setJoin] = useState<JoinType>(relation.join_type === "none" ? "left" : relation.join_type);
  const [childColumn, setChildColumn] = useState(relation.child_columns.join(","));
  const [parentColumn, setParentColumn] = useState(relation.parent_columns.join(","));
  const save = useSaveRelation(projectId);
  const update = useUpdateRelation(projectId);
  const remove = useDeleteRelation(projectId);
  const edited =
    join !== relation.join_type || childColumn !== relation.child_columns.join(",") || parentColumn !== relation.parent_columns.join(",");
  const draft = { ...columnsOf(relation), child_columns: childColumn.split(","), parent_columns: parentColumn.split(","), join_type: join };
  const checked = useQuery({
    queryKey: ["model", projectId, "check", draft],
    queryFn: () => modelApi.check(projectId, draft),
    enabled: edited,
    retry: false,
  });
  const check: RelationCheck | null | undefined = edited ? checked.data : relation.check;
  const pending = save.isPending || update.isPending || remove.isPending;
  const failure = save.error ?? update.error ?? remove.error ?? (edited ? checked.error : null);

  function accept(status: "accepted" | "rejected") {
    save.mutate({ ...draft, status }, { onSuccess: onDone });
  }

  return (
    <article className="flex flex-col gap-4" aria-labelledby="model-relation-title">
      <header className="space-y-1.5">
        <h2 id="model-relation-title" className="text-sm font-semibold break-words text-ink-1">
          {relationTitle(relation, names)}
        </h2>
        <p className="flex flex-wrap gap-x-2.5 gap-y-1 text-xs">
          <span className={relation.status === "accepted" ? "text-ok" : "text-ink-2"}>
            {t(relation.status === "accepted" ? "model.relation.accepted" : "model.relation.suggested")}
          </span>
          <span className={many ? "text-bad" : "text-ink-2"}>{cardinalityLabel(relation)}</span>
          {relation.confidence && (
            <span className="text-ink-3">{t(`model.relation.confidence.${relation.confidence}` as MessageKey)}</span>
          )}
        </p>
      </header>
      {relation.problem && <ErrorNote>{t("model.relation.problem", { message: relation.problem })}</ErrorNote>}
      {check && <CheckSummary check={check} childName={childName} parentName={parentName} />}

      {relation.reasons.some((reason) => !reason.code.startsWith("R-JOIN")) && (
        <section className="space-y-2">
          <SectionTitle>{t("model.relation.why")}</SectionTitle>
          <ReasonList reasons={relation.reasons.filter((reason) => !reason.code.startsWith("R-JOIN"))} />
        </section>
      )}

      <section className="space-y-2">
        <SectionTitle>{t("model.relation.whyJoin")}</SectionTitle>
        <ReasonList reasons={check?.reasons ?? []} />
        {!many && (
          <div className="grid grid-cols-[minmax(0,1fr)] gap-2 pt-1">
            <Field label={t("model.relation.join")}>
              <Select value={join} onChange={(event) => setJoin(event.target.value as JoinType)}>
                {JOINS.map((value) => (
                  <option key={value} value={value}>
                    {joinLabel(value)}
                    {check?.recommended === value ? ` (${t("model.relation.recommendedShort")})` : ""}
                  </option>
                ))}
              </Select>
            </Field>
            <p className="text-xs text-ink-3">{t(`model.join.${join}Help` as MessageKey, { child: childName, parent: parentName })}</p>
          </div>
        )}
        {(check?.warnings.length ?? 0) > 0 && (
          <ul className="space-y-1" aria-live="polite">
            {check?.warnings.map((warning, index) => (
              <li key={index} className="flex gap-2 text-xs text-warn">
                <code className="shrink-0 font-mono text-[11px]">{warning.code}</code>
                <span>
                  <Ticked text={warning.text} />
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>

      {child && parent && (
        <section className="space-y-2">
          <SectionTitle>{t("model.relation.columns")}</SectionTitle>
          <div className="grid grid-cols-2 gap-2">
            <Field label={t("model.relation.childColumn", { name: childName })}>
              <ColumnSelect table={child} value={childColumn} onChange={setChildColumn} />
            </Field>
            <Field label={t("model.relation.parentColumn", { name: parentName })}>
              <ColumnSelect table={parent} value={parentColumn} onChange={setParentColumn} keysFirst />
            </Field>
          </div>
        </section>
      )}

      {failure && <ErrorNote>{failure.message}</ErrorNote>}
      <div className="flex flex-wrap gap-2">
        {relation.status === "suggested" ? (
          <>
            <Button variant="primary" icon={Check} disabled={pending} onClick={() => accept("accepted")}>
              {t("model.relation.accept")}
            </Button>
            <Button icon={X} disabled={pending} onClick={() => accept("rejected")}>
              {t("model.relation.reject")}
            </Button>
          </>
        ) : (
          <>
            <Button
              variant="primary"
              disabled={!edited || pending}
              onClick={() =>
                update.mutate({
                  id: relation.id,
                  body: { join_type: join, child_columns: draft.child_columns, parent_columns: draft.parent_columns },
                })
              }
            >
              {t("model.relation.save")}
            </Button>
            <Button variant="ghost" icon={Trash2} disabled={pending} onClick={() => remove.mutate(relation.id, { onSuccess: onDone })}>
              {t("model.relation.delete")}
            </Button>
          </>
        )}
      </div>
      {relation.status === "accepted" && relation.cardinality === "n:1" && child && parent && (
        <Aggregate projectId={projectId} relation={relation} childName={childName} parentName={parentName} />
      )}
    </article>
  );
}

function ColumnSelect({ table, value, onChange, keysFirst = false }: { table: ModelTable; value: string; onChange: (value: string) => void; keysFirst?: boolean }) {
  const keys = new Set(table.keys.filter((key) => key.columns.length === 1).map((key) => key.columns[0]));
  const composite = table.keys.filter((key) => key.columns.length === 2);
  const linkable = table.columns.filter((column) => column.linkable);
  const ordered = keysFirst ? [...linkable].sort((a, b) => Number(keys.has(b.name)) - Number(keys.has(a.name))) : linkable;
  const known = ordered.some((column) => column.name === value) || composite.some((key) => key.columns.join(",") === value);
  return (
    <Select value={value} onChange={(event) => onChange(event.target.value)} className="w-full min-w-0">
      {!known && <option value={value}>{value.replace(",", " + ")}</option>}
      {composite.map((key) => (
        <option key={key.columns.join(",")} value={key.columns.join(",")}>
          {key.columns.join(" + ")}
        </option>
      ))}
      {ordered.map((column) => (
        <option key={column.name} value={column.name}>
          {column.name}
          {keys.has(column.name) ? ` (${t("model.relation.unique")})` : ""}
        </option>
      ))}
    </Select>
  );
}

function CheckSummary({ check, childName, parentName }: { check: RelationCheck; childName: string; parentName: string }) {
  const share = check.child_rows ? check.matched_rows / check.child_rows : 0;
  const rows: [string, number, boolean][] = [
    [t("model.relation.unmatched"), check.orphan_rows, check.orphan_rows > 0],
    [t("model.relation.empty"), check.child_empty, check.child_empty > 0],
    [t("model.relation.repeated", { parent: parentName }), check.duplicate_keys, check.duplicate_keys > 0],
    [t("model.relation.unused", { parent: parentName }), check.unused_keys, false],
  ];
  return (
    <section className="space-y-2.5">
      <p className="text-[13px] text-ink-1">
        <span className="text-lg font-semibold tabular-nums">{percent(share)}</span>{" "}
        <span className="text-ink-2">{t("model.relation.matched", { child: childName, parent: parentName })}</span>
      </p>
      <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1 text-xs">
        {rows.map(([label, value, bad]) => (
          <div key={label} className="contents">
            <dt className="truncate text-ink-2">{label}</dt>
            <dd className={`text-right tabular-nums ${bad ? "font-medium text-warn" : "text-ink-1"}`}>{formatCount(value)}</dd>
          </div>
        ))}
      </dl>
      {check.orphans.length > 0 && <Examples label={t("model.relation.examples")} items={check.orphans} />}
      {check.duplicates.length > 0 && <Examples label={t("model.relation.duplicates")} items={check.duplicates} />}
    </section>
  );
}

function Examples({ label, items }: { label: string; items: { value: string; rows: number }[] }) {
  return (
    <div className="space-y-1">
      <p className="text-xs text-ink-3">{label}</p>
      <ul className="flex flex-wrap gap-1.5">
        {items.map((item) => (
          <li key={item.value} className="rounded-[4px] bg-surface-2 px-1.5 py-0.5 font-mono text-[11px] text-ink-1">
            {item.value} <span className="text-ink-3 tabular-nums">×{formatCount(item.rows)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Advanced (onion rule): nothing is computed before the user asks. */
function Aggregate({ projectId, relation, childName, parentName }: { projectId: string; relation: Relation; childName: string; parentName: string }) {
  const aggregate = useAggregate(projectId);
  return (
    <details className="group rounded-md border border-line">
      <summary className="cursor-pointer list-none px-3 py-2 text-xs font-medium text-ink-2 select-none hover:text-ink-1 [&::-webkit-details-marker]:hidden">
        {t("model.advanced")}
      </summary>
      <div className="space-y-2.5 border-t border-line px-3 py-3">
        <p className="text-xs font-semibold text-ink-1">{t("model.aggregate.title", { child: childName, parent: parentName })}</p>
        <p className="text-xs leading-relaxed text-ink-2">{t("model.aggregate.body", { child: childName, parent: parentName })}</p>
        {aggregate.isError && <ErrorNote>{aggregate.error.message}</ErrorNote>}
        {aggregate.data ? (
          <p className="text-xs text-ok">
            {t("model.aggregate.done")}{" "}
            <Link to="/d/$datasetId" params={{ datasetId: aggregate.data.dataset_id }} className="text-accent underline underline-offset-4">
              {t("model.aggregate.open")}
            </Link>
          </p>
        ) : (
          <Button disabled={aggregate.isPending} onClick={() => aggregate.mutate({ relation_id: relation.id })}>
            {aggregate.isPending ? t("model.aggregate.creating") : t("model.aggregate.button")}
          </Button>
        )}
      </div>
    </details>
  );
}

// --- One table --------------------------------------------------------------------------------

function TableDetail({ projectId, model, table, onSelect }: { projectId: string; model: Model; table: ModelTable; onSelect: (selection: Selection) => void }) {
  const key = primaryKey(table);
  const others = table.keys.filter((item) => !item.primary);
  const primary = table.keys.find((item) => item.primary);
  const layout = useSaveLayout(projectId);
  return (
    <article className="flex flex-col gap-4" aria-labelledby="model-table-title">
      <header className="space-y-1">
        <h2 id="model-table-title" className="text-sm font-semibold break-words text-ink-1">
          {table.name}
        </h2>
        <p className="text-xs text-ink-2 tabular-nums">
          {roleLabel(table)} · {t("model.card.rows", { rows: formatCount(table.rows) })}
        </p>
      </header>
      <section className="space-y-2">
        <SectionTitle>{t("model.table.why", { role: roleLabel(table).toLowerCase() })}</SectionTitle>
        <ReasonList reasons={table.role_reasons} />
      </section>
      <section className="space-y-2">
        <SectionTitle>{t("model.table.key")}</SectionTitle>
        <p className="flex items-center gap-1.5 text-xs">
          <KeyRound size={12} strokeWidth={1.5} className="text-ink-3" aria-hidden />
          {key ? <span className="font-mono text-ink-1">{key.join(" + ")}</span> : <span className="text-ink-3">{t("model.card.noKey")}</span>}
        </p>
        {primary && <ReasonList reasons={primary.reasons} />}
        {others.length > 0 && (
          <p className="text-xs text-ink-3">
            {t("model.table.otherKeys")}: <span className="font-mono text-ink-2">{others.map((item) => item.columns.join(" + ")).join(", ")}</span>
          </p>
        )}
      </section>
      {table.versions.length > 1 && (
        <Field label={t("model.table.version")}>
          <Select
            value={table.id}
            disabled={layout.isPending}
            onChange={(event) => {
              layout.mutate(
                { positions: {}, versions: { [table.root_id]: event.target.value } },
                { onSuccess: () => onSelect({ kind: "table", id: event.target.value }) },
              );
            }}
          >
            {table.versions.map((version) => (
              <option key={version.id} value={version.id}>
                {version.seq === 0 ? t("model.table.original") : t("model.card.version", { seq: String(version.seq) })}
              </option>
            ))}
          </Select>
        </Field>
      )}
      <AddRelation projectId={projectId} model={model} table={table} onSelect={onSelect} />
      <section className="space-y-2">
        <SectionTitle>{t("model.table.columns")}</SectionTitle>
        <ul className="divide-y divide-line rounded-md border border-line bg-surface-1 text-xs">
          {table.columns.map((column) => (
            <li key={column.name} className="flex items-baseline gap-2 px-2.5 py-1.5" title={column.excluded_by ? t("model.table.cantLink", { text: column.excluded_by.text }) : undefined}>
              <span className="min-w-0 flex-1 truncate font-mono text-ink-1">{column.name}</span>
              <span className="shrink-0 text-ink-3">{t(`model.category.${column.category}` as MessageKey)}</span>
            </li>
          ))}
        </ul>
      </section>
    </article>
  );
}

/** A relation by hand: a column of this table that refers to a key of another (R-REL-01: tables of
 *  the project, never itself). Checked before it is saved. */
function AddRelation({ projectId, model, table, onSelect }: { projectId: string; model: Model; table: ModelTable; onSelect: (selection: Selection) => void }) {
  const linkable = table.columns.filter((column) => column.linkable);
  const targets = model.tables.filter((other) => other.id !== table.id);
  const [column, setColumn] = useState(linkable[0]?.name ?? "");
  const [targetId, setTargetId] = useState(targets[0]?.id ?? "");
  const target = targets.find((other) => other.id === targetId);
  const targetKeys = target ? target.columns.filter((item) => item.linkable) : [];
  const [key, setKey] = useState("");
  const chosenKey = targetKeys.some((item) => item.name === key) ? key : (primaryKey(target ?? table)?.[0] ?? targetKeys[0]?.name ?? "");
  const save = useSaveRelation(projectId);
  if (linkable.length === 0 || targets.length === 0) return null;
  const body = { child_id: table.id, child_columns: [column], parent_id: targetId, parent_columns: [chosenKey] };
  return (
    <details className="rounded-md border border-line">
      <summary className="cursor-pointer list-none px-3 py-2 text-xs font-medium text-ink-2 select-none hover:text-ink-1 [&::-webkit-details-marker]:hidden">
        {t("model.table.add")}
      </summary>
      <div className="grid grid-cols-[minmax(0,1fr)] gap-2.5 border-t border-line px-3 py-3">
        <Field label={t("model.table.addColumn", { name: table.name })}>
          <Select value={column} onChange={(event) => setColumn(event.target.value)}>
            {linkable.map((item) => (
              <option key={item.name} value={item.name}>
                {item.name}
              </option>
            ))}
          </Select>
        </Field>
        <div className="grid grid-cols-2 gap-2">
          <Field label={t("model.table.addTable")}>
            <Select value={targetId} onChange={(event) => setTargetId(event.target.value)} className="w-full min-w-0">
              {targets.map((other) => (
                <option key={other.id} value={other.id}>
                  {other.name}
                </option>
              ))}
            </Select>
          </Field>
          <Field label={t("model.table.addKey")}>
            <Select value={chosenKey} onChange={(event) => setKey(event.target.value)} className="w-full min-w-0">
              {targetKeys.map((item) => (
                <option key={item.name} value={item.name}>
                  {item.name}
                </option>
              ))}
            </Select>
          </Field>
        </div>
        {save.isError && <ErrorNote>{save.error.message}</ErrorNote>}
        <Button
          className="w-fit"
          disabled={save.isPending || !column || !chosenKey}
          onClick={() =>
            save.mutate(
              { ...body, status: "accepted", origin: "user" },
              {
                onSuccess: (model) => {
                  const added = model.relations.find(
                    (relation) =>
                      relation.status === "accepted" &&
                      relation.child_id === table.id &&
                      relation.child_columns.join(",") === column &&
                      relation.parent_id === targetId,
                  );
                  if (added) onSelect({ kind: "relation", id: added.id });
                },
              },
            )
          }
        >
          {t("model.table.addButton")}
        </Button>
      </div>
    </details>
  );
}
