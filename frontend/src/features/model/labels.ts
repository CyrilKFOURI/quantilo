import { formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import type { Model, ModelTable, Relation } from "./api";

/** Words of the Model view (088), from the model the server sends. */

export const joinLabel = (join: string): string => t(`model.join.${join}` as MessageKey);

export function cardinalityLabel(relation: Pick<Relation, "cardinality">): string {
  if (relation.cardinality === "1:1") return t("model.relation.oneToOne");
  if (relation.cardinality === "n:m") return t("model.relation.manyToMany");
  return t("model.relation.oneToMany");
}

/** The marks at the ends of a line: the child's (many side) then the parent's (one side). */
export function marks(relation: Pick<Relation, "cardinality">): [string, string] {
  if (relation.cardinality === "1:1") return ["1", "1"];
  if (relation.cardinality === "n:m") return ["*", "*"];
  return ["*", "1"];
}

export function tableNames(model: Model): Map<string, string> {
  return new Map(model.tables.map((table) => [table.id, table.name]));
}

export function relationTitle(relation: Relation, names: Map<string, string>): string {
  return t("model.relation.title", {
    child: names.get(relation.child_id) ?? "?",
    childColumns: relation.child_columns.join(" + "),
    parent: names.get(relation.parent_id) ?? "?",
    parentColumns: relation.parent_columns.join(" + "),
  });
}

export function percent(share: number): string {
  if (share >= 1) return "100%";
  if (share >= 0.995) return "99.5%+";
  return `${(share * 100).toFixed(1).replace(/\.0$/, "")}%`;
}

function plural(count: number, one: MessageKey, many: MessageKey): string {
  return count === 1 ? t(one) : t(many, { count: formatCount(count) });
}

/** The conclusion above the diagram (UI_UX: a sentence says what the visual shows). */
export function summary(model: Model): string {
  if (model.tables.length < 2) return t("model.summary.few");
  const usable = model.relations.filter((relation) => relation.cardinality !== "n:m");
  if (model.relations.length === 0) return t("model.summary.none", { count: String(model.tables.length) });
  const relations = plural(model.relations.length, "model.summary.relationOne", "model.summary.relations");
  const pending = model.relations.filter((relation) => relation.status === "suggested").length;
  const review = pending ? ` ${t("model.summary.toReview", { count: String(pending) })}` : "";
  const fact = model.tables.find((table) => table.id === model.facts[0]);
  if (!fact || usable.length === 0) return t("model.summary.noFact", { relations }) + review;
  const dimensions = model.tables.filter((table) => table.role === "dimension").length;
  return (
    t("model.summary.fact", {
      fact: fact.name,
      relations,
      shape: t(`model.shape.${model.shape}` as MessageKey),
      dimensions: plural(dimensions, "model.summary.dimensionOne", "model.summary.dimensions"),
    }) + review
  );
}

/** "3 measures · 1 attribute" */
export function cardCounts(table: Pick<ModelTable, "measures" | "attributes">): string {
  return t("model.card.counts", {
    measures: plural(table.measures, "model.card.measureOne", "model.card.measures"),
    attributes: plural(table.attributes, "model.card.attributeOne", "model.card.attributes"),
  });
}

export function roleLabel(table: Pick<ModelTable, "role">): string {
  return t(`model.role.${table.role}` as MessageKey);
}

export function primaryKey(table: ModelTable): string[] | null {
  return table.keys.find((key) => key.primary)?.columns ?? null;
}
