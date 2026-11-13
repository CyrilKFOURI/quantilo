import type { EntityCluster, EntityClusters, Step, StepOf, Suggestion } from "./api";

/** N36: the review of an entity number (A-20) cluster by cluster, as N35 reviews duplicates: per
 *  cluster, untick it (its spellings stay apart: each its own number) or split it (the spellings
 *  ticked are another entity, together). The state is read from the draft's step, never kept
 *  apart; every write gives the step back with exactly what the review decided. Pure functions. */

export type EntityStep = StepOf<"entity_column">;

/** The rules whose fix numbers entities. */
export const ENTITY_RULES = new Set(["A-20"]);

export const entityOf = (step: Step | null | undefined): EntityStep | null => (step?.op === "entity_column" ? step : null);

/** A fix that numbers entities, reviewed by cluster. */
export function isEntityFix(suggestion: Suggestion): boolean {
  return ENTITY_RULES.has(suggestion.rule) && entityOf(suggestion.step) !== null;
}

/** The step as the rule wrote it (the review cleared): its clusters do not move while reviewed. */
export function ruleEntityStep(step: EntityStep): EntityStep {
  return { ...step, not_entities: null, split: null };
}

/** The draft for the clusters: with this fix (turned on), the review cleared, nothing after it. */
export function stepsForClusters(steps: Step[], stepId: string): Step[] {
  const at = steps.findIndex((step) => step.id === stepId);
  return steps.slice(0, at + 1).map((step) => {
    const entity = step.id === stepId ? entityOf(step) : null;
    return entity ? ({ ...ruleEntityStep(entity), enabled: true } as Step) : step;
  });
}

/** The cluster is said not one entity: each of its spellings keeps its own number. */
export function isApart(cluster: Pick<EntityCluster, "index">, step: EntityStep | null): boolean {
  return (step?.not_entities ?? []).includes(cluster.index);
}

/** The step with this cluster said apart (or one entity again). */
export function withApart(step: EntityStep, cluster: Pick<EntityCluster, "index">, apart: boolean): EntityStep {
  const others = (step.not_entities ?? []).filter((index) => index !== cluster.index);
  if (apart) others.push(cluster.index);
  return { ...step, not_entities: others.length ? others.sort((a, b) => a - b) : null };
}

/** Every shown cluster said apart, or one entity again (Tick all / Untick all). */
export function withAllApart(step: EntityStep, clusters: Pick<EntityCluster, "index">[], apart: boolean): EntityStep {
  return clusters.reduce((next, cluster) => withApart(next, cluster, apart), step);
}

/** The spellings (by their index in the cluster) taken out of it: another entity, together. */
export function takenOut(cluster: Pick<EntityCluster, "index">, step: EntityStep | null): Set<number> {
  return new Set(step?.split?.[String(cluster.index)] ?? []);
}

/** The step with spelling `record` taken out of its cluster (or put back). Taking every spelling
 *  out is the cluster as it was: nothing written. */
export function withTaken(step: EntityStep, cluster: Pick<EntityCluster, "index" | "records">, record: number, out: boolean): EntityStep {
  const taken = takenOut(cluster, step);
  if (out) taken.add(record);
  else taken.delete(record);
  const split = { ...(step.split ?? {}) };
  const key = String(cluster.index);
  if (taken.size === 0 || taken.size >= cluster.records.length) delete split[key];
  else split[key] = [...taken].sort((a, b) => a - b);
  return { ...step, split: Object.keys(split).length ? split : null };
}

/** The entity numbers the clusters give now: one per cluster, two for a split one, one per
 *  spelling for a cluster said apart (the clusters shown count their spellings; those not shown,
 *  the records the step lists). */
export function entitiesNow(found: Pick<EntityClusters, "total_clusters" | "clusters">, step: EntityStep | null): number {
  const apart = step?.not_entities ?? [];
  const extra = apart.reduce((sum, index) => {
    const shown = found.clusters.find((cluster) => cluster.index === index);
    const records = shown?.records.length ?? step?.groups[index]?.length ?? 1;
    return sum + records - 1;
  }, 0);
  const split = Object.keys(step?.split ?? {}).filter((key) => !apart.includes(Number(key))).length;
  return found.total_clusters + extra + split;
}

/** The review changed something: a cluster said apart or split. */
export function isEntityReviewed(step: EntityStep | null): boolean {
  return Boolean(step && (step.not_entities?.length || Object.keys(step.split ?? {}).length));
}
