import type { FamilyRow } from "../../modules/clean/lineage";
import type { Project } from "./api";

/** A group of the home list (087): a project, or the datasets in none (`project: null`). */
export type ProjectGroup = { project: Project | null; rows: FamilyRow[] };

/** The key of a group, for the folded list and the drop target ("none" = Not in a project). */
export const groupKey = (project: Project | null) => project?.id ?? "none";

/** The family of a row: its original first, then the versions (065). Moving any of them moves all. */
export function familyIds(row: FamilyRow, rows: FamilyRow[]): string[] {
  const rootId = row.version?.parentId ?? row.dataset.id;
  const root = rows.find((candidate) => candidate.dataset.id === rootId);
  return root ? [root.dataset.id, ...root.children.map((child) => child.id)] : [row.dataset.id];
}

/** The rows (families in the order of the list) split by project: projects in the server's order
 *  (by name, R-PRJ-09), then Not in a project. A version always sits in its original's group
 *  (R-PRJ-04), whatever the list of members says. */
export function groupByProject(rows: FamilyRow[], projects: Project[]): ProjectGroup[] {
  const projectOfRoot = new Map<string, Project>();
  for (const project of projects) for (const id of project.dataset_ids) projectOfRoot.set(id, project);
  const groups = new Map<string, ProjectGroup>(projects.map((project) => [project.id, { project, rows: [] }]));
  const none: ProjectGroup = { project: null, rows: [] };
  for (const row of rows) {
    const project = projectOfRoot.get(row.version?.parentId ?? row.dataset.id);
    (project ? groups.get(project.id)! : none).rows.push(row);
  }
  return [...groups.values(), none];
}
