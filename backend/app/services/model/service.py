"""The model of a project (088, docs/features/088-modele-projet/spec.md).

The engine (`quantilo.model`) profiles the tables, proposes relations and builds the flat table;
this service adds the state: which datasets are the project's tables (one per family, R-REL-01),
the user's decisions on relations (accepted, rejected), the layout of the Model view, and the
lineage of the analysis datasets, which are imported like Segments' "Save as dataset"
(`new_upload_path`, `start_import`, and `after_import`, which copies the source columns' types).
Everything runs in the request (no thread): the detection is bounded (R-REL-10) and cached."""

import json
import re
import sqlite3
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import duckdb
from quantilo import model as engine
from quantilo.catalog import ColumnChange, ColumnView, to_view
from quantilo.cleaning.rules.tables import TableLink
from quantilo.errors import NotFoundError
from quantilo.ingestion.options import ImportOptions
from quantilo.model.build import Filter, JoinType, OutputColumn
from quantilo.model.dates import write_date_table
from quantilo.model.keys import Match, TableInput, TableProfile

from app.core import db
from app.core.errors import AppError
from app.core.storage import Workspace
from app.schemas import model as m
from app.services.columns import repository as columns_repo
from app.services.columns.service import ColumnService
from app.services.datasets.repository import DatasetRecord, now
from app.services.datasets.service import DatasetService
from app.services.projects.service import ProjectService

CACHED = 16  # detections and checks kept in memory


@dataclass
class Family:
    root: str
    versions: list[tuple[str, int]]  # (dataset id, seq), the original first
    used: str  # the dataset the model reads


@dataclass
class Stored:
    id: str
    child_root: str
    child_columns: list[str]
    parent_root: str
    parent_columns: list[str]
    join_type: JoinType
    status: str
    origin: str
    active: bool = True  # C1 (R-REL-13)
    match: Match = "exact"  # C1 (R-REL-14)


@dataclass
class Context:
    """A project's tables, as the model sees them right now."""

    project_id: str
    families: dict[str, Family]  # by root
    tables: dict[str, TableInput]  # by used dataset id
    records: dict[str, DatasetRecord]
    positions: dict[str, Any]
    versions: dict[str, str]
    used_of: dict[str, str] = field(default_factory=dict)  # root → used id
    root_of: dict[str, str] = field(default_factory=dict)  # used id → root

    def stamp(self) -> tuple[tuple[str, str | None, int], ...]:
        return tuple(
            (tid, self.records[tid].imported_at, self.records[tid].types_version)
            for tid in sorted(self.tables)
        )


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.\- ]+", "_", name).strip() or "dataset"


def _reasons(items: list[engine.Reason]) -> list[m.ReasonOut]:
    return [m.ReasonOut(code=item.code, text=item.text) for item in items]


def _check_out(check: engine.LinkCheck, warnings: list[engine.Reason]) -> m.CheckOut:
    return m.CheckOut(
        child_rows=check.child_rows,
        child_empty=check.child_empty,
        matched_rows=check.matched_rows,
        orphan_rows=check.orphan_rows,
        orphans=[m.ExampleOut(value=e.value, rows=e.rows) for e in check.orphans],
        child_distinct=check.child_distinct,
        parent_rows=check.parent_rows,
        parent_distinct=check.parent_distinct,
        duplicate_keys=check.duplicate_keys,
        duplicates=[m.ExampleOut(value=e.value, rows=e.rows) for e in check.duplicates],
        unused_keys=check.unused_keys,
        cardinality=check.cardinality,
        recommended=check.recommended,
        reasons=_reasons(check.reasons),
        warnings=_reasons(warnings),
        normalized_gain=check.normalized_gain,
        near_misses=[m.NearMissOut(value=a, key=b) for a, b in check.near_misses],
    )


class ModelService:
    def __init__(
        self,
        workspace: Workspace,
        datasets: DatasetService,
        columns: ColumnService,
        projects: ProjectService,
    ) -> None:
        self.workspace = workspace
        self.datasets = datasets
        self.columns = columns
        self.projects = projects
        self._detections: OrderedDict[tuple[Any, ...], engine.Detection] = OrderedDict()
        self._checks: OrderedDict[tuple[Any, ...], engine.LinkCheck] = OrderedDict()
        # analysis datasets being imported: their columns' types, by upload path
        self._pending: dict[str, list[ColumnChange]] = {}

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        connection = db.connect(self.workspace.db_path)
        try:
            yield connection
        finally:
            connection.close()

    # --- The project's tables ------------------------------------------------------------------

    def _built(self, connection: sqlite3.Connection) -> set[str]:
        """Analysis datasets, never a table of the model (R-REL-01); a generated date table
        (C1, R-BLD-08) is one: a dimension."""
        return {
            row[0]
            for row in connection.execute(
                "SELECT dataset_id FROM model_builds WHERE kind <> 'dates'"
            )
        }

    def _context(self, project_id: str) -> Context:
        project = self.projects.get(project_id)
        with self._db() as connection:
            built = self._built(connection)
            parents = {
                row["dataset_id"]: (row["parent_id"], int(row["seq"]))
                for row in connection.execute(
                    "SELECT dataset_id, parent_id, seq FROM clean_versions"
                )
            }
            saved = connection.execute(
                "SELECT positions, versions FROM project_models WHERE project_id = ?", (project_id,)
            ).fetchone()
        positions = json.loads(saved["positions"]) if saved else {}
        versions = json.loads(saved["versions"]) if saved else {}
        records: dict[str, DatasetRecord] = {}
        families: dict[str, Family] = {}
        for dataset_id in project.dataset_ids:
            if dataset_id in built:
                continue
            try:
                record = self.datasets.get(dataset_id)
            except NotFoundError:
                continue
            if record.status != "ready":
                continue
            records[dataset_id] = record
            root, seq = parents.get(dataset_id, (dataset_id, 0))
            family = families.setdefault(root, Family(root, [], root))
            family.versions.append((dataset_id, seq))
        context = Context(project_id, {}, {}, records, positions, versions)
        for root, family in families.items():
            family.versions.sort(key=lambda item: item[1])
            ids = [dataset_id for dataset_id, _ in family.versions]
            wanted = versions.get(root)
            family.used = wanted if wanted in ids else (root if root in ids else ids[0])
            context.families[root] = family
            context.used_of[root] = family.used
            context.root_of[family.used] = root
            views = self.columns.columns(family.used)
            record = records[family.used]
            context.tables[family.used] = TableInput(
                family.used, record.name, record.storage_path, views
            )
        return context

    def _detect(self, context: Context) -> engine.Detection:
        key = context.stamp()
        if key in self._detections:
            self._detections.move_to_end(key)
            return self._detections[key]
        found = engine.detect_tables(
            list(context.tables.values()), families={t: context.root_of[t] for t in context.tables}
        )
        self._detections[key] = found
        while len(self._detections) > CACHED:
            self._detections.popitem(last=False)
        return found

    def _check(
        self,
        context: Context,
        child_id: str,
        child_columns: list[str],
        parent_id: str,
        parent_columns: list[str],
        match: Match = "exact",
    ) -> engine.LinkCheck:
        child, parent = context.tables[child_id], context.tables[parent_id]
        key = (
            match,
            child_id,
            context.records[child_id].imported_at,
            context.records[child_id].types_version,
            tuple(child_columns),
            parent_id,
            context.records[parent_id].imported_at,
            context.records[parent_id].types_version,
            tuple(parent_columns),
        )
        if key in self._checks:
            self._checks.move_to_end(key)
            return self._checks[key]
        connection = duckdb.connect()
        try:
            found = engine.check(connection, child, child_columns, parent, parent_columns, match)
        except engine.ModelError as error:
            raise AppError(error.message, code=error.code, status_code=422) from None
        finally:
            connection.close()
        self._checks[key] = found
        while len(self._checks) > CACHED * 8:
            self._checks.popitem(last=False)
        return found

    def _stored(self, project_id: str) -> list[Stored]:
        with self._db() as connection:
            rows = connection.execute(
                "SELECT * FROM dataset_relations WHERE project_id = ? ORDER BY created_at, id",
                (project_id,),
            ).fetchall()
        return [
            Stored(
                id=row["id"],
                child_root=row["child_root"],
                child_columns=json.loads(row["child_columns"]),
                parent_root=row["parent_root"],
                parent_columns=json.loads(row["parent_columns"]),
                join_type=row["join_type"],
                status=row["status"],
                origin=row["origin"],
                active=bool(row["active"]),
                match=row["match"],
            )
            for row in rows
        ]

    # --- The model ----------------------------------------------------------------------------

    def model(self, project_id: str) -> m.ModelOut:
        started = time.perf_counter()
        context = self._context(project_id)
        found = self._detect(context)
        relations, engine_relations = self._relations(context, found)
        roles = engine.assign(found.profiles, engine_relations)
        link_columns = {
            (r.child_id, column) for r in engine_relations for column in r.child_columns
        }
        tables = [
            self._table_out(context, profile, roles.tables[profile.id], link_columns)
            for profile in found.profiles
        ]
        return m.ModelOut(
            project_id=project_id,
            tables=tables,
            relations=relations,
            facts=roles.facts,
            shape=roles.shape,
            positions={
                root: m.PositionIn.model_validate(position)
                for root, position in context.positions.items()
                if root in context.families
            },
            builds=self._builds(project_id, context),
            notes=_reasons(found.notes),
            elapsed_ms=round((time.perf_counter() - started) * 1000),
        )

    def _relations(
        self, context: Context, found: engine.Detection
    ) -> tuple[list[m.RelationOut], list[engine.Relation]]:
        """Accepted relations (checked again, R-REL-11) then the proposals nobody decided on."""
        out: list[m.RelationOut] = []
        as_engine: list[engine.Relation] = []
        decided: set[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = set()
        taken: set[tuple[str, tuple[str, ...]]] = set()
        for stored in self._stored(context.project_id):
            decided.add(
                (
                    stored.child_root,
                    tuple(stored.child_columns),
                    stored.parent_root,
                    tuple(stored.parent_columns),
                )
            )
            if stored.status != "accepted":
                continue
            child_id = context.used_of.get(stored.child_root)
            parent_id = context.used_of.get(stored.parent_root)
            if child_id is None or parent_id is None:
                continue
            taken.add((stored.child_root, tuple(stored.child_columns)))
            relation = self._accepted_out(
                context, stored, child_id, parent_id, self._why(found, stored, child_id, parent_id)
            )
            out.append(relation)
            if relation.problem is None and stored.active:  # R-REL-13: roles from active ones
                as_engine.append(
                    engine.Relation(
                        child_id=child_id,
                        child_columns=stored.child_columns,
                        parent_id=parent_id,
                        parent_columns=stored.parent_columns,
                        cardinality=relation.cardinality,
                        name_score=1,
                        inclusion=1,
                        coverage=1,
                        confidence="high",
                        score=1,
                    )
                )
        for proposal in found.relations:
            child_root = context.root_of[proposal.child_id]
            parent_root = context.root_of[proposal.parent_id]
            slot = (
                child_root,
                tuple(proposal.child_columns),
                parent_root,
                tuple(proposal.parent_columns),
            )
            if slot in decided or (child_root, tuple(proposal.child_columns)) in taken:
                continue
            check = self._check(
                context,
                proposal.child_id,
                proposal.child_columns,
                proposal.parent_id,
                proposal.parent_columns,
            )
            recommended = check.recommended
            join_type = recommended if recommended != "none" else "left"
            warnings = engine.warnings(
                check,
                join_type,
                context.tables[proposal.child_id].name,
                context.tables[proposal.parent_id].name,
            )
            out.append(
                m.RelationOut(
                    id=f"s-{proposal.key}",
                    status="suggested",
                    origin="detected",
                    child_id=proposal.child_id,
                    child_columns=proposal.child_columns,
                    parent_id=proposal.parent_id,
                    parent_columns=proposal.parent_columns,
                    cardinality=proposal.cardinality if check.cardinality != "n:m" else "n:m",
                    join_type=recommended,
                    recommended=recommended,
                    confidence=proposal.confidence,
                    score=proposal.score,
                    inclusion=proposal.inclusion,
                    reasons=_reasons(proposal.reasons),
                    check=_check_out(check, warnings),
                )
            )
            as_engine.append(proposal)
        return out, as_engine

    @staticmethod
    def _why(
        found: engine.Detection, stored: Stored, child_id: str, parent_id: str
    ) -> list[engine.Reason]:
        """Why an accepted relation was proposed; or that the user added it."""
        for proposal in found.relations:
            if (
                proposal.child_id == child_id
                and proposal.parent_id == parent_id
                and proposal.child_columns == stored.child_columns
                and proposal.parent_columns == stored.parent_columns
            ):
                return proposal.reasons
        return [engine.Reason("R-REL-01", "added by hand, checked on the data below")]

    def _accepted_out(
        self,
        context: Context,
        stored: Stored,
        child_id: str,
        parent_id: str,
        why: list[engine.Reason],
    ) -> m.RelationOut:
        child, parent = context.tables[child_id], context.tables[parent_id]
        base = {
            "id": stored.id,
            "status": "accepted",
            "origin": stored.origin,
            "child_id": child_id,
            "child_columns": stored.child_columns,
            "parent_id": parent_id,
            "parent_columns": stored.parent_columns,
            "join_type": stored.join_type,
            "confidence": None,
            "score": None,
            "inclusion": None,
            "active": stored.active,
            "match": stored.match,
        }
        try:
            check = self._check(
                context,
                child_id,
                stored.child_columns,
                parent_id,
                stored.parent_columns,
                stored.match,
            )
        except AppError as error:
            return m.RelationOut(
                **base,
                cardinality="n:1",
                recommended="left",
                reasons=[],
                check=None,
                problem=error.message,
            )
        warnings = engine.warnings(check, stored.join_type, child.name, parent.name)
        return m.RelationOut(
            **base,
            cardinality=check.cardinality,
            recommended=check.recommended,
            reasons=_reasons(why),
            check=_check_out(check, warnings),
        )

    def _table_out(
        self,
        context: Context,
        profile: TableProfile,
        role: engine.RoleOf,
        link_columns: set[tuple[str, str]],
    ) -> m.ModelTableOut:
        table = context.tables[profile.id]
        primary = profile.primary_key
        keys = set(primary.columns) if primary else set()
        own_links = {column for tid, column in link_columns if tid == profile.id}
        picks = set(engine.default_columns(table, keys | own_links, profile))
        family = context.families[context.root_of[profile.id]]
        columns = []
        for column in profile.columns:
            category = "key" if column.name in own_links else column.category
            columns.append(
                m.ModelColumnOut(
                    name=column.name,
                    type=column.type,
                    role=column.role,
                    category=category,
                    linkable=column.linkable,
                    excluded_by=(
                        m.ReasonOut(code=column.excluded_by.code, text=column.excluded_by.text)
                        if column.excluded_by
                        else None
                    ),
                    default_pick=column.name in picks,
                )
            )
        counts = {
            name: sum(1 for c in columns if c.category == name)
            for name in ("measure", "attribute", "date")
        }
        return m.ModelTableOut(
            id=profile.id,
            root_id=family.root,
            name=profile.name,
            rows=profile.rows,
            columns=columns,
            keys=[
                m.KeyOut(columns=key.columns, primary=key.primary, reasons=_reasons(key.reasons))
                for key in profile.keys
            ],
            measures=counts["measure"],
            attributes=counts["attribute"],
            dates=counts["date"],
            role=role.role,
            depth=role.depth,
            primary=role.primary,
            role_reasons=_reasons(role.reasons),
            versions=[m.VersionOut(id=tid, seq=seq) for tid, seq in family.versions],
        )

    # --- Decisions --------------------------------------------------------------------------------

    def _in_project(self, context: Context, dataset_id: str) -> str:
        """The root of a dataset of the project (409 when it is not one of its tables)."""
        if dataset_id in context.root_of:
            return context.root_of[dataset_id]
        for root, family in context.families.items():
            if any(tid == dataset_id for tid, _ in family.versions):
                return root
        self.datasets.get(dataset_id)  # 404 when it does not exist at all
        raise AppError(
            "This dataset is not one of the project's tables.",
            code="not_in_project",
            status_code=409,
        )

    def links_for(self, dataset_id: str) -> list[TableLink]:
        """N32: the accepted relations of every project the dataset is a table of, with the
        dataset itself on its side (the version Clean is opened on, which may not be the one the
        model uses) and the model's version of the other table. Clean checks them (T-01, T-02)."""
        links: list[TableLink] = []
        for project in self.projects.list_all():
            if dataset_id not in project.dataset_ids:
                continue
            context = self._context(project.id)
            root = next(
                (
                    root
                    for root, family in context.families.items()
                    if any(tid == dataset_id for tid, _ in family.versions)
                ),
                None,
            )
            if root is None:  # an analysis dataset, or not ready
                continue
            record = context.records[dataset_id]
            mine = TableInput(
                dataset_id, record.name, record.storage_path, self.columns.columns(dataset_id)
            )
            for stored in self._stored(project.id):
                if stored.status != "accepted":
                    continue
                if stored.child_root == root and stored.parent_root in context.used_of:
                    parent = context.tables[context.used_of[stored.parent_root]]
                    child = mine
                elif stored.parent_root == root and stored.child_root in context.used_of:
                    child = context.tables[context.used_of[stored.child_root]]
                    parent = mine
                else:
                    continue
                links.append(
                    TableLink(
                        child,
                        stored.child_columns,
                        parent,
                        stored.parent_columns,
                        stored.match,
                        stored.id,
                        project.name,
                    )
                )
        return links

    def check(self, project_id: str, body: m.CheckIn) -> m.CheckOut:
        context = self._context(project_id)
        child_id = context.used_of[self._in_project(context, body.child_id)]
        parent_id = context.used_of[self._in_project(context, body.parent_id)]
        found = self._check(
            context, child_id, body.child_columns, parent_id, body.parent_columns, body.match
        )
        join_type: JoinType = body.join_type or (
            found.recommended if found.recommended != "none" else "left"
        )
        warnings = engine.warnings(
            found, join_type, context.tables[child_id].name, context.tables[parent_id].name
        )
        return _check_out(found, warnings)

    def save_relation(self, project_id: str, body: m.RelationIn) -> m.ModelOut:
        context = self._context(project_id)
        child_root = self._in_project(context, body.child_id)
        parent_root = self._in_project(context, body.parent_id)
        if child_root == parent_root:
            raise AppError(
                "A table can't be related to itself (or to its own version) (R-REL-01).",
                code="same_table",
                status_code=422,
            )
        join_type: JoinType = body.join_type or "left"
        if body.status == "accepted":
            found = self._check(
                context,
                context.used_of[child_root],
                body.child_columns,
                context.used_of[parent_root],
                body.parent_columns,
                body.match,
            )
            if body.join_type is None and found.recommended != "none":
                join_type = cast(JoinType, found.recommended)
        stamp = now()
        with self._db() as connection, connection:
            connection.execute(
                "INSERT INTO dataset_relations (id, project_id, child_root, child_columns, "
                "parent_root, parent_columns, join_type, status, origin, created_at, updated_at, "
                "active, match) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (project_id, child_root, child_columns, parent_root, parent_columns) "
                "DO UPDATE SET join_type = excluded.join_type, status = excluded.status, "
                "updated_at = excluded.updated_at, active = excluded.active, "
                "match = excluded.match",
                (
                    uuid.uuid4().hex[:12],
                    project_id,
                    child_root,
                    json.dumps(body.child_columns),
                    parent_root,
                    json.dumps(body.parent_columns),
                    join_type,
                    body.status,
                    body.origin,
                    stamp,
                    stamp,
                    int(body.active),
                    body.match,
                ),
            )
        return self.model(project_id)

    def _stored_one(self, project_id: str, relation_id: str) -> Stored:
        found = next((s for s in self._stored(project_id) if s.id == relation_id), None)
        if found is None or found.status != "accepted":
            raise NotFoundError("Relation")
        return found

    def update_relation(
        self, project_id: str, relation_id: str, body: m.RelationPatch
    ) -> m.ModelOut:
        stored = self._stored_one(project_id, relation_id)
        child_columns = body.child_columns or stored.child_columns
        parent_columns = body.parent_columns or stored.parent_columns
        match = body.match or stored.match
        active = stored.active if body.active is None else body.active
        if body.child_columns or body.parent_columns or body.match:
            context = self._context(project_id)
            child_id = context.used_of.get(stored.child_root)
            parent_id = context.used_of.get(stored.parent_root)
            if child_id and parent_id:
                self._check(context, child_id, child_columns, parent_id, parent_columns, match)
        with self._db() as connection, connection:
            try:
                connection.execute(
                    "UPDATE dataset_relations SET child_columns = ?, parent_columns = ?, "
                    "join_type = ?, active = ?, match = ?, updated_at = ? WHERE id = ?",
                    (
                        json.dumps(child_columns),
                        json.dumps(parent_columns),
                        body.join_type or stored.join_type,
                        int(active),
                        match,
                        now(),
                        relation_id,
                    ),
                )
            except sqlite3.IntegrityError:
                raise AppError(
                    "This relation already exists.", code="relation_exists", status_code=409
                ) from None
        return self.model(project_id)

    def delete_relation(self, project_id: str, relation_id: str) -> m.ModelOut:
        """Deleting an accepted relation makes it a proposal again (if the data still says so)."""
        self._stored_one(project_id, relation_id)
        with self._db() as connection, connection:
            connection.execute("DELETE FROM dataset_relations WHERE id = ?", (relation_id,))
        return self.model(project_id)

    def save_layout(self, project_id: str, body: m.LayoutIn) -> m.ModelOut:
        context = self._context(project_id)
        positions = dict(context.positions)
        for root, position in body.positions.items():
            if root in context.families:
                positions[root] = position.model_dump()
        versions = dict(context.versions)
        if body.versions is not None:
            for root, dataset_id in body.versions.items():
                family = context.families.get(root)
                if family is None or dataset_id not in {tid for tid, _ in family.versions}:
                    raise AppError(
                        "This version is not in the family.", code="not_in_project", status_code=409
                    )
                if dataset_id == root:
                    versions.pop(root, None)
                else:
                    versions[root] = dataset_id
        with self._db() as connection, connection:
            connection.execute(
                "INSERT INTO project_models (project_id, positions, versions, updated_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT (project_id) DO UPDATE SET "
                "positions = excluded.positions, versions = excluded.versions, "
                "updated_at = excluded.updated_at",
                (project_id, json.dumps(positions), json.dumps(versions), now()),
            )
        return self.model(project_id)

    # --- The analysis dataset ------------------------------------------------------------------

    def _links(self, context: Context) -> list[engine.Link]:
        links = []
        for stored in self._stored(context.project_id):
            child_id = context.used_of.get(stored.child_root)
            parent_id = context.used_of.get(stored.parent_root)
            if stored.status == "accepted" and child_id and parent_id:
                links.append(
                    engine.Link(
                        child_id,
                        stored.child_columns,
                        parent_id,
                        stored.parent_columns,
                        stored.join_type,
                        stored.id,
                        stored.active,
                        stored.match,
                    )
                )
        return links

    def _plan(self, context: Context, body: m.BuildIn) -> tuple[engine.Plan, str]:
        found = self._detect(context)
        links = self._links(context)
        fact_id = body.fact_id
        if fact_id is None:
            roles = engine.assign(
                found.profiles,
                [
                    engine.Relation(
                        link.child_id,
                        link.child_columns,
                        link.parent_id,
                        link.parent_columns,
                        "n:1",
                        1,
                        1,
                        1,
                        "high",
                        1,
                    )
                    for link in links
                ],
            )
            if not roles.facts:
                raise AppError(
                    "Accept at least one relation first: the analysis dataset starts from a fact "
                    "table and its relations (R-BLD-01).",
                    code="no_relation",
                    status_code=409,
                )
            fact_id = roles.facts[0]
        elif fact_id not in context.tables:
            fact_id = context.used_of[self._in_project(context, fact_id)]
        profiles = {profile.id: profile for profile in found.profiles}
        filters = [
            Filter(
                context.used_of.get(
                    context.root_of.get(item.table_id, item.table_id), item.table_id
                ),
                item.column,
                item.op,
                list(item.values),
            )
            for item in body.filters
        ]
        try:
            plan = engine.plan(
                fact_id, context.tables, links, body.picks or None, profiles, filters
            )
        except engine.ModelError as error:
            raise AppError(error.message, code=error.code, status_code=422) from None
        return plan, fact_id

    def preview(self, project_id: str, body: m.BuildIn) -> m.PreviewOut:
        context = self._context(project_id)
        plan, fact_id = self._plan(context, body)
        connection = duckdb.connect()
        try:
            result = engine.preview(connection, plan)
        except engine.ModelError as error:
            raise AppError(error.message, code=error.code, status_code=422) from None
        finally:
            connection.close()
        return self._preview_out(context, fact_id, result)

    def _preview_out(self, context: Context, fact_id: str, result: engine.Preview) -> m.PreviewOut:
        steps = []
        for step in result.steps:
            child, parent = context.tables[step.link.child_id], context.tables[step.link.parent_id]
            warnings = engine.warnings(step.check, step.link.join_type, child.name, parent.name)
            steps.append(
                m.StepOut(
                    relation_id=step.link.id or "",
                    child_id=step.link.child_id,
                    parent_id=step.link.parent_id,
                    prefix=step.prefix,
                    join_type=step.link.join_type,
                    columns=step.columns,
                    check=_check_out(step.check, warnings),
                )
            )
        return m.PreviewOut(
            fact_id=fact_id,
            rows_before=result.rows_before,
            rows_after=result.rows_after,
            columns=[
                m.OutputColumnOut(name=c.name, table_id=c.table_id, column=c.column)
                for c in result.columns
            ],
            steps=steps,
            refused=_reasons(result.refused),
            skipped=_reasons(result.skipped),
            name=f"{context.tables[fact_id].name} · analysis",
            filtered_out=result.filtered_out,
        )

    def build(self, project_id: str, body: m.BuildIn) -> m.CreatedOut:
        context = self._context(project_id)
        plan, fact_id = self._plan(context, body)
        name = body.name or f"{context.tables[fact_id].name} · analysis"
        target = self.datasets.new_upload_path(f"{_safe_name(name)}.parquet")
        report = self._write_flat(context, plan, target)
        spec = {
            "fact_root": context.root_of[fact_id],
            "links": [
                {
                    "child_root": context.root_of[step.link.child_id],
                    "child_columns": step.link.child_columns,
                    "parent_root": context.root_of[step.link.parent_id],
                    "parent_columns": step.link.parent_columns,
                    "join_type": step.link.join_type,
                    "relation_id": step.link.id,
                }
                for step in plan.steps
            ],
            "picks": {context.root_of[step.link.parent_id]: step.columns for step in plan.steps},
            "filters": [
                {**item.model_dump(), "table_id": context.root_of.get(item.table_id, item.table_id)}
                for item in body.filters
            ],
        }
        dataset_id = self._import(context, target, name, plan.columns)
        self._record(dataset_id, project_id, "flat", spec, context, self._sources(plan), report)
        return m.CreatedOut(dataset_id=dataset_id)

    def _sources(self, plan: engine.Plan) -> list[str]:
        return sorted({plan.fact.id, *(step.link.parent_id for step in plan.steps)})

    def _write_flat(self, context: Context, plan: engine.Plan, target: Path) -> dict[str, Any]:
        connection = duckdb.connect()
        try:
            result = engine.write_flat(connection, plan, target)
        except engine.ModelError as error:
            raise AppError(error.message, code=error.code, status_code=422) from None
        finally:
            connection.close()
        return {
            "rows_before": result.rows_before,
            "rows_after": result.rows_after,
            "columns": len(result.columns),
        }

    def _changes(self, context: Context, columns: list[OutputColumn]) -> list[ColumnChange]:
        """The source columns' types, roles and orders for the new dataset (R-BLD-04)."""
        changes = []
        for column in columns:
            table = context.tables.get(column.table_id)
            view = (
                next((v for v in table.columns if v.name == column.column), None) if table else None
            )
            if view is None:
                continue
            changes.append(
                ColumnChange(
                    name=column.name,
                    type=view.type,
                    role=view.role or "",
                    order=view.order or [],
                    excluded=view.excluded,
                )
            )
        return changes

    def _import(
        self, context: Context, target: Path, name: str, columns: list[OutputColumn]
    ) -> str:
        self._pending[str(self.datasets.resolve(str(target)))] = self._changes(context, columns)
        record, _ = self.datasets.start_import(str(target), name, ImportOptions())
        self.projects.move([record.id], context.project_id)
        return record.id

    def _record(
        self,
        dataset_id: str,
        project_id: str,
        kind: str,
        spec: dict[str, Any],
        context: Context,
        sources: list[str],
        report: dict[str, Any],
    ) -> None:
        stamps = [
            {
                "id": source_id,
                "root": context.root_of[source_id],
                "name": context.records[source_id].name,
                "imported_at": context.records[source_id].imported_at,
                "types_version": context.records[source_id].types_version,
            }
            for source_id in sources
        ]
        stamp = now()
        with self._db() as connection, connection:
            connection.execute(
                "INSERT INTO model_builds (dataset_id, project_id, kind, spec, sources, report, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (dataset_id) DO UPDATE SET spec = excluded.spec, "
                "sources = excluded.sources, report = excluded.report, "
                "updated_at = excluded.updated_at",
                (
                    dataset_id,
                    project_id,
                    kind,
                    json.dumps(spec),
                    json.dumps(stamps),
                    json.dumps(report),
                    stamp,
                    stamp,
                ),
            )

    def aggregate(self, project_id: str, body: m.AggregateIn) -> m.CreatedOut:
        context = self._context(project_id)
        link = next((x for x in self._links(context) if x.id == body.relation_id), None)
        if link is None:
            raise NotFoundError("Relation")
        child, parent = context.tables[link.child_id], context.tables[link.parent_id]
        measures = (
            [engine.Measure(item.column, item.function) for item in body.measures]
            if body.measures is not None
            else engine.default_measures(child, set(link.child_columns))
        )
        name = body.name or f"{parent.name} · {child.name} per {parent.name}"
        target = self.datasets.new_upload_path(f"{_safe_name(name)}.parquet")
        rows, added = self._write_aggregate(child, link, parent, measures, target)
        columns = [OutputColumn(v.name, parent.id, v.name) for v in parent.columns]
        dataset_id = self._import(context, target, name, columns)
        spec = {
            "child_root": context.root_of[child.id],
            "child_columns": link.child_columns,
            "parent_root": context.root_of[parent.id],
            "parent_columns": link.parent_columns,
            "relation_id": link.id,
            "measures": [{"column": x.column, "function": x.function} for x in measures],
        }
        report = {"rows": rows, "added": [column.name for column in added]}
        self._record(
            dataset_id,
            project_id,
            "aggregate",
            spec,
            context,
            sorted({child.id, parent.id}),
            report,
        )
        return m.CreatedOut(dataset_id=dataset_id)

    def _write_aggregate(
        self,
        child: TableInput,
        link: engine.Link,
        parent: TableInput,
        measures: list[engine.Measure],
        target: Path,
    ) -> tuple[int, list[OutputColumn]]:
        connection = duckdb.connect()
        try:
            return engine.aggregate(connection, child, link, parent, measures, target)
        except engine.ModelError as error:
            raise AppError(error.message, code=error.code, status_code=422) from None
        finally:
            connection.close()

    # --- A generated date table (C1, R-BLD-08) ---------------------------------------------------

    def _write_dates(self, fact: TableInput, column: str, target: Path) -> dict[str, Any]:
        connection = duckdb.connect()
        try:
            made = write_date_table(connection, fact, column, target)
        except engine.ModelError as error:
            raise AppError(error.message, code=error.code, status_code=422) from None
        finally:
            connection.close()
        return {"rows": made.rows, "first": made.first.isoformat(), "last": made.last.isoformat()}

    def dates(self, project_id: str, body: m.DatesIn) -> m.CreatedOut:
        """One row per day of the column's range, filed in the project and related to the table
        by the date (an accepted relation, made by the user's click)."""
        context = self._context(project_id)
        root = self._in_project(context, body.dataset_id)
        fact = context.tables[context.used_of[root]]
        name = body.name or f"{fact.name} · dates"
        target = self.datasets.new_upload_path(f"{_safe_name(name)}.parquet")
        report = self._write_dates(fact, body.column, target)
        dataset_id = self._import(context, target, name, [])
        spec = {"fact_root": root, "column": body.column}
        self._record(dataset_id, project_id, "dates", spec, context, [fact.id], report)
        stamp = now()
        with self._db() as connection, connection:
            connection.execute(
                "INSERT OR IGNORE INTO dataset_relations (id, project_id, child_root, "
                "child_columns, parent_root, parent_columns, join_type, status, origin, "
                "created_at, updated_at, active, match) "
                "VALUES (?, ?, ?, ?, ?, ?, 'left', 'accepted', 'user', ?, ?, 1, 'exact')",
                (
                    uuid.uuid4().hex[:12],
                    project_id,
                    root,
                    json.dumps([body.column]),
                    dataset_id,
                    json.dumps(["date"]),
                    stamp,
                    stamp,
                ),
            )
        return m.CreatedOut(dataset_id=dataset_id)

    # --- Lineage: out of date, rebuild ---------------------------------------------------------

    def _builds(self, project_id: str, context: Context) -> list[m.BuildOut]:
        with self._db() as connection:
            rows = connection.execute(
                "SELECT * FROM model_builds WHERE project_id = ? ORDER BY created_at", (project_id,)
            ).fetchall()
        stored = {s.id: s for s in self._stored(project_id) if s.status == "accepted"}
        out = []
        for row in rows:
            if row["kind"] == "join":  # 089: said on the joined dataset's page (C1)
                continue
            try:
                record = self.datasets.get(row["dataset_id"])
            except NotFoundError:
                continue
            why, can = self._staleness(row, context, stored)
            out.append(
                m.BuildOut(
                    dataset_id=record.id,
                    name=record.name,
                    kind=row["kind"],
                    status=record.status,
                    rows=record.row_count,
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    out_of_date=why,
                    can_rebuild=can,
                )
            )
        return out

    def _staleness(
        self, row: sqlite3.Row, context: Context, stored: dict[str, Stored]
    ) -> tuple[list[str], bool]:
        """R-BLD-05: a source reimported or retyped, a relation changed or deleted."""
        why: list[str] = []
        can = True
        for source in json.loads(row["sources"]):
            used = context.used_of.get(source["root"])
            if used is None:
                why.append(f"{source['name']} is no longer in the project")
                can = False
                continue
            record = context.records[used]
            if used != source["id"]:
                why.append(f"{record.name}: another version is used")
            elif record.imported_at != source["imported_at"]:
                why.append(f"{record.name} was reimported")
            elif record.types_version != source["types_version"]:
                why.append(f"{record.name}: column types changed")
        spec = json.loads(row["spec"])
        links = spec.get("links") or ([spec] if row["kind"] == "aggregate" else [])
        for link in links:
            relation = stored.get(link.get("relation_id") or "")
            if relation is None:
                why.append("a relation it uses was deleted")
                can = can and row["kind"] == "flat"
            elif (
                relation.join_type != link.get("join_type", relation.join_type)
                or relation.child_columns != link["child_columns"]
                or relation.parent_columns != link["parent_columns"]
            ):
                why.append("a relation it uses changed")
        return why, can

    def rebuild(self, project_id: str, dataset_id: str) -> m.CreatedOut:
        """Write the analysis dataset again with the current data and relations, then reimport it
        (the previous version is kept, 031)."""
        with self._db() as connection:
            row = connection.execute(
                "SELECT * FROM model_builds WHERE dataset_id = ? AND project_id = ?",
                (dataset_id, project_id),
            ).fetchone()
        if row is None:
            raise NotFoundError("Analysis dataset")
        record = self.datasets.get(dataset_id)
        context = self._context(project_id)
        spec = json.loads(row["spec"])
        target = Path(record.source_path)
        partial = target.with_name(target.name + ".partial")
        if row["kind"] == "flat":
            fact_id = context.used_of.get(spec["fact_root"])
            if fact_id is None:
                raise AppError(
                    "Its fact table is no longer in the project.",
                    code="not_in_project",
                    status_code=409,
                )
            picks = {
                context.used_of[root]: columns
                for root, columns in spec.get("picks", {}).items()
                if root in context.used_of
            }
            filters = [m.FilterIn.model_validate(item) for item in spec.get("filters", [])]
            plan, _ = self._plan(context, m.BuildIn(fact_id=fact_id, picks=picks, filters=filters))
            report = self._write_flat(context, plan, partial)
            columns, sources = plan.columns, self._sources(plan)
            spec["links"] = [
                {
                    "child_root": context.root_of[step.link.child_id],
                    "child_columns": step.link.child_columns,
                    "parent_root": context.root_of[step.link.parent_id],
                    "parent_columns": step.link.parent_columns,
                    "join_type": step.link.join_type,
                    "relation_id": step.link.id,
                }
                for step in plan.steps
            ]
        elif row["kind"] == "dates":  # C1 (R-BLD-08): the days of the fact's dates again
            fact_id = context.used_of.get(spec["fact_root"])
            if fact_id is None:
                raise AppError(
                    "Its table is no longer in the project.", code="not_in_project", status_code=409
                )
            report = self._write_dates(context.tables[fact_id], spec["column"], partial)
            columns, sources = [], [fact_id]
        else:
            link = next((x for x in self._links(context) if x.id == spec.get("relation_id")), None)
            if link is None:
                raise AppError(
                    "The relation it aggregates was deleted.", code="no_relation", status_code=409
                )
            child, parent = context.tables[link.child_id], context.tables[link.parent_id]
            measures = [engine.Measure(x["column"], x["function"]) for x in spec["measures"]]
            rows, added = self._write_aggregate(child, link, parent, measures, partial)
            report = {"rows": rows, "added": [column.name for column in added]}
            columns = [OutputColumn(v.name, parent.id, v.name) for v in parent.columns]
            sources = sorted({child.id, parent.id})
        partial.replace(target)
        self._pending[str(self.datasets.resolve(str(target)))] = self._changes(context, columns)
        self.datasets.reimport(dataset_id)
        self._record(dataset_id, project_id, row["kind"], spec, context, sources, report)
        return m.CreatedOut(dataset_id=dataset_id)

    # --- After an import ------------------------------------------------------------------------

    def after_import(self, dataset_id: str, storage_path: str) -> None:
        """An analysis dataset gets its sources' types and roles where the detection differs."""
        record = self.datasets.get(dataset_id)
        changes = self._pending.pop(record.source_path, None)
        if not changes:
            return
        with self._db() as connection:
            current = {
                r.name: to_view(r) for r in columns_repo.list_columns(connection, dataset_id)
            }
        needed = [change for change in changes if self._differs(current.get(change.name), change)]
        if needed:
            self.columns.apply(dataset_id, storage_path, needed)

    @staticmethod
    def _differs(view: ColumnView | None, change: ColumnChange) -> bool:
        if view is None:
            return False
        return (
            view.type != change.type
            or (view.role or "") != (change.role or "")
            or view.excluded != change.excluded
            or (view.order or []) != (change.order or [])
        )
