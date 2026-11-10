"""Facts, dimensions, star or snowflake (088, R-ROLE-01 to 03)."""

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from quantilo.model.keys import Reason, TableProfile
from quantilo.model.relations import Relation

TableRole = Literal["fact", "dimension", "unrelated"]
Shape = Literal["star", "snowflake", "none"]


@dataclass
class RoleOf:
    role: TableRole
    depth: int | None = None  # links from the nearest fact (a fact: 0)
    primary: bool = False  # the fact in the centre, picked by default to build (R-ROLE-02)
    reasons: list[Reason] = field(default_factory=list)


@dataclass
class Roles:
    tables: dict[str, RoleOf]
    facts: list[str]  # primary first
    shape: Shape


def assign(profiles: list[TableProfile], relations: list[Relation]) -> Roles:
    """R-ROLE-01: pointed to = dimension; points and is never pointed to = fact."""
    links = [r for r in relations if r.cardinality != "n:m"]
    rows = {p.id: p.rows for p in profiles}
    names = {p.id: p.name for p in profiles}
    outgoing: dict[str, list[Relation]] = {p.id: [] for p in profiles}
    incoming: dict[str, list[Relation]] = {p.id: [] for p in profiles}
    for relation in links:
        outgoing[relation.child_id].append(relation)
        incoming[relation.parent_id].append(relation)
    roles: dict[str, RoleOf] = {}
    for profile in profiles:
        out, into = outgoing[profile.id], incoming[profile.id]
        if not out and not into:
            roles[profile.id] = RoleOf(
                "unrelated", reasons=[Reason("R-ROLE-01", "no relation with another table")]
            )
        elif into:
            pointers = sorted({names[r.child_id] for r in into})
            roles[profile.id] = RoleOf(
                "dimension",
                reasons=[
                    Reason("R-ROLE-01", f"described by its key, used by {', '.join(pointers)}")
                ],
            )
        else:
            targets = sorted({names[r.parent_id] for r in out})
            roles[profile.id] = RoleOf(
                "fact",
                reasons=[
                    Reason("R-ROLE-01", f"refers to {', '.join(targets)} and nothing refers to it")
                ],
            )
    facts = [tid for tid, role in roles.items() if role.role == "fact"]
    if not facts:
        related = [tid for tid, role in roles.items() if role.role != "unrelated"]
        if related:
            # every related table is pointed to (a cycle): the biggest one is the fact
            biggest = max(related, key=lambda tid: (rows[tid], len(outgoing[tid])))
            roles[biggest] = RoleOf(
                "fact",
                reasons=[
                    Reason("R-ROLE-01", "every table is referred to: the biggest one is the fact")
                ],
            )
            facts = [biggest]
    facts.sort(key=lambda tid: (-rows[tid], -len(outgoing[tid]), names[tid]))
    if facts:
        roles[facts[0]].primary = True
        if len(facts) > 1:
            roles[facts[0]].reasons.append(
                Reason("R-ROLE-02", "the biggest fact table: in the centre and built by default")
            )
    _depths(roles, facts, outgoing)
    depths = [role.depth for role in roles.values() if role.role == "dimension" and role.depth]
    shape: Shape = (
        "none" if not facts or not depths else ("star" if max(depths) == 1 else "snowflake")
    )
    return Roles(roles, facts, shape)


def _depths(
    roles: dict[str, RoleOf], facts: list[str], outgoing: dict[str, list[Relation]]
) -> None:
    queue: deque[str] = deque()
    for fact in facts:
        roles[fact].depth = 0
        queue.append(fact)
    while queue:
        current = queue.popleft()
        depth = roles[current].depth or 0
        for relation in outgoing[current]:
            target = roles[relation.parent_id]
            if target.depth is None:
                target.depth = depth + 1
                queue.append(relation.parent_id)
