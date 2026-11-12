"""Exact and near duplicates of an image set, and the train/test leaks they cause (105, § 4.2).

- **R-VQ-09**, exact duplicates: the same `pixel_sha` (the decoded pixels, so a PNG saved again as
  BMP is still a copy). One image per group is kept, the first in the table's order.
- **R-VQ-10**, near duplicates: pHash distance <= 6 **and** dHash distance <= 10 (out of 64 bits).
  Crops, recompressions and bursts of a camera.
- Groups are the connected parts of the "is a copy of" graph (union-find): if A ~ B and B ~ C, the
  three are one group, so a split by group never puts A in train and C in test.

No N^2 comparison: the 64 bits of the pHash are cut into `t + 1` bands (7 bands of 9 or 10 bits
for the default `t = 6`). Two hashes at most `t` bits apart differ in at most
`t` bands, so at least one band is identical (pigeonhole principle): only the hashes sharing a band
are compared, with numpy's popcount. Close to linear on real sets, and the same code runs in
Pyodide (no thread, no subprocess).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from quantilo.vision.hashes import HASH_BITS

NEAR_PHASH = 6  # R-VQ-10
NEAR_DHASH = 10
BUCKET_LIMIT = 4_000  # a band shared by more hashes than this is compared in chunks

GroupKind = Literal["exact", "near"]


@dataclass(frozen=True)
class DuplicateGroup:
    """Images that are copies of each other. `rows` are table positions, `keep` the one kept
    (the first exact copy) when the group is exact, None for a near group (reviewed by a person)."""

    group: int
    kind: GroupKind
    rows: tuple[int, ...]
    keep: int | None


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[i] != root:  # path compression
            self.parent[i], i = root, int(self.parent[i])
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)  # the first row stays the root


def _bands(threshold: int) -> list[tuple[int, int]]:
    """(shift, width) of `threshold + 1` bands covering the 64 bits."""
    count = min(HASH_BITS, max(1, threshold + 1))
    edges = np.linspace(0, HASH_BITS, count + 1).round().astype(int)
    return [(int(lo), int(hi - lo)) for lo, hi in itertools.pairwise(edges)]


def _popcount(x: np.ndarray) -> np.ndarray:
    return np.bitwise_count(x.view(np.uint64)).astype(np.int64)


def near_pairs(
    phash: Sequence[int | None] | np.ndarray,
    dhash: Sequence[int | None] | np.ndarray,
    *,
    max_phash: int = NEAR_PHASH,
    max_dhash: int = NEAR_DHASH,
) -> list[tuple[int, int]]:
    """Pairs (i < j) of rows whose pHash differ by <= `max_phash` bits and dHash by
    <= `max_dhash`. Rows without hashes (not decoded) are left out."""
    p = np.asarray([0 if v is None else int(v) for v in phash], dtype=np.int64)
    d = np.asarray([0 if v is None else int(v) for v in dhash], dtype=np.int64)
    valid = np.asarray([a is not None and b is not None for a, b in zip(phash, dhash, strict=True)])
    rows = np.flatnonzero(valid)
    if len(rows) < 2:
        return []
    pv, dv = p[rows], d[rows]
    unsigned = pv.view(np.uint64)
    found: set[tuple[int, int]] = set()
    for shift, width in _bands(max_phash):
        mask = np.uint64((1 << width) - 1)
        key = (unsigned >> np.uint64(shift)) & mask
        order = np.argsort(key, kind="stable")
        sorted_key = key[order]
        starts = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1]])
        ends = np.r_[starts[1:], len(order)]
        for start, end in zip(starts, ends, strict=True):
            if end - start < 2:
                continue
            members = order[start:end]
            for chunk in range(0, len(members), BUCKET_LIMIT):
                head = members[chunk : chunk + BUCKET_LIMIT]
                # each head row against the members after it in the bucket
                for offset, i in enumerate(head):
                    others = members[chunk + offset + 1 :]
                    if len(others) == 0:
                        continue
                    close = (_popcount(np.bitwise_xor(pv[others], pv[i])) <= max_phash) & (
                        _popcount(np.bitwise_xor(dv[others], dv[i])) <= max_dhash
                    )
                    for j in others[close]:
                        a, b = int(rows[i]), int(rows[j])
                        found.add((a, b) if a < b else (b, a))
    return sorted(found)


def duplicate_groups(
    pixel_sha: Sequence[str | None],
    phash: Sequence[int | None] | np.ndarray,
    dhash: Sequence[int | None] | np.ndarray,
    *,
    max_phash: int = NEAR_PHASH,
    max_dhash: int = NEAR_DHASH,
    near: bool = True,
) -> list[DuplicateGroup]:
    """Exact groups (R-VQ-09), then near groups (R-VQ-10) joined by union-find. A near group that
    holds exact copies is one group of kind "near" (its exact copies are still flagged R-VQ-09).
    Groups are numbered from 0 in the order of their first row."""
    n = len(pixel_sha)
    uf = _UnionFind(n)
    by_sha: dict[str, int] = {}
    for i, sha in enumerate(pixel_sha):
        if sha is None:
            continue
        if sha in by_sha:
            uf.union(by_sha[sha], i)
        else:
            by_sha[sha] = i
    exact_of = exact_copies(pixel_sha)
    if near:
        for a, b in near_pairs(phash, dhash, max_phash=max_phash, max_dhash=max_dhash):
            uf.union(a, b)
    members: dict[int, list[int]] = {}
    for i in range(n):
        members.setdefault(uf.find(i), []).append(i)
    groups: list[DuplicateGroup] = []
    for root in sorted(members, key=lambda r: members[r][0]):
        rows = members[root]
        if len(rows) < 2:
            continue
        inside = set(rows)
        exact = [m for m in exact_of if m[0] in inside]
        only_exact = len(exact) == 1 and len(exact[0]) == len(rows)
        groups.append(
            DuplicateGroup(
                group=len(groups),
                kind="exact" if only_exact else "near",
                rows=tuple(rows),
                keep=rows[0] if only_exact else None,
            )
        )
    return groups


def exact_copies(pixel_sha: Sequence[str | None]) -> list[list[int]]:
    """Rows sharing a `pixel_sha`, first row first (R-VQ-09); the first one is kept."""
    seen: dict[str, list[int]] = {}
    for i, sha in enumerate(pixel_sha):
        if sha is not None:
            seen.setdefault(sha, []).append(i)
    return [rows for rows in seen.values() if len(rows) > 1]


def group_ids(n: int, groups: Sequence[DuplicateGroup]) -> np.ndarray:
    """One group per row for a grouped split (GroupKFold): the duplicate group, or the row's own
    singleton group. Copies always fall on the same side of a split."""
    ids = np.full(n, -1, dtype=np.int64)
    for g in groups:
        ids[list(g.rows)] = g.group
    alone = np.flatnonzero(ids < 0)
    ids[alone] = len(groups) + np.arange(len(alone))
    return ids


@dataclass(frozen=True)
class Leak:
    """R-VQ-11: a duplicate group with images in train and in an evaluation split."""

    group: int
    train_rows: tuple[int, ...]
    eval_rows: tuple[int, ...]  # the test (or val) images that have a copy in train


def leaks(groups: Sequence[DuplicateGroup], split: Sequence[str | None]) -> list[Leak]:
    """Groups spanning `train` and `test` or `val` (R-VQ-11)."""
    found = []
    for g in groups:
        train = tuple(r for r in g.rows if split[r] == "train")
        evaluation = tuple(r for r in g.rows if split[r] in ("test", "val"))
        if train and evaluation:
            found.append(Leak(g.group, train, evaluation))
    return found


def label_conflicts(
    groups: Sequence[DuplicateGroup], label: Sequence[str | None]
) -> list[DuplicateGroup]:
    """Groups whose images carry two labels or more (R-VQ-12)."""
    return [g for g in groups if len({label[r] for r in g.rows if label[r] is not None}) > 1]


__all__ = [
    "NEAR_DHASH",
    "NEAR_PHASH",
    "DuplicateGroup",
    "Leak",
    "duplicate_groups",
    "exact_copies",
    "group_ids",
    "label_conflicts",
    "leaks",
    "near_pairs",
]
