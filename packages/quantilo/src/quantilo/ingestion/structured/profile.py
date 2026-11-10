"""The profile of every path of a sample of records: types, presence, list lengths, keys.

A path is a tuple of segments from the record: a key, `[]` for the elements of a list, `*` for
the values of a dictionary (R-SST-12), whose keys are never paths.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from quantilo.ingestion.structured.readers import MAX_NESTING, too_deep

LIST = "[]"
VALUES = "*"
Path = tuple[str, ...]

KEY_CAP = 1_000  # distinct keys kept per object path (beyond: "more than 200" is known)
DISTINCT_CAP = 31  # R-SST-14 joins up to 30 distinct values
INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1
MAX_NUMBER_TEXT = 40  # longer numeric texts (R-LIM-03 integers) stay text

MAP_MIN_KEYS = 10  # R-SST-12
MAP_VALUE_LIKE = 0.8
MAP_MAX_KEYS = 200  # DuckDB's map_inference_threshold
MAP_PRESENCE = 0.1  # DuckDB's field_appearance_threshold

INT_TEXT = re.compile(r"\s*[-+]?\d+\s*")
NUM_TEXT = re.compile(r"\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\s*")
VALUE_LIKE = re.compile(
    r"[-+]?\d+(?:[.,]\d+)?"  # a number
    r"|\d{4}-\d{2}(?:-\d{2})?(?:[T ][\d:.]+(?:Z|[+-]\d{2}:?\d{2})?)?"  # an ISO date
    r"|\d{4}-?[QW]\d{1,2}"  # a quarter, a week
    r"|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}"  # a day-first or month-first date
    r"|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"  # a UUID
    r"|[0-9a-fA-F]{16,}"  # a hash
)


@dataclass
class PathStat:
    count: int = 0  # values seen (explicit nulls included)
    nulls: int = 0
    bools: int = 0
    ints: int = 0
    big_ints: int = 0  # beyond 64 bits
    floats: int = 0
    strings: int = 0
    numeric_strings: int = 0
    int_strings: int = 0
    objects: int = 0
    lists: int = 0
    lengths: list[int] = field(default_factory=list)
    keys: dict[str, int] = field(default_factory=dict)  # objects: how many contain each key
    key_overflow: bool = False
    key_total: int = 0  # key occurrences
    distinct: set[tuple[str, Any]] = field(default_factory=set)  # scalars, capped
    children: list[str] = field(default_factory=list)  # segments, in order of first sight

    @property
    def scalars(self) -> int:
        return self.bools + self.ints + self.big_ints + self.floats + self.strings

    @property
    def distinct_keys(self) -> int:
        return len(self.keys) + (1 if self.key_overflow else 0)

    def length_summary(self) -> dict[str, float]:
        if not self.lengths:
            return {}
        ordered = sorted(self.lengths)
        p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
        return {
            "min": ordered[0],
            "median": statistics.median(ordered),
            "p95": p95,
            "max": ordered[-1],
        }


class Profile:
    def __init__(self, maps: set[Path] | None = None) -> None:
        self.maps = maps or set()
        self.stats: dict[Path, PathStat] = {(): PathStat()}
        self.records = 0

    def add(self, record: Any) -> None:
        self.records += 1
        self.walk(record, ())

    def stat(self, path: Path) -> PathStat:
        found = self.stats.get(path)
        if found is None:
            found = self.stats[path] = PathStat()
            parent = self.stats[path[:-1]]
            parent.children.append(path[-1])
        return found

    def walk(self, value: Any, path: Path) -> None:
        if len(path) > MAX_NESTING:
            raise too_deep()
        stat = self.stats[()] if not path else self.stat(path)
        stat.count += 1
        if value is None:
            stat.nulls += 1
        elif isinstance(value, dict):
            stat.objects += 1
            self.census(stat, value)
            if path in self.maps:
                for item in value.values():
                    self.walk(item, (*path, VALUES))
            else:
                for key, item in value.items():
                    self.walk(item, (*path, key))
        elif isinstance(value, list):
            stat.lists += 1
            stat.lengths.append(len(value))
            for item in value:
                self.walk(item, (*path, LIST))
        else:
            self.scalar(stat, value)

    @staticmethod
    def census(stat: PathStat, value: dict[str, Any]) -> None:
        stat.key_total += len(value)
        for key in value:
            if key in stat.keys:
                stat.keys[key] += 1
            elif len(stat.keys) < KEY_CAP:
                stat.keys[key] = 1
            else:
                stat.key_overflow = True

    @staticmethod
    def scalar(stat: PathStat, value: Any) -> None:
        if isinstance(value, bool):
            stat.bools += 1
        elif isinstance(value, int):
            if INT64_MIN <= value <= INT64_MAX:
                stat.ints += 1
            else:
                stat.big_ints += 1
        elif isinstance(value, float):
            stat.floats += 1
        else:
            value = str(value)
            stat.strings += 1
            if NUM_TEXT.fullmatch(value) and len(value) <= MAX_NUMBER_TEXT:
                stat.numeric_strings += 1
                if INT_TEXT.fullmatch(value):
                    stat.int_strings += 1
        if len(stat.distinct) < DISTINCT_CAP:
            stat.distinct.add((type(value).__name__, value))

    def presence(self, path: Path) -> float:
        """Share of the parent objects holding this key (1.0 for list elements and map values)."""
        if not path or path[-1] in (LIST, VALUES):
            return 1.0
        parent = self.stats.get(path[:-1])
        if parent is None or not parent.objects:
            return 0.0
        return self.stats[path].count / parent.objects


def map_reason(stat: PathStat) -> str | None:
    """R-SST-12: why the objects at this path are a dictionary of values, not a record."""
    keys = stat.distinct_keys
    if keys > MAP_MAX_KEYS:
        many = f"more than {KEY_CAP:,}" if stat.key_overflow else f"{keys:,}"
        return f"{many} distinct keys (over {MAP_MAX_KEYS})"
    if keys < MAP_MIN_KEYS or not stat.objects:
        return None
    like = sum(1 for key in stat.keys if VALUE_LIKE.fullmatch(key))
    if like / keys >= MAP_VALUE_LIKE:
        return f"{like} of its {keys} keys look like values (dates, numbers, identifiers)"
    average = stat.key_total / keys / stat.objects
    if average < MAP_PRESENCE:
        return f"{keys} keys, each in {average:.0%} of the objects on average"
    return None


def is_value_map(value: dict[str, Any]) -> str | None:
    """R-SST-07: a root object that is itself a dictionary (dates → values)."""
    stat = PathStat(objects=1)
    Profile.census(stat, value)
    keys = stat.distinct_keys
    if keys > MAP_MAX_KEYS:
        return f"{keys:,} keys (over {MAP_MAX_KEYS})"
    if keys >= MAP_MIN_KEYS:
        like = sum(1 for key in stat.keys if VALUE_LIKE.fullmatch(key))
        if like / keys >= MAP_VALUE_LIKE:
            return f"{like} of its {keys} keys look like values (dates, numbers, identifiers)"
    return None


def user_path(path: Path) -> str:
    """The dotted path a user writes: list levels dropped, `*` for dictionary values."""
    return ".".join(segment for segment in path if segment != LIST)


def profile_records(
    records: Iterable[Any], forced_flat: set[str], rounds: int = 4
) -> tuple[Profile, dict[Path, str]]:
    """Profile, then fold the dictionaries found (R-SST-12) and profile again, until stable."""
    sample = list(records)
    maps: dict[Path, str] = {}
    profile = Profile()
    for _ in range(rounds + 1):
        profile = Profile(set(maps))
        for record in sample:
            profile.add(record)
        found = {}
        for path, stat in profile.stats.items():
            if (
                not path
                or path in maps
                or path[-1] in (LIST, VALUES)
                or user_path(path) in forced_flat
            ):
                continue  # never the record itself, nor list elements or map values
            if any(path[:i] in maps for i in range(1, len(path))):
                continue
            reason = map_reason(stat) if stat.objects else None
            if reason:
                found[path] = reason
        # keep only the outermost new dictionaries; the inner ones are found in the next round
        outer = {
            p: r for p, r in found.items() if not any(p[:i] in found for i in range(1, len(p)))
        }
        if not outer:
            break
        maps.update(outer)
    return profile, maps
