"""Durations in Compare (062): survival curves when the data has a start and an end, or a
duration and an event; status transitions when the same entity is seen several times.

Every reading is a written rule with its reason (no model guesses which column is what):

- R-SRV-01  a duration (role "duration", or a name such as tenure, days, months, lifetime) and an
            event (yes/no or 0/1 column named churn, event, death, failure, cancel, exited...):
            survival of the duration, the event closing it, the others still followed (censored).
- R-SRV-02  two dates, one named like a start (start, begin, signup, opened, created, since...),
            the other like an end (end, stop, close, cancel, churn, until...): the duration is
            the days between them; an empty end means still running, followed until the latest
            date in the data (censored).
- R-SRV-03  the event's value: a boolean's True, a 0/1's 1, or the level whose name says it
            happened (yes, true, churned, dead, failed...). A name with "censor" reverses it
            (1 = censored). Otherwise the rarer level, said so: check it.
- R-SRV-04  groups: at most 6 curves, the most frequent levels; the log-rank test compares them.
- R-MKV-01  an entity key that repeats (role identifier or *_key, or a name with "id"), a date and
            a status (role status, or a category of 2 to 12 levels named status, state, stage,
            plan, tier, segment...): consecutive observations of one entity are transitions.
- R-MKV-02  at most 12 states, the most frequent; rows of other states are not paired.
- R-MKV-03  a state never left in the data (every exit stays in it, or none observed) is read as
            absorbing; the chain then gives the expected steps before reaching it.
"""

import re
from dataclasses import dataclass, field
from typing import Literal

import duckdb
import numpy as np

from quantilo.compare import survival as sv
from quantilo.compare import transitions as tr
from quantilo.compare.analyze import (
    CompareColumn,
    NotComparableError,
    label_sql,
    number_label_sql,
    number_sql,
)
from quantilo.typing.parsers import typed_sql
from quantilo.typing.types import AnalysisType, Parser

MAX_CURVES = 6  # R-SRV-04
MAX_STATES = 12  # R-MKV-02
MIN_ROWS = 10
MAX_ROWS = 2_000_000
ALPHA = 0.05

NUMBERS = {"numeric_continuous", "numeric_discrete"}
LEVELS = {"categorical_nominal", "categorical_ordinal", "boolean", "numeric_discrete"}
CATEGORIES = {"categorical_nominal", "categorical_ordinal", "boolean"}

DURATION_WORDS = {
    "tenure", "duration", "lifetime", "days", "weeks", "months", "years", "time", "survival",
    "followup", "follow", "length", "anciennete", "duree", "seniority",
}  # fmt: skip
EVENT_WORDS = {
    "churn", "churned", "event", "death", "dead", "died", "failure", "failed", "fail", "cancel",
    "cancelled", "canceled", "attrition", "exited", "exit", "left", "default", "status",
    "outcome", "censor", "censored", "deceased", "relapse", "resigned", "quit", "lapsed",
}  # fmt: skip
START_WORDS = {
    "start", "begin", "began", "signup", "opened", "open", "created", "creation", "since",
    "joined", "join", "entry", "enrolled", "hire", "hired", "subscribed", "activation", "from",
    "debut", "ouverture",
}  # fmt: skip
END_WORDS = {
    "end", "ended", "stop", "stopped", "close", "closed", "cancel", "cancelled", "canceled",
    "churn", "churned", "until", "termination", "terminated", "exit", "left", "leave", "to",
    "expiry", "expired", "death", "fin", "cloture", "resiliation",
}  # fmt: skip
HAPPENED = {
    "yes", "true", "1", "y", "oui", "churned", "churn", "dead", "died", "death", "deceased",
    "failed", "failure", "fail", "event", "cancelled", "canceled", "left", "exited", "default",
    "defaulted", "closed", "lost", "quit", "attrited", "attrited customer", "relapse",
}  # fmt: skip
ID_WORDS = {"id", "key", "code", "no", "number", "customer", "client", "user", "account"}
STATUS_WORDS = {
    "status", "state", "stage", "phase", "plan", "tier", "statut", "etat", "lifecycle",
}  # fmt: skip


def name_words(name: str) -> set[str]:
    """Words of a column name: "SignupDate", "signup_date", "Signup date" → signup, date."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return set(re.findall(r"[a-z0-9]+", spaced.lower()))


@dataclass
class DurationColumn:
    name: str
    type: AnalysisType
    parser: Parser
    role: str | None = None
    distinct: int | None = None
    rows: int | None = None
    order: list[str] | None = None
    minimum: float | None = None  # numbers: the smallest value, when the summary has it

    def compare_column(self) -> CompareColumn:
        return CompareColumn(self.name, self.type, self.parser, self.order, self.role)


# --- Survival --------------------------------------------------------------------------------


@dataclass
class SurvivalSetup:
    kind: Literal["duration", "dates"]
    duration: str | None = None
    event: str | None = None
    start: str | None = None
    end: str | None = None
    event_value: str | None = None  # the level read as "it happened" (R-SRV-03)
    event_reason: str | None = None
    unit: str | None = None  # days for two dates; else the duration's own unit (unknown)
    detected: bool = True
    reason: str = ""


@dataclass
class SurvivalChoices:
    durations: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)


@dataclass
class SurvivalCurve:
    label: str
    n: int
    events: int
    times: list[float]
    survival: list[float]
    lower: list[float]
    upper: list[float]
    at_risk: list[int]
    median: float | None
    median_lower: float | None
    median_upper: float | None


@dataclass
class LogRankResult:
    statistic: float
    df: int
    p_value: float
    significant: bool
    observed: list[int]
    expected: list[float]


@dataclass
class Survival:
    setup: SurvivalSetup | None
    choices: SurvivalChoices
    group: str | None = None
    rows: int = 0
    events: int = 0
    censored: int = 0
    dropped: int = 0  # rows left out: negative duration, end before start, missing values
    curves: list[SurvivalCurve] = field(default_factory=list)
    log_rank: LogRankResult | None = None
    others: int = 0  # group levels beyond MAX_CURVES, left out (R-SRV-04)
    sentence: str | None = None
    message: str | None = None


def _is_duration(column: DurationColumn) -> bool:
    if column.type not in NUMBERS or (column.minimum is not None and column.minimum < 0):
        return False
    return column.role == "duration" or bool(name_words(column.name) & DURATION_WORDS)


def _is_event(column: DurationColumn) -> bool:
    two = column.type == "boolean" or (column.type in LEVELS and column.distinct == 2)
    return two and bool(name_words(column.name) & EVENT_WORDS)


def _date_words(column: DurationColumn, words: set[str]) -> bool:
    return column.type == "datetime" and bool(name_words(column.name) & words)


def survival_choices(columns: list[DurationColumn]) -> SurvivalChoices:
    return SurvivalChoices(
        durations=[c.name for c in columns if c.type in NUMBERS],
        events=[
            c.name for c in columns if c.type == "boolean" or (c.type in LEVELS and c.distinct == 2)
        ],
        dates=[c.name for c in columns if c.type == "datetime"],
        groups=[
            c.name
            for c in columns
            if c.type in LEVELS and c.distinct is not None and 2 <= c.distinct <= 20
        ],
    )


def detect_survival(columns: list[DurationColumn]) -> SurvivalSetup | None:
    """The first reading a rule allows, with its reason; None when no rule fires."""
    durations = [c for c in columns if _is_duration(c)]
    events = [c for c in columns if _is_event(c)]
    if durations and events:
        duration, event = durations[0], events[0]
        return SurvivalSetup(
            kind="duration",
            duration=duration.name,
            event=event.name,
            reason=(
                f"R-SRV-01: {duration.name!r} reads as a duration and {event.name!r} as the event "
                "that ends it (their names); the rows without the event are still followed "
                "(censored). Check that your data was read correctly."
            ),
        )
    starts = [c for c in columns if _date_words(c, START_WORDS)]
    ends = [c for c in columns if _date_words(c, END_WORDS) and c not in starts]
    if starts and ends:
        start, end = starts[0], ends[0]
        return SurvivalSetup(
            kind="dates",
            start=start.name,
            end=end.name,
            unit="days",
            reason=(
                f"R-SRV-02: {start.name!r} reads as a start and {end.name!r} as an end (their "
                "names): the duration is the days between them; an empty end is still running, "
                "followed until the latest date in the data (censored). Check that your data was "
                "read correctly."
            ),
        )
    return None


def _event_level(
    column: DurationColumn, counts: dict[str, int], chosen: str | None
) -> tuple[str, str]:
    """The level read as "it happened", and why (R-SRV-03)."""
    if chosen is not None:
        if chosen not in counts:
            raise NotComparableError(f"{chosen!r} is not a value of {column.name!r}.")
        return chosen, f"{column.name} = {chosen} read as the event (your choice)."
    censor = bool(name_words(column.name) & {"censor", "censored"})
    labels = list(counts)
    if column.type == "boolean":
        level = "False" if censor else "True"
    else:
        positive = [label for label in labels if label.strip().lower() in HAPPENED]
        zero_one = sorted(labels) == ["0", "1"]
        if zero_one:
            level = "0" if censor else "1"
        elif len(positive) == 1:
            level = positive[0]
            if censor:
                level = next(label for label in labels if label != level)
        else:
            level = min(labels, key=lambda label: (counts[label], label))
            return level, (
                f"R-SRV-03: {column.name} = {level} read as the event (the rarer value: no "
                "value says it happened). Check that your data was read correctly."
            )
    flip = " (the name says censored, so the other value is the event)" if censor else ""
    return level, f"R-SRV-03: {column.name} = {level} read as the event{flip}."


def _curve(label: str, durations: np.ndarray, happened: np.ndarray) -> SurvivalCurve:
    km = sv.kaplan_meier(durations, happened)
    return SurvivalCurve(
        label,
        km.n,
        km.total_events,
        km.times,
        km.survival,
        km.lower,
        km.upper,
        km.at_risk,
        km.median,
        km.median_lower,
        km.median_upper,
    )


def _amount(value: float | None, unit: str | None) -> str:
    if value is None:
        return "not reached"
    text = f"{value:,.0f}" if abs(value) >= 100 else f"{value:.3g}"
    return f"{text} {unit}" if unit else text


def _p(value: float) -> str:
    return "p < 0.0001" if value < 1e-4 else f"p = {value:.2g}"


def _sentence(result: Survival, unit: str | None) -> str:
    if len(result.curves) == 1:
        curve = result.curves[0]
        if curve.median is None:
            last = curve.survival[-1]
            return (
                f"More than half never reach the event while followed: {last:.0%} are still "
                f"without it at {_amount(curve.times[-1], unit)}."
            )
        band = (
            f" (95% interval {_amount(curve.median_lower, unit)} to "
            f"{_amount(curve.median_upper, unit)})"
            if curve.median_lower is not None
            else ""
        )
        return f"Half reach the event by {_amount(curve.median, unit)}{band}."
    medians = ", ".join(f"{c.label} {_amount(c.median, unit)}" for c in result.curves)
    test = result.log_rank
    if test is None:
        return f"Median time to the event: {medians}."
    if test.significant:
        return (
            f"The curves differ (log-rank {_p(test.p_value)}). Median time to the event: {medians}."
        )
    return (
        f"No clear difference between the curves (log-rank {_p(test.p_value)}). "
        f"Median time to the event: {medians}."
    )


def survival(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    setup: SurvivalSetup,
    by_name: dict[str, DurationColumn],
    group: str | None = None,
    event_value: str | None = None,
) -> Survival:
    choices = survival_choices(list(by_name.values()))
    result = Survival(setup=setup, choices=choices, group=group)
    group_column = by_name.get(group) if group else None
    if group and group_column is None:
        raise NotComparableError(f"Unknown or excluded column {group!r}.")
    group_sql = (
        (
            number_label_sql(group_column)
            if group_column.type in NUMBERS
            else label_sql(group_column.compare_column())
        )
        if group_column
        else "NULL"
    )
    total = int((connection.execute(f"SELECT count(*) FROM {source}").fetchone() or (0,))[0])
    if setup.kind == "duration":
        duration_column = by_name.get(setup.duration or "")
        event_column = by_name.get(setup.event or "")
        if duration_column is None or event_column is None:
            raise NotComparableError("Pick a duration and an event column.")
        if duration_column.type not in NUMBERS:
            raise NotComparableError(f"{duration_column.name!r} is not a number.")
        event_sql = (
            number_label_sql(event_column)
            if event_column.type in NUMBERS
            else label_sql(event_column.compare_column())
        )
        rows = connection.execute(
            f"SELECT d, e, g FROM (SELECT {number_sql(duration_column)} AS d, {event_sql} AS e, "
            f"{group_sql} AS g FROM {source}) WHERE d IS NOT NULL AND isfinite(d) AND d >= 0 "
            f"AND e IS NOT NULL LIMIT {MAX_ROWS}"
        ).fetchall()
        counts: dict[str, int] = {}
        for _, e, _ in rows:
            counts[str(e)] = counts.get(str(e), 0) + 1
        if len(counts) > 2:
            raise NotComparableError(
                f"{event_column.name!r} has {len(counts)} values: the event needs two (it "
                "happened or not)."
            )
        if not counts:
            result.message = "No row has both a duration and an event."
            return result
        level, why = _event_level(event_column, counts, event_value or setup.event_value)
        setup.event_value, setup.event_reason = level, why
        durations = np.array([float(d) for d, _, _ in rows])
        happened = np.array([str(e) == level for _, e, _ in rows])
        labels = np.array(["" if g is None else str(g) for _, _, g in rows])
    else:
        start_column = by_name.get(setup.start or "")
        end_column = by_name.get(setup.end or "")
        if start_column is None or end_column is None:
            raise NotComparableError("Pick a start and an end date.")
        start_sql = f"CAST({typed_sql(start_column.name, start_column.parser)} AS TIMESTAMP)"
        end_sql = f"CAST({typed_sql(end_column.name, end_column.parser)} AS TIMESTAMP)"
        latest = connection.execute(
            f"SELECT greatest(max({start_sql}), max({end_sql})) FROM {source}"
        ).fetchone()
        if not latest or latest[0] is None:
            result.message = "The dates are empty."
            return result
        rows = connection.execute(
            f"SELECT date_diff('second', s, coalesce(e, {_ts(latest[0])})) / 86400.0, "
            f"e IS NOT NULL, g FROM (SELECT {start_sql} AS s, {end_sql} AS e, {group_sql} AS g "
            f"FROM {source}) WHERE s IS NOT NULL AND (e IS NULL OR e >= s) LIMIT {MAX_ROWS}"
        ).fetchall()
        setup.unit = "days"
        setup.event_value = None
        setup.event_reason = (
            f"R-SRV-02: a filled {end_column.name} is the event; an empty one is still running, "
            f"followed until {latest[0]:%Y-%m-%d} (the latest date in the data)."
        )
        durations = np.array([float(d) for d, _, _ in rows])
        happened = np.array([bool(e) for _, e, _ in rows])
        labels = np.array(["" if g is None else str(g) for _, _, g in rows])
    result.dropped = total - len(rows)
    if group_column is not None:
        keep = labels != ""
        durations, happened, labels = durations[keep], happened[keep], labels[keep]
    result.rows = len(durations)
    result.events = int(happened.sum())
    result.censored = result.rows - result.events
    if result.rows < MIN_ROWS:
        result.message = f"Needs at least {MIN_ROWS} rows with a duration (got {result.rows})."
        return result
    if result.events == 0:
        result.message = "No row has the event: every curve would stay at 100%."
        return result
    if group_column is None:
        result.curves = [_curve("All rows", durations, happened)]
    else:
        levels, sizes = np.unique(labels, return_counts=True)
        kept = [str(level) for level, _ in sorted(zip(levels, sizes, strict=True), key=_largest)]
        kept = kept[:MAX_CURVES]
        result.others = len(levels) - len(kept)
        mask = np.isin(labels, kept)
        durations, happened, labels = durations[mask], happened[mask], labels[mask]
        if order := group_column.order:
            kept.sort(key=lambda x: order.index(x) if x in order else 99)
        result.curves = [
            _curve(level, durations[labels == level], happened[labels == level]) for level in kept
        ]
        test = sv.log_rank(durations, happened, labels)
        if test is not None:
            result.log_rank = LogRankResult(
                test.statistic,
                test.df,
                test.p_value,
                bool(test.p_value < ALPHA),
                test.observed,
                test.expected,
            )
    result.sentence = _sentence(result, setup.unit)
    return result


def _largest(item: tuple) -> tuple:
    level, size = item
    return (-int(size), str(level))


def _ts(value) -> str:
    return f"TIMESTAMP '{value:%Y-%m-%d %H:%M:%S}'"


# --- Transitions -----------------------------------------------------------------------------


@dataclass
class TransitionSetup:
    entity: str
    time: str
    status: str
    detected: bool = True
    reason: str = ""


@dataclass
class TransitionChoices:
    entities: list[str] = field(default_factory=list)
    times: list[str] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)


@dataclass
class Transitions:
    setup: TransitionSetup | None
    choices: TransitionChoices
    entities: int = 0
    rows: int = 0
    chain: tr.Chain | None = None
    step_days: float | None = None  # median days between two observations of one entity
    others: int = 0  # states beyond MAX_STATES, not paired (R-MKV-02)
    sentence: str | None = None
    message: str | None = None


def _is_entity(column: DurationColumn) -> bool:
    if column.type not in CATEGORIES | {"identifier", "numeric_discrete", "text"}:
        return False
    repeats = (
        column.distinct is not None
        and column.rows is not None
        and 2 <= column.distinct < 0.9 * column.rows
    )
    named = (
        column.role == "identifier"
        or (column.role or "").endswith("_key")
        or bool(name_words(column.name) & ID_WORDS)
    )
    return repeats and named


def _is_status(column: DurationColumn) -> bool:
    if column.type not in CATEGORIES or column.distinct is None or column.role == "region":
        return False
    if not 2 <= column.distinct <= MAX_STATES:
        return False
    return column.role == "status" or bool(name_words(column.name) & STATUS_WORDS)


def transition_choices(columns: list[DurationColumn]) -> TransitionChoices:
    return TransitionChoices(
        entities=[
            c.name
            for c in columns
            if c.type in CATEGORIES | {"identifier", "numeric_discrete", "text"}
            and (c.distinct or 0) >= 2
        ],
        times=[c.name for c in columns if c.type in {"datetime", *NUMBERS}],
        statuses=[
            c.name
            for c in columns
            if c.type in CATEGORIES and c.distinct is not None and 2 <= c.distinct <= 30
        ],
    )


def detect_transitions(columns: list[DurationColumn]) -> TransitionSetup | None:
    entities = [c for c in columns if _is_entity(c)]
    statuses = [c for c in columns if _is_status(c) and c not in entities]
    times = [c for c in columns if c.type == "datetime"]
    if not (entities and statuses and times):
        return None
    entity, status, time = entities[0], statuses[0], times[0]
    return TransitionSetup(
        entity=entity.name,
        time=time.name,
        status=status.name,
        reason=(
            f"R-MKV-01: {entity.name!r} repeats ({entity.distinct} values over {entity.rows} rows)"
            f" and {status.name!r} reads as a status ({status.distinct} values): each entity's "
            f"consecutive rows by {time.name!r} are transitions between statuses. Check that "
            "your data was read correctly."
        ),
    )


def transitions(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    setup: TransitionSetup,
    by_name: dict[str, DurationColumn],
) -> Transitions:
    result = Transitions(setup=setup, choices=transition_choices(list(by_name.values())))
    entity, time, status = (by_name.get(n) for n in (setup.entity, setup.time, setup.status))
    if entity is None or time is None or status is None:
        raise NotComparableError("Pick an entity, a time and a status column.")
    if len({setup.entity, setup.time, setup.status}) < 3:
        raise NotComparableError("Pick three different columns.")
    entity_sql = f"CAST({typed_sql(entity.name, entity.parser)} AS VARCHAR)"
    if time.type == "datetime":
        time_sql = f"epoch(CAST({typed_sql(time.name, time.parser)} AS TIMESTAMP)) / 86400.0"
    elif time.type in NUMBERS:
        time_sql = number_sql(time)
    else:
        raise NotComparableError(f"{time.name!r} is neither a date nor a number.")
    status_sql = label_sql(status.compare_column())
    rows = connection.execute(
        f"SELECT i, t, s FROM (SELECT {entity_sql} AS i, {time_sql} AS t, {status_sql} AS s "
        f"FROM {source}) WHERE i IS NOT NULL AND t IS NOT NULL AND s IS NOT NULL "
        f"ORDER BY i, t LIMIT {MAX_ROWS}"
    ).fetchall()
    result.rows = len(rows)
    if not rows:
        result.message = "No row has an entity, a time and a status."
        return result
    ids = np.array([r[0] for r in rows], dtype=object)
    times = np.array([float(r[1]) for r in rows])
    states = np.array([str(r[2]) for r in rows], dtype=object)
    result.entities = len(set(ids.tolist()))
    labels, sizes = np.unique(states, return_counts=True)
    ranked = [str(level) for level, _ in sorted(zip(labels, sizes, strict=True), key=_largest)]
    levels = ranked[:MAX_STATES]
    if order := status.order:
        levels.sort(key=lambda x: order.index(x) if x in order else 99)
    else:
        levels.sort()
    result.others = len(ranked) - len(levels)
    counts = tr.count_transitions(ids, states, levels)
    if counts.sum() == 0:
        result.message = (
            f"No entity is seen twice: {entity.name!r} needs to repeat to give transitions."
        )
        return result
    if np.trace(counts) == counts.sum():
        result.message = (
            f"No entity ever changes {status.name!r}: it is a fixed attribute of each "
            f"{entity.name!r}, not a status that moves."
        )
        return result
    same = ids[1:] == ids[:-1]
    gaps = (times[1:] - times[:-1])[same]
    gaps = gaps[gaps > 0]
    if time.type == "datetime" and gaps.size:
        result.step_days = float(np.median(gaps))
    result.chain = tr.chain(counts, levels)
    result.sentence = _chain_sentence(result.chain, result.step_days)
    return result


def _steps(value: float, step_days: float | None) -> str:
    text = f"{value:.1f} steps"
    if step_days:
        days = value * step_days
        text += f" (about {days / 30.44:.1f} months)" if days >= 60 else f" (about {days:.0f} days)"
    return text


def _chain_sentence(chain: tr.Chain, step_days: float | None) -> str:
    if chain.absorbing and chain.steps_to_absorption:
        target = " or ".join(chain.absorbing)
        start, steps = max(chain.steps_to_absorption.items(), key=lambda kv: kv[1])
        return (
            f"{target} is never left: from {start}, it takes {_steps(steps, step_days)} on "
            "average to get there."
        )
    if chain.stationary:
        share, state = max(zip(chain.stationary, chain.states, strict=True))
        return f"In the long run, {share:.0%} of the time is spent in {state}."
    return f"{chain.transitions} transitions between {len(chain.states)} states."
