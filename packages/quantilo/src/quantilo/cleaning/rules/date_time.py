"""B-20, N19 (audit 2026-11-12, § 4.3): a date column and a time-of-day column of one moment.

UCI Air Quality keeps `Date` and `Time` apart, the NYC collisions `crash_date` and `crash_time`:
each reads well alone, but a timeline by the hour needs them together. The pair is found by the
names (`date_parts.date_time_pairs`), then checked on the values: when 95 % of the rows holding
both make a valid moment, an `assemble_date` step taking the year, month and day from the date
column and the hour and minute from the time column is proposed (mode S), as a new column; both
columns stay as they are. A date column stored as text is read once B-11 has stored it as dates
(only an ISO date reads here).

Called from `basic.run` after `b20_assemble_dates` (imported there, inside `run`, as this module
reads `basic`).
"""

from quantilo.cleaning import model as m
from quantilo.cleaning.compile import Stage, assembled_date_sql, is_temporal, is_text
from quantilo.cleaning.rules.basic import DATE_COVERAGE, MAX_EXAMPLES, Column, Context, _id, n
from quantilo.ingestion.sql import ident
from quantilo.typing import date_parts


def _has_time(context: Context, column: Column) -> bool:
    """A TIMESTAMP whose values are not all at midnight: the date already holds its time."""
    if not column.dtype.startswith("TIMESTAMP"):
        return False
    name = ident(column.name)
    (timed,) = context.one(
        f"SELECT count(*) FROM {context.source} WHERE {name} IS NOT NULL AND "
        f"CAST({name} AS TIMESTAMP) <> CAST(CAST({name} AS DATE) AS TIMESTAMP)"
    )
    return bool(timed)


def b20_date_and_time(context: Context) -> None:
    by_name = {c.name: c for c in context.columns}
    names = set(by_name)
    stage = Stage("source", [(c.name, c.dtype) for c in context.columns])
    for pair in date_parts.date_time_pairs(c.name for c in context.columns):
        day, clock = by_name[pair.date], by_name[pair.time]
        if not (is_temporal(day.dtype) or is_text(day.dtype)) or _has_time(context, day):
            continue
        target = pair.name
        while target in names:
            target = f"{target} (assembled)"
        # the year, the month and the day all come from the date column
        step = m.AssembleDate(
            id=_id(), year=pair.date, month=pair.date, day=pair.date, time=pair.time, to=target
        )
        tries = [step]
        if not (is_text(clock.dtype) or clock.dtype.startswith("TIME")):
            # a whole hour (0 to 23) rather than a time of day
            tries.insert(0, step.model_copy(update={"time": None, "hour": pair.time}))
        given = f"{ident(pair.date)} IS NOT NULL AND {ident(pair.time)} IS NOT NULL"
        for step in tries:
            made_sql = assembled_date_sql(step, stage)
            filled, made = context.one(
                f"SELECT count(*) FILTER (WHERE {given}), count({made_sql}) FROM {context.source}"
            )
            if filled and made >= DATE_COVERAGE * filled:
                break
        else:
            continue
        examples: list[tuple[str | None, str | None]] = [
            (f"{a} · {b}", str(made_at))
            for a, b, made_at in context.connection.execute(
                f"SELECT {ident(pair.date)}, {ident(pair.time)}, CAST({made_sql} AS VARCHAR) "
                f"FROM {context.source} WHERE {made_sql} IS NOT NULL LIMIT {MAX_EXAMPLES}"
            ).fetchall()
        ]
        clock_word = "hour" if step.hour else "time of day"
        context.add(
            "B-20",
            "S",
            day,
            f"Assemble a date and time from {pair.date}, {pair.time}",
            f"{pair.date} (date) and {pair.time} ({clock_word}) hold one moment: {n(made)} of "
            f"{n(filled)} rows with both ({made / filled:.1%}) make a valid date and time. A new "
            f"column '{target}' holds it, for a timeline by the hour; both columns stay as they "
            "are, and a row missing one of them stays empty. Check your data.",
            step,
            cells=made,
            examples=examples,
            key="time",
        )
        context.found[-1].columns = [pair.date, pair.time]
        names.add(target)
