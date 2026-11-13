"""Public holidays of a country as a calendar driver of the forecast (028 A5).

The calendar comes from the `holidays` package (MIT, pure Python), imported lazily: without it
the driver is offered greyed with the reason. A holiday is known in advance (the calendar covers
the forecast periods too), so it enters every driver model as a flag:

- hourly and daily periods: 1 on a public holiday, 0 otherwise;
- weekly periods: the share of the week's 7 days that are public holidays (0 to 1);
- other periods: not offered (a month always holds about the same holidays).

The Prophet-like model gives each holiday its own effect, as Prophet does (one column per holiday
name seen in the history); the other models get the single flag.
"""

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import cache
from typing import Any

import numpy as np

from quantilo.errors import ExplorerError
from quantilo.timeseries import series as ts

PERIODS: tuple[ts.Period, ...] = ("hour", "day", "week")
LANGUAGE = "en_US"  # holiday names in English when the country's calendar has them
# Common names that are neither ISO codes nor the package's English names.
ALIASES = {
    "usa": "US",
    "unitedstatesofamerica": "US",
    "america": "US",
    "uk": "GB",
    "greatbritain": "GB",
    "britain": "GB",
    "england": "GB",
    "holland": "NL",
    "russia": "RU",
    "southkorea": "KR",
    "korea": "KR",
}


class HolidayError(ExplorerError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="bad_driver", status_code=422)


@dataclass(frozen=True)
class Subdivision:
    code: str
    name: str | None  # from the package's aliases, when it has one


@dataclass(frozen=True)
class Country:
    code: str  # ISO 3166-1 alpha-2
    name: str
    subdivisions: list[Subdivision]


def missing() -> str | None:
    """Why the calendar cannot be used here (None: it can)."""
    try:
        import holidays  # noqa: F401
    except ImportError:
        return "Needs the holidays package (pip install holidays), absent from this installation."
    return None


def _module() -> Any:
    try:
        import holidays
    except ImportError as error:
        raise HolidayError(missing() or str(error)) from error
    return holidays


def _spaced(name: str) -> str:
    """'UnitedStates' -> 'United States' (the package's class names)."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)


@cache
def countries() -> tuple[Country, ...]:
    """Every country the calendar knows, by English name."""
    holidays = _module()
    from holidays import registry

    found = []
    for entry in registry.COUNTRIES.values():
        name, code = entry[0], entry[1]
        cls = getattr(holidays, code, None)
        if cls is None:
            continue
        aliases = getattr(cls, "subdivisions_aliases", {}) or {}
        named: dict[str, str] = {}
        for alias, sub in aliases.items():
            if len(alias) > 3 and sub not in named:
                named[sub] = alias
        subdivisions = [Subdivision(str(s), named.get(str(s))) for s in cls.subdivisions]
        found.append(Country(code, _spaced(name), subdivisions))
    return tuple(sorted(found, key=lambda c: c.name))


def _key(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


@cache
def _lookup() -> dict[str, str]:
    from holidays import registry

    table = dict(ALIASES)
    for entry in registry.COUNTRIES.values():
        name, code, alpha3 = entry[0], entry[1], entry[2]
        table[_key(name)] = code
        table[code.lower()] = code
        table[alpha3.lower()] = code
    return table


def country_code(value: object) -> str | None:
    """The ISO code of a country written as a code (FR, FRA) or an English name (France);
    None when the calendar does not know it."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if missing():
        return None
    if len(text) in (2, 3) and text.isalpha():
        return _lookup().get(text.lower())
    return _lookup().get(_key(text))


def parse(text: str) -> tuple[str, str | None]:
    """'FR' -> ('FR', None); 'US-CA' -> ('US', 'CA') (ISO 3166-2 style)."""
    country, _, subdiv = text.strip().partition("-")
    return country.upper(), (subdiv.upper() or None)


def label(country: str, subdiv: str | None) -> str:
    return f"Public holidays ({country}{'-' + subdiv if subdiv else ''})"


def calendar(country: str, subdiv: str | None, years: range) -> dict[date, list[str]]:
    """The public holidays of those years: date -> names (in English when available)."""
    holidays = _module()
    cls = getattr(holidays, country.upper(), None) if country else None
    if cls is None or not isinstance(country, str) or len(country) != 2:
        raise HolidayError(f"The holiday calendar does not know the country {country!r}.")
    if subdiv is not None and subdiv not in [str(s) for s in cls.subdivisions]:
        raise HolidayError(f"{country} has no subdivision {subdiv!r} in the holiday calendar.")
    languages = getattr(cls, "supported_languages", ()) or ()
    found = holidays.country_holidays(
        country.upper(),
        subdiv=subdiv,
        years=list(years),
        language=LANGUAGE if LANGUAGE in languages else None,
    )
    return {day: [n.strip() for n in found.get_list(day)] for day in found}


def _days(moment: datetime, period: ts.Period) -> list[date]:
    if period == "week":
        start = moment.date()
        return [start + timedelta(days=i) for i in range(7)]
    return [moment.date()]


def coverage(country: str) -> tuple[int | None, int | None]:
    """First and last years the package defines for the country."""
    cls = getattr(_module(), country.upper(), None)
    return getattr(cls, "start_year", None), getattr(cls, "end_year", None)


def flags(
    country: str,
    subdiv: str | None,
    moments: list[datetime],
    future: list[datetime],
    period: ts.Period,
    separate: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """The driver's columns on the series' periods then the forecast periods: one flag (the share
    of holiday days for a week), or with `separate` one per holiday name seen in the history.
    Also the column names and the notes (years outside the calendar)."""
    if period not in PERIODS:
        raise HolidayError(
            f"Public holidays need hourly, daily or weekly periods, not {period}: a month or a "
            "longer period always holds about the same holidays."
        )
    every = [*moments, *future]
    if not every:
        return np.empty((0, 1)), np.empty((0, 1)), [label(country, subdiv)], []
    first = min(m.year for m in every)
    last = max((m + timedelta(days=6)).year for m in every)
    known = calendar(country, subdiv, range(first, last + 1))
    notes: list[str] = []
    start, end = coverage(country)
    if (start is not None and first < start) or (end is not None and last > end):
        notes.append(
            f"{label(country, subdiv)}: the calendar covers {start} to {end}; the periods outside "
            "count as having no holiday."
        )

    def share(moment: datetime, wanted: str | None) -> float:
        days = _days(moment, period)
        hits = sum(1 for day in days if day in known and (wanted is None or wanted in known[day]))
        return hits / len(days)

    if not separate:
        past = np.array([share(m, None) for m in moments], dtype=float).reshape(-1, 1)
        ahead = np.array([share(m, None) for m in future], dtype=float).reshape(-1, 1)
        return past, ahead, [label(country, subdiv)], notes
    seen: list[str] = []
    for moment in moments:
        for day in _days(moment, period):
            for name in known.get(day, []):
                if name not in seen:
                    seen.append(name)
    if not seen:
        seen = ["(none in the history)"]
    past = np.array([[share(m, n) for n in seen] for m in moments], dtype=float)
    ahead = np.array([[share(m, n) for n in seen] for m in future], dtype=float)
    return past.reshape(len(moments), -1), ahead.reshape(len(future), -1), seen, notes
