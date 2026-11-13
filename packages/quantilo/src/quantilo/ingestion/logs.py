"""Text log files → one table, every reading rule named (090, R-LOG).

    table, reading = read_log(Path("access.log"))
    reading.format      # "apache_combined"
    reading.notes       # ["Read as an Apache/Nginx access log (combined format): R-LOG-01. …"]

Known line formats are tried on the first lines (R-LOG-06); the one that matches at least 80 %
of them reads the whole file. A line that does not match is the continuation of the previous
entry when it starts with a space or a tab (a stack trace, R-LOG-07), otherwise it is kept on its
own with its text in `message`. The messages are then grouped into templates by a small Drain
(He et al., 2017): variable tokens (numbers, addresses, identifiers) become `<*>` (R-LOG-08).

Standard library and pyarrow only: it runs in the browser (Pyodide). No LLM: every format is a
regular expression written here, and the app says which one read the file."""

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa

from quantilo.ingestion.errors import IngestionError

SNIFF_LINES = 200  # lines the format is chosen on (R-LOG-06)
MATCH_SHARE = 0.8  # share of those lines a format must read (R-LOG-06, R-DET-05)
MAX_CONTINUATION = 200  # continuation lines glued to one entry at most (R-LOG-07)
MAX_MESSAGE = 10_000  # characters kept per message
DRAIN_DEPTH = 4  # Drain: tokens of the prefix tree (length, then 2 tokens)
DRAIN_SIMILARITY = 0.5  # Drain: share of equal tokens to join a template
DRAIN_CHILDREN = 100  # Drain: branches per node, then "<*>"
WILDCARD = "<*>"

_MONTHS = {
    m: i
    for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1
    )
}
_LEVELS = {
    "TRACE": "TRACE",
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "NOTICE": "NOTICE",
    "WARN": "WARN",
    "WARNING": "WARN",
    "ERROR": "ERROR",
    "ERR": "ERROR",
    "SEVERE": "ERROR",
    "CRIT": "CRITICAL",
    "CRITICAL": "CRITICAL",
    "FATAL": "FATAL",
    "ALERT": "ALERT",
    "EMERG": "EMERGENCY",
}
_SEVERITIES = ["emergency", "alert", "critical", "error", "warning", "notice", "info", "debug"]

_APACHE = (
    r"(?P<host>\S+) (?P<ident>\S+) (?P<user>\S+) \[(?P<time>[^\]]+)\] "
    r'"(?P<request>[^"]*)" (?P<status>\d{3}|-) (?P<bytes>\d+|-)'
)
APACHE_COMBINED = re.compile(_APACHE + r' "(?P<referrer>[^"]*)" "(?P<user_agent>[^"]*)".*$')
APACHE_COMMON = re.compile(_APACHE + r"\s*$")
SYSLOG_5424 = re.compile(
    r"<(?P<pri>\d{1,3})>1 (?P<time>\S+) (?P<host>\S+) (?P<app>\S+) (?P<procid>\S+) "
    r"(?P<msgid>\S+) (?P<sd>-|(?:\[[^\]]*\])+) ?(?P<message>.*)$"
)
SYSLOG_3164 = re.compile(
    r"(?:<(?P<pri>\d{1,3})>)?(?P<month>[A-Z][a-z]{2}) {1,2}(?P<day>\d{1,2}) "
    r"(?P<clock>\d{2}:\d{2}:\d{2}) (?P<host>\S+) (?P<program>[^\s:\[]+)(?:\[(?P<pid>\d+)\])?: ?"
    r"(?P<message>.*)$"
)
ISO_LINE = re.compile(
    r"\[?(?P<time>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?"
    r"(?:Z|[+-]\d{2}:?\d{2})?)\]?[\s|,;-]*"
    r"(?:\[?(?P<level>TRACE|DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|ERR|SEVERE|CRIT|CRITICAL|FATAL"
    r"|ALERT|EMERG)\]?\b[\s:|-]*)?(?P<message>.*)$",
    re.IGNORECASE,
)
# R-DET-05: a line that starts like a log entry (ISO date or syslog header)
LOG_START = re.compile(
    r"^(?:\[?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}|<\d{1,3}>"
    r"|[A-Z][a-z]{2} {1,2}\d{1,2} \d{2}:\d{2}:\d{2}"
    r"|\S+ \S+ \S+ \[\d{2}/[A-Z][a-z]{2}/\d{4}:)"
)

# Drain's preprocessing: variable tokens masked before grouping (He et al., 2017, § III-A)
_MASKS = [
    re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"),
    re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?$"),
    re.compile(r"^(?:0x)?[0-9a-fA-F]*\d[0-9a-fA-F]*$"),  # numbers and hexadecimal
    re.compile(r"^[-+]?\d+(?:[.,:]\d+)*[a-zA-Z%]{0,3}$"),  # 12, 3.5, 10:20, 250ms, 80%
    re.compile(r"^(?:/[^/\s]+){2,}/?$"),  # paths
    re.compile(r"^\w+[=:][^\s]+$"),  # key=value
]


@dataclass(frozen=True)
class LogFormat:
    key: str
    rule: str
    label: str
    pattern: re.Pattern[str]
    columns: tuple[str, ...]


FORMATS: tuple[LogFormat, ...] = (
    LogFormat(
        "apache_combined",
        "R-LOG-01",
        "an Apache/Nginx access log (combined format)",
        APACHE_COMBINED,
        (
            "time",
            "host",
            "user",
            "method",
            "path",
            "protocol",
            "status",
            "bytes",
            "referrer",
            "user_agent",
        ),
    ),
    LogFormat(
        "apache_common",
        "R-LOG-02",
        "an Apache/Nginx access log (common format)",
        APACHE_COMMON,
        ("time", "host", "user", "method", "path", "protocol", "status", "bytes"),
    ),
    LogFormat(
        "syslog_5424",
        "R-LOG-04",
        "a syslog file (RFC 5424)",
        SYSLOG_5424,
        ("time", "host", "app", "process", "message_id", "severity", "facility", "message"),
    ),
    LogFormat(
        "syslog_3164",
        "R-LOG-03",
        "a syslog file (BSD format, RFC 3164)",
        SYSLOG_3164,
        ("time", "host", "program", "process", "message"),
    ),
    LogFormat(
        "iso_lines",
        "R-LOG-05",
        "a log whose lines start with a date and time",
        ISO_LINE,
        ("time", "level", "message"),
    ),
)
BY_KEY = {fmt.key: fmt for fmt in FORMATS}

# R-LOG-11 (C1): a line pattern the user declares, `{time} [{level}] {message}` or a regular
# expression with named groups `(?P<time>…)`
USER = "user"
MAX_PATTERN = 500
MAX_FIELDS = 30
_FIELD = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]{0,40})\}")


def user_format(pattern: str) -> LogFormat:
    """R-LOG-11: the user's pattern as a format. A template's `{name}` fields read as little as
    they can (the last one, the rest of the line), the text between them as written (runs of
    spaces: one or more); `(?P<name>…)` makes it a regular expression, used as it is."""
    text = pattern.strip()
    if not text or len(text) > MAX_PATTERN:
        raise IngestionError(f"A line pattern has 1 to {MAX_PATTERN} characters.")
    if "(?P<" in text:
        try:
            compiled = re.compile(text if text.startswith("^") else f"^{text}")
        except re.error as error:
            raise IngestionError(f"Invalid line pattern: {error}.") from None
        names = tuple(compiled.groupindex)
    else:
        names = tuple(_FIELD.findall(text))
        if not names:
            raise IngestionError("A line pattern names its fields in braces: {time} {message}.")
        if len(set(names)) != len(names):
            raise IngestionError("Each field of a line pattern is named once.")
        parts: list[str] = []
        last = 0
        fields = list(_FIELD.finditer(text))
        for index, found in enumerate(fields):
            parts.append(_literal(text[last : found.start()]))
            greedy = index == len(fields) - 1
            parts.append(f"(?P<{found.group(1)}>{'.*' if greedy else '.*?'})")
            last = found.end()
        parts.append(_literal(text[last:]))
        compiled = re.compile("^" + "".join(parts) + "$")
    if not names or len(names) > MAX_FIELDS or len(set(names)) != len(names):
        raise IngestionError(f"A line pattern has 1 to {MAX_FIELDS} fields, each named once.")
    columns = names if "message" in names else (*names, "message")
    return LogFormat(USER, "R-LOG-11", f"your line pattern ({', '.join(names)})", compiled, columns)


def _literal(text: str) -> str:
    return r"\s+".join(re.escape(piece) for piece in re.split(r"\s+", text)) if text else ""


@dataclass
class LogReading:
    """How the file was read: the format, the rules that applied, the counts."""

    format: str  # a FORMATS key, or "lines" (no known format, R-LOG-09)
    rule: str
    label: str
    lines: int = 0
    entries: int = 0
    continued: int = 0  # lines glued to the entry above (R-LOG-07)
    unmatched: int = 0  # lines that started a new entry without matching the format
    templates: int = 0
    year: int | None = None  # R-LOG-03: the year a BSD syslog leaves out
    notes: list[str] = field(default_factory=list)


def _lines(path: Path, limit: int | None = None) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                return
            yield line.rstrip("\r\n")


def _sample(path: Path) -> list[str]:
    return [line for line in _lines(path, SNIFF_LINES * 2) if line.strip()][:SNIFF_LINES]


def choose_format(lines: list[str]) -> LogFormat | None:
    """R-LOG-06: the first known format that reads at least 80 % of the sampled lines that do not
    start with a space (continuations are not entries)."""
    entries = [line for line in lines if line and not line[0].isspace()]
    if not entries:
        return None
    for fmt in FORMATS:
        hits = sum(1 for line in entries if fmt.pattern.match(line))
        if hits / len(entries) >= MATCH_SHARE:
            return fmt
    return None


def looks_like_log(path: Path) -> bool:
    """R-DET-05: at least 80 % of the first lines start like a log entry (an ISO date and time,
    a syslog header or an access-log line), or a known format reads them."""
    try:
        lines = _sample(path)
    except OSError:
        return False
    entries = [line for line in lines if not line[0].isspace()]
    if len(entries) < 3:
        return False
    starts = sum(1 for line in entries if LOG_START.match(line))
    return starts / len(entries) >= MATCH_SHARE or choose_format(lines) is not None


# --- Values -----------------------------------------------------------------------------------


def _apache_time(text: str) -> datetime | None:
    try:
        return datetime.strptime(text, "%d/%b/%Y:%H:%M:%S %z")
    except ValueError:
        return None


def _iso_time(text: str) -> datetime | None:
    text = text.replace(",", ".").replace(" ", "T", 1)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    fraction = re.search(r"\.(\d+)", text)
    if fraction and len(fraction.group(1)) > 6:  # nanoseconds: cut to microseconds
        text = text[: fraction.start(1) + 6] + text[fraction.end(1) :]
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _naive_utc(value: datetime | None) -> datetime | None:
    """R-LOG-10: times with an offset are stored in UTC; times without one as written."""
    if value is None or value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _int(text: str | None) -> int | None:
    return int(text) if text and text.isdigit() else None


def _request(text: str) -> tuple[str | None, str | None, str | None]:
    parts = text.split(" ")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return None, text or None, None


def _dash(text: str | None) -> str | None:
    return None if text in (None, "-", "") else text


def parse(fmt: LogFormat, match: re.Match[str], year: int) -> dict[str, object]:
    """One matched line → its columns."""
    g = match.groupdict()
    if fmt.key.startswith("apache"):
        method, target, protocol = _request(g["request"])
        row: dict[str, object] = {
            "time": _naive_utc(_apache_time(g["time"])),
            "host": g["host"],
            "user": _dash(g["user"]),
            "method": method,
            "path": target,
            "protocol": protocol,
            "status": _int(g["status"]),
            "bytes": _int(g["bytes"]),
        }
        if fmt.key == "apache_combined":
            row["referrer"] = _dash(g["referrer"])
            row["user_agent"] = _dash(g["user_agent"])
        return row
    if fmt.key == "syslog_5424":
        pri = int(g["pri"])
        return {
            "time": _naive_utc(_iso_time(g["time"])) if g["time"] != "-" else None,
            "host": _dash(g["host"]),
            "app": _dash(g["app"]),
            "process": _dash(g["procid"]),
            "message_id": _dash(g["msgid"]),
            "severity": _SEVERITIES[pri % 8],
            "facility": pri // 8,
            "message": g["message"],
        }
    if fmt.key == "syslog_3164":
        hour, minute, second = (int(x) for x in g["clock"].split(":"))
        try:
            time = datetime(year, _MONTHS[g["month"]], int(g["day"]), hour, minute, second)
        except (KeyError, ValueError):
            time = None
        return {
            "time": time,
            "host": g["host"],
            "program": g["program"],
            "process": g["pid"],
            "message": g["message"],
        }
    level = g.get("level")
    return {
        "time": _naive_utc(_iso_time(g["time"])),
        "level": _LEVELS.get(level.upper(), level.upper()) if level else None,
        "message": g["message"],
    }


# --- Drain ------------------------------------------------------------------------------------


def _mask(token: str) -> str:
    return WILDCARD if any(mask.match(token) for mask in _MASKS) else token


class Drain:
    """A fixed-depth prefix tree of templates (He et al., 2017): messages are split on spaces,
    variable tokens masked, then routed by length and their first tokens; a message joins the
    most similar template of its leaf if at least half its tokens are equal, otherwise it starts
    a new one. Differing tokens of a joined template become `<*>`. Deterministic."""

    def __init__(self) -> None:
        self.tree: dict[object, object] = {}
        self.templates: list[list[str]] = []

    def add(self, message: str) -> int:
        tokens = [_mask(token) for token in message.split()]
        leaf = self._leaf(tokens)
        best, best_score = -1, -1.0
        for index in leaf:
            template = self.templates[index]
            same = sum(1 for a, b in zip(template, tokens, strict=True) if a == b)
            score = same / len(tokens) if tokens else 1.0
            if score > best_score:
                best, best_score = index, score
        if best >= 0 and best_score >= DRAIN_SIMILARITY:
            template = self.templates[best]
            self.templates[best] = [
                a if a == b else WILDCARD for a, b in zip(template, tokens, strict=True)
            ]
            return best
        self.templates.append(tokens)
        leaf.append(len(self.templates) - 1)
        return len(self.templates) - 1

    def _leaf(self, tokens: list[str]) -> list[int]:
        node = self.tree.setdefault(len(tokens), {})
        for depth in range(DRAIN_DEPTH - 2):
            assert isinstance(node, dict)
            if depth >= len(tokens):
                break
            token = tokens[depth]
            if any(ch.isdigit() for ch in token):
                token = WILDCARD
            if token not in node and len(node) >= DRAIN_CHILDREN:
                token = WILDCARD
            node = node.setdefault(token, {})
        assert isinstance(node, dict)
        return node.setdefault(None, [])  # type: ignore[return-value]

    def text(self, index: int) -> str:
        return " ".join(self.templates[index])


# --- Reading ----------------------------------------------------------------------------------


def _year(path: Path) -> int:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).year
    except OSError:
        return datetime.now().year


def read_log(
    path: Path, limit: int | None = None, pattern: str | None = None
) -> tuple[pa.Table, LogReading]:
    """The whole file (or its first `limit` entries, for a preview) as one table. `pattern`:
    the user's line pattern (R-LOG-11) instead of the known formats."""
    fmt = user_format(pattern) if pattern else choose_format(_sample(path))
    if fmt is None:
        reading = LogReading("lines", "R-LOG-09", "lines of text")
        columns: tuple[str, ...] = ("message",)
    else:
        reading = LogReading(fmt.key, fmt.rule, fmt.label)
        columns = fmt.columns
    year = _year(path)
    rows: list[dict[str, object]] = []
    numbers: list[int] = []
    glued = 0
    for number, line in enumerate(_lines(path), start=1):
        reading.lines += 1
        if not line.strip():
            continue
        glue = fmt is not None and "message" in fmt.columns
        if rows and line[0].isspace() and glued < MAX_CONTINUATION and glue:
            message = str(rows[-1].get("message") or "")
            if len(message) < MAX_MESSAGE:
                rows[-1]["message"] = (message + "\n" + line)[:MAX_MESSAGE]
            reading.continued += 1
            glued += 1
            continue
        if limit is not None and len(rows) >= limit:
            break
        glued = 0
        match = fmt.pattern.match(line) if fmt is not None else None
        if fmt is not None and match is not None and fmt.key == USER:
            row = {k: v for k, v in match.groupdict().items() if v is not None}  # text (R-LOG-11)
        elif fmt is not None and match is not None:
            row = parse(fmt, match, year)
        else:
            row: dict[str, object] = {"message": line[:MAX_MESSAGE]}
            if fmt is not None:
                reading.unmatched += 1
        rows.append(row)
        numbers.append(number)
    reading.entries = len(rows)
    drain = Drain()
    template_of: list[int] = []
    for row in rows:
        text = row.get("message")
        if text is None:  # access logs: the request is the message
            text = f"{row.get('method') or ''} {row.get('path') or ''} {row.get('status') or ''}"
        template_of.append(drain.add(str(text).split("\n", 1)[0]))
    reading.templates = len(drain.templates)
    if fmt is not None and fmt.key == "syslog_3164":
        reading.year = year
    data: dict[str, list[object]] = {"line_no": list(numbers)}
    for name in columns:
        data[name] = [row.get(name) for row in rows]
    data["template_id"] = [index + 1 for index in template_of]
    data["template"] = [drain.text(index) for index in template_of]
    typed = fmt is None or fmt.key != USER  # the user's fields stay text; the import types them
    table = pa.table({name: _array(name, values, typed) for name, values in data.items()})
    reading.notes = _notes(path, reading)
    return table, reading


def _array(name: str, values: list[object], typed: bool = True) -> pa.Array:
    if name == "time" and typed:
        return pa.array(values, type=pa.timestamp("us"))
    if name in ("line_no", "template_id") or (typed and name in ("status", "bytes", "facility")):
        return pa.array(values, type=pa.int64())
    return pa.array([None if v is None else str(v) for v in values], type=pa.string())


def _notes(path: Path, reading: LogReading) -> list[str]:
    """What the app says where the user looks (import preview), one sentence per rule."""
    check = "Check that your data was read correctly."
    if reading.format == "lines":
        out = [
            f"{path.name}: no known log format matched 80 % of its first lines; one row per "
            f"line, its text in `message` (R-LOG-09). {check}"
        ]
    elif reading.format == USER:
        read = reading.entries - reading.unmatched
        out = [
            f"{path.name} read with {reading.label}: R-LOG-11. {read:,} of {reading.entries:,} "
            f"entries follow it; its fields are read as text, then typed like any column. {check}"
        ]
    else:
        out = [f"{path.name} read as {reading.label}: {reading.rule}. {check}"]
    if reading.continued:
        out.append(
            f"{reading.continued:,} indented lines (stack traces) joined to the entry above them "
            "(R-LOG-07)."
        )
    if reading.unmatched:
        out.append(
            f"{reading.unmatched:,} lines do not follow the format: kept with their text in "
            "`message`, other columns empty (R-LOG-06)."
        )
    if reading.year is not None:
        out.append(
            f"BSD syslog lines have no year: {reading.year}, the year the file was last "
            "modified, is used (R-LOG-03)."
        )
    if reading.format.startswith("apache") or reading.format in ("syslog_5424", "iso_lines"):
        out.append("Times with a UTC offset are converted to UTC (R-LOG-10).")
    out.append(
        f"{reading.templates:,} message templates found: numbers, addresses and identifiers "
        "replaced by <*> (Drain, R-LOG-08)."
    )
    return out
