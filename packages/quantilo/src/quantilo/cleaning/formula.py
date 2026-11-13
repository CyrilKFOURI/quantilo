"""Formulas of the calculation steps (M5: a computed column, a conditional step's test or value).

A formula is data, never code (C-02): it is read token by token and rebuilt as SQL from pieces
that cannot do anything else. A column is a bare name (`quantity`) or a double-quoted one
(`"unit price"`), a text is single-quoted (`'France'`), numbers, the operators `+ - * / % ||`,
the comparisons `= <> != < <= > >=`, a few words (`and`, `or`, `not`, `is`, `null`, `true`,
`false`, `case`, `when`, `then`, `else`, `end`, `between`, `in`) and the functions of `FUNCTIONS`
(a list kept short on purpose: no file, no network, no clock). Anything else is refused with what
was not understood; DuckDB then checks the grammar (a step that cannot run is reported, C-02).
"""

import re
from dataclasses import dataclass

from quantilo.cleaning.model import CleanError
from quantilo.ingestion.sql import ident, lit

FUNCTIONS = frozenset(
    {
        "abs",
        "round",
        "floor",
        "ceil",
        "sqrt",
        "exp",
        "ln",
        "log10",
        "power",
        "sign",
        "least",
        "greatest",
        "coalesce",
        "nullif",
        "if",
        "lower",
        "upper",
        "trim",
        "length",
        "concat",
        "left",
        "right",
        "substr",
        "replace",
        "strip_accents",
        "year",
        "month",
        "day",
        "date_diff",
        "date_add",
        "to_days",
        "to_months",
        "to_years",
    }
)
WORDS = frozenset(
    {"and", "or", "not", "is", "null", "true", "false", "case", "when", "then", "else", "end"}
    | {"between", "in"}
)
TOKENS = re.compile(
    r"\s+"
    r"|(?P<number>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\.\d+)"
    r"|(?P<text>'(?:[^']|'')*')"
    r"|(?P<quoted>\"(?:[^\"]|\"\")+\")"
    r"|(?P<word>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<operator><=|>=|<>|!=|\|\||[-+*/%=<>(),])"
)


@dataclass
class Token:
    kind: str  # number, text, column, function, word, operator
    value: str  # the column's name, the text's content, the word or operator as written


def _tokens(formula: str) -> list[Token]:
    found: list[Token] = []
    position = 0
    while position < len(formula):
        match = TOKENS.match(formula, position)
        if match is None or match.end() == position:
            shown = formula[position : position + 20]
            raise CleanError(f"Formula: cannot read '{shown}' (at character {position + 1}).")
        position = match.end()
        kind = match.lastgroup
        if kind is None:
            continue  # spaces
        text = match.group(kind)
        if kind == "text":
            found.append(Token("text", text[1:-1].replace("''", "'")))
        elif kind == "quoted":
            found.append(Token("column", text[1:-1].replace('""', '"')))
        elif kind == "word":
            found.append(Token("word", text))
        else:
            found.append(Token(kind, text))
    for index, token in enumerate(found):  # a word before '(' is a function, else a column
        if token.kind != "word":
            continue
        following = found[index + 1] if index + 1 < len(found) else None
        if following is not None and following.value == "(":
            token.kind = "function"
        elif token.value.lower() not in WORDS:
            token.kind = "column"
    return found


def _column(name: str, columns: list[str]) -> str:
    if name in columns:
        return name
    folded = [c for c in columns if c.lower() == name.lower()]
    if len(folded) == 1:
        return folded[0]
    raise CleanError(
        f"Formula: no column '{name}' at this step (write a name with spaces in "
        'double quotes: "unit price").'
    )


def to_sql(formula: str, columns: list[str]) -> str:
    """The formula as a DuckDB expression on `columns`, in parentheses; refused when it names
    something else than these columns, the words and the functions above."""
    tokens = _tokens(formula)
    if not tokens:
        raise CleanError("Formula: empty.")
    parts: list[str] = []
    depth = 0
    for token in tokens:
        if token.kind == "column":
            parts.append(ident(_column(token.value, columns)))
        elif token.kind == "function":
            name = token.value.lower()
            if name not in FUNCTIONS:
                raise CleanError(
                    f"Formula: '{token.value}' is not one of the functions a step may use "
                    f"({', '.join(sorted(FUNCTIONS))})."
                )
            parts.append(name)
        elif token.kind == "word":
            parts.append(token.value.upper())
        elif token.kind == "text":
            parts.append(lit(token.value))
        elif token.kind == "number":
            parts.append(token.value)
        else:
            depth += token.value == "("
            depth -= token.value == ")"
            if depth < 0:
                raise CleanError("Formula: a ')' closes nothing.")
            parts.append("<>" if token.value == "!=" else token.value)
    if depth:
        raise CleanError("Formula: a '(' is not closed.")
    return f"({' '.join(parts)})"


def columns_in(formula: str | None) -> list[str]:
    """The column names a formula reads, as written (for the replay's matching, 101)."""
    if not formula:
        return []
    try:
        return list(dict.fromkeys(t.value for t in _tokens(formula) if t.kind == "column"))
    except CleanError:
        return []


def renamed(formula: str | None, names: dict[str, str]) -> str | None:
    """The formula with its column names replaced (`names`: old → new), written quoted."""
    if not formula or not names:
        return formula
    try:
        tokens = _tokens(formula)
    except CleanError:
        return formula
    if not any(t.kind == "column" and t.value in names for t in tokens):
        return formula
    out: list[str] = []
    for token in tokens:
        if token.kind == "column":
            out.append(quoted(names.get(token.value, token.value)))
        elif token.kind == "text":
            out.append("'" + token.value.replace("'", "''") + "'")
        else:
            out.append(token.value)
    return " ".join(out)


def quoted(name: str) -> str:
    """A column name as a formula writes it: bare when it can be, else double-quoted."""
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) and name.lower() not in WORDS:
        return name
    return '"' + name.replace('"', '""') + '"'
