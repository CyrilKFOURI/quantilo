"""Small helpers to build DuckDB SQL safely from values we control."""


def lit(value: str) -> str:
    """SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def ident(name: str) -> str:
    """SQL quoted identifier."""
    return '"' + name.replace('"', '""') + '"'
