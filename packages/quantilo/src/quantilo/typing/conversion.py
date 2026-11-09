"""Choose how to read a column when the user switches it to another analysis type."""

from quantilo.typing.knowledge_base import load
from quantilo.typing.types import NUMERIC_TYPES, AnalysisType, Parser


def default_boolean_parser() -> Parser:
    kb = load()
    trues = sorted({word for pair in kb.boolean_pairs for word in pair[0]})
    falses = sorted({word for pair in kb.boolean_pairs for word in pair[1]} - set(trues))
    return Parser(kind="boolean", true_values=trues, false_values=falses)


def parser_for(
    target: AnalysisType,
    candidates: dict[str, Parser],
    detected_type: AnalysisType,
    detected_parser: Parser,
) -> Parser:
    """Best parser for `target`, reusing what detection already measured on the sample."""
    if target == detected_type:
        return detected_parser
    if target in NUMERIC_TYPES:
        return candidates.get("number", Parser(kind="number"))
    if target == "datetime":
        return candidates.get("date", Parser(kind="date", format="%Y-%m-%d"))
    if target == "boolean":
        return candidates.get("boolean", default_boolean_parser())
    if target == "ignored":
        return Parser(kind="raw")
    return candidates.get("text", Parser(kind="text"))
