"""XML → dicts, lists and text (the xmltodict convention), on the standard library's expat.

`@name` for an attribute, `#text` for text next to attributes or children, repeated children
become a list. Namespace prefixes are dropped unless two names collide in one element.

Safety, without defusedxml (083 spec): the first entity declaration of any kind is refused,
external DTDs and entities are never loaded, undeclared entities are refused, nesting is bounded.
expat (>= 2.6) already stops entity bombs; this second layer does not depend on its version
(Pyodide's may differ).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn
from xml.parsers import expat

from quantilo.ingestion.structured.errors import StructuredError
from quantilo.ingestion.structured.readers import MAX_NESTING, check_size, too_deep

ENTITIES_REFUSED = (
    "Entity declarations are not allowed in XML files: they can hide an entity-expansion bomb "
    "or read local files."
)


def refuse_entity(*_: object) -> NoReturn:
    raise StructuredError(ENTITIES_REFUSED, rule="R-SEC-01")


def refuse_external(*_: object) -> NoReturn:
    raise StructuredError(
        "External entities are not allowed in XML files: they can read local files.",
        rule="R-SEC-01",
    )


def refuse_skipped(name: str, _is_parameter: int) -> NoReturn:
    raise StructuredError(
        f"Undeclared entity &{name}; (entities are not allowed in XML files).", rule="R-SEC-01"
    )


def local(name: str) -> str:
    return name.rsplit(":", 1)[-1]


@dataclass
class Frame:
    name: str
    attrs: dict[str, str]
    children: dict[str, Any] = field(default_factory=dict)
    text: list[str] = field(default_factory=list)

    def add(self, name: str, value: Any) -> None:
        if name not in self.children:
            self.children[name] = value
            return
        existing = self.children[name]
        if isinstance(existing, list):  # only repetition makes a list
            existing.append(value)
        else:
            self.children[name] = [existing, value]

    def value(self) -> Any:
        attrs = {
            key: value
            for key, value in self.attrs.items()
            if key != "xmlns" and not key.startswith("xmlns:")
        }
        text = "".join(self.text).strip()
        if not attrs and not self.children:
            return text or None
        items = [("@" + key, value) for key, value in attrs.items()]
        items += list(self.children.items())
        locals_ = [
            ("@" + local(key[1:])) if key.startswith("@") else local(key) for key, _ in items
        ]
        counts: dict[str, int] = {}
        for name in locals_:
            counts[name] = counts.get(name, 0) + 1
        result: dict[str, Any] = {}
        for (key, value), name in zip(items, locals_, strict=True):
            result[name if counts[name] == 1 else key] = value
        if text:
            result["#text"] = text
        return result


class TreeBuilder:
    def __init__(self) -> None:
        self.stack: list[Frame] = []
        self.root: dict[str, Any] | None = None

    def start(self, name: str, attrs: dict[str, str]) -> None:
        if len(self.stack) >= MAX_NESTING:
            raise too_deep()
        self.stack.append(Frame(name, attrs))

    def end(self, _name: str) -> None:
        frame = self.stack.pop()
        value = frame.value()
        if self.stack:
            self.stack[-1].add(frame.name, value)
        else:
            self.root = {local(frame.name): value}

    def data(self, text: str) -> None:
        if self.stack:
            self.stack[-1].text.append(text)


def parse_xml(source: Path | bytes) -> dict[str, Any]:
    """The document as {root element: value}; a refused or malformed file raises StructuredError."""
    parser = expat.ParserCreate()  # no namespace processing: "prefix:name" is kept, then dropped
    parser.buffer_text = True
    parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    parser.EntityDeclHandler = refuse_entity
    parser.UnparsedEntityDeclHandler = refuse_entity
    parser.ExternalEntityRefHandler = refuse_external
    parser.SkippedEntityHandler = refuse_skipped
    builder = TreeBuilder()
    parser.StartElementHandler = builder.start
    parser.EndElementHandler = builder.end
    parser.CharacterDataHandler = builder.data
    try:
        if isinstance(source, bytes):
            parser.Parse(source, True)
        else:
            with source.open("rb") as stream:
                parser.ParseFile(stream)
    except expat.ExpatError as error:
        message = expat.ErrorString(error.code) if error.code else str(error)
        raise StructuredError(
            f"This XML file is not well-formed: {message} (line {error.lineno}, "
            f"column {error.offset + 1})."
        ) from error
    except RecursionError as error:
        raise too_deep() from error
    if builder.root is None:
        raise StructuredError("This XML file has no root element.")
    return builder.root


def load_xml(path: Path, max_bytes: int | None) -> dict[str, Any]:
    check_size(path, max_bytes, "XML file", "split it into smaller files")
    return parse_xml(path)


def list_paths(node: Any, path: tuple[str, ...], found: set[tuple[str, ...]]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            child = (*path, key)
            if isinstance(value, list):
                found.add(child)
                for item in value:
                    list_paths(item, child, found)
            else:
                list_paths(value, child, found)


def wrap_singletons(node: Any, path: tuple[str, ...], lists: set[tuple[str, ...]]) -> int:
    """R-SST-06: an element repeated under one parent is a list under every parent."""
    wrapped = 0
    if isinstance(node, dict):
        for key, value in node.items():
            child = (*path, key)
            if child in lists and not isinstance(value, list):
                node[key] = value = [value]
                wrapped += 1
            if isinstance(value, list):
                for item in value:
                    wrapped += wrap_singletons(item, child, lists)
            else:
                wrapped += wrap_singletons(value, child, lists)
    return wrapped


def normalize_lists(root: dict[str, Any]) -> int:
    """Wrap the lone elements of repeated paths (R-SST-06); returns how many were wrapped."""
    lists: set[tuple[str, ...]] = set()
    list_paths(root, (), lists)
    return wrap_singletons(root, (), lists) if lists else 0
