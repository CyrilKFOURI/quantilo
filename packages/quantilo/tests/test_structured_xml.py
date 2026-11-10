"""XML → linked tables (083): the xmltodict convention on expat, and the refused documents."""

from pathlib import Path

import pytest

from quantilo.ingestion.structured import StructuredError, StructuredLayout, read_structured
from quantilo.ingestion.structured.xmltree import parse_xml

FIXTURES = Path(__file__).parent / "fixtures" / "structured"


class TestCatalogue:
    @pytest.fixture(scope="class")
    @classmethod
    def layout(cls) -> StructuredLayout:
        return read_structured(FIXTURES / "catalog.xml")

    def test_repeated_element_is_the_record(self, layout: StructuredLayout) -> None:
        assert layout.format == "xml"
        assert layout.records_rule == "R-SST-03"
        assert layout.records_path == "catalog.book"
        assert layout.tables[0].name == "book"
        assert layout.tables[0].rows == 3
        assert layout.table("book").locator == "$.catalog.book[*]"

    def test_attributes_text_and_namespaces(self, layout: StructuredLayout) -> None:
        first = layout.rows("book", 1)[0]
        assert first["@id"] == "bk101"
        assert first["price.@currency"] == "USD"
        assert first["price.#text"] == "44.95"  # text: the type detection (002) reads it
        assert first["description"].startswith("An in-depth")  # dc: prefix dropped
        assert layout.table("book").natural_key == "@id"

    def test_lone_element_is_a_list_everywhere(self, layout: StructuredLayout) -> None:
        assert any(hit.rule == "R-SST-06" for hit in layout.reasons)
        tags = layout.arrow("book.tag").to_pylist()
        assert [(row["_parent_id"], row["value"]) for row in tags] == [
            (0, "xml"),
            (0, "programming"),
            (1, "fantasy"),
        ]
        assert layout.arrow("book").column("tag.count").to_pylist() == [2, 1, None]

    def test_root_attributes_in_meta(self, layout: StructuredLayout) -> None:
        assert layout.rows("_meta")[0] == {"catalog.@updated": "2024-05-01"}


class TestFeed:
    def test_rss_items_with_namespace_collision(self) -> None:
        layout = read_structured(FIXTURES / "feed.xml")
        assert layout.records_path == "rss.channel.item"
        assert layout.rows("item", 1)[0]["creator"] == "Cyril"
        meta = layout.rows("_meta")[0]
        # <link> and <atom:link> collide once the prefix is dropped: the prefix is kept
        assert meta["rss.channel.link"] == "https://example.com/"
        assert meta["rss.channel.atom_link.@rel"] == "self"
        assert "rss.@xmlns:atom" not in meta and "rss.@xmlns_atom" not in meta
        assert layout.table("item.category").rows == 3


class TestConvention:
    def test_leaf_mixed_and_empty_elements(self) -> None:
        tree = parse_xml(b'<r a="1"><e/><t>hi</t><m k="v">text</m><n>  </n></r>')
        assert tree == {
            "r": {"@a": "1", "e": None, "t": "hi", "m": {"@k": "v", "#text": "text"}, "n": None}
        }

    def test_repeated_children_become_a_list(self) -> None:
        assert parse_xml(b"<r><i>1</i><i>2</i></r>") == {"r": {"i": ["1", "2"]}}

    def test_latin1_declaration(self) -> None:
        raw = '<?xml version="1.0" encoding="ISO-8859-1"?><r><v>Médoc</v></r>'.encode("latin-1")
        assert parse_xml(raw) == {"r": {"v": "Médoc"}}

    def test_doctype_without_entities_is_fine(self) -> None:
        assert parse_xml(b'<!DOCTYPE r SYSTEM "r.dtd"><r><v>1</v></r>') == {"r": {"v": "1"}}


class TestRefused:
    """Security (083 spec): no entity is ever declared, expanded or loaded."""

    def test_billion_laughs(self) -> None:
        with pytest.raises(StructuredError, match="Entity declarations are not allowed") as error:
            read_structured(FIXTURES / "bomb.xml")
        assert error.value.rule == "R-SEC-01"

    def test_external_entity(self) -> None:
        with pytest.raises(StructuredError, match="Entity declarations are not allowed"):
            read_structured(FIXTURES / "xxe.xml")

    def test_external_parameter_entity(self) -> None:
        with pytest.raises(StructuredError, match="Entity declarations are not allowed"):
            read_structured(FIXTURES / "xxe_param.xml")

    def test_undeclared_entity(self) -> None:
        with pytest.raises(StructuredError, match="not well-formed: undefined entity"):
            parse_xml(b"<r>&nope;</r>")

    def test_undeclared_entity_with_external_dtd(self) -> None:
        with pytest.raises(StructuredError, match="Undeclared entity &x;"):
            parse_xml(b'<!DOCTYPE r SYSTEM "file:///c:/windows/win.ini"><r>&x;</r>')

    def test_malformed(self) -> None:
        with pytest.raises(StructuredError, match=r"not well-formed.*line 1"):
            parse_xml(b"<r><a></r>")

    def test_nesting_too_deep(self) -> None:
        deep = b"<a>" * 70 + b"</a>" * 70
        with pytest.raises(StructuredError, match="Nesting too deep"):
            parse_xml(deep)

    def test_size_limit(self) -> None:
        from quantilo.ingestion.structured import StructuredOptions

        with pytest.raises(StructuredError, match="over the"):
            read_structured(FIXTURES / "catalog.xml", StructuredOptions(max_bytes=200))
