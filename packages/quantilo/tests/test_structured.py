"""JSON and JSON Lines → linked tables (083): records, flattening rules, keys, limits, Parquet."""

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

import quantilo as dx
from quantilo.ingestion.structured import (
    StructuredError,
    StructuredLayout,
    StructuredOptions,
    read_structured,
)

FIXTURES = Path(__file__).parent / "fixtures" / "structured"


def write_json(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def column_rules(layout: StructuredLayout, table: str) -> dict[str, str]:
    return {column.name: column.rule for column in layout.table(table).columns}


class TestOrders:
    """Orders with line items: a parent table and linked child tables (R-SST-15, R-SST-14)."""

    @pytest.fixture(scope="class")
    @classmethod
    def layout(cls) -> StructuredLayout:
        return dx.read_structured(FIXTURES / "orders.json")

    def test_three_linked_tables(self, layout: StructuredLayout) -> None:
        assert [table.name for table in layout.tables] == ["orders", "orders.items", "orders.tags"]
        assert layout.records_rule == "R-SST-01"
        assert layout.summary() == (
            "3 tables from orders.json: orders (4 rows), orders.items (8), orders.tags (6)."
        )

    def test_objects_flattened_into_dotted_columns(self, layout: StructuredLayout) -> None:
        rules = column_rules(layout, "orders")
        assert rules["customer.address.city"] == "R-SST-10"
        assert rules["items.count"] == "R-SST-15"
        assert layout.rows("orders", 1)[0]["customer.name"] == "Ana Lopez"
        assert layout.table("orders").column("customer.name").reason == "text"

    def test_child_rows_point_to_their_parent(self, layout: StructuredLayout) -> None:
        items = layout.arrow("orders.items").to_pylist()
        assert items[0] == {
            "_id": 0,
            "_parent_id": 0,
            "_position": 0,
            "sku": "A-1",
            "qty": 2,
            "price": 9.5,
        }
        assert [row["_parent_id"] for row in items] == [0, 0, 1, 2, 2, 2, 3, 3]
        counts = layout.arrow("orders").column("items.count").to_pylist()
        assert sum(counts) == layout.table("orders.items").rows

    def test_relations(self, layout: StructuredLayout) -> None:
        relations = [(r.parent, r.child, r.parent_key, r.child_key) for r in layout.relations]
        assert relations == [
            ("orders", "orders.items", "_id", "_parent_id"),
            ("orders", "orders.tags", "_id", "_parent_id"),
        ]
        assert layout.table("orders.items").locator == "$[*].items[*]"

    def test_tags_child_table_and_joined_column(self, layout: StructuredLayout) -> None:
        assert layout.table("orders.tags").kind == "scalars"
        assert [c.name for c in layout.table("orders.tags").columns] == [
            "_parent_id",
            "_position",
            "value",
        ]
        assert layout.arrow("orders").column("tags").to_pylist() == [
            "gift; express",
            "",
            "express",
            "gift; fragile; express",
        ]

    def test_missing_and_explicit_nulls_are_both_counted(self, layout: StructuredLayout) -> None:
        status = layout.table("orders").column("status")
        assert (status.present, status.missing, status.nulls) == (3, 1, 0)
        coupon = layout.table("orders").column("coupon")
        assert (coupon.present, coupon.nulls, coupon.missing) == (2, 1, 2)
        assert any("50%" in note for note in coupon.notes)

    def test_natural_key_and_types(self, layout: StructuredLayout) -> None:
        assert layout.table("orders").natural_key == "id"
        assert layout.table("orders.items").column("price").dtype == "float"
        assert layout.table("orders.items").column("qty").dtype == "int"

    def test_confident(self, layout: StructuredLayout) -> None:
        assert layout.confidence == 1.0
        assert layout.level == "high"

    def test_same_result_twice(self, layout: StructuredLayout) -> None:
        again = dx.read_structured(FIXTURES / "orders.json")
        assert dx.to_dict(again) == dx.to_dict(layout)
        for table in layout.tables:
            assert again.arrow(table.name).equals(layout.arrow(table.name))

    def test_to_dict_is_json(self, layout: StructuredLayout) -> None:
        data = layout.to_dict()
        json.dumps(data)
        assert data["level"] == "high"
        assert data["tables"][1]["columns"][3]["rule"] == "R-SST-10"


class TestRecords:
    def test_api_response_records_and_meta(self) -> None:
        layout = read_structured(FIXTURES / "api_response.json")
        assert layout.records_path == "results"
        assert layout.records_rule == "R-SST-02"
        meta = layout.table("_meta")
        assert meta.selected is False
        assert meta.rule == "R-SST-05"
        assert layout.rows("_meta")[0]["meta.total"] == 42
        assert layout.rows("_meta")[0]["links.next"].startswith("https://")
        assert layout.table("results").selected

    def test_numbers_and_numeric_text_read_as_numbers(self) -> None:
        layout = read_structured(FIXTURES / "api_response.json")
        score = layout.table("results").column("profile.score")
        assert (score.dtype, score.rule) == ("float", "R-SST-16")
        assert layout.arrow("results").column("profile.score").to_pylist() == [
            12.5,
            8.0,
            15.25,
            9.0,
            11.0,
        ]

    def test_null_object_leaves_its_columns_empty(self) -> None:
        layout = read_structured(FIXTURES / "api_response.json")
        assert layout.arrow("results").column("last_login.ip").to_pylist() == [
            "10.0.0.1",
            None,
            None,
            "10.0.0.7",
            "10.0.0.9",
        ]

    def test_candidates_are_listed(self, tmp_path: Path) -> None:
        doc = {"data": [{"a": 1}] * 6, "included": [{"b": 2}] * 4}
        layout = read_structured(write_json(tmp_path / "two.json", doc))
        assert [(c.path, c.count) for c in layout.candidates] == [("data", 6), ("included", 4)]
        assert layout.records_path == "data"
        assert layout.confidence == 0.8  # another candidate holds >= 50 % as many (R-CONF-01)
        assert layout.level == "high"

    def test_records_path_chosen(self, tmp_path: Path) -> None:
        doc = {"data": [{"a": 1}] * 6, "included": [{"b": 2}] * 4}
        layout = dx.read_structured(write_json(tmp_path / "two.json", doc), records_path="included")
        assert layout.records_rule == "R-OVR"
        assert layout.table("included").rows == 4

    def test_unknown_records_path(self, tmp_path: Path) -> None:
        with pytest.raises(StructuredError, match="not in the document"):
            dx.read_structured(write_json(tmp_path / "a.json", {"a": []}), records_path="b.c")

    def test_records_path_with_dotted_keys(self, tmp_path: Path) -> None:
        doc = {"1. data": {"rows": [{"x": 1}, {"x": 2}]}}
        layout = dx.read_structured(
            write_json(tmp_path / "d.json", doc), records_path="1. data.rows"
        )
        assert layout.table("rows").rows == 2

    def test_config_is_records_with_much_around_them(self) -> None:
        layout = read_structured(FIXTURES / "config.json")
        assert layout.records_path == "servers"
        assert layout.level == "high"
        assert layout.confidence == 0.8
        assert any("check the records path" in hit.message for hit in layout.reasons)

    def test_config_as_one_record(self) -> None:
        layout = dx.read_structured(FIXTURES / "config.json", records_path="")
        table = layout.tables[0]
        assert table.name == "config"
        assert table.rows == 1
        row = layout.rows("config")[0]
        assert row["database.primary.pool.max"] == 10
        assert row["app.features.export.parquet"] is True
        assert row["servers.count"] == 2
        assert layout.table("config.servers").rows == 2
        # two URLs, always two: positional columns (R-SST-13)
        assert row["allowed_origins[1]"] == "https://b.example"

    def test_root_list_of_values(self, tmp_path: Path) -> None:
        layout = read_structured(write_json(tmp_path / "v.json", [3, 1, 4]))
        assert layout.arrow("v").column("value").to_pylist() == [3, 1, 4]

    def test_root_dictionary_of_dates(self, tmp_path: Path) -> None:
        doc = {f"2024-01-{day:02d}": day * 10 for day in range(1, 16)}
        layout = read_structured(write_json(tmp_path / "daily.json", doc))
        assert layout.records_rule == "R-SST-07"
        assert layout.rows("daily", 2) == [
            {"_id": 0, "key": "2024-01-01", "value": 10},
            {"_id": 1, "key": "2024-01-02", "value": 20},
        ]


class TestFlattening:
    def test_dictionary_of_dates_becomes_a_long_table(self) -> None:
        layout = read_structured(FIXTURES / "daily_prices.json")
        series = layout.tables[1]
        assert series.kind == "map"
        assert series.rule == "R-SST-12"
        assert series.rows == 12
        assert "look like values" in series.reason
        first = layout.rows(series.name, 1)[0]
        assert first["key"] == "2024-01-02"
        assert first["1._open"] == "185.00"
        assert first["_parent_id"] == 0
        assert len(layout.tables[0].columns) == 4  # _id and the three Meta Data fields

    def test_many_sparse_keys_are_a_dictionary(self, tmp_path: Path) -> None:
        records = [
            {"id": i, "scores": {f"player{j}": j for j in range(i * 3, i * 3 + 3)}}
            for i in range(40)
        ]
        layout = read_structured(write_json(tmp_path / "s.json", records))
        scores = layout.table("s.scores")
        assert scores.kind == "map"
        assert "on average" in scores.reason
        assert scores.rows == 120
        assert scores.column("value").dtype == "int"

    def test_more_than_200_keys_is_a_dictionary(self, tmp_path: Path) -> None:
        doc = [{"counts": {f"w{j}": j for j in range(250)}}]
        layout = read_structured(write_json(tmp_path / "w.json", doc))
        assert layout.table("w.counts").rows == 250
        assert "over 200" in layout.table("w.counts").reason

    def test_wide_records_are_never_a_dictionary(self, tmp_path: Path) -> None:
        records = [{f"c{j}": j for j in range(300)} for _ in range(3)]
        layout = read_structured(write_json(tmp_path / "wide.json", records))
        assert len(layout.tables[0].columns) == 301

    def test_point_coordinates_become_positional_columns(self) -> None:
        layout = read_structured(FIXTURES / "places.geojson")
        rules = column_rules(layout, "features")
        assert rules["geometry.coordinates[0]"] == "R-SST-13"
        assert layout.rows("features", 1)[0]["geometry.coordinates[1]"] == 48.8566
        assert layout.table("_meta").selected is False

    def test_polygons_stay_json_text(self) -> None:
        layout = read_structured(FIXTURES / "shapes.geojson")
        column = layout.table("features").column("geometry.coordinates")
        assert (column.dtype, column.rule) == ("json", "R-SST-17b")
        assert json.loads(layout.rows("features", 1)[0]["geometry.coordinates"])[0][1] == [1, 0]

    def test_mixed_shapes_stay_json_text(self, tmp_path: Path) -> None:
        records = [{"v": 1}, {"v": {"a": 2}}, {"v": [1, 2]}, {"v": "x"}]
        layout = read_structured(write_json(tmp_path / "m.json", records))
        column = layout.table("m").column("v")
        assert (column.dtype, column.rule) == ("json", "R-SST-17")
        assert layout.arrow("m").column("v").to_pylist() == ["1", '{"a":2}', "[1,2]", '"x"']
        assert layout.confidence == 0.95

    def test_mixed_scalars_kept_as_text(self, tmp_path: Path) -> None:
        records = [{"v": 1}, {"v": "high"}, {"v": 2.5}, {"v": True}]
        layout = read_structured(write_json(tmp_path / "t.json", records))
        column = layout.table("t").column("v")
        assert (column.dtype, column.rule) == ("string", "R-SST-16b")
        assert layout.arrow("t").column("v").to_pylist() == ["1", "high", "2.5", "true"]

    def test_unreadable_values_left_empty_and_counted(self, tmp_path: Path) -> None:
        records = [{"v": i} for i in range(40)] + [{"v": "n/a"}]
        layout = read_structured(write_json(tmp_path / "u.json", records))
        column = layout.table("u").column("v")
        assert (column.dtype, column.rule, column.unreadable) == ("int", "R-SST-16", 1)
        assert column.examples == ["n/a"]
        assert layout.arrow("u").column("v").to_pylist()[-1] is None
        assert layout.confidence == 0.95

    def test_singleton_lists_read_as_objects(self, tmp_path: Path) -> None:
        records = [{"id": 1, "address": [{"city": "Lyon"}]}, {"id": 2, "address": []}]
        layout = read_structured(write_json(tmp_path / "p.json", records))
        assert len(layout.tables) == 1
        assert column_rules(layout, "p") == {
            "_id": "R-KEY-01",
            "id": "R-SST-10",
            "address.city": "R-SST-10",
        }

    def test_lists_always_empty_keep_their_length(self, tmp_path: Path) -> None:
        layout = read_structured(write_json(tmp_path / "e.json", [{"a": []}, {"a": []}]))
        assert column_rules(layout, "e")["a.count"] == "R-SST-14b"

    def test_deep_objects_go_to_extra(self, tmp_path: Path) -> None:
        deep = {"a": {"b": {"c": {"d": {"e": {"f": 1}}}}}, "x": 1}
        layout = read_structured(write_json(tmp_path / "deep.json", [deep, deep]))
        rules = column_rules(layout, "deep")
        assert "a.b.c.d.e.f" not in rules
        assert rules["_extra"] == "R-SST-18"
        assert json.loads(layout.rows("deep", 1)[0]["_extra"]) == {"a.b.c.d.e": {"f": 1}}
        assert layout.confidence == 0.9

    def test_column_limit(self, tmp_path: Path) -> None:
        records = [{f"c{j}": j for j in range(12)}]
        options = StructuredOptions(max_columns=10)
        layout = read_structured(write_json(tmp_path / "c.json", records), options)
        names = [c.name for c in layout.tables[0].columns]
        assert names[-1] == "_extra"
        assert len(names) == 1 + 10 + 1  # _id, 10 columns, _extra
        assert json.loads(layout.rows("c")[0]["_extra"]) == {"c10": 10, "c11": 11}

    def test_too_many_tables_folds_the_deepest_lists(self, tmp_path: Path) -> None:
        record: dict[str, Any] = {f"l{j}": [{"a": 1}, {"a": 2}] for j in range(4)}
        record["l0"] = [{"a": 1, "inner": [{"b": 1}, {"b": 2}]}, {"a": 2, "inner": [{"b": 3}]}]
        options = StructuredOptions(max_tables=4)
        layout = read_structured(write_json(tmp_path / "t.json", [record, record]), options)
        assert len(layout.tables) == 4
        assert "t.l0.inner" not in [table.name for table in layout.tables]
        assert layout.table("t.l0").column("inner").rule == "R-SST-21"
        assert any(hit.rule == "R-SST-21" for hit in layout.reasons)

    def test_duplicate_keys_last_wins(self, tmp_path: Path) -> None:
        path = tmp_path / "dup.json"
        path.write_text('[{"a": 1, "a": 2}, {"a": 3}]', encoding="utf-8")
        layout = read_structured(path)
        assert layout.arrow("dup").column("a").to_pylist() == [2, 3]
        assert any(hit.rule == "R-SST-23" and "(a)" in hit.message for hit in layout.reasons)

    def test_nan_and_big_integers(self, tmp_path: Path) -> None:
        path = tmp_path / "n.json"
        big = "9" * 5000
        path.write_text(f'[{{"x": NaN, "y": {big}}}, {{"x": 1.5, "y": 1}}]', encoding="utf-8")
        layout = read_structured(path)
        rules = {hit.rule for hit in layout.reasons}
        assert {"R-LIM-03", "R-LIM-04"} <= rules
        assert layout.table("n").column("y").dtype == "string"
        assert layout.arrow("n").column("y").to_pylist()[0] == big

    def test_names_sanitized_and_reserved(self, tmp_path: Path) -> None:
        records = [{"_id": "abc", "prénom": "Zoé", "a b/c": 1, "a_b_c": 2}]
        layout = read_structured(write_json(tmp_path / "n.json", records))
        assert [c.name for c in layout.tables[0].columns] == [
            "_id",
            "_id~2",
            "prénom",
            "a_b_c",
            "a_b_c~2",
        ]


class TestOptions:
    def test_one_table_explodes_the_only_list(self) -> None:
        layout = dx.read_structured(FIXTURES / "orders.json", one_table=True)
        main = layout.tables[0]
        assert main.rule == "R-SST-15b"
        assert main.rows == 8
        assert "4 orders → 8 rows (×2.0)" in main.reason
        rows = layout.rows("orders", 3)
        assert [(r["id"], r["items._position"], r["items.sku"]) for r in rows] == [
            (1001, 0, "A-1"),
            (1001, 1, "B-7"),
            (1002, 0, "A-1"),
        ]
        assert "orders.items" not in [table.name for table in layout.tables]
        assert "orders.tags" in [table.name for table in layout.tables]

    def test_one_table_refused_with_two_lists(self, tmp_path: Path) -> None:
        record = {"a": [{"x": 1}, {"x": 2}], "b": [{"y": 1}, {"y": 2}]}
        with pytest.raises(StructuredError, match="exactly one list of objects"):
            dx.read_structured(write_json(tmp_path / "two.json", [record]), one_table=True)

    def test_overrides(self) -> None:
        layout = dx.read_structured(
            FIXTURES / "orders.json",
            overrides={"items": "json", "customer.address": "skip", "tags": "separate"},
        )
        rules = column_rules(layout, "orders")
        assert rules["items"] == "R-OVR"
        assert "customer.address.city" not in rules
        assert [table.name for table in layout.tables] == ["orders", "orders.tags"]
        assert json.loads(layout.rows("orders", 1)[0]["items"])[1]["sku"] == "B-7"

    def test_flatten_a_dictionary(self) -> None:
        layout = dx.read_structured(
            FIXTURES / "daily_prices.json", overrides={"Time Series (Daily)": "flatten"}
        )
        assert len(layout.tables) == 1
        assert "Time_Series__Daily_.2024_01_02.1._open" in column_rules(layout, "daily_prices")

    def test_separate_object_table(self) -> None:
        layout = dx.read_structured(FIXTURES / "orders.json", overrides={"customer": "separate"})
        customer = layout.table("orders.customer")
        assert customer.kind == "object"
        assert layout.rows("orders.customer", 1)[0]["address.city"] == "Lyon"

    def test_flatten_list_of_objects_refused(self) -> None:
        with pytest.raises(StructuredError, match="can't be flattened"):
            dx.read_structured(FIXTURES / "orders.json", overrides={"items": "flatten"})

    def test_unknown_choice(self) -> None:
        with pytest.raises(StructuredError, match="Unknown choice"):
            dx.read_structured(FIXTURES / "orders.json", overrides={"items": "explode"})  # type: ignore[dict-item]

    def test_limit_reads_the_first_records(self) -> None:
        layout = dx.read_structured(FIXTURES / "orders.json", limit=2)
        assert layout.truncated
        assert layout.records == 2
        assert layout.table("orders.items").rows == 3
        assert layout.warnings


class TestJsonLines:
    def test_events_with_a_truncated_last_line(self) -> None:
        layout = read_structured(FIXTURES / "events.jsonl")
        assert layout.format == "jsonl"
        assert layout.records == 30
        assert layout.skipped_lines == [31]
        assert any(hit.rule == "R-LIM-05" for hit in layout.reasons)
        names = [table.name for table in layout.tables]
        assert "events.payload.commits" in names
        assert "events.payload.issue.labels" in names
        commits = layout.table("events.payload.commits")
        assert commits.rows == sum((i // 3) % 3 + 1 for i in range(0, 30, 3))

    def test_too_many_invalid_lines(self, tmp_path: Path) -> None:
        lines = [json.dumps({"a": i}) for i in range(190)] + ["{bad"] * 10
        path = tmp_path / "bad.jsonl"
        path.write_text("\n".join(lines), encoding="utf-8")
        with pytest.raises(StructuredError, match="over the 1 %"):
            read_structured(path)

    def test_json_file_holding_json_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "export.json"
        path.write_text("\n".join(json.dumps({"a": i}) for i in range(5)), encoding="utf-8")
        layout = read_structured(path)
        assert layout.format == "jsonl"
        assert layout.reasons[0].rule == "R-SST-04"
        assert any(hit.rule == "R-DET-02" for hit in layout.reasons)

    def test_schema_drift_after_the_sample(self, tmp_path: Path) -> None:
        lines = [{"a": i} for i in range(10)] + [{"a": 10, "late": "x", "obj": {"k": 1}}]
        path = tmp_path / "drift.jsonl"
        path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
        layout = read_structured(path, StructuredOptions(sample=5))
        late = layout.table("drift").column("late")
        assert (late.rule, late.present, late.missing) == ("R-SST-20", 1, 10)
        assert layout.arrow("drift").column("late").to_pylist()[-2:] == [None, "x"]
        assert layout.arrow("drift").column("obj").to_pylist()[-1] == '{"k":1}'
        assert layout.sampled == 5
        assert layout.confidence == 0.9

    def test_shape_drift_goes_to_extra(self, tmp_path: Path) -> None:
        lines = [{"a": i} for i in range(5)] + [{"a": {"oops": 1}}]
        path = tmp_path / "shape.jsonl"
        path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
        layout = read_structured(path, StructuredOptions(sample=5))
        assert json.loads(layout.rows("shape")[-1]["_extra"]) == {"a": {"oops": 1}}
        assert layout.table("shape").warnings

    def test_batches_keep_columns_aligned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("quantilo.ingestion.structured.build.BATCH", 3)
        lines = [{"a": i} for i in range(7)] + [{"a": 7, "late": "x"}]
        path = tmp_path / "b.jsonl"
        path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
        table = read_structured(path, StructuredOptions(sample=2)).arrow("b")
        assert table.column("a").num_chunks == 3
        assert table.column("a").to_pylist() == list(range(8))
        assert table.column("late").to_pylist() == [None] * 7 + ["x"]

    def test_limit_on_json_lines(self) -> None:
        layout = dx.read_structured(FIXTURES / "events.jsonl", limit=4)
        assert layout.truncated
        assert layout.tables[0].rows == 4


class TestLimits:
    def test_nesting_too_deep(self, tmp_path: Path) -> None:
        path = tmp_path / "deep.json"
        path.write_text("[" * 70 + "]" * 70, encoding="utf-8")
        with pytest.raises(StructuredError, match="Nesting too deep"):
            read_structured(path)

    def test_nesting_far_too_deep(self, tmp_path: Path) -> None:
        path = tmp_path / "deeper.json"
        path.write_text('{"a":' * 100_000 + "1" + "}" * 100_000, encoding="utf-8")
        with pytest.raises(StructuredError, match="Nesting too deep"):
            read_structured(path)

    def test_size_limit(self) -> None:
        with pytest.raises(StructuredError, match="convert it to JSON Lines") as error:
            dx.read_structured(FIXTURES / "orders.json", max_bytes=100)
        assert error.value.rule == "R-LIM-01"

    def test_not_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text('{"a": 1,', encoding="utf-8")
        with pytest.raises(StructuredError, match="not valid JSON"):
            read_structured(path)

    def test_neither_json_nor_xml(self, tmp_path: Path) -> None:
        path = tmp_path / "data.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        with pytest.raises(StructuredError, match="not a JSON"):
            read_structured(path)

    def test_utf16_json(self, tmp_path: Path) -> None:
        path = tmp_path / "u16.json"
        path.write_text(json.dumps([{"a": "é"}]), encoding="utf-16")
        assert read_structured(path).rows("u16")[0]["a"] == "é"


class TestOutputs:
    def test_parquet_one_file_per_selected_table(self, tmp_path: Path) -> None:
        layout = read_structured(FIXTURES / "api_response.json")
        written = layout.write_parquet(tmp_path / "out")
        assert sorted(written) == ["results", "results.roles"]
        for name, path in written.items():
            assert pq.read_table(path).equals(layout.arrow(name))
        everything = layout.write_parquet(tmp_path / "all", selected_only=False)
        assert "_meta" in everything

    def test_open_as_dataset(self, tmp_path: Path) -> None:
        layout = read_structured(FIXTURES / "orders.json")
        dataset = layout.open("orders.items", cache_dir=tmp_path)
        assert len(dataset) == 8
        assert "price" in dataset.columns

    def test_unknown_table(self) -> None:
        layout = read_structured(FIXTURES / "orders.json")
        with pytest.raises(KeyError, match=r"orders\.items"):
            layout.arrow("nope")
