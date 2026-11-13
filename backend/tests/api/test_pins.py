"""099 — Pinned rows over HTTP: pin with a label and a colour, one colour per label, rename,
unpin, rows found again after a re-import, what distinguishes the pinned rows, and the labels as
the column of a cleaned version."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core.tools import ToolContext, ToolError
from tests.api.test_clean import save
from tests.api.test_profiling import import_path

BIG = [3, 17, 42, 55, 80, 101, 150, 199]


def shop(path: Path, n: int = 300) -> Path:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "amount": rng.gamma(4, 50, n).round(2),
            "channel": rng.choice(["store", "web"], n, p=[0.7, 0.3]),
        }
    )
    frame.loc[BIG, "amount"] = rng.gamma(4, 600, len(BIG)).round(2)
    frame.loc[BIG, "channel"] = "web"
    frame.to_csv(path, index=False)
    return path


def url(dataset: str, tail: str = "") -> str:
    return f"/api/v1/datasets/{dataset}/pins{tail}"


def test_pin_label_colour_and_unpin(client: TestClient, tmp_path: Path) -> None:
    dataset = import_path(client, shop(tmp_path / "shop.csv"))
    assert client.get(url(dataset)).json() == {
        "pins": [],
        "labels": [],
        "missing": 0,
        "limit": 5000,
        "parent": None,
        "inherited": 0,
    }

    first = client.put(url(dataset), json={"rows": [3, 17], "label": " big ", "color": 2}).json()
    assert [(p["row"], p["label"], p["color"], p["found"]) for p in first["pins"]] == [
        (3, "big", 2, True),
        (17, "big", 2, True),
    ]
    assert first["pins"][0]["key"].startswith("row:")  # no identifier: a fingerprint (R-PIN-01)

    # R-PIN-03: one colour per label; a row pinned again takes the new label
    again = client.put(url(dataset), json={"rows": [42], "label": "big", "color": 5}).json()
    assert {p["color"] for p in again["pins"]} == {5}
    other = client.put(url(dataset), json={"rows": [17], "label": "check", "color": 1}).json()
    assert other["labels"] == [
        {"label": "big", "color": 5, "rows": 2},
        {"label": "check", "color": 1, "rows": 1},
    ]

    renamed = client.patch(url(dataset, "/labels"), json={"label": "check", "to": "big"}).json()
    assert renamed["labels"] == [{"label": "big", "color": 5, "rows": 3}]
    recoloured = client.patch(url(dataset, "/labels"), json={"label": "big", "color": 7}).json()
    assert recoloured["labels"][0]["color"] == 7
    missing = client.patch(url(dataset, "/labels"), json={"label": "nope", "color": 1})
    assert missing.status_code == 404

    keys = [p["key"] for p in recoloured["pins"][:2]]
    left = client.post(url(dataset, "/remove"), json={"keys": keys}).json()
    assert len(left["pins"]) == 1


def test_pins_are_refused_beyond_the_rows_and_without_a_label(
    client: TestClient, tmp_path: Path
) -> None:
    dataset = import_path(client, shop(tmp_path / "shop.csv"))
    outside = client.put(url(dataset), json={"rows": [5_000], "label": "x", "color": 1})
    assert outside.status_code == 422 and outside.json()["error"]["code"] == "unknown_row"
    blank = client.put(url(dataset), json={"rows": [1], "label": "   ", "color": 1})
    assert blank.status_code == 422
    colour = client.put(url(dataset), json={"rows": [1], "label": "x", "color": 9})
    assert colour.status_code == 422


def test_pins_follow_their_rows_after_a_reimport(client: TestClient, tmp_path: Path) -> None:
    path = tmp_path / "shop.csv"
    frame = pd.read_csv(shop(path))
    dataset = import_path(client, path)
    client.put(url(dataset), json={"rows": [3, 17], "label": "big", "color": 2})

    # the file comes back with its rows reversed, and row 17 removed
    flipped = frame.drop(index=17).iloc[::-1]
    flipped.to_csv(path, index=False)
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 10))
    assert client.post(f"/api/v1/datasets/{dataset}/reimport").json()["status"] == "done"

    found = client.get(url(dataset)).json()
    assert found["missing"] == 1  # R-PIN-02: said, never dropped silently
    assert [(p["row"], p["found"]) for p in found["pins"]] == [
        (len(flipped) - 1 - 3, True),
        (17, False),
    ]


def test_what_distinguishes_pinned_rows(client: TestClient, tmp_path: Path) -> None:
    dataset = import_path(client, shop(tmp_path / "shop.csv"))
    client.put(url(dataset), json={"rows": BIG, "label": "big", "color": 2})
    client.put(url(dataset), json={"rows": [7], "label": "odd", "color": 3})

    found = client.get(url(dataset, "/compare"), params={"label": "big"}).json()
    assert found["label"] == "big" and found["pinned"] == len(BIG)
    assert found["others"] == 300 - len(BIG)
    assert [t["column"] for t in found["traits"][:2]] == ["amount", "channel"]
    assert found["warnings"] == [f"{len(BIG)} pinned rows: read these differences as hints."]
    every = client.get(url(dataset, "/compare")).json()
    assert every["label"] is None and every["pinned"] == len(BIG) + 1
    alone = client.get(url(dataset, "/compare"), params={"label": "odd"}).json()
    assert alone["traits"] == [] and alone["reading"].startswith("Pin at least 2 rows")
    unknown = client.get(url(dataset, "/compare"), params={"label": "nope"})
    assert unknown.status_code == 404


def test_the_labels_become_a_column_of_a_cleaned_version(
    client: TestClient, tmp_path: Path
) -> None:
    dataset = import_path(client, shop(tmp_path / "shop.csv"))
    pins = client.put(url(dataset), json={"rows": [3, 17], "label": "big", "color": 2}).json()
    labels = {str(p["row"]): p["label"] for p in pins["pins"]}
    step = {"id": "s1", "op": "label_rows", "to": "pin_label", "labels": labels, "other": "none"}
    child = save(client, dataset, [step])

    names = [c["name"] for c in client.get(f"/api/v1/datasets/{child}/columns").json()]
    assert names[-1] == "pin_label"
    rows = client.get(f"/api/v1/datasets/{child}/rows", params={"limit": 20}).json()
    index = 1 + [c["name"] for c in rows["columns"]].index("pin_label")
    assert [row[index] for row in rows["rows"][:5]] == ["none", "none", "none", "big", "none"]


def test_the_agent_compares_pinned_rows(client: TestClient, tmp_path: Path) -> None:
    dataset = import_path(client, shop(tmp_path / "shop.csv"))
    client.put(url(dataset), json={"rows": BIG, "label": "big", "color": 2})
    registry = client.app.state.agent.registry  # type: ignore[attr-defined]
    tool = registry.get("compare_pinned_rows")
    assert tool.kind == "compute"
    output = tool.run(ToolContext(dataset), tool.parse({"label": "big"}))
    assert output.content["pinned_rows"] == len(BIG) and output.link.page == "data"
    assert output.content["differences"][0]["text"].startswith("amount: ")
    wrong = tool.parse({"label": "nope"})
    with pytest.raises(ToolError, match="No pinned row"):
        tool.run(ToolContext(dataset), wrong)


def test_a_cleaned_version_follows_the_parent_pins_and_data_shows_them_alone(
    client: TestClient, tmp_path: Path
) -> None:
    """099 bis: R-PIN-16, the parent's pins on the rows the cleaning kept; R-PIN-17, the Data
    page's rows narrowed to the pinned ones."""
    dataset = import_path(client, shop(tmp_path / "shop.csv"))
    client.put(url(dataset), json={"rows": [0, 3, 17], "label": "big", "color": 2})
    child = save(client, dataset, [{"id": "s1", "op": "drop_rows", "rows": [0]}])

    found = client.get(url(child)).json()
    assert [(p["row"], p["label"], p["inherited"]) for p in found["pins"]] == [
        (2, "big", True),
        (16, "big", True),
    ]
    assert found["inherited"] == 2 and "R-PIN-16" in found["parent"]
    assert "1 were on rows the cleaning removed" in found["parent"]
    # the child's own pin on a row takes over the inherited one
    client.put(url(child), json={"rows": [2], "label": "test", "color": 4})
    mine = {p["row"]: (p["label"], p["inherited"]) for p in client.get(url(child)).json()["pins"]}
    assert mine[2] == ("test", False) and mine[16] == ("big", True)

    rows = client.get(f"/api/v1/datasets/{child}/rows", params={"pinned": True}).json()
    assert rows["total"] == 2 and [r[0] for r in rows["rows"]] == [3, 17]  # shown from 1
    big = client.get(
        f"/api/v1/datasets/{child}/rows", params={"pinned": True, "pin_label": "big"}
    ).json()
    assert big["total"] == 1
    none = client.get(f"/api/v1/datasets/{dataset}/rows", params={"pinned": True, "pin_label": "x"})
    assert none.json()["total"] == 0
