# ruff: noqa: RUF001  (derived names use their separator, as dates do)
"""058 and 059 through the API: derived columns, distinctive words, topics."""

import csv
import io
import json
import os
import random
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.api.test_profiling import import_path
from tests.api.test_segments import BASE, wait

API = "/api/v1/datasets"
THEMES = {
    "fit": ["size", "small", "large", "fit", "tight", "loose", "ordered"],
    "fabric": ["fabric", "soft", "cotton", "material", "thin", "quality", "wash"],
    "refund": ["refund", "return", "money", "late", "shipping", "order", "broken"],
}
FILLER = ["the", "dress", "was", "and", "really", "it", "is", "very", "my", "for"]
LANGUAGES = ["Python", "SQL", "JavaScript", "Rust", "Go", "Java"]


def rows(count: int = 600, swap: str | None = None) -> list[dict[str, object]]:
    rng = random.Random(4)
    found = []
    for i in range(count):
        theme = list(THEMES)[i % 3]
        words = rng.sample(THEMES[theme], 3) + rng.sample(FILLER, 4)
        rng.shuffle(words)
        text = " ".join(words).capitalize() + ("?" if i % 7 == 0 else ".")
        if swap:
            text = text.replace("refund", swap)
        found.append(
            {
                "review": text,
                "rating": 1 if theme == "refund" else 4 + i % 2,
                "age": 20 + i % 50,
                "skills": ";".join(rng.sample(LANGUAGES, 1 + i % 3)),
            }
        )
    return found


def write(path: Path, data: list[dict[str, object]]) -> Path:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, list(data[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(data)
    path.write_text(buffer.getvalue(), encoding="utf-8")
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 10))
    return path


@pytest.fixture
def source(tmp_path: Path) -> Path:
    return write(tmp_path / "reviews.csv", rows())


@pytest.fixture
def reviews(client: TestClient, source: Path) -> str:
    return import_path(client, source)


def test_derived_columns_are_listed_with_their_rule(client: TestClient, reviews: str) -> None:
    found = client.get(f"{API}/{reviews}/columns/derived").json()
    by_name = {c["name"]: c for c in found}
    length = by_name["review › Length"]
    assert length["type"] == "numeric_discrete"
    assert length["parser"]["kind"] == "derived"
    assert length["parser"]["derived"] == {
        "source": "review", "part": "length", "option": None, "topics": []
    }  # fmt: skip
    assert "(R-DER-01)" in length["reasons"][0]
    assert by_name["skills › has Python"]["type"] == "boolean"
    assert "skills › Count" in by_name
    plain = client.get(f"{API}/{reviews}/columns").json()
    assert not any("›" in c["name"] for c in plain)  # R-DER-07


def test_a_derived_column_filters_every_page(client: TestClient, reviews: str) -> None:
    filters = json.dumps([{"column": "review › Has question", "op": "in", "values": ["True"]}])
    overview = client.get(f"{API}/{reviews}/overview", params={"filters": filters}).json()
    assert overview["rows"] == len([i for i in range(600) if i % 7 == 0])
    chart = client.get(
        f"{API}/{reviews}/columns/chart",
        params={"column": "skills › has Python", "filters": filters},
    )
    assert chart.status_code == 200, chart.text


def test_compare_a_derived_column_and_a_text(client: TestClient, reviews: str) -> None:
    groups = client.get(f"{API}/{reviews}/compare", params={"x": "rating", "y": "review › Words"})
    assert groups.status_code == 200 and groups.json()["kind"] == "groups"
    words = client.get(f"{API}/{reviews}/compare", params={"x": "rating", "y": "review"}).json()
    assert words["kind"] == "words"
    first = {g["group"]: g for g in words["words"]["groups"]}["1"]["words"][0]
    assert first["word"] in THEMES["refund"] and first["elsewhere"] == 0


def test_topics_kept_as_a_column(client: TestClient, reviews: str) -> None:
    found = client.get(f"{API}/{reviews}/text/topics", params={"column": "review", "k": 3})
    assert found.status_code == 200, found.text
    topics = found.json()
    assert topics["k"] == 3 and not topics["kept"] and topics["agreement"] > 0.9
    kept = client.post(f"{API}/{reviews}/text/topics/keep", json={"column": "review", "k": 3})
    assert kept.status_code == 200 and kept.json()["kept"]
    names = {c["name"] for c in client.get(f"{API}/{reviews}/columns/derived").json()}
    assert "review › Topic" in names
    label = topics["topics"][0]["label"]
    filters = json.dumps([{"column": "review › Topic", "op": "in", "values": [label]}])
    overview = client.get(f"{API}/{reviews}/overview", params={"filters": filters}).json()
    assert overview["rows"] == topics["topics"][0]["texts"]
    again = client.get(f"{API}/{reviews}/text/topics", params={"column": "review", "k": 3}).json()
    assert again["kept"]
    assert (
        client.delete(f"{API}/{reviews}/text/topics", params={"column": "review"}).status_code
        == 204
    )
    names = {c["name"] for c in client.get(f"{API}/{reviews}/columns/derived").json()}
    assert "review › Topic" not in names


def test_topics_refuse_a_non_text_column(client: TestClient, reviews: str) -> None:
    response = client.get(f"{API}/{reviews}/text/topics", params={"column": "rating"})
    assert response.status_code == 422


def test_words_of_each_segment(client: TestClient, reviews: str) -> None:
    started = client.post(
        f"{BASE}/datasets/{reviews}/runs", json={"columns": ["rating", "review › Length"]}
    )
    assert started.status_code == 202, started.text
    job = wait(client, started.json())
    assert job["status"] == "done", job
    found = client.get(
        f"{BASE}/datasets/{reviews}/runs/{job['run_id']}/words", params={"column": "review"}
    )
    assert found.status_code == 200, found.text
    assert found.json()["by"] == "segment" and found.json()["groups"]
    refused = client.get(
        f"{BASE}/datasets/{reviews}/runs/{job['run_id']}/words", params={"column": "skills"}
    )
    assert refused.status_code == 422


def test_words_between_versions(client: TestClient, reviews: str, source: Path) -> None:
    write(source, rows(swap="parcel"))
    job = client.post(f"{API}/{reviews}/reimport").json()
    assert job["status"] == "done", job
    found = client.get(f"{API}/{reviews}/changes/words", params={"column": "review"}).json()
    assert found["available"], found
    groups = {g["group"]: g for g in found["words"]["groups"]}
    assert groups["this version"]["words"][0]["word"] == "parcel"
    assert any(w["word"] == "refund" for w in next(
        g for name, g in groups.items() if name != "this version"
    )["words"])  # fmt: skip
