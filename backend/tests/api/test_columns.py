from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import default_samples_dir


def by_name(columns: list[dict]) -> dict[str, dict]:
    return {column["name"]: column for column in columns}


@pytest.fixture
def traps_id(client: TestClient) -> str:
    path = default_samples_dir() / "typing_traps.csv"
    response = client.post("/api/v1/datasets", json={"path": str(path)})
    assert response.status_code == 201, response.text
    return response.json()["dataset"]["id"]


def patch(client: TestClient, dataset_id: str, *changes: dict) -> dict[str, dict]:
    response = client.patch(f"/api/v1/datasets/{dataset_id}/columns", json={"changes": changes})
    assert response.status_code == 200, response.text
    return by_name(response.json())


def test_types_are_detected_at_import(client: TestClient, traps_id: str) -> None:
    columns = by_name(client.get(f"/api/v1/datasets/{traps_id}/columns").json())

    price = columns["price_eur"]
    assert (price["type"], price["role"], price["role_label"]) == (
        "numeric_continuous",
        "amount",
        "Amount",
    )
    assert price["summary"]["invalid_count"] == 0
    assert price["summary"]["examples"]
    assert len(price["summary"]["histogram"]) == 12
    assert columns["segment"]["order"] == ["low", "medium", "high"]
    assert [c["label"] for c in columns["segment"]["summary"]["histogram"]] == [
        "low",
        "medium",
        "high",
    ]
    assert columns["income"]["summary"]["null_count"] > 0  # "N/A" values


def test_reasons_come_with_their_codes(client: TestClient, traps_id: str) -> None:
    """M02: each reason with its template code and vars, read back from the database."""
    columns = client.get(f"/api/v1/datasets/{traps_id}/columns").json()
    for column in columns:
        assert [d["text"] for d in column["reason_details"]] == column["reasons"]
    price = by_name(columns)["price_eur"]["reason_details"]
    assert {"text": "name suggests amount", "code": "TYP.name", "vars": {"role": "amount"}} in price


def test_changing_a_type_reports_invalid_values(client: TestClient, traps_id: str) -> None:
    before = client.get(f"/api/v1/datasets/{traps_id}").json()["types_version"]

    city = patch(client, traps_id, {"name": "city", "type": "numeric_continuous"})["city"]

    assert city["type"] == "numeric_continuous"
    assert city["overridden"] is True
    assert city["summary"]["invalid_count"] == 600
    assert client.get(f"/api/v1/datasets/{traps_id}").json()["types_version"] == before + 1


def test_conversion_preview_does_not_save(client: TestClient, traps_id: str) -> None:
    response = client.get(
        f"/api/v1/datasets/{traps_id}/columns/conversion",
        params={"column": "zip_code", "type": "numeric_discrete"},
    )

    assert response.status_code == 200
    assert response.json()["parser"]["kind"] == "number"
    zip_code = by_name(client.get(f"/api/v1/datasets/{traps_id}/columns").json())["zip_code"]
    assert zip_code["type"] == "categorical_nominal"


def test_role_exclusion_order_and_reset(client: TestClient, traps_id: str) -> None:
    columns = patch(
        client,
        traps_id,
        {"name": "comment", "excluded": True},
        {"name": "segment", "order": ["high", "medium", "low"]},
        {"name": "age", "role": ""},
    )
    assert columns["comment"]["excluded"] is True
    assert columns["segment"]["order"] == ["high", "medium", "low"]
    assert columns["age"]["role"] is None

    reset = patch(
        client, traps_id, {"name": "segment", "reset": True}, {"name": "age", "reset": True}
    )
    assert reset["segment"]["order"] == ["low", "medium", "high"]
    assert reset["age"]["role"] == "age"
    assert reset["age"]["overridden"] is False


def test_unknown_role_is_rejected(client: TestClient, traps_id: str) -> None:
    response = client.patch(
        f"/api/v1/datasets/{traps_id}/columns",
        json={"changes": [{"name": "age", "role": "astrology"}]},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_role"


def test_choosing_the_detected_type_clears_the_override(client: TestClient, traps_id: str) -> None:
    patch(client, traps_id, {"name": "country", "type": "text"})

    country = patch(client, traps_id, {"name": "country", "type": "categorical_nominal"})["country"]

    assert country["overridden"] is False


def test_rerunning_detection_keeps_corrections(client: TestClient, traps_id: str) -> None:
    patch(client, traps_id, {"name": "satisfaction", "type": "categorical_ordinal"})

    response = client.post(f"/api/v1/datasets/{traps_id}/columns/detect")

    assert by_name(response.json())["satisfaction"]["type"] == "categorical_ordinal"


def test_review_marks_dataset(client: TestClient, traps_id: str) -> None:
    assert client.get(f"/api/v1/datasets/{traps_id}").json()["types_reviewed_at"] is None

    response = client.post(f"/api/v1/datasets/{traps_id}/columns/review")

    assert response.json()["types_reviewed_at"] is not None


def test_columns_of_a_dataset_imported_before_detection_are_detected_lazily(
    client: TestClient, traps_id: str, tmp_path: Path
) -> None:
    from app.core import db

    connection = db.connect(Path(client.app.state.workspace.db_path))  # type: ignore[attr-defined]
    connection.execute("DELETE FROM columns WHERE dataset_id = ?", (traps_id,))
    connection.commit()
    connection.close()

    columns = client.get(f"/api/v1/datasets/{traps_id}/columns").json()

    assert len(columns) == 24


def test_roles_are_listed(client: TestClient) -> None:
    roles = client.get("/api/v1/column-roles").json()

    assert {"id": "postal_code", "label": "Postal code"} in roles
