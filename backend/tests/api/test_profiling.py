from pathlib import Path

from fastapi.testclient import TestClient

from app.config import default_samples_dir
from tests.factories import write_csv


def import_path(client: TestClient, path: Path) -> str:
    response = client.post("/api/v1/datasets", json={"path": str(path)})
    assert response.status_code == 201, response.text
    return response.json()["dataset"]["id"]


def test_overview_of_typing_traps(client: TestClient) -> None:
    dataset_id = import_path(client, default_samples_dir() / "typing_traps.csv")

    overview = client.get(f"/api/v1/datasets/{dataset_id}/overview").json()

    assert (overview["rows"], overview["columns"], overview["duplicate_rows"]) == (600, 24, 0)
    assert overview["types"]["numeric_continuous"] == 5
    assert overview["missing_pct"] > 0
    kinds = {alert["kind"] for alert in overview["alerts"]}
    assert {"ignored", "skewed", "high_cardinality"} <= kinds
    assert overview["column_stats"]["price_eur"]["skewness"] > 2


def test_duplicates_are_reported(client: TestClient, tmp_path: Path) -> None:
    path = write_csv(tmp_path / "dup.csv", "a,b,c\n1,x,x\n1,x,x\n2,y,y\n3,z,z\n")
    dataset_id = import_path(client, path)

    overview = client.get(f"/api/v1/datasets/{dataset_id}/overview").json()

    assert overview["duplicate_rows"] == 1
    assert overview["duplicate_columns"] == [["b", "c"]]
    assert {"duplicate_rows", "duplicate_columns"} <= {a["kind"] for a in overview["alerts"]}


def test_columns_with_evenly_repeated_values_are_not_duplicates(
    client: TestClient, tmp_path: Path
) -> None:
    # Each value appears an even number of times: an XOR of hashes would cancel out.
    path = write_csv(tmp_path / "even.csv", "a,b\nx,p\nx,p\ny,q\ny,q\n")
    dataset_id = import_path(client, path)

    overview = client.get(f"/api/v1/datasets/{dataset_id}/overview").json()

    assert overview["duplicate_columns"] == []


def test_excluding_a_column_refreshes_the_overview(client: TestClient) -> None:
    dataset_id = import_path(client, default_samples_dir() / "sales_timeseries.csv")
    before = client.get(f"/api/v1/datasets/{dataset_id}/overview").json()

    client.patch(
        f"/api/v1/datasets/{dataset_id}/columns",
        json={"changes": [{"name": "promo", "excluded": True}]},
    )
    after = client.get(f"/api/v1/datasets/{dataset_id}/overview").json()

    assert after["used_columns"] == before["used_columns"] - 1


def test_chart_endpoint(client: TestClient) -> None:
    dataset_id = import_path(client, default_samples_dir() / "sales_timeseries.csv")

    timeline = client.get(
        f"/api/v1/datasets/{dataset_id}/columns/chart", params={"column": "date"}
    ).json()
    months = client.get(
        f"/api/v1/datasets/{dataset_id}/columns/chart",
        params={"column": "date", "view": "month"},
    ).json()
    stores = client.get(
        f"/api/v1/datasets/{dataset_id}/columns/chart", params={"column": "store", "top_n": 2}
    ).json()

    assert (timeline["kind"], timeline["step"]) == ("timeline", "month")
    assert timeline["views"] == ["timeline", "month", "day_of_week"]
    assert [bar["label"] for bar in months["bars"]][:3] == ["Jan", "Feb", "Mar"]
    assert len(stores["bars"]) == 2
    assert stores["others"] == {"count": 1095, "categories": 1}


def test_date_profile_in_column_summary(client: TestClient) -> None:
    dataset_id = import_path(client, default_samples_dir() / "sales_timeseries.csv")

    columns = {c["name"]: c for c in client.get(f"/api/v1/datasets/{dataset_id}/columns").json()}

    date = columns["date"]["summary"]["date"]
    assert date["step"] == "daily"
    assert date["has_time"] is False
    assert "day_of_week" in date["parts"]
    assert "hour" not in date["parts"]


def test_unknown_chart_column_is_404(client: TestClient) -> None:
    dataset_id = import_path(client, default_samples_dir() / "sales_timeseries.csv")

    response = client.get(f"/api/v1/datasets/{dataset_id}/columns/chart", params={"column": "nope"})

    assert response.status_code == 404
