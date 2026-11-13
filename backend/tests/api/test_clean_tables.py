"""N32: Clean > Advanced of a project's table checks its accepted relations (T-01, T-02)."""

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from tests.api.test_model import _accept_all, _model, _project

BASE = "/api/v1/modules/clean"


def _advanced(client: TestClient, dataset_id: str) -> dict[str, dict[str, Any]]:
    response = client.get(f"{BASE}/datasets/{dataset_id}/suggestions/advanced")
    assert response.status_code == 200, response.text
    return {s["id"]: s for s in response.json()["suggestions"]}


def test_orphans_once_the_relation_is_accepted(client: TestClient, tmp_path: Path) -> None:
    project_id, ids = _project(client, tmp_path)
    sales, customers = ids["sales"], ids["customers"]
    # only proposed: nothing in Clean (the user has not said the tables are related)
    assert not [s for s in _advanced(client, sales) if s.startswith("T-")]
    _accept_all(client, project_id, _model(client, project_id))
    found = _advanced(client, sales)
    orphans = found[f"T-01:customer_id:{customers}"]
    assert orphans["mode"] == "I" and orphans["layer"] == "advanced"
    assert orphans["step"] is None and orphans["catalogue"] == "T-01"
    assert orphans["title"] == "2 rows of sales point to a customer_id missing from customers"
    assert orphans["where"] == [
        {"column": "customer_id", "op": "in", "value": None, "values": ["C999"]}
    ]
    # the step ids of the other suggestions are untouched; the dimension has nothing to say
    assert not [s for s in _advanced(client, customers) if s.startswith("T-")]
    # the cached Advanced layer is read again with the relation's rules added each time
    assert f"T-01:customer_id:{customers}" in _advanced(client, sales)
