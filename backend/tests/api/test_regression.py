import json

import pytest
from fastapi.testclient import TestClient

from app.config import default_samples_dir
from tests.api.test_profiling import import_path


@pytest.fixture
def traps(client: TestClient) -> str:
    return import_path(client, default_samples_dir() / "typing_traps.csv")


def url(dataset_id: str, path: str = "") -> str:
    return f"/api/v1/datasets/{dataset_id}/regression{path}"


def test_first_plane(client: TestClient, traps: str) -> None:
    response = client.get(
        url(traps), params={"target": "price_eur", "columns": "age,income,segment"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["target"] == "price_eur" and body["model"] == "ols"
    assert body["se"]["kind"] == "hc3" and body["se"]["rule"] == "R-REG-SE-05"
    assert body["terms"][0]["term"] == "(Intercept)"
    assert any(t["column"] == "segment" and t["level"] for t in body["terms"])
    assert body["quality"]["rows"] > 0 and body["multiple"]["tests"] == len(body["terms"]) - 1
    assert body["sentence"] and {c["key"] for c in body["checks"]} >= {"spread", "collinearity"}
    assert body["plots"]["points"] == body["quality"]["rows"]


def test_suggestion_then_regression_without_parameters(client: TestClient, traps: str) -> None:
    found = client.get(url(traps, "/suggestion")).json()
    assert found["target"]["column"] == "price_eur"  # the amount (R-REG-20)
    assert found["columns"] and all(p["reason"].startswith("R-REG-21") for p in found["columns"])
    body = client.get(url(traps)).json()
    assert body["target"] == "price_eur"
    assert body["columns"] == [p["column"] for p in found["columns"]]


def test_sections(client: TestClient, traps: str) -> None:
    params = {"target": "price_eur", "columns": "age,income"}
    errors = client.get(url(traps, "/standard-errors"), params=params).json()
    assert [k["kind"] for k in errors["kinds"]][:5] == ["classic", "hc0", "hc1", "hc2", "hc3"]
    assert next(k for k in errors["kinds"] if k["kind"] == "hc3")["chosen"]
    influence = client.get(url(traps, "/influence"), params=params).json()
    assert len(influence["rows"]) == 20 and len(influence["rows"][0]["dfbetas"]) == 3
    terms = client.get(url(traps, "/terms"), params=params).json()
    assert {v["term"] for v in terms["vifs"]} == {"age", "income"}
    more = client.get(url(traps, "/more"), params=params).json()
    assert more["white"]["name"] == "White"


def test_forced_standard_errors_and_filters(client: TestClient, traps: str) -> None:
    body = client.get(
        url(traps),
        params={
            "target": "price_eur",
            "columns": "age",
            "se": "classic",
            "filters": json.dumps([{"column": "country", "op": "in", "values": ["FR"]}]),
        },
    ).json()
    assert body["se"]["kind"] == "classic" and body["se"]["rule"] is None
    assert body["quality"]["total"] < 600


def test_errors_are_explained(client: TestClient, traps: str) -> None:
    unknown = client.get(url(traps), params={"target": "price_eur", "columns": "nope"})
    assert unknown.status_code == 404
    category = client.get(url(traps), params={"target": "city", "columns": "age"})
    assert category.status_code == 422 and "levels" in category.json()["error"]["message"]


def test_families(client: TestClient, traps: str) -> None:
    """069: the family follows the target, can be forced, and More fits the niche ones."""
    counts = client.get(url(traps), params={"target": "nb_children", "columns": "age,segment"})
    assert counts.status_code == 200, counts.text
    body = counts.json()
    assert body["model"] == "poisson" and body["family"]["rule"] == "R-REG-02"
    assert body["family"]["reads"] == "ratio" and body["terms"][1]["ratio"] is not None
    assert body["se"]["kind"] == "hc0" and any(c["key"] == "overdispersion" for c in body["checks"])
    classes = client.get(url(traps), params={"target": "country", "columns": "age"}).json()
    assert classes["model"] == "mnlogit" and classes["family"]["outcomes"][0]
    params = {"target": "price_eur", "columns": "age,income", "family": "log_ols"}
    logged = client.get(url(traps), params=params).json()
    assert logged["model"] == "log_ols" and logged["family"]["smearing"] > 0
    more = client.get(url(traps, "/families"), params=params)
    assert more.status_code == 200, more.text
    found = {a["kind"]: a for a in more.json()["alternatives"]}
    assert found["boxcox"]["available"] and found["boxcox"]["effects"]
    assert not found["probit"]["available"]


def test_agent_tool(client: TestClient, traps: str) -> None:
    from tests.api.test_agent import run_tool

    found = run_tool(
        client, traps, "fit_regression", {"target": "price_eur", "columns": ["age", "income"]}
    )
    assert found.content["standard_errors"]["rule"] == "R-REG-SE-05"
    assert [t["term"] for t in found.content["terms"]] == ["(Intercept)", "age", "income"]
    assert found.content["checks"] and found.content["conclusion"]
    assert found.link.page == "regression"
    assert found.link.params == {"target": "price_eur", "columns": "age,income"}
    counts = run_tool(
        client, traps, "fit_regression", {"target": "nb_children", "columns": ["age"]}
    )
    assert counts.content["family"]["kind"] == "poisson"
    assert counts.content["family"]["rule"] == "R-REG-02"
    assert counts.content["terms"][1]["ratio"] is not None
    forced = run_tool(
        client,
        traps,
        "fit_regression",
        {"target": "price_eur", "columns": ["age"], "family": "gamma"},
    )
    assert forced.content["family"]["kind"] == "gamma" and forced.link.params["family"] == "gamma"
    playbook = run_tool(client, traps, "get_playbook", {"topic": "regression"})
    assert "HC3" in playbook.content


def test_panel_and_instruments(client: TestClient, tmp_path) -> None:
    """071: the Panel section of a store x week panel, then IV with a named instrument."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(3)
    stores, weeks = 40, 20
    store = np.repeat(np.arange(1, stores + 1), weeks)
    date = np.tile(pd.date_range("2021-01-04", periods=weeks, freq="W-MON"), stores)
    alpha = rng.normal(0, 2, stores)[store - 1]
    z, u = rng.normal(size=stores * weeks), rng.normal(size=stores * weeks)
    price = 0.8 * alpha + 0.7 * z + u + rng.normal(size=stores * weeks)
    sales = 2 * price + alpha + 2 * u + rng.normal(size=stores * weeks)
    path = tmp_path / "panel.csv"
    pd.DataFrame({"Store": store, "Date": date, "Sales": sales, "Price": price, "Cost": z}).to_csv(
        path, index=False
    )
    dataset_id = import_path(client, path)
    params = {"target": "Sales", "columns": "Price"}
    first = client.get(url(dataset_id), params=params).json()
    assert any(c["key"] == "panel" and c["action"] == "panel" for c in first["checks"])
    panel = client.get(url(dataset_id, "/panel"), params=params)
    assert panel.status_code == 200, panel.text
    body = panel.json()
    assert body["available"] and body["entity"] == "Store"
    assert [e["key"] for e in body["estimators"]][:2] == ["pooled", "entity"]
    assert body["recommended"] in ("entity", "twoway")
    missing = client.get(url(dataset_id, "/instruments"), params=params)
    assert missing.status_code == 422 and "never guessed" in missing.json()["error"]["message"]
    iv = client.get(
        url(dataset_id, "/instruments"),
        params={**params, "endogenous": "Price", "instruments": "Cost"},
    )
    assert iv.status_code == 200, iv.text
    found = iv.json()
    assert found["se"]["kind"] == "cluster"
    assert found["first_stages"][0]["effective_f"] > 10
    assert {t["key"] for t in found["tests"]} == {"endogeneity", "anderson_rubin", "overid"}


def test_terms_partial_robust_and_quantiles(client: TestClient, traps: str) -> None:
    """070 and 072: terms in the query, the partial plots, the robust and quantile routes."""
    params = {
        "target": "price_eur",
        "columns": "age,income,segment",
        "terms": "spline:age,interaction:income|segment",
        "center": "true",
    }
    body = client.get(url(traps), params=params).json()
    names = [t["term"] for t in body["terms"]]
    assert "age · curve 1" in names
    assert any(n.startswith("income × segment = ") for n in names)  # noqa: RUF001
    assert any(t["group"] == "spline:age" for t in body["terms"])
    terms = client.get(url(traps, "/terms"), params=params).json()
    assert terms["specs"] == ["spline:age", "interaction:income|segment"] and terms["center"]
    assert any(t["question"] == "Is the effect of `age` a curve?" for t in terms["tests"])
    assert terms["slopes"] and terms["curves"]
    shape = client.get(url(traps, "/partial"), params={**params, "column": "income"}).json()
    assert shape["available"] and shape["slope"] is not None
    bad = client.get(url(traps), params={**params, "terms": "curve:age"})
    assert bad.status_code == 422
    plain = {"target": "price_eur", "columns": "age,income"}
    robust = client.get(url(traps, "/robust"), params={**plain, "quantile": 0.75}).json()
    assert robust["available"] and [c["kind"] for c in robust["columns"]] == [
        "ols", "huber", "tukey", "quantile",
    ]  # fmt: skip
    assert robust["quantile"] == 0.75
    assert client.get(url(traps, "/robust"), params={**plain, "quantile": 0.3}).status_code == 422
    process = client.get(url(traps, "/quantiles"), params=plain).json()
    assert process["available"] and process["taus"][0] == 0.1 and process["curves"]
