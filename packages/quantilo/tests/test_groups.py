"""Several series at once (039): every group, the global model, coherent totals."""

import subprocess
import sys
import time
from concurrent.futures import wait

import numpy as np
import pytest

from quantilo.timeseries import global_model as gm
from quantilo.timeseries import groups as gr
from quantilo.timeseries import hierarchy as hi

RNG = np.random.default_rng(7)
N, H, SEASON = 72, 6, 12


def simulated_hierarchy(seed: int = 3, n: int = N) -> tuple[list[str], list[str], np.ndarray]:
    """Two regions of four stores: a shared regional pattern plus noisy store-level demand."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    stores, regions, values = [], [], []
    for r, region in enumerate(("North", "South")):
        regional = 50 + 10 * np.sin(2 * np.pi * t / 12 + r) + np.cumsum(rng.normal(0, 1.5, n))
        for s in range(4):
            stores.append(f"{region[0]}{s}")
            regions.append(region)
            share = 0.5 + 0.25 * s
            values.append(share * regional + rng.normal(0, 6, n) + 20)
    return stores, regions, np.array(values)


def nodes_of(tree: hi.Tree, leaves: np.ndarray) -> list[gr.Node]:
    totals = tree.summing() @ leaves
    return [
        gr.Node(label, level, [float(v) for v in totals[i]])
        for i, (label, level) in enumerate(zip(tree.labels, tree.levels, strict=True))
    ]


def run(nodes, tree, keys, reconcile=None, use_global=False, workers: int | None = 1, n=N):
    job = gr.GroupJob("t", "running", 0, len(nodes))
    periods = [str(i) for i in range(n)]
    meta = {"group": "store", "groups": len(nodes), "kept": len(nodes), "pooled": 0, "limit": 200}
    return gr.forecast_groups(
        job,
        nodes,
        tree,
        keys,
        H,
        SEASON,
        12,
        periods,
        [f"f{i}" for i in range(H)],
        meta,
        lambda: False,
        use_global=use_global,
        reconcile=reconcile,
        workers=workers,
    )


# --- The hierarchy -------------------------------------------------------------------------------


def test_tree_and_summing_matrix() -> None:
    tree = hi.build_tree(["a", "b", "c"], "store", [("region", ["N", "N", "S"])])

    assert tree.labels == ["Total", "N", "S", "a", "b", "c"]
    assert tree.levels == ["Total", "region", "region", "store", "store", "store"]
    assert tree.parents == [None, 0, 0, 1, 1, 2]
    assert tree.summing().tolist() == [
        [1, 1, 1],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    assert hi.is_function_of([("a", "N"), ("b", "N"), ("a", "N")])
    assert not hi.is_function_of([("a", "N"), ("a", "S")])


@pytest.mark.parametrize("method", hi.METHODS)
def test_every_method_gives_coherent_forecasts(method: hi.Method) -> None:
    stores, regions, leaves = simulated_hierarchy()
    tree = hi.build_tree(stores, "store", [("region", regions)])
    history = tree.summing() @ leaves
    base = history[:, -H:] + RNG.normal(0, 5, (len(tree.labels), H))  # incoherent
    residuals = RNG.normal(0, 3, (len(tree.labels), 40))

    coherent = hi.reconcile(tree, base, hi.projection(tree, method, residuals, history))

    assert hi.incoherence(tree, base) > 1
    assert hi.incoherence(tree, coherent) < hi.TOLERANCE
    if method == "bottom_up":
        assert coherent[tree.bottom] == pytest.approx(base[tree.bottom])
    if method == "top_down":
        assert coherent[0] == pytest.approx(base[0])  # the total is kept, then split


def test_shrinkage_goes_to_the_diagonal_when_nodes_are_independent() -> None:
    independent = RNG.normal(0, 1, (500, 4))
    linked = independent + RNG.normal(0, 3, (500, 1))  # a strong shared error

    _, free = hi.shrink_covariance(independent)
    covariance, tied = hi.shrink_covariance(linked)

    assert free > 0.5 and tied < 0.1
    assert np.allclose(covariance, covariance.T)


def test_mint_is_at_least_as_accurate_as_bottom_up_on_a_simulated_hierarchy() -> None:
    """Spec 039: on a simulated hierarchy (fixed seed), MinT's mean error over every level is
    no higher than bottom-up's."""
    stores, regions, leaves = simulated_hierarchy()
    tree = hi.build_tree(stores, "store", [("region", regions)])
    nodes = nodes_of(tree, leaves)

    errors = {}
    for method in ("bottom_up", "mint_shrink"):
        found = run(nodes, tree, ["naive", "seasonal_naive", "ses"], reconcile=method)
        assert found is not None and found.hierarchy is not None
        assert found.hierarchy.method == method and found.hierarchy.gap < hi.TOLERANCE
        levels = found.hierarchy.errors
        assert [e.level for e in levels] == ["Total", "region", "store"]
        assert all(e.mase_after is not None for e in levels)
        errors[method] = np.mean([e.mase_after or 0.0 for e in levels])

    assert errors["mint_shrink"] <= errors["bottom_up"]


def test_sums_of_children_equal_parents_at_every_date() -> None:
    stores, regions, leaves = simulated_hierarchy()
    tree = hi.build_tree(stores, "store", [("region", regions)])

    found = run(nodes_of(tree, leaves), tree, ["seasonal_naive", "ses"], reconcile="auto")

    assert found is not None and found.hierarchy is not None
    assert found.hierarchy.method == "mint_shrink" and found.hierarchy.method_auto
    rows = {(r.level, r.label): r for r in found.rows}
    for region in ("North", "South"):
        children = [r for r in found.rows if r.parent == region]
        assert len(children) == 4
        total = np.sum([c.forecast for c in children], axis=0)
        assert np.max(np.abs(total - rows[("region", region)].forecast)) < 1e-6
    regions_sum = rows[("region", "North")].forecast + np.array(rows[("region", "South")].forecast)
    assert np.max(np.abs(regions_sum - rows[("Total", "Total")].forecast)) < 1e-6
    assert [r.level for r in found.rows][:8] == ["store"] * 8  # leaves first


# --- One series and every series -----------------------------------------------------------------


def test_short_series_use_the_seasonal_naive_and_are_flagged() -> None:
    values = [float(v) for v in 10 + np.sin(np.arange(20) / 2)]

    outcome = gr.forecast_series(gr.SeriesTask("new", values, ["ets_auto"], 3, 12))
    old = gr.forecast_series(gr.SeriesTask("old", [0.0] * 40 + values, ["ses"], 3, 12, 40))

    assert outcome.short_history and outcome.models == ["seasonal_naive"]
    assert outcome.best == "seasonal_naive" and len(outcome.forecast) == 3
    assert old.short_history  # 40 empty periods before the first sale do not count


def test_intermittent_series_get_croston_type_models() -> None:
    values = [0.0, 0, 3, 0, 0, 0, 5, 0, 0, 2, 0, 0] * 5

    outcome = gr.forecast_series(gr.SeriesTask("rare", values, ["seasonal_naive"], 3, 12))

    assert {"croston", "sba", "tsb"} <= set(outcome.models)
    assert outcome.demand is not None and outcome.demand.intermittent


def test_parallel_and_serial_give_the_same_forecasts() -> None:
    _, _, leaves = simulated_hierarchy()
    nodes = [gr.Node(f"s{i}", "store", [float(v) for v in row]) for i, row in enumerate(leaves)]

    serial = run(nodes, None, ["seasonal_naive", "ses"], workers=1)
    parallel = run(nodes, None, ["seasonal_naive", "ses"], workers=4)

    assert serial is not None and parallel is not None
    assert [r.forecast for r in serial.rows] == [r.forecast for r in parallel.rows]
    assert parallel.workers == 4


SCRIPT = """
import numpy as np
from quantilo.timeseries import groups as gr

print("TOP")
rng = np.random.default_rng(1)
tasks = [
    gr.SeriesTask(f"s{i}", list(rng.normal(50, 5, 48)), ["seasonal_naive"], 6, 12) for i in range(4)
]
print("DONE", len(gr.forecast_all(tasks, 2)))
"""


def test_a_script_without_a_main_guard_runs_once(tmp_path) -> None:
    """Processes are started without the caller's script: it runs once, not once per process."""
    script = tmp_path / "forecast.py"
    script.write_text(SCRIPT, encoding="utf-8")

    out = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=240, cwd=tmp_path
    )

    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split() == ["TOP", "DONE", "4"]
    assert "bootstrapping phase" not in out.stderr


@pytest.mark.slow
def test_thirty_series_in_under_thirty_seconds() -> None:
    """Spec 039: 30 series, 4 fast methods, 5 origins, in under 30 s on a laptop (at rest: run
    with -m slow; ETS alone fits 15 forms at 6 windows per series)."""
    wait(gr.warm_up())  # the app starts its processes when a group is chosen
    rng = np.random.default_rng(1)
    t = np.arange(60)
    nodes = [
        gr.Node(f"store {i}", "store", [float(v) for v in 100 + 20 * np.sin(2 * np.pi * t / 12)
                                        + rng.normal(0, 5, 60) + i])
        for i in range(30)
    ]  # fmt: skip
    started = time.perf_counter()

    found = run(
        nodes, None, ["naive", "seasonal_naive", "ets_auto", "combination"], workers=None, n=60
    )

    assert time.perf_counter() - started < 30
    assert found is not None and len(found.rows) == 30
    assert all(r.mase is not None for r in found.rows)
    assert sum(s.series for s in found.best_models) == 30


def test_cancel_stops_before_the_remaining_series() -> None:
    nodes = [gr.Node(f"s{i}", "store", [float(i + j % 12) for j in range(48)]) for i in range(6)]
    job = gr.GroupJob("t", "running", 0, 6)
    calls = iter([False, False, True, True, True, True, True])

    found = gr.forecast_groups(
        job, nodes, None, ["naive"], 3, 12, 12, [str(i) for i in range(48)], ["a", "b", "c"],
        {"group": "g", "groups": 6, "kept": 6, "pooled": 0, "limit": 200},
        lambda: next(calls), workers=1,
    )  # fmt: skip

    assert found is None and job.done == 2


# --- The global model ----------------------------------------------------------------------------


def test_the_global_model_learns_a_pattern_shared_by_short_series() -> None:
    """Many short series with the same shape: one model on all of them."""
    rng = np.random.default_rng(5)
    t = np.arange(40)
    shape = np.sin(2 * np.pi * t / 12)
    values = np.array([(10 + i) * (2 + shape) + rng.normal(0, 0.3, 40) for i in range(12)])

    found = gm.forecast(values, 6, 12)

    assert len(found) == 12
    for i, result in enumerate(found):
        assert result.scores is not None and result.scores.mase is not None
        assert result.scores.mase < 1.5 and len(result.forecast) == 6
        assert result.low95[0] <= result.forecast[0] <= result.high95[0]
        expected = (10 + i) * (2 + np.sin(2 * np.pi * np.arange(40, 46) / 12))
        assert np.mean(np.abs(np.array(result.forecast) - expected)) < (10 + i) * 0.5
    assert gm.library() in ("lightgbm", "scikit-learn")


def test_the_global_model_competes_with_each_series_best() -> None:
    rng = np.random.default_rng(5)
    t = np.arange(40)
    shape = np.sin(2 * np.pi * t / 12)
    nodes = [
        gr.Node(f"p{i}", "product", [float(v) for v in (10 + i) * (2 + shape) + noise])
        for i, noise in enumerate(rng.normal(0, 0.3, (12, 40)))
    ]

    found = run(nodes, None, ["naive"], use_global=True, n=40)

    assert found is not None and found.global_model is not None
    assert found.global_model.series == 12 and found.global_model.wins >= 6
    assert any(r.best == gm.KEY for r in found.rows)
    assert all(r.global_mase is not None for r in found.rows)


# --- Export --------------------------------------------------------------------------------------


def test_long_csv_escapes_formulas_and_keeps_negative_numbers() -> None:
    nodes = [
        gr.Node("=HYPERLINK(1)", "store", [float(-5 - (j % 3)) for j in range(30)]),
        gr.Node("Lyon, Part-Dieu", "store", [float(5 + j % 4) for j in range(30)]),
    ]

    found = run(nodes, None, ["naive"], n=30)
    assert found is not None
    lines = gr.long_csv(found).splitlines()

    assert lines[0] == "group,level,period,forecast,low80,high80,low95,high95,model"
    assert len(lines) == 1 + 2 * H
    assert lines[1].startswith("'=HYPERLINK(1),store,f0,-")
    assert any(line.startswith('"Lyon, Part-Dieu",store,f0,') for line in lines)


def test_a_dead_process_pool_falls_back_to_one_series_after_the_other(monkeypatch) -> None:
    """A process killed (out of memory…) breaks the pool: the series still get forecast."""
    from concurrent.futures.process import BrokenProcessPool

    class Broken:
        def submit(self, *args, **kwargs):
            raise BrokenProcessPool("A child process terminated abruptly")

        def shutdown(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(gr, "shared_pool", lambda workers: Broken())
    tasks = [
        gr.SeriesTask(f"s{i}", [float(i + j % 12) for j in range(40)], ["naive"], 3, 12)
        for i in range(5)
    ]

    found = gr.forecast_all(tasks, 4)

    assert found is not None and [o.label for o in found] == [f"s{i}" for i in range(5)]
    assert gr.default_workers(1_000) <= gr.MAX_WORKERS
