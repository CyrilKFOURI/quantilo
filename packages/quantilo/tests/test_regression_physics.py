"""Physics and metrology (075): the fit by uncertainties against `curve_fit` and statsmodels,
York on Pearson's data, Deming, total least squares, curves against R's `nls` on Puromycin."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as statsmodels_api
from scipy import optimize

import quantilo as dx
from quantilo.regression import physics as ph

sm: Any = statsmodels_api


def _measured(seed: int = 11, scale: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 80
    t = np.linspace(0, 10, n)
    err = rng.uniform(0.2, 1.0, n)
    flux = 3.0 + 1.5 * t + rng.normal(0, 1, n) * err * scale
    return pd.DataFrame({"time": t, "flux": flux, "flux_err": err})


@pytest.fixture(scope="module")
def measured(tmp_path_factory) -> dx.Dataset:
    return dx.from_pandas(_measured(), "measured", cache_dir=tmp_path_factory.mktemp("m"))


def test_the_uncertainty_role_is_read_from_the_name(measured):
    assert measured.column("flux_err").role == "uncertainty"
    offered = measured.regression_methods("flux", ["time"])
    assert offered.sigma is not None and offered.sigma.column == "flux_err"
    method = next(m for m in offered.methods if m.kind == "measurement")
    assert method.proposed and "R-REG-PHY-01" in method.reason


def test_the_weighted_fit_matches_curve_fit_and_statsmodels(measured):
    table = measured.regression_method("measurement", "flux", ["time"])
    assert isinstance(table, ph.MeasurementTable) and table.uncertainty is not None
    found = table.uncertainty
    frame = _measured()
    t, y, s = frame["time"].to_numpy(), frame["flux"].to_numpy(), frame["flux_err"].to_numpy()

    def line(x, a, b):
        return a + b * x

    absolute, cov_abs = optimize.curve_fit(line, t, y, sigma=s, absolute_sigma=True)
    _, cov_scaled = optimize.curve_fit(line, t, y, sigma=s, absolute_sigma=False)
    assert [term.estimate for term in found.terms] == pytest.approx(list(absolute), rel=1e-7)
    assert [term.absolute_se for term in found.terms] == pytest.approx(
        list(np.sqrt(np.diag(cov_abs))), rel=1e-6
    )
    assert [term.scaled_se for term in found.terms] == pytest.approx(
        list(np.sqrt(np.diag(cov_scaled))), rel=1e-6
    )
    wls = sm.WLS(y, sm.add_constant(t), weights=1 / s**2).fit()
    assert [term.scaled_se for term in found.terms] == pytest.approx(list(wls.bse), rel=1e-6)
    expected = float((((y - line(t, *absolute)) / s) ** 2).sum())
    assert found.chi2 == pytest.approx(expected, rel=1e-9) and found.df == len(y) - 2
    assert found.absolute and "absolute sigma" in found.absolute_reason


def test_underestimated_uncertainties_scale_the_intervals(tmp_path):
    ds = dx.from_pandas(_measured(scale=3.0), "under", cache_dir=tmp_path)
    table = ds.regression_method("measurement", "flux", ["time"])
    assert isinstance(table, ph.MeasurementTable) and table.uncertainty is not None
    found = table.uncertainty
    assert found.reduced > found.band_high and not found.absolute
    assert "too small" in found.absolute_reason
    assert found.pull_sd > 2


# York (1966) weights on Pearson's (1901) data; the published line (York et al. 2004, Cantrell 2008)
PEARSON_X = np.array([0.0, 0.9, 1.8, 2.6, 3.3, 4.4, 5.2, 6.1, 6.5, 7.4])
PEARSON_Y = np.array([5.9, 5.4, 4.4, 4.6, 3.5, 3.7, 2.8, 2.8, 2.4, 1.5])
WX = np.array([1000, 1000, 500, 800, 200, 80, 60, 20, 1.8, 1.0])
WY = np.array([1, 1.8, 4, 8, 20, 20, 70, 70, 100, 500])


def test_york_on_pearson_data_gives_the_published_line():
    b, a, sb, sa = ph.york(PEARSON_X, PEARSON_Y, 1 / np.sqrt(WX), 1 / np.sqrt(WY))
    assert b == pytest.approx(-0.48053, abs=5e-5)
    assert a == pytest.approx(5.47991, abs=5e-5)
    assert 0 < sb < 0.1 and 0 < sa < 0.5


def test_deming_reaches_least_squares_and_total_least_squares():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 2, 300)
    y = 1 + 0.7 * x + rng.normal(0, 1, 300)
    slope, _ = ph.deming(x, y, 1e9)
    assert slope == pytest.approx(np.polyfit(x, y, 1)[0], rel=1e-6)
    orthogonal, _ = ph.deming(x, y, 1.0)
    assert orthogonal == pytest.approx(float(ph.total_least_squares(x[:, None], y)[0]), rel=1e-9)
    se, _ = ph.deming_jackknife(x, y, 1.0)
    slopes = [ph.deming(np.delete(x, i), np.delete(y, i), 1.0)[0] for i in range(len(x))]
    expected = np.sqrt((len(x) - 1) / len(x) * ((np.array(slopes) - np.mean(slopes)) ** 2).sum())
    assert se == pytest.approx(expected, rel=1e-6)


def test_errors_on_x_say_the_attenuation(tmp_path):
    rng = np.random.default_rng(4)
    n = 400
    truth = rng.normal(0, 1, n)
    frame = pd.DataFrame(
        {"x": truth + rng.normal(0, 0.7, n), "y": 2 * truth + rng.normal(0, 0.7, n)}
    )
    ds = dx.from_pandas(frame, "attenuated", cache_dir=tmp_path)
    table = ds.regression_method("measurement", "y", ["x"])
    assert isinstance(table, ph.MeasurementTable) and table.errors is not None
    deming = next(line for line in table.errors.lines if line.method == "deming")
    ols = next(line for line in table.errors.lines if line.method == "ols")
    assert ols.slope < deming.slope
    assert "understate" in table.errors.sentence and table.uncertainty is None


# Puromycin, treated (Treloar 1974); R: nls(rate ~ Vm*conc/(K+conc)) gives Vm 212.68 (6.947),
# K 0.06412 (0.008281)
CONC = [0.02, 0.02, 0.06, 0.06, 0.11, 0.11, 0.22, 0.22, 0.56, 0.56, 1.10, 1.10]
RATE = [76, 47, 97, 107, 123, 139, 159, 152, 191, 201, 207, 200]


def test_michaelis_menten_matches_r_nls_on_puromycin():
    params, covariance, _, converged = ph.fit_curve(
        "michaelis", np.array(CONC), np.array(RATE, dtype=float)
    )
    se = np.sqrt(np.diag(covariance))
    assert converged
    assert params[0] == pytest.approx(212.68, abs=0.01) and se[0] == pytest.approx(6.947, abs=1e-3)
    assert params[1] == pytest.approx(0.06412, abs=1e-5) and se[1] == pytest.approx(
        0.008281, abs=1e-5
    )


def test_curves_rank_the_saturation_first(tmp_path):
    rng = np.random.default_rng(8)
    conc = np.repeat(np.linspace(0.05, 5, 40), 3)
    rate = 10 * conc / (0.8 + conc) + rng.normal(0, 0.2, len(conc))
    ds = dx.from_pandas(pd.DataFrame({"conc": conc, "rate": rate}), "mm", cache_dir=tmp_path)
    table = ds.regression_method("curves", "rate", ["conc"])
    assert isinstance(table, ph.CurvesTable) and table.available
    assert table.best in ("michaelis", "logistic") and table.fits[0].delta_aic == 0
    line = next(f for f in table.fits if f.kind == "line")
    assert line.delta_aic is not None and line.delta_aic > 20
    exp = next(f for f in table.fits if f.kind == "exponential")
    reference, _ = optimize.curve_fit(
        lambda x, a, b, c: a * np.exp(b * x) + c, conc, rate, p0=[-10, -1, 10], maxfev=10_000
    )
    assert [p.estimate for p in exp.parameters] == pytest.approx(list(reference), rel=1e-3)
