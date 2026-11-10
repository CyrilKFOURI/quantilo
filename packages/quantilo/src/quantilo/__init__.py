"""quantilo: exploratory data analysis with statistical depth, in Python.

    import quantilo as dx

    ds = dx.open("titanic.csv")
    ds.types                        # detected types, with the reasons
    ds.variable("Age")              # statistics, normality, outliers
    ds.compare("Age", "Survived")   # the test follows the types, with an effect size
    ds.correlations()               # mixed association matrix, ranked pairs, insights
    ds.experiment("group", ["converted"]).analyze()   # an A/B test, Holm-adjusted
    dx.sample_size(baseline=0.10, effect=0.05)       # units per group to plan one
    dx.bandit_simulation(baseline=0.10, effect=0.05, daily=1_000)   # fixed split vs bandit
    dx.plot(ds.compare("Age", "Survived"))   # the app's chart (the plot extra)
    model = dx.load_model("churn_lightgbm.zip")   # a model exported by the app (037)

The engine of the Quantilo app: the same code computes the app's results.
"""

import os
from typing import TYPE_CHECKING

from quantilo.catalog import ColumnView
from quantilo.dataset import Dataset, DatasetInfo, from_arrow, from_pandas, open
from quantilo.errors import ColumnNotFound, ExplorerError, NotFoundError
from quantilo.experiments.facade import bandit_simulation, sample_size
from quantilo.filters.model import Condition
from quantilo.results import to_dict, to_pandas
from quantilo.timeseries.drivers import Driver, ScenarioChange
from quantilo.typing.engine import DEFAULT_THRESHOLDS, Thresholds
from quantilo.version import __version__

if TYPE_CHECKING:
    from quantilo.ml.saved import LoadedModel
    from quantilo.recipes import Recipe


def __getattr__(name: str) -> object:
    """`dx.Recipe` loads on first use (it reads the machine-learning settings: scikit-learn)."""
    if name == "Recipe":
        from quantilo.recipes import Recipe

        return Recipe
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def plot(result: object, view: str | None = None, **options: object) -> object:
    """A result drawn as the app draws it, as a plotly Figure (the `plot` extra); see
    `quantilo.plots.plot`."""
    from quantilo.plots import plot as draw

    return draw(result, view, **options)  # type: ignore[arg-type]


def load_model(path: str | os.PathLike[str]) -> "LoadedModel":
    """A model exported by the app or by `model.export` (the ml extra): its card is checked
    before the pipeline is loaded, a zip without one is refused; see
    `quantilo.ml.saved.load_model`."""
    from quantilo.ml.saved import load_model as load

    return load(path)


__all__ = [
    "DEFAULT_THRESHOLDS",
    "ColumnNotFound",
    "ColumnView",
    "Condition",
    "Dataset",
    "DatasetInfo",
    "Driver",
    "ExplorerError",
    "NotFoundError",
    "Recipe",
    "ScenarioChange",
    "Thresholds",
    "__version__",
    "bandit_simulation",
    "from_arrow",
    "from_pandas",
    "load_model",
    "open",
    "plot",
    "sample_size",
    "to_dict",
    "to_pandas",
]
