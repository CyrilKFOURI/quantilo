"""The model catalogue (019): every model by family, declared once and served to the interface."""

from quantilo.ml.catalog import linear, other, spaces, trees
from quantilo.ml.catalog.base import (
    Context,
    ModelSpec,
    Rule,
    Setting,
    SettingsError,
    resolve,
    visible,
)
from quantilo.ml.preparation import ModelProfile
from quantilo.ml.target import Task

FAMILIES = [
    "Linear models",
    "Discriminant analysis",
    "Nearest neighbours",
    "Support vector machines",
    "Naive Bayes",
    "Trees",
    "Bagging ensembles",
    "Boosting ensembles",
    "Neural networks",
    "Kernel & probabilistic",
]
CATALOGUE: dict[str, ModelSpec] = {
    spec.key: spec for spec in [*linear.MODELS, *other.MODELS, *trees.MODELS]
}
LINEAR = linear.LINEAR
REGULARIZED = linear.REGULARIZED


def default_model(task: Task) -> str:
    """The reference model trained first: readable, fast (016)."""
    return "linear_regression" if task == "regression" else "logistic_regression"


def profile_for(spec: ModelSpec, values: dict) -> ModelProfile:
    return spec.model_profile(values)


def _boosting() -> str:
    return next(
        key for key in ("xgboost", "lightgbm", "hist_gradient_boosting") if CATALOGUE[key].installed
    )


def recommend(context: Context) -> list[tuple[str, str]]:
    """Two or three models chosen by rules on the data (019 spec), each with its reason."""
    reference = default_model(context.task)
    if context.rows < 1_000:
        return [
            (reference, "Small data: a simple model is hard to beat."),
            ("random_forest", "Small data: robust without tuning."),
        ]
    if context.rows > 100_000:
        return [
            (_boosting(), "Large data: boosting is accurate and fast."),
            ("sgd", "Large data: a linear model fitted a few rows at a time."),
        ]
    found = [(_boosting(), "The reference on tabular data.")]
    if context.categorical_share > 0.3 or context.missing_share > 0.05:
        found[0] = (
            _boosting(),
            "Many categories or empty cells: boosting handles them well.",
        )
    found += [
        (reference, "Readable: every weight can be explained."),
        ("random_forest", "Robust without tuning."),
    ]
    return found


__all__ = [
    "CATALOGUE",
    "FAMILIES",
    "LINEAR",
    "REGULARIZED",
    "Context",
    "ModelSpec",
    "Rule",
    "Setting",
    "SettingsError",
    "default_model",
    "profile_for",
    "recommend",
    "resolve",
    "spaces",
    "visible",
]
