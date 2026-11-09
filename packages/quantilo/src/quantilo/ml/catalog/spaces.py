"""Where the optimizer searches, model by model (021): the settings that matter, with bounds that
make sense (narrower than what a user may type), on a log scale where the effect multiplies."""

from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["float", "int", "categorical"]


@dataclass
class Param:
    key: str
    kind: Kind
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: list[str] = field(default_factory=list)


def _f(key: str, low: float, high: float, log: bool = False) -> Param:
    return Param(key, "float", low, high, log)


def _i(key: str, low: int, high: int, log: bool = False) -> Param:
    return Param(key, "int", low, high, log)


def _c(key: str, *choices: str) -> Param:
    return Param(key, "categorical", choices=list(choices))


_TREES = [_i("max_depth", 2, 20), _i("min_samples_leaf", 1, 100, log=True)]
_FOREST = [
    _i("n_estimators", 100, 600),
    _i("max_depth", 3, 40),
    _i("min_samples_leaf", 1, 50, log=True),
    _c("max_features", "sqrt", "log2", "0.5", "1.0"),
]
_NB_ALPHA = [_f("alpha", 1e-3, 10, log=True)]

SPACES: dict[str, list[Param]] = {
    "linear_regression": [
        _c("regularization", "ridge", "lasso", "elasticnet"),
        _f("alpha", 1e-4, 100, log=True),
        _f("l1_ratio", 0.05, 0.95),
    ],
    "huber": [_f("epsilon", 1.0, 3.0), _f("alpha", 1e-6, 1, log=True)],
    "glm_poisson": [_f("alpha", 1e-4, 10, log=True)],
    "glm_gamma": [_f("alpha", 1e-4, 10, log=True)],
    "glm_tweedie": [_f("alpha", 1e-4, 10, log=True), _f("power", 1.1, 1.9)],
    "logistic_regression": [_f("C", 1e-3, 1e3, log=True), _c("penalty", "l2", "l1")],
    "ridge_classifier": [_f("alpha", 1e-3, 1e3, log=True)],
    "sgd": [_f("alpha", 1e-6, 1e-1, log=True), _c("penalty", "l2", "l1", "elasticnet")],
    "lda": [_c("solver", "lsqr", "eigen")],
    "qda": [_f("reg_param", 0.0, 1.0)],
    "knn": [
        _i("n_neighbors", 1, 100, log=True),
        _c("weights", "uniform", "distance"),
        _f("p", 1.0, 3.0),
    ],
    "linear_svm": [_f("C", 1e-3, 1e3, log=True)],
    "kernel_svm": [
        _f("C", 1e-2, 1e3, log=True),
        _c("gamma", "scale", "auto"),
        _c("kernel", "rbf", "poly"),
    ],
    "nu_svm": [_f("nu", 0.05, 0.95), _c("gamma", "scale", "auto")],
    "gaussian_nb": [_f("var_smoothing", 1e-12, 1e-3, log=True)],
    "multinomial_nb": _NB_ALPHA,
    "complement_nb": _NB_ALPHA,
    "bernoulli_nb": _NB_ALPHA,
    "decision_tree": _TREES,
    "extra_tree": _TREES,
    "random_forest": _FOREST,
    "extra_trees": _FOREST,
    "bagging": [
        _i("n_estimators", 10, 200),
        _f("max_samples", 0.3, 1.0),
        _f("max_features", 0.3, 1.0),
    ],
    "hist_gradient_boosting": [
        _f("learning_rate", 0.01, 0.3, log=True),
        _i("max_leaf_nodes", 8, 128, log=True),
        _i("min_samples_leaf", 5, 100, log=True),
        _f("l2_regularization", 1e-4, 10, log=True),
    ],
    "gradient_boosting": [
        _i("n_estimators", 50, 500),
        _f("learning_rate", 0.01, 0.3, log=True),
        _i("max_depth", 2, 6),
        _f("subsample", 0.5, 1.0),
    ],
    "adaboost": [_i("n_estimators", 20, 400), _f("learning_rate", 0.01, 2.0, log=True)],
    "xgboost": [
        _i("max_depth", 2, 10),
        _f("learning_rate", 0.01, 0.3, log=True),
        _f("subsample", 0.5, 1.0),
        _f("colsample_bytree", 0.5, 1.0),
        _f("min_child_weight", 1, 10, log=True),
        _f("reg_lambda", 1e-3, 10, log=True),
    ],
    "lightgbm": [
        _i("num_leaves", 8, 256, log=True),
        _f("learning_rate", 0.01, 0.3, log=True),
        _i("min_child_samples", 5, 100, log=True),
        _f("colsample_bytree", 0.5, 1.0),
        _f("reg_lambda", 1e-3, 10, log=True),
    ],
    "catboost": [
        _i("depth", 3, 10),
        _f("l2_leaf_reg", 1e-2, 30, log=True),
    ],
    "mlp": [
        _f("alpha", 1e-6, 1e-1, log=True),
        _c("hidden_layer_sizes", "32", "64", "64, 32", "128, 64"),
        _f("learning_rate_init", 1e-4, 1e-2, log=True),
    ],
    "kernel_ridge": [_f("alpha", 1e-3, 10, log=True), _c("kernel", "rbf", "laplacian")],
}


def space(model: str) -> list[Param]:
    """The default search space of a model (empty: nothing worth tuning)."""
    return [
        Param(p.key, p.kind, p.low, p.high, p.log, list(p.choices)) for p in SPACES.get(model, [])
    ]
