# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# (settings reach the libraries as validated keyword arguments: the stubs cannot follow them)
"""Linear models and discriminant analysis."""

from typing import Any

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    GammaRegressor,
    HuberRegressor,
    Lasso,
    LinearRegression,
    LogisticRegression,
    PoissonRegressor,
    QuantileRegressor,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
    SGDRegressor,
    TweedieRegressor,
)
from sklearn.multiclass import OneVsRestClassifier

from quantilo.ml.catalog.base import (
    Context,
    ModelSpec,
    Rule,
    Setting,
    class_weight_setting,
    seed_setting,
    weight,
)
from quantilo.ml.preparation import ModelProfile

FAMILY = "Linear models"
LINEAR = ModelProfile(needs_scaling=True, sensitive_to_skew=True, sensitive_to_collinearity=True)
REGULARIZED = ModelProfile(needs_scaling=True, sensitive_to_skew=True)


def _strength(
    default: float = 1.0, why: str = "1 is a moderate penalty on standardised inputs.", **kw
) -> Setting:
    zero = default == 0  # a penalty that may be switched off: no log scale
    return Setting(
        "alpha",
        "Strength",
        "number",
        default,
        minimum=0.0 if zero else 1e-6,
        maximum=1e4,
        log=not zero,
        why=why,
        group="Regularization",
        param="alpha",
        **kw,
    )


def _iterations(default: int = 1_000, level: str = "expert") -> list[Setting]:
    return [
        Setting(
            "max_iter",
            "Max iterations",
            "int",
            default,
            level=level,
            minimum=10,
            maximum=100_000,
            group="Optimizer",
            param="max_iter",
            why="Enough to converge on most data.",
        ),  # type: ignore[arg-type]
        Setting(
            "tol",
            "Tolerance",
            "number",
            1e-4,
            level="expert",
            minimum=1e-10,
            maximum=1e-1,
            log=True,
            group="Optimizer",
            param="tol",
            why="Stops when an iteration improves less than this.",
        ),
    ]


# --- Linear regression (OLS, ridge, lasso, elastic net) -------------------------------------------


def _linear_settings(context: Context) -> list[Setting]:
    return [
        Setting(
            "regularization",
            "Regularization",
            "select",
            "none",
            options=["none", "ridge", "lasso", "elasticnet"],
            option_labels={
                "none": "None",
                "ridge": "Ridge (L2)",
                "lasso": "Lasso (L1)",
                "elasticnet": "Elastic net",
            },
            why="None keeps the coefficients readable, with standard errors and p-values.",
            group="Regularization",
            param="penalty",
        ),
        _strength(shown_when={"regularization": ["ridge", "lasso", "elasticnet"]}),
        Setting(
            "l1_ratio",
            "L1 share",
            "number",
            0.5,
            level="advanced",
            minimum=0.0,
            maximum=1.0,
            why="Half L1, half L2.",
            shown_when={"regularization": ["elasticnet"]},
            group="Regularization",
            param="l1_ratio",
        ),
        Setting(
            "fit_intercept",
            "Intercept",
            "bool",
            True,
            level="advanced",
            why="A constant term, almost always wanted.",
            group="Structure",
            param="fit_intercept",
        ),
        Setting(
            "solver",
            "Method",
            "select",
            "auto",
            level="expert",
            options=["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
            why="Auto picks the exact method for the data.",
            shown_when={"regularization": ["ridge"]},
            group="Optimizer",
            param="solver",
        ),
        *_iterations(),
    ]


def _linear_build(values: dict[str, Any], context: Context):
    kind, intercept = values["regularization"], values["fit_intercept"]
    if kind == "ridge":
        return Ridge(
            alpha=values["alpha"],
            solver=values["solver"],
            fit_intercept=intercept,
            max_iter=values["max_iter"],
            tol=values["tol"],
            random_state=42,
        )
    if kind == "lasso":
        return Lasso(
            alpha=values["alpha"],
            fit_intercept=intercept,
            max_iter=values["max_iter"],
            tol=values["tol"],
        )
    if kind == "elasticnet":
        return ElasticNet(
            alpha=values["alpha"],
            l1_ratio=values["l1_ratio"],
            fit_intercept=intercept,
            max_iter=values["max_iter"],
            tol=values["tol"],
        )
    return LinearRegression(fit_intercept=intercept)


def _linear_profile(values: dict[str, Any]) -> ModelProfile:
    """A regularized linear model is not sensitive to collinearity."""
    return REGULARIZED if values.get("regularization", "none") != "none" else LINEAR


# --- Robust and Bayesian regressions --------------------------------------------------------------


def _huber_settings(context: Context) -> list[Setting]:
    return [
        Setting(
            "epsilon",
            "Outlier threshold",
            "number",
            1.35,
            minimum=1.0,
            maximum=10.0,
            why="1.35 keeps 95 % efficiency on normal errors while resisting outliers.",
            group="Structure",
            param="epsilon",
        ),
        _strength(1e-4, "A tiny L2 penalty for stability."),
        *_iterations(200, "advanced"),
    ]


def _quantile_settings(context: Context) -> list[Setting]:
    return [
        Setting(
            "quantile",
            "Quantile predicted",
            "number",
            0.5,
            minimum=0.01,
            maximum=0.99,
            why="0.5 predicts the median: robust to extreme values.",
            group="Structure",
            param="quantile",
        ),
        _strength(0.0, "No penalty: the plain quantile regression.", level="advanced"),
    ]


def _bayes_settings(context: Context) -> list[Setting]:
    return _iterations(300)


def _glm_settings(power: float | None) -> Any:
    def settings(context: Context) -> list[Setting]:
        found = [_strength(1.0, "1 is the library's moderate penalty.")]
        if power is None:
            found.append(
                Setting(
                    "power",
                    "Power",
                    "number",
                    1.5,
                    minimum=0.0,
                    maximum=3.0,
                    why="Between 1 (Poisson) and 2 (Gamma): amounts with exact zeros.",
                    group="Structure",
                    param="power",
                )
            )
        found.append(
            Setting(
                "solver",
                "Solver",
                "select",
                "lbfgs",
                level="expert",
                options=["lbfgs", "newton-cholesky"],
                why="L-BFGS works everywhere.",
                group="Optimizer",
                param="solver",
            )
        )
        return [*found, *_iterations(300)]

    return settings


# --- Logistic regression --------------------------------------------------------------------------

L1_SOLVERS = ["saga", "liblinear"]


def _logistic_settings(context: Context) -> list[Setting]:
    return [
        Setting(
            "C",
            "Regularization strength C",
            "number",
            1.0,
            minimum=1e-4,
            maximum=1e4,
            log=True,
            why="C = 1 is the usual moderate penalty; lower C = simpler model.",
            group="Regularization",
            param="C",
        ),
        class_weight_setting(context),
        Setting(
            "penalty",
            "Penalty",
            "select",
            "l2",
            level="advanced",
            options=["l2", "l1", "elasticnet", "none"],
            option_labels={"l2": "L2", "l1": "L1", "elasticnet": "Elastic net", "none": "None"},
            why="L2 shrinks every weight a little and keeps them all.",
            group="Regularization",
            param="l1_ratio",
        ),
        Setting(
            "l1_ratio",
            "L1 share",
            "number",
            0.5,
            level="advanced",
            minimum=0.0,
            maximum=1.0,
            why="Half L1, half L2.",
            shown_when={"penalty": ["elasticnet"]},
            group="Regularization",
            param="l1_ratio",
        ),
        Setting(
            "multiclass",
            "Multiclass strategy",
            "select",
            "multinomial",
            level="advanced",
            options=["multinomial", "ovr"],
            option_labels={"multinomial": "Multinomial", "ovr": "One-vs-rest"},
            why="One model of all the classes together.",
            group="Structure",
        ),
        Setting(
            "solver",
            "Solver",
            "select",
            "lbfgs",
            level="expert",
            options=["lbfgs", "newton-cg", "newton-cholesky", "liblinear", "sag", "saga"],
            why="L-BFGS: fast and exact for L2.",
            group="Optimizer",
            param="solver",
        ),
        *_iterations(1_000),
        Setting(
            "intercept_scaling",
            "Intercept scaling",
            "number",
            1.0,
            level="expert",
            minimum=1e-3,
            maximum=1e3,
            log=True,
            why="Only used by liblinear.",
            shown_when={"solver": ["liblinear"]},
            group="Optimizer",
            param="intercept_scaling",
        ),
        seed_setting(),
    ]


LOGISTIC_RULES = [
    Rule(
        {"penalty": ["l1"]}, "solver", L1_SOLVERS, "L1 penalty needs the liblinear or saga solver."
    ),
    Rule({"penalty": ["elasticnet"]}, "solver", ["saga"], "Elastic net needs the saga solver."),
    Rule(
        {"penalty": ["none"]},
        "solver",
        ["lbfgs", "newton-cg", "newton-cholesky", "sag", "saga"],
        "liblinear always penalises: choose another solver.",
    ),
]


def _logistic_build(values: dict[str, Any], context: Context):
    penalty = values["penalty"]
    ratio = {"l2": 0.0, "l1": 1.0, "none": 0.0}.get(penalty, values["l1_ratio"])
    model = LogisticRegression(
        C=np.inf if penalty == "none" else values["C"],
        l1_ratio=ratio,
        solver=values["solver"],
        class_weight=weight(values),
        max_iter=values["max_iter"],
        tol=values["tol"],
        intercept_scaling=values["intercept_scaling"],
        random_state=values["random_state"],
    )
    # liblinear fits one class at a time: with several classes it always runs one-vs-rest
    if context.task == "multiclass" and (
        values["multiclass"] == "ovr" or values["solver"] == "liblinear"
    ):
        return OneVsRestClassifier(model)
    return model


# --- Other linear classifiers and SGD -------------------------------------------------------------


def _ridge_classifier_settings(context: Context) -> list[Setting]:
    return [_strength(), class_weight_setting(context)]


def _perceptron_settings(context: Context) -> list[Setting]:
    return [class_weight_setting(context), *_iterations(1_000, "advanced"), seed_setting()]


def _sgd_settings(context: Context) -> list[Setting]:
    regression = context.task == "regression"
    losses = (
        ["squared_error", "huber", "epsilon_insensitive"]
        if regression
        else ["log_loss", "hinge", "modified_huber"]
    )
    return [
        Setting(
            "loss",
            "Loss",
            "select",
            losses[0],
            options=losses,
            option_labels={
                "squared_error": "Squared error",
                "huber": "Huber",
                "epsilon_insensitive": "Epsilon-insensitive",
                "log_loss": "Log loss (logistic)",
                "hinge": "Hinge (linear SVM)",
                "modified_huber": "Modified Huber",
            },
            why="Squared error for numbers; log loss gives probabilities.",
            group="Structure",
            param="loss",
        ),
        Setting(
            "penalty",
            "Penalty",
            "select",
            "l2",
            options=["l2", "l1", "elasticnet", "none"],
            option_labels={"l2": "L2", "l1": "L1", "elasticnet": "Elastic net", "none": "None"},
            why="L2 shrinks every weight a little.",
            group="Regularization",
            param="penalty",
        ),
        _strength(1e-4, "The library's default for stochastic gradient descent."),
        *([] if regression else [class_weight_setting(context)]),
        Setting(
            "learning_rate",
            "Learning-rate schedule",
            "select",
            "invscaling" if regression else "optimal",
            level="expert",
            options=["optimal", "constant", "invscaling", "adaptive"],
            why="The library's default for the problem.",
            group="Optimizer",
            param="learning_rate",
        ),
        Setting(
            "eta0",
            "Initial rate",
            "number",
            0.01,
            level="expert",
            minimum=1e-6,
            maximum=10.0,
            log=True,
            why="Used by the constant, invscaling and adaptive schedules.",
            shown_when={"learning_rate": ["constant", "invscaling", "adaptive"]},
            group="Optimizer",
            param="eta0",
        ),
        Setting(
            "power_t",
            "Power",
            "number",
            0.25,
            level="expert",
            minimum=0.0,
            maximum=1.0,
            why="How fast invscaling lowers the rate.",
            shown_when={"learning_rate": ["invscaling"]},
            group="Optimizer",
            param="power_t",
        ),
        Setting(
            "average",
            "Averaging (ASGD)",
            "bool",
            False,
            level="expert",
            why="Averaging the weights smooths the noise of stochastic steps.",
            group="Optimizer",
            param="average",
        ),
        Setting(
            "early_stopping",
            "Early stopping",
            "bool",
            True,
            level="expert",
            why="Stops on 10 % of the training rows held out (never the test rows).",
            group="Stopping",
            param="early_stopping",
        ),
        *_iterations(1_000),
        seed_setting(),
    ]


def _sgd_build(values: dict[str, Any], context: Context):
    shared = dict(
        loss=values["loss"],
        penalty=None if values["penalty"] == "none" else values["penalty"],
        alpha=values["alpha"],
        learning_rate=values["learning_rate"],
        eta0=values["eta0"],
        power_t=values["power_t"],
        average=values["average"],
        early_stopping=values["early_stopping"],
        max_iter=values["max_iter"],
        tol=values["tol"],
        random_state=values["random_state"],
    )
    if context.task == "regression":
        return SGDRegressor(**shared)
    return SGDClassifier(class_weight=weight(values), **shared)


def _sgd_probabilities(values: dict[str, Any]) -> bool:
    return values.get("loss") in ("log_loss", "modified_huber")


# --- Discriminant analysis ----------------------------------------------------------------------


def _lda_settings(context: Context) -> list[Setting]:
    return [
        Setting(
            "solver",
            "Solver",
            "select",
            "lsqr",
            options=["svd", "lsqr", "eigen"],
            why="lsqr allows shrinkage.",
            group="Structure",
            param="solver",
        ),
        Setting(
            "shrinkage",
            "Shrinkage",
            "select",
            "auto",
            options=["auto", "none"],
            option_labels={"auto": "Auto (Ledoit-Wolf)", "none": "None"},
            why="Stabilises the covariance estimate with many inputs.",
            shown_when={"solver": ["lsqr", "eigen"]},
            group="Regularization",
            param="shrinkage",
        ),
    ]


def _qda_settings(context: Context) -> list[Setting]:
    return [
        Setting(
            "reg_param",
            "Regularization",
            "number",
            0.1,
            minimum=0.0,
            maximum=1.0,
            why="Shrinks each class covariance towards the identity: QDA overfits easily.",
            group="Regularization",
            param="reg_param",
        )
    ]


def _fast(context: Context) -> list[str]:
    return ["Fast", "Interpretable"]


MODELS = [
    ModelSpec(
        "linear_regression",
        "Linear regression",
        FAMILY,
        "A weighted sum of the features: fast, and every weight can be read.",
        ("regression",),
        LINEAR,
        _linear_settings,
        _linear_build,
        badges=_fast,
        linear=True,
        profile_for=_linear_profile,
    ),
    ModelSpec(
        "huber",
        "Huber regression",
        FAMILY,
        "A linear model that gives less weight to extreme errors.",
        ("regression",),
        REGULARIZED,
        _huber_settings,
        lambda v, c: HuberRegressor(
            epsilon=v["epsilon"], alpha=v["alpha"], max_iter=v["max_iter"], tol=v["tol"]
        ),
        badges=lambda c: ["Robust", "Interpretable"],
        linear=True,
    ),
    ModelSpec(
        "quantile",
        "Quantile regression",
        FAMILY,
        "Predicts the median (or another percentile) instead of the mean.",
        ("regression",),
        REGULARIZED,
        _quantile_settings,
        lambda v, c: QuantileRegressor(quantile=v["quantile"], alpha=v["alpha"], solver="highs"),
        badges=lambda c: ["Robust"] + (["Slow on this data"] if c.rows > 20_000 else []),
        limit=lambda c: (
            "Quantile regression needs ≤ 50,000 training rows." if c.rows > 50_000 else None
        ),
        linear=True,
    ),
    ModelSpec(
        "bayesian_ridge",
        "Bayesian ridge",
        FAMILY,
        "A ridge regression whose penalty is learnt from the data, "
        "with uncertainty on the weights.",
        ("regression",),
        REGULARIZED,
        _bayes_settings,
        lambda v, c: BayesianRidge(max_iter=v["max_iter"], tol=v["tol"]),
        badges=_fast,
        linear=True,
    ),
    ModelSpec(
        "ard",
        "ARD regression",
        FAMILY,
        "Bayesian regression that switches off useless features by itself.",
        ("regression",),
        REGULARIZED,
        _bayes_settings,
        lambda v, c: ARDRegression(max_iter=v["max_iter"], tol=v["tol"]),
        badges=lambda c: ["Interpretable"],
        linear=True,
    ),
    ModelSpec(
        "glm_poisson",
        "Poisson regression",
        FAMILY,
        "For counts (≥ 0): the log of the expected count is linear in the features.",
        ("regression",),
        REGULARIZED,
        _glm_settings(1.0),
        lambda v, c: PoissonRegressor(
            alpha=v["alpha"], solver=v["solver"], max_iter=v["max_iter"], tol=v["tol"]
        ),
        badges=lambda c: ["Counts"],
        linear=True,
    ),
    ModelSpec(
        "glm_gamma",
        "Gamma regression",
        FAMILY,
        "For positive amounts whose spread grows with their size.",
        ("regression",),
        REGULARIZED,
        _glm_settings(2.0),
        lambda v, c: GammaRegressor(
            alpha=v["alpha"], solver=v["solver"], max_iter=v["max_iter"], tol=v["tol"]
        ),
        badges=lambda c: ["Positive amounts"],
        linear=True,
    ),
    ModelSpec(
        "glm_tweedie",
        "Tweedie regression",
        FAMILY,
        "For amounts with many exact zeros (claims, sales).",
        ("regression",),
        REGULARIZED,
        _glm_settings(None),
        lambda v, c: TweedieRegressor(
            power=v["power"],
            alpha=v["alpha"],
            link="log",
            solver=v["solver"],
            max_iter=v["max_iter"],
            tol=v["tol"],
        ),
        badges=lambda c: ["Amounts with zeros"],
        linear=True,
    ),
    ModelSpec(
        "logistic_regression",
        "Logistic regression",
        FAMILY,
        "A weighted sum of the features turned into a probability: fast and interpretable.",
        ("binary", "multiclass"),
        REGULARIZED,
        _logistic_settings,
        _logistic_build,
        rules=LOGISTIC_RULES,
        badges=_fast,
        linear=True,
    ),
    ModelSpec(
        "ridge_classifier",
        "Ridge classifier",
        FAMILY,
        "A ridge regression on the classes: very fast.",
        ("binary", "multiclass"),
        REGULARIZED,
        _ridge_classifier_settings,
        lambda v, c: RidgeClassifier(alpha=v["alpha"], class_weight=weight(v)),
        badges=lambda c: ["Fast"],
        probabilities=False,
        linear=True,
    ),
    ModelSpec(
        "perceptron",
        "Perceptron",
        FAMILY,
        "The simplest neural unit: a linear frontier learnt by corrections.",
        ("binary", "multiclass"),
        REGULARIZED,
        _perceptron_settings,
        lambda v, c: SGDClassifier(
            loss="perceptron",
            penalty=None,
            learning_rate="constant",
            eta0=1.0,
            class_weight=weight(v),
            max_iter=v["max_iter"],
            tol=v["tol"],
            random_state=v["random_state"],
        ),
        badges=lambda c: ["Fast"],
        probabilities=False,
        linear=True,
    ),
    ModelSpec(
        "passive_aggressive",
        "Passive-aggressive",
        FAMILY,
        "Updates only when wrong, by just enough to be right: for large or streaming data.",
        ("binary", "multiclass"),
        REGULARIZED,
        _perceptron_settings,
        lambda v, c: SGDClassifier(
            loss="hinge",
            penalty=None,
            learning_rate="pa1",
            eta0=1.0,
            class_weight=weight(v),
            max_iter=v["max_iter"],
            tol=v["tol"],
            random_state=v["random_state"],
        ),
        badges=lambda c: ["Fast"],
        probabilities=False,
        linear=True,
    ),
    ModelSpec(
        "sgd",
        "SGD (stochastic gradient descent)",
        FAMILY,
        "A linear model fitted a few rows at a time: made for large data.",
        ("regression", "binary", "multiclass"),
        REGULARIZED,
        _sgd_settings,
        _sgd_build,
        badges=lambda c: ["Fast", "Large data"],
        probabilities=_sgd_probabilities,
        linear=True,
    ),
    ModelSpec(
        "lda",
        "Linear discriminant analysis",
        "Discriminant analysis",
        "Each class as a bell shape with a shared spread: a linear frontier.",
        ("binary", "multiclass"),
        LINEAR,
        _lda_settings,
        lambda v, c: LinearDiscriminantAnalysis(
            solver=v["solver"],
            shrinkage=None if v["solver"] == "svd" or v["shrinkage"] == "none" else "auto",
        ),
        badges=_fast,
    ),
    ModelSpec(
        "qda",
        "Quadratic discriminant analysis",
        "Discriminant analysis",
        "Each class as a bell shape with its own spread: curved frontiers.",
        ("binary", "multiclass"),
        LINEAR,
        _qda_settings,
        lambda v, c: QuadraticDiscriminantAnalysis(reg_param=v["reg_param"]),
        badges=lambda c: ["Fast"],
    ),
]
