# ruff: noqa: RUF001  (Greek letters and symbols are the usual names of settings)
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# (settings reach the libraries as validated keyword arguments: the stubs cannot follow them)
"""Nearest neighbours, support vector machines, Naive Bayes, neural networks, kernel models."""

import math
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR, NuSVC, NuSVR

from quantilo.ml.catalog.base import (
    Context,
    ModelSpec,
    Rule,
    Setting,
    class_weight_setting,
    layers,
    seed_setting,
    weight,
)
from quantilo.ml.preparation import ModelProfile

DISTANCE = ModelProfile(needs_scaling=True, sensitive_to_skew=True)
NONNEGATIVE = ModelProfile(nonnegative_inputs=True)
GAUSSIAN = ModelProfile(sensitive_to_skew=True)
CLASSES = ("binary", "multiclass")
EVERY = ("regression", "binary", "multiclass")


def _regression(context: Context) -> bool:
    return context.task == "regression"


# --- Nearest neighbours ---------------------------------------------------------------------------


def neighbours(context: Context) -> int:
    """√n bounded to 5 to 50, odd for two classes (no ties)."""
    k = min(50, max(5, round(math.sqrt(max(context.rows, 1)))))
    if context.task == "binary" and k % 2 == 0:
        k = k - 1 if k == 50 else k + 1
    return k


def _knn_settings(context: Context) -> list[Setting]:
    k = neighbours(context)
    odd = ", odd for two classes" if context.task == "binary" else ""
    return [
        Setting(
            "n_neighbors",
            "Neighbours k",
            "int",
            k,
            minimum=1,
            maximum=500,
            why=f"√n bounded to 5 to 50{odd}: k = {k}.",
            group="Structure",
            param="n_neighbors",
        ),
        Setting(
            "weights",
            "Weights",
            "select",
            "distance",
            options=["uniform", "distance"],
            option_labels={"uniform": "Equal", "distance": "Closer counts more"},
            why="Closer neighbours are more alike.",
            group="Structure",
            param="weights",
        ),
        Setting(
            "metric",
            "Distance",
            "select",
            "minkowski",
            level="expert",
            options=["minkowski", "manhattan", "cosine"],
            option_labels={
                "minkowski": "Euclidean / Minkowski",
                "manhattan": "Manhattan",
                "cosine": "Cosine",
            },
            why="Straight-line distance on standardised inputs.",
            group="Optimizer",
            param="metric",
        ),
        Setting(
            "p",
            "Minkowski p",
            "number",
            2.0,
            level="expert",
            minimum=1.0,
            maximum=10.0,
            why="2 = Euclidean, 1 = Manhattan.",
            shown_when={"metric": ["minkowski"]},
            group="Optimizer",
            param="p",
        ),
        Setting(
            "algorithm",
            "Search",
            "select",
            "auto",
            level="expert",
            options=["auto", "ball_tree", "kd_tree", "brute"],
            why="Auto picks the fastest index for the data.",
            group="Optimizer",
            param="algorithm",
        ),
    ]


KNN_RULES = [
    Rule(
        {"metric": ["cosine"]},
        "algorithm",
        ["brute", "auto"],
        "Cosine distance only works with brute-force search.",
    )
]


def _knn_build(values: dict[str, Any], context: Context):
    algorithm = "brute" if values["metric"] == "cosine" else values["algorithm"]
    shared = dict(
        n_neighbors=values["n_neighbors"],
        weights=values["weights"],
        metric=values["metric"],
        p=values["p"],
        algorithm=algorithm,
    )
    return KNeighborsRegressor(**shared) if _regression(context) else KNeighborsClassifier(**shared)


def _knn_badges(context: Context) -> list[str]:
    return ["No training"] + (["Slow to predict on this data"] if context.rows > 100_000 else [])


# --- Support vector machines ----------------------------------------------------------------------

KERNEL_LIMIT, KERNEL_SLOW = 50_000, 10_000


def _c(default: float = 1.0) -> Setting:
    return Setting(
        "C",
        "C (penalty of errors)",
        "number",
        default,
        minimum=1e-4,
        maximum=1e4,
        log=True,
        why="1 balances a smooth frontier and few training errors.",
        group="Regularization",
        param="C",
    )


def _linear_svm_settings(context: Context) -> list[Setting]:
    regression = _regression(context)
    losses = (
        ["epsilon_insensitive", "squared_epsilon_insensitive"]
        if regression
        else ["squared_hinge", "hinge"]
    )
    return [
        _c(),
        *([] if regression else [class_weight_setting(context)]),
        Setting(
            "loss",
            "Loss",
            "select",
            losses[0],
            level="expert",
            options=losses,
            why="The library's default.",
            group="Optimizer",
            param="loss",
        ),
        Setting(
            "max_iter",
            "Max iterations",
            "int",
            5_000,
            level="expert",
            minimum=100,
            maximum=100_000,
            why="Enough to converge on standardised inputs.",
            group="Optimizer",
            param="max_iter",
        ),
        seed_setting(),
    ]


def _linear_svm_build(values: dict[str, Any], context: Context):
    if _regression(context):
        return LinearSVR(
            C=values["C"],
            loss=values["loss"],
            max_iter=values["max_iter"],
            random_state=values["random_state"],
        )
    return LinearSVC(
        C=values["C"],
        loss=values["loss"],
        class_weight=weight(values),
        max_iter=values["max_iter"],
        random_state=values["random_state"],
    )


def _kernel_settings(nu: bool):
    def settings(context: Context) -> list[Setting]:
        found = [
            Setting(
                "kernel",
                "Kernel",
                "select",
                "rbf",
                options=["rbf", "poly", "sigmoid"],
                option_labels={"rbf": "RBF (Gaussian)", "poly": "Polynomial", "sigmoid": "Sigmoid"},
                why="RBF bends the frontier smoothly: the usual first choice.",
                group="Structure",
                param="kernel",
            ),
            Setting(
                "nu",
                "ν (share of support vectors)",
                "number",
                0.5,
                minimum=0.01,
                maximum=1.0,
                why="Upper bound on training errors, lower bound on support vectors.",
                group="Regularization",
                param="nu",
            )
            if nu
            else _c(),
            Setting(
                "gamma",
                "γ (reach of each point)",
                "select",
                "scale",
                options=["scale", "auto"],
                why="Scale: 1 / (inputs × variance).",
                group="Structure",
                param="gamma",
            ),
            Setting(
                "degree",
                "Degree",
                "int",
                3,
                level="advanced",
                minimum=1,
                maximum=10,
                why="Cubic.",
                shown_when={"kernel": ["poly"]},
                group="Structure",
                param="degree",
            ),
            Setting(
                "coef0",
                "Kernel offset",
                "number",
                0.0,
                level="expert",
                minimum=-10.0,
                maximum=10.0,
                shown_when={"kernel": ["poly", "sigmoid"]},
                group="Optimizer",
                param="coef0",
            ),
            Setting(
                "tol",
                "Tolerance",
                "number",
                1e-3,
                level="expert",
                minimum=1e-8,
                maximum=1e-1,
                log=True,
                group="Optimizer",
                param="tol",
            ),
            Setting(
                "cache_size",
                "Cache (MB)",
                "int",
                500,
                level="expert",
                minimum=50,
                maximum=8_000,
                why="More memory for the kernel speeds training up.",
                group="Optimizer",
                param="cache_size",
            ),
            Setting(
                "shrinking",
                "Shrinking heuristic",
                "bool",
                True,
                level="expert",
                group="Optimizer",
                param="shrinking",
            ),
        ]
        if not _regression(context):
            found.insert(2, class_weight_setting(context))
        elif not nu:
            found.insert(
                2,
                Setting(
                    "epsilon",
                    "ε (tolerated error)",
                    "number",
                    0.1,
                    level="advanced",
                    minimum=0.0,
                    maximum=100.0,
                    why="Errors smaller than ε cost nothing (on the target's scale).",
                    group="Regularization",
                    param="epsilon",
                ),
            )
        return found

    return settings


def _kernel_build(nu: bool):
    def build(values: dict[str, Any], context: Context):
        shared = dict(
            kernel=values["kernel"],
            gamma=values["gamma"],
            degree=values["degree"],
            coef0=values["coef0"],
            tol=values["tol"],
            cache_size=values["cache_size"],
            shrinking=values["shrinking"],
        )
        if _regression(context):
            return (
                NuSVR(nu=values["nu"], **shared)
                if nu
                else SVR(C=values["C"], epsilon=values["epsilon"], **shared)
            )
        common = dict(class_weight=weight(values), probability=True, random_state=42, **shared)
        return NuSVC(nu=values["nu"], **common) if nu else SVC(C=values["C"], **common)

    return build


def _kernel_limit(context: Context) -> str | None:
    if context.rows > KERNEL_LIMIT:
        return (
            f"Kernel SVMs need ≤ {KERNEL_LIMIT:,} training rows "
            "(their cost grows with the square of the rows)."
        )
    return None


def _kernel_badges(context: Context) -> list[str]:
    return [f"Slow on {context.rows:,} rows"] if context.rows > KERNEL_SLOW else []


# --- Naive Bayes ----------------------------------------------------------------------------------


def _alpha(context: Context) -> list[Setting]:
    return [
        Setting(
            "alpha",
            "Smoothing α",
            "number",
            1.0,
            minimum=1e-6,
            maximum=100.0,
            log=True,
            why="Laplace smoothing: a category never seen in a class is not impossible.",
            group="Regularization",
            param="alpha",
        )
    ]


def _gaussian_nb_settings(context: Context) -> list[Setting]:
    return [
        Setting(
            "var_smoothing",
            "Variance smoothing",
            "number",
            1e-9,
            minimum=1e-12,
            maximum=1.0,
            log=True,
            why="A tiny share of the largest variance added for stability.",
            group="Regularization",
            param="var_smoothing",
        )
    ]


# --- Neural network -------------------------------------------------------------------------------


def _mlp_settings(context: Context) -> list[Setting]:
    return [
        Setting(
            "hidden_layer_sizes",
            "Hidden layers",
            "text",
            "64, 32",
            why="Two layers, wide then narrow: enough for most tabular data.",
            group="Structure",
            param="hidden_layer_sizes",
        ),
        Setting(
            "alpha",
            "Regularization",
            "number",
            1e-4,
            minimum=1e-7,
            maximum=10.0,
            log=True,
            why="A light L2 penalty on the weights.",
            group="Regularization",
            param="alpha",
        ),
        Setting(
            "activation",
            "Activation",
            "select",
            "relu",
            level="advanced",
            options=["relu", "tanh", "logistic", "identity"],
            why="ReLU trains fast and well.",
            group="Structure",
            param="activation",
        ),
        Setting(
            "early_stopping",
            "Early stopping",
            "bool",
            True,
            level="advanced",
            why="Stops when 10 % of the training rows held out stop improving (not the test rows).",
            group="Stopping",
            param="early_stopping",
        ),
        Setting(
            "validation_fraction",
            "Held-out share",
            "number",
            0.1,
            level="advanced",
            minimum=0.05,
            maximum=0.4,
            shown_when={"early_stopping": [True]},
            group="Stopping",
            param="validation_fraction",
        ),
        Setting(
            "batch_size",
            "Batch size",
            "int",
            min(200, max(1, context.rows)),
            level="advanced",
            minimum=1,
            maximum=10_000,
            why="min(200, rows).",
            group="Sampling",
            param="batch_size",
        ),
        Setting(
            "solver",
            "Optimizer",
            "select",
            "adam",
            level="expert",
            options=["adam", "sgd", "lbfgs"],
            option_labels={"adam": "Adam", "sgd": "SGD", "lbfgs": "L-BFGS"},
            why="Adam adapts the step of each weight: robust default.",
            group="Optimizer",
            param="solver",
        ),
        Setting(
            "learning_rate_init",
            "Initial learning rate",
            "number",
            1e-3,
            level="expert",
            minimum=1e-6,
            maximum=1.0,
            log=True,
            shown_when={"solver": ["adam", "sgd"]},
            group="Optimizer",
            param="learning_rate_init",
        ),
        Setting(
            "learning_rate",
            "Schedule",
            "select",
            "constant",
            level="expert",
            options=["constant", "invscaling", "adaptive"],
            shown_when={"solver": ["sgd"]},
            group="Optimizer",
            param="learning_rate",
        ),
        Setting(
            "momentum",
            "Momentum",
            "number",
            0.9,
            level="expert",
            minimum=0.0,
            maximum=1.0,
            shown_when={"solver": ["sgd"]},
            group="Optimizer",
            param="momentum",
        ),
        Setting(
            "nesterovs_momentum",
            "Nesterov",
            "bool",
            True,
            level="expert",
            shown_when={"solver": ["sgd"]},
            group="Optimizer",
            param="nesterovs_momentum",
        ),
        Setting(
            "beta_1",
            "β1",
            "number",
            0.9,
            level="expert",
            minimum=0.0,
            maximum=0.9999,
            shown_when={"solver": ["adam"]},
            group="Optimizer",
            param="beta_1",
        ),
        Setting(
            "beta_2",
            "β2",
            "number",
            0.999,
            level="expert",
            minimum=0.0,
            maximum=0.99999,
            shown_when={"solver": ["adam"]},
            group="Optimizer",
            param="beta_2",
        ),
        Setting(
            "epsilon",
            "ε",
            "number",
            1e-8,
            level="expert",
            minimum=1e-12,
            maximum=1e-3,
            log=True,
            shown_when={"solver": ["adam"]},
            group="Optimizer",
            param="epsilon",
        ),
        Setting(
            "max_iter",
            "Max epochs",
            "int",
            200,
            level="expert",
            minimum=10,
            maximum=10_000,
            group="Optimizer",
            param="max_iter",
        ),
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
        ),
        Setting(
            "n_iter_no_change",
            "Patience (epochs)",
            "int",
            10,
            level="expert",
            minimum=1,
            maximum=500,
            group="Stopping",
            param="n_iter_no_change",
        ),
        seed_setting(),
    ]


MLP_RULES = [
    Rule(
        {"solver": ["lbfgs"]},
        "early_stopping",
        [False],
        "L-BFGS has no early stopping: it uses all the training rows at once.",
    )
]


def _mlp_build(values: dict[str, Any], context: Context):
    shared = {
        key: values[key]
        for key in (
            "alpha",
            "activation",
            "early_stopping",
            "validation_fraction",
            "batch_size",
            "solver",
            "learning_rate_init",
            "learning_rate",
            "momentum",
            "nesterovs_momentum",
            "beta_1",
            "beta_2",
            "epsilon",
            "max_iter",
            "tol",
            "n_iter_no_change",
            "random_state",
        )
    }
    shared["hidden_layer_sizes"] = layers(values["hidden_layer_sizes"])
    return MLPRegressor(**shared) if _regression(context) else MLPClassifier(**shared)


# --- Kernel and probabilistic ---------------------------------------------------------------------

GP_LIMIT, KERNEL_RIDGE_LIMIT = 5_000, 10_000


def _kernel_ridge_settings(context: Context) -> list[Setting]:
    return [
        Setting(
            "alpha",
            "Strength",
            "number",
            1.0,
            minimum=1e-6,
            maximum=1e4,
            log=True,
            why="1 is a moderate penalty.",
            group="Regularization",
            param="alpha",
        ),
        Setting(
            "kernel",
            "Kernel",
            "select",
            "rbf",
            options=["rbf", "linear", "poly", "laplacian"],
            why="RBF: smooth curves.",
            group="Structure",
            param="kernel",
        ),
        Setting(
            "degree",
            "Degree",
            "int",
            3,
            level="advanced",
            minimum=1,
            maximum=10,
            shown_when={"kernel": ["poly"]},
            group="Structure",
            param="degree",
        ),
    ]


def _gp_settings(context: Context) -> list[Setting]:
    return [
        Setting(
            "n_restarts_optimizer",
            "Restarts of the kernel optimizer",
            "int",
            0,
            level="expert",
            minimum=0,
            maximum=10,
            why="L-BFGS-B fits the kernel; restarts avoid local optima at a cost.",
            group="Optimizer",
            param="n_restarts_optimizer",
        ),
        seed_setting(),
    ]


def _gp_build(values: dict[str, Any], context: Context):
    if _regression(context):
        kernel = ConstantKernel() * RBF() + WhiteKernel()
        return GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=values["n_restarts_optimizer"],
            random_state=values["random_state"],
        )
    return GaussianProcessClassifier(
        kernel=ConstantKernel() * RBF(),
        n_restarts_optimizer=values["n_restarts_optimizer"],
        random_state=values["random_state"],
    )


def _limit(maximum: int, name: str):
    def limit(context: Context) -> str | None:
        return (
            f"{name} needs ≤ {maximum:,} training rows (its cost grows with the cube of the rows)."
            if context.rows > maximum
            else None
        )

    return limit


# --- Radius neighbours and Categorical Naive Bayes (019b) -----------------------------------------


def radius(context: Context) -> float:
    """On standardised inputs, distances grow like the square root of the number of inputs."""
    return round(math.sqrt(max(context.features, 1)), 1)


def _radius_settings(context: Context) -> list[Setting]:
    r = radius(context)
    return [
        Setting(
            "radius",
            "Radius",
            "number",
            r,
            minimum=0.01,
            maximum=1_000.0,
            log=True,
            why=f"√(inputs) on standardised inputs: {r}.",
            group="Structure",
            param="radius",
        ),
        Setting(
            "weights",
            "Weights",
            "select",
            "distance",
            options=["uniform", "distance"],
            option_labels={"uniform": "Equal", "distance": "Closer counts more"},
            why="Closer neighbours are more alike.",
            group="Structure",
            param="weights",
        ),
    ]


class RadiusRegressor(BaseEstimator, RegressorMixin):
    """Radius neighbours that predict the training mean where a row has no neighbour."""

    def __init__(self, radius: float = 1.0, weights: str = "distance") -> None:
        self.radius = radius
        self.weights = weights

    def fit(self, X, y):
        self.model_ = RadiusNeighborsRegressor(radius=self.radius, weights=self.weights).fit(X, y)
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            found = np.asarray(self.model_.predict(X), dtype=float)
        return np.where(np.isfinite(found), found, self.mean_)


def _radius_build(values: dict[str, Any], context: Context):
    if _regression(context):
        return RadiusRegressor(values["radius"], values["weights"])
    return RadiusNeighborsClassifier(
        radius=values["radius"], weights=values["weights"], outlier_label="most_frequent"
    )


def _categorical_nb_settings(context: Context) -> list[Setting]:
    return [
        *_alpha(context),
        Setting(
            "bins",
            "Bins per number",
            "int",
            10,
            level="advanced",
            minimum=2,
            maximum=50,
            why="Numbers are cut into 10 quantile bins: the model only counts categories.",
            group="Structure",
            param="n_bins",
        ),
    ]


def _categorical_nb_build(values: dict[str, Any], context: Context):
    bins = int(values["bins"])
    return make_pipeline(
        KBinsDiscretizer(
            n_bins=bins,
            encode="ordinal",
            strategy="quantile",
            quantile_method="averaged_inverted_cdf",
        ),
        CategoricalNB(alpha=values["alpha"], min_categories=bins),
    )


MODELS = [
    ModelSpec(
        "knn",
        "k-nearest neighbours",
        "Nearest neighbours",
        "Predicts like the k most similar rows.",
        EVERY,
        DISTANCE,
        _knn_settings,
        _knn_build,
        rules=KNN_RULES,
        badges=_knn_badges,
    ),
    ModelSpec(
        "linear_svm",
        "Linear SVM",
        "Support vector machines",
        "The widest straight frontier between classes (or a tube around the numbers).",
        EVERY,
        DISTANCE,
        _linear_svm_settings,
        _linear_svm_build,
        badges=lambda c: ["Fast"],
        probabilities=False,
        linear=True,
    ),
    ModelSpec(
        "kernel_svm",
        "Kernel SVM",
        "Support vector machines",
        "A frontier that bends through a kernel: powerful on small and medium data.",
        EVERY,
        DISTANCE,
        _kernel_settings(False),
        _kernel_build(False),
        badges=_kernel_badges,
        limit=_kernel_limit,
    ),
    ModelSpec(
        "nu_svm",
        "Nu-SVM",
        "Support vector machines",
        "A kernel SVM set by the share of support vectors instead of C.",
        EVERY,
        DISTANCE,
        _kernel_settings(True),
        _kernel_build(True),
        badges=_kernel_badges,
        limit=_kernel_limit,
    ),
    ModelSpec(
        "radius_neighbours",
        "Radius neighbours",
        "Nearest neighbours",
        "Predicts like every row within a given distance (not a fixed number of neighbours).",
        EVERY,
        DISTANCE,
        _radius_settings,
        _radius_build,
        badges=lambda c: ["No training"],
    ),
    ModelSpec(
        "categorical_nb",
        "Categorical Naive Bayes",
        "Naive Bayes",
        "Counts how often each category (numbers cut into bins) appears in each class.",
        CLASSES,
        NONNEGATIVE,
        _categorical_nb_settings,
        _categorical_nb_build,
        badges=lambda c: ["Fast"],
    ),
    ModelSpec(
        "gaussian_nb",
        "Gaussian Naive Bayes",
        "Naive Bayes",
        "Each feature a bell curve per class, taken as independent: very fast.",
        CLASSES,
        GAUSSIAN,
        _gaussian_nb_settings,
        lambda v, c: GaussianNB(var_smoothing=v["var_smoothing"]),
        badges=lambda c: ["Fast"],
    ),
    ModelSpec(
        "multinomial_nb",
        "Multinomial Naive Bayes",
        "Naive Bayes",
        "Counts per class, for non-negative inputs (inputs are scaled to [0, 1]).",
        CLASSES,
        NONNEGATIVE,
        _alpha,
        lambda v, c: MultinomialNB(alpha=v["alpha"]),
        badges=lambda c: ["Fast"],
    ),
    ModelSpec(
        "complement_nb",
        "Complement Naive Bayes",
        "Naive Bayes",
        "Multinomial Naive Bayes made robust to imbalanced classes.",
        CLASSES,
        NONNEGATIVE,
        _alpha,
        lambda v, c: ComplementNB(alpha=v["alpha"]),
        badges=lambda c: ["Fast"],
    ),
    ModelSpec(
        "bernoulli_nb",
        "Bernoulli Naive Bayes",
        "Naive Bayes",
        "Yes / no features per class (numbers are cut at 0 after standardising).",
        CLASSES,
        ModelProfile(needs_scaling=True),
        _alpha,
        lambda v, c: BernoulliNB(alpha=v["alpha"]),
        badges=lambda c: ["Fast"],
    ),
    ModelSpec(
        "mlp",
        "Neural network (MLP)",
        "Neural networks",
        "Layers of neurons that learn curved relations between the features.",
        EVERY,
        DISTANCE,
        _mlp_settings,
        _mlp_build,
        rules=MLP_RULES,
        badges=lambda c: ["Needs lots of data"] if c.rows < 2_000 else [],
    ),
    ModelSpec(
        "kernel_ridge",
        "Kernel ridge",
        "Kernel & probabilistic",
        "A ridge regression through a kernel: smooth curves on small data.",
        ("regression",),
        DISTANCE,
        _kernel_ridge_settings,
        lambda v, c: KernelRidge(alpha=v["alpha"], kernel=v["kernel"], degree=v["degree"]),
        limit=_limit(KERNEL_RIDGE_LIMIT, "Kernel ridge"),
        badges=lambda c: ["Small data"],
    ),
    ModelSpec(
        "gaussian_process",
        "Gaussian process",
        "Kernel & probabilistic",
        "A distribution over smooth functions, with honest uncertainty: small data only.",
        EVERY,
        DISTANCE,
        _gp_settings,
        _gp_build,
        limit=_limit(GP_LIMIT, "A Gaussian process"),
        badges=lambda c: ["Small data", "Uncertainty"],
    ),
]
