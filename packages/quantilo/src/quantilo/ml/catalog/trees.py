# ruff: noqa: RUF001  (Greek letters and symbols are the usual names of settings)
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# (settings reach the libraries as validated keyword arguments: the stubs cannot follow them)
"""Trees, bagging and boosting ensembles (XGBoost, LightGBM, CatBoost when installed)."""

import importlib
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from sklearn.utils import ClassifierTags, RegressorTags
from sklearn.utils.class_weight import compute_sample_weight

from quantilo.ml.catalog.base import (
    Context,
    ModelSpec,
    Setting,
    class_weight_setting,
    seed_setting,
    weight,
)
from quantilo.ml.preparation import ModelProfile

TREE = ModelProfile(tree_based=True, handles_missing=True)  # scikit-learn trees split on NaN
TREE_FILLED = ModelProfile(tree_based=True)
EVERY = ("regression", "binary", "multiclass")
HELD_OUT = 0.1  # early stopping: share of the training rows held out (never the test rows)


def _regression(context: Context) -> bool:
    return context.task == "regression"


def _criteria(context: Context) -> tuple[list[str], dict[str, str]]:
    if _regression(context):
        return ["squared_error", "absolute_error", "poisson"], {
            "squared_error": "Squared error",
            "absolute_error": "Absolute error",
            "poisson": "Poisson",
        }
    return ["gini", "entropy", "log_loss"], {
        "gini": "Gini",
        "entropy": "Entropy",
        "log_loss": "Log loss",
    }


def _criterion(context: Context) -> Setting:
    options, labels = _criteria(context)
    return Setting(
        "criterion",
        "Split quality",
        "select",
        options[0],
        level="expert",
        options=options,
        option_labels=labels,
        why="The library's default.",
        group="Optimizer",
        param="criterion",
    )


def _depth(default: int | None, why: str) -> Setting:
    return Setting(
        "max_depth",
        "Tree depth",
        "int",
        default if default is not None else 0,
        minimum=0,
        maximum=100,
        why=why + " (0 = unlimited)",
        group="Structure",
        param="max_depth",
    )


def _leaf(context: Context, small: int = 1, large: int = 5) -> Setting:
    value = large if context.rows > 100_000 else small
    return Setting(
        "min_samples_leaf",
        "Min. rows per leaf",
        "int",
        value,
        minimum=1,
        maximum=10_000,
        why=f"{value}{' (more than 100 000 rows: steadier leaves)' if value == large else ''}.",
        group="Structure",
        param="min_samples_leaf",
    )


def _none_if_zero(value: int) -> int | None:
    return value or None


def _monotonic_setting(param: str) -> Setting:
    return Setting(
        "monotonic",
        "Monotonic constraints",
        "text",
        "",
        level="advanced",
        why="Optional, e.g. “size:+1, age:-1”: the prediction may only rise with size and only "
        "fall with age (business knowledge the model must respect).",
        group="Structure",
        param=param,
    )


# --- Single trees ---------------------------------------------------------------------------------


def _tree_settings(context: Context) -> list[Setting]:
    return [
        _depth(5, "5 levels stay readable"),
        Setting(
            "min_samples_leaf",
            "Min. rows per leaf",
            "int",
            20,
            minimum=1,
            maximum=10_000,
            why="20 avoids leaves built on a handful of rows.",
            group="Structure",
            param="min_samples_leaf",
        ),
        *([] if _regression(context) else [class_weight_setting(context)]),
        _criterion(context),
        Setting(
            "splitter",
            "Splitter",
            "select",
            "best",
            level="expert",
            options=["best", "random"],
            group="Optimizer",
            param="splitter",
        ),
        Setting(
            "ccp_alpha",
            "Pruning (ccp α)",
            "number",
            0.0,
            level="expert",
            minimum=0.0,
            maximum=1e3,
            why="0: no cost-complexity pruning.",
            group="Regularization",
            param="ccp_alpha",
        ),
        seed_setting(),
    ]


def _tree_build(extra: bool):
    def build(values: dict[str, Any], context: Context):
        shared = dict(
            max_depth=_none_if_zero(values["max_depth"]),
            min_samples_leaf=values["min_samples_leaf"],
            criterion=values["criterion"],
            ccp_alpha=values["ccp_alpha"],
            random_state=values["random_state"],
        )
        if _regression(context):
            return (ExtraTreeRegressor if extra else DecisionTreeRegressor)(
                splitter=values["splitter"], **shared
            )
        return (ExtraTreeClassifier if extra else DecisionTreeClassifier)(
            splitter=values["splitter"], class_weight=weight(values), **shared
        )

    return build


# --- Bagging ensembles ----------------------------------------------------------------------------


def _forest_settings(context: Context) -> list[Setting]:
    return [
        Setting(
            "n_estimators",
            "Number of trees",
            "int",
            300,
            minimum=10,
            maximum=5_000,
            why="300 trees: more rarely helps, it only costs time.",
            group="Structure",
            param="n_estimators",
        ),
        _depth(None, "Unlimited: each tree grows until its leaves are small"),
        _leaf(context),
        Setting(
            "max_features",
            "Features tried per split",
            "select",
            "sqrt" if not _regression(context) else "1.0",
            level="advanced",
            options=["sqrt", "log2", "0.5", "1.0"],
            option_labels={
                "sqrt": "√ features",
                "log2": "log₂ features",
                "0.5": "Half",
                "1.0": "All",
            },
            why="The library's default for the problem.",
            group="Sampling",
            param="max_features",
        ),
        *([] if _regression(context) else [class_weight_setting(context)]),
        _criterion(context),
        Setting(
            "bootstrap",
            "Bootstrap rows",
            "bool",
            True,
            level="expert",
            why="Each tree sees a random draw of the rows.",
            group="Sampling",
            param="bootstrap",
        ),
        Setting(
            "max_samples",
            "Rows per tree",
            "number",
            1.0,
            level="expert",
            minimum=0.05,
            maximum=1.0,
            shown_when={"bootstrap": [True]},
            group="Sampling",
            param="max_samples",
        ),
        seed_setting(),
    ]


def _features(value: str) -> str | float:
    return value if value in ("sqrt", "log2") else float(value)


def _forest_build(extra: bool):
    def build(values: dict[str, Any], context: Context):
        shared = dict(
            n_estimators=values["n_estimators"],
            max_depth=_none_if_zero(values["max_depth"]),
            min_samples_leaf=values["min_samples_leaf"],
            max_features=_features(values["max_features"]),
            criterion=values["criterion"],
            bootstrap=values["bootstrap"],
            max_samples=values["max_samples"]
            if values["bootstrap"] and values["max_samples"] < 1
            else None,
            n_jobs=-1,
            random_state=values["random_state"],
        )
        if _regression(context):
            return (ExtraTreesRegressor if extra else RandomForestRegressor)(**shared)
        return (ExtraTreesClassifier if extra else RandomForestClassifier)(
            class_weight=weight(values), **shared
        )

    return build


def _bagging_settings(context: Context) -> list[Setting]:
    return [
        Setting(
            "n_estimators",
            "Number of models",
            "int",
            50,
            minimum=2,
            maximum=2_000,
            why="50 full trees, each on a random draw of rows.",
            group="Structure",
            param="n_estimators",
        ),
        Setting(
            "max_samples",
            "Rows per model",
            "number",
            1.0,
            level="advanced",
            minimum=0.05,
            maximum=1.0,
            group="Sampling",
            param="max_samples",
        ),
        Setting(
            "max_features",
            "Features per model",
            "number",
            1.0,
            level="advanced",
            minimum=0.05,
            maximum=1.0,
            group="Sampling",
            param="max_features",
        ),
        seed_setting(),
    ]


def _bagging_build(values: dict[str, Any], context: Context):
    shared = dict(
        n_estimators=values["n_estimators"],
        max_samples=values["max_samples"],
        max_features=values["max_features"],
        n_jobs=-1,
        random_state=values["random_state"],
    )
    return BaggingRegressor(**shared) if _regression(context) else BaggingClassifier(**shared)


# --- scikit-learn boosting ------------------------------------------------------------------------


def _hist_settings(context: Context) -> list[Setting]:
    return [
        Setting(
            "max_iter",
            "Max iterations",
            "int",
            300,
            minimum=10,
            maximum=10_000,
            why="300, cut short by early stopping on 10 % of the training rows.",
            group="Structure",
            param="max_iter",
        ),
        Setting(
            "learning_rate",
            "Learning rate",
            "number",
            0.1,
            minimum=1e-3,
            maximum=1.0,
            log=True,
            why="0.1: the usual balance of speed and accuracy.",
            group="Structure",
            param="learning_rate",
        ),
        Setting(
            "max_leaf_nodes",
            "Max leaves per tree",
            "int",
            31,
            minimum=2,
            maximum=1_024,
            why="31 leaves, like LightGBM.",
            group="Structure",
            param="max_leaf_nodes",
        ),
        Setting(
            "l2_regularization",
            "L2 penalty",
            "number",
            0.0,
            level="advanced",
            minimum=0.0,
            maximum=100.0,
            group="Regularization",
            param="l2_regularization",
        ),
        Setting(
            "min_samples_leaf",
            "Min. rows per leaf",
            "int",
            20,
            level="advanced",
            minimum=1,
            maximum=10_000,
            group="Structure",
            param="min_samples_leaf",
        ),
        Setting(
            "early_stopping",
            "Early stopping",
            "bool",
            True,
            level="advanced",
            why="Held-out 10 % of the training rows, never the test rows.",
            group="Stopping",
            param="early_stopping",
        ),
        *([] if _regression(context) else [class_weight_setting(context)]),
        Setting(
            "max_bins",
            "Max bins",
            "int",
            255,
            level="expert",
            minimum=2,
            maximum=255,
            group="Optimizer",
            param="max_bins",
        ),
        _monotonic_setting("monotonic_cst"),
        seed_setting(),
    ]


def _hist_build(values: dict[str, Any], context: Context):
    shared = dict(
        max_iter=values["max_iter"],
        learning_rate=values["learning_rate"],
        max_leaf_nodes=values["max_leaf_nodes"],
        l2_regularization=values["l2_regularization"],
        min_samples_leaf=values["min_samples_leaf"],
        early_stopping=values["early_stopping"],
        validation_fraction=HELD_OUT,
        max_bins=values["max_bins"],
        random_state=values["random_state"],
    )
    if _regression(context):
        return HistGradientBoostingRegressor(**shared)
    return HistGradientBoostingClassifier(class_weight=weight(values), **shared)


def _gb_settings(context: Context) -> list[Setting]:
    return [
        Setting(
            "n_estimators",
            "Number of trees",
            "int",
            200,
            minimum=10,
            maximum=5_000,
            why="200 small trees, cut short when 10 % held-out rows stop improving.",
            group="Structure",
            param="n_estimators",
        ),
        Setting(
            "learning_rate",
            "Learning rate",
            "number",
            0.1,
            minimum=1e-3,
            maximum=1.0,
            log=True,
            group="Structure",
            param="learning_rate",
        ),
        Setting(
            "max_depth",
            "Tree depth",
            "int",
            3,
            minimum=1,
            maximum=20,
            why="Shallow trees, each correcting the previous ones.",
            group="Structure",
            param="max_depth",
        ),
        Setting(
            "subsample",
            "Rows per tree",
            "number",
            1.0,
            level="advanced",
            minimum=0.1,
            maximum=1.0,
            why="Below 1: stochastic gradient boosting.",
            group="Sampling",
            param="subsample",
        ),
        Setting(
            "n_iter_no_change",
            "Early stopping patience",
            "int",
            10,
            level="advanced",
            minimum=0,
            maximum=500,
            why="Trees without improvement before stopping (0 = never).",
            group="Stopping",
            param="n_iter_no_change",
        ),
        seed_setting(),
    ]


def _gb_build(values: dict[str, Any], context: Context):
    shared = dict(
        n_estimators=values["n_estimators"],
        learning_rate=values["learning_rate"],
        max_depth=values["max_depth"],
        subsample=values["subsample"],
        n_iter_no_change=values["n_iter_no_change"] or None,
        validation_fraction=HELD_OUT,
        random_state=values["random_state"],
    )
    return (
        GradientBoostingRegressor(**shared)
        if _regression(context)
        else GradientBoostingClassifier(**shared)
    )


def _ada_settings(context: Context) -> list[Setting]:
    return [
        Setting(
            "n_estimators",
            "Number of estimators",
            "int",
            100,
            minimum=2,
            maximum=5_000,
            group="Structure",
            param="n_estimators",
        ),
        Setting(
            "learning_rate",
            "Learning rate",
            "number",
            1.0,
            minimum=1e-3,
            maximum=10.0,
            log=True,
            group="Structure",
            param="learning_rate",
        ),
        Setting(
            "base_depth",
            "Depth of each tree",
            "int",
            1 if not _regression(context) else 3,
            level="expert",
            minimum=1,
            maximum=3,
            why="Stumps for classes, depth 3 for numbers.",
            group="Optimizer",
            param="estimator.max_depth",
        ),
        seed_setting(),
    ]


def _ada_build(values: dict[str, Any], context: Context):
    if _regression(context):
        return AdaBoostRegressor(
            DecisionTreeRegressor(max_depth=values["base_depth"]),
            n_estimators=values["n_estimators"],
            learning_rate=values["learning_rate"],
            random_state=values["random_state"],
        )
    return AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=values["base_depth"]),
        n_estimators=values["n_estimators"],
        learning_rate=values["learning_rate"],
        random_state=values["random_state"],
    )


# --- XGBoost, LightGBM, CatBoost ------------------------------------------------------------------


class EarlyStopped(BaseEstimator):
    """Fits a boosting library with early stopping on a share of the training rows held out (never
    the test rows), with labels encoded as 0..k-1 and balanced weights when asked."""

    def __init__(
        self,
        estimator=None,
        library: str = "xgboost",
        patience: int = 50,
        balanced: bool = False,
        classes: bool = False,
    ) -> None:
        self.estimator = estimator
        self.library = library
        self.patience = patience
        self.balanced = balanced
        self.classes = classes

    def fit(self, X, y):
        target = np.asarray(y)
        if self.classes:
            self.encoder_ = LabelEncoder().fit(target)
            self.classes_ = self.encoder_.classes_
            target = self.encoder_.transform(target)
        counts = np.unique(target, return_counts=True)[1] if self.classes else None
        stratify = target if counts is not None and counts.min() >= 2 else None
        x_fit, x_held, y_fit, y_held = train_test_split(
            X, target, test_size=HELD_OUT, random_state=42, stratify=stratify
        )
        weights = (
            compute_sample_weight("balanced", y_fit) if self.balanced and self.classes else None
        )
        model = clone(self.estimator)
        if self.library == "xgboost":
            model.set_params(early_stopping_rounds=self.patience)
            model.fit(
                x_fit, y_fit, sample_weight=weights, eval_set=[(x_held, y_held)], verbose=False
            )
            self.best_iteration_ = int(model.best_iteration) + 1
        elif self.library == "lightgbm":
            lightgbm = importlib.import_module("lightgbm")
            model.fit(
                x_fit,
                y_fit,
                sample_weight=weights,
                eval_set=[(x_held, y_held)],
                callbacks=[lightgbm.early_stopping(self.patience, verbose=False)],
            )
            self.best_iteration_ = int(model.best_iteration_ or model.n_estimators)
        else:
            model.fit(
                x_fit,
                y_fit,
                sample_weight=weights,
                eval_set=(x_held, y_held),
                early_stopping_rounds=self.patience,
                verbose=False,
            )
            self.best_iteration_ = int(model.get_best_iteration()) + 1
        self.model_ = model
        return self

    def predict(self, X):
        predicted = np.asarray(self.model_.predict(X)).ravel()
        return self.encoder_.inverse_transform(predicted.astype(int)) if self.classes else predicted

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def __sklearn_tags__(self):
        """A classifier when it predicts classes: scikit-learn's scorers (roc_auc while tuning) then
        take the positive class's probability instead of the whole matrix."""
        tags = super().__sklearn_tags__()
        if self.classes:
            tags.estimator_type = "classifier"
            tags.classifier_tags = ClassifierTags()
        else:
            tags.estimator_type = "regressor"
            tags.regressor_tags = RegressorTags()
        return tags

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_


def _xgb_settings(context: Context) -> list[Setting]:
    depth = 4 if context.rows < 5_000 else 6
    rate = 0.05 if context.rows > 100_000 else 0.1
    return [
        Setting(
            "n_estimators",
            "Number of trees",
            "int",
            1_000,
            minimum=10,
            maximum=20_000,
            why="1 000, cut short by early stopping: it finds the right number.",
            group="Structure",
            param="n_estimators",
        ),
        Setting(
            "max_depth",
            "Tree depth",
            "int",
            depth,
            minimum=1,
            maximum=20,
            why="4 under 5 000 rows, 6 otherwise.",
            group="Structure",
            param="max_depth",
        ),
        Setting(
            "learning_rate",
            "Learning rate",
            "number",
            rate,
            minimum=1e-3,
            maximum=1.0,
            log=True,
            why="0.05 above 100 000 rows, 0.1 otherwise.",
            group="Structure",
            param="learning_rate",
        ),
        Setting(
            "subsample",
            "Rows per tree",
            "number",
            0.8,
            level="advanced",
            minimum=0.1,
            maximum=1.0,
            group="Sampling",
            param="subsample",
        ),
        Setting(
            "colsample_bytree",
            "Features per tree",
            "number",
            0.8,
            level="advanced",
            minimum=0.1,
            maximum=1.0,
            group="Sampling",
            param="colsample_bytree",
        ),
        Setting(
            "min_child_weight",
            "Min. weight per leaf",
            "number",
            1.0,
            level="advanced",
            minimum=0.0,
            maximum=100.0,
            group="Structure",
            param="min_child_weight",
        ),
        Setting(
            "gamma",
            "Min. gain to split",
            "number",
            0.0,
            level="advanced",
            minimum=0.0,
            maximum=100.0,
            group="Regularization",
            param="gamma",
        ),
        Setting(
            "reg_lambda",
            "L2 penalty",
            "number",
            1.0,
            level="advanced",
            minimum=0.0,
            maximum=100.0,
            group="Regularization",
            param="reg_lambda",
        ),
        Setting(
            "reg_alpha",
            "L1 penalty",
            "number",
            0.0,
            level="advanced",
            minimum=0.0,
            maximum=100.0,
            group="Regularization",
            param="reg_alpha",
        ),
        *([] if _regression(context) else [class_weight_setting(context)]),
        Setting(
            "patience",
            "Early stopping patience",
            "int",
            50,
            level="advanced",
            minimum=5,
            maximum=1_000,
            group="Stopping",
            param="early_stopping_rounds",
        ),
        Setting(
            "booster",
            "Booster",
            "select",
            "gbtree",
            level="expert",
            options=["gbtree", "dart", "gblinear"],
            group="Optimizer",
            param="booster",
        ),
        Setting(
            "tree_method",
            "Tree construction",
            "select",
            "hist",
            level="expert",
            options=["hist", "approx", "exact"],
            group="Optimizer",
            param="tree_method",
        ),
        Setting(
            "max_bin",
            "Max bins",
            "int",
            256,
            level="expert",
            minimum=2,
            maximum=1_024,
            group="Optimizer",
            param="max_bin",
        ),
        _monotonic_setting("monotone_constraints"),
        seed_setting(),
    ]


def _xgb_build(values: dict[str, Any], context: Context):
    xgboost = importlib.import_module("xgboost")
    keys = (
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "gamma",
        "reg_lambda",
        "reg_alpha",
        "booster",
        "tree_method",
        "max_bin",
        "random_state",
    )
    shared = {key: values[key] for key in keys} | {"n_jobs": -1}
    model = (
        xgboost.XGBRegressor(**shared) if _regression(context) else xgboost.XGBClassifier(**shared)
    )
    return EarlyStopped(
        model,
        "xgboost",
        values["patience"],
        values.get("class_weight") == "balanced",
        not _regression(context),
    )


def _lgbm_settings(context: Context) -> list[Setting]:
    return [
        Setting(
            "num_leaves",
            "Number of leaves",
            "int",
            31,
            minimum=2,
            maximum=1_024,
            why="31: the library's balanced default.",
            group="Structure",
            param="num_leaves",
        ),
        Setting(
            "learning_rate",
            "Learning rate",
            "number",
            0.1,
            minimum=1e-3,
            maximum=1.0,
            log=True,
            group="Structure",
            param="learning_rate",
        ),
        Setting(
            "n_estimators",
            "Number of trees",
            "int",
            1_000,
            minimum=10,
            maximum=20_000,
            why="1 000, cut short by early stopping.",
            group="Structure",
            param="n_estimators",
        ),
        Setting(
            "min_child_samples",
            "Min. rows per leaf",
            "int",
            20,
            level="advanced",
            minimum=1,
            maximum=10_000,
            group="Structure",
            param="min_data_in_leaf",
        ),
        Setting(
            "colsample_bytree",
            "Features per tree",
            "number",
            1.0,
            level="advanced",
            minimum=0.1,
            maximum=1.0,
            group="Sampling",
            param="feature_fraction",
        ),
        Setting(
            "reg_lambda",
            "L2 penalty",
            "number",
            0.0,
            level="advanced",
            minimum=0.0,
            maximum=100.0,
            group="Regularization",
            param="lambda_l2",
        ),
        Setting(
            "reg_alpha",
            "L1 penalty",
            "number",
            0.0,
            level="advanced",
            minimum=0.0,
            maximum=100.0,
            group="Regularization",
            param="lambda_l1",
        ),
        *([] if _regression(context) else [class_weight_setting(context)]),
        Setting(
            "patience",
            "Early stopping patience",
            "int",
            50,
            level="advanced",
            minimum=5,
            maximum=1_000,
            group="Stopping",
            param="early_stopping_round",
        ),
        Setting(
            "boosting_type",
            "Boosting",
            "select",
            "gbdt",
            level="expert",
            options=["gbdt", "dart", "goss"],
            group="Optimizer",
            param="boosting",
        ),
        _monotonic_setting("monotone_constraints"),
        seed_setting(),
    ]


def _lgbm_build(values: dict[str, Any], context: Context):
    lightgbm = importlib.import_module("lightgbm")
    keys = (
        "num_leaves",
        "learning_rate",
        "n_estimators",
        "min_child_samples",
        "colsample_bytree",
        "reg_lambda",
        "reg_alpha",
        "boosting_type",
        "random_state",
    )
    shared = {key: values[key] for key in keys} | {"verbose": -1, "n_jobs": -1}
    model = (
        lightgbm.LGBMRegressor(**shared)
        if _regression(context)
        else lightgbm.LGBMClassifier(**shared)
    )
    return EarlyStopped(
        model,
        "lightgbm",
        values["patience"],
        values.get("class_weight") == "balanced",
        not _regression(context),
    )


def _cat_settings(context: Context) -> list[Setting]:
    return [
        Setting(
            "iterations",
            "Iterations",
            "int",
            1_000,
            minimum=10,
            maximum=20_000,
            why="1 000, cut short by early stopping.",
            group="Structure",
            param="iterations",
        ),
        Setting(
            "depth", "Depth", "int", 6, minimum=1, maximum=16, group="Structure", param="depth"
        ),
        Setting(
            "l2_leaf_reg",
            "L2 penalty",
            "number",
            3.0,
            level="advanced",
            minimum=0.0,
            maximum=100.0,
            group="Regularization",
            param="l2_leaf_reg",
        ),
        *([] if _regression(context) else [class_weight_setting(context)]),
        Setting(
            "patience",
            "Early stopping patience",
            "int",
            50,
            level="advanced",
            minimum=5,
            maximum=1_000,
            group="Stopping",
            param="early_stopping_rounds",
        ),
        Setting(
            "border_count",
            "Borders per feature",
            "int",
            254,
            level="expert",
            minimum=2,
            maximum=1_024,
            group="Optimizer",
            param="border_count",
        ),
        seed_setting(),
    ]


def _cat_build(values: dict[str, Any], context: Context):
    catboost = importlib.import_module("catboost")
    shared = dict(
        iterations=values["iterations"],
        depth=values["depth"],
        l2_leaf_reg=values["l2_leaf_reg"],
        border_count=values["border_count"],
        random_seed=values["random_state"],
        verbose=False,
        allow_writing_files=False,
    )
    model = (
        catboost.CatBoostRegressor(**shared)
        if _regression(context)
        else catboost.CatBoostClassifier(**shared)
    )
    return EarlyStopped(
        model,
        "catboost",
        values["patience"],
        values.get("class_weight") == "balanced",
        not _regression(context),
    )


def _boosting_badges(context: Context) -> list[str]:
    return ["Missing ok", "Accurate"] + (["Fast"] if context.rows < 500_000 else [])


FAMILY_TREES, FAMILY_BAGGING, FAMILY_BOOSTING = "Trees", "Bagging ensembles", "Boosting ensembles"
MODELS = [
    ModelSpec(
        "decision_tree",
        "Decision tree",
        FAMILY_TREES,
        "Questions on one feature at a time, drawn as a tree: easy to read.",
        EVERY,
        TREE,
        _tree_settings,
        _tree_build(False),
        badges=lambda c: ["Interpretable", "Missing ok"],
    ),
    ModelSpec(
        "extra_tree",
        "Extra tree",
        FAMILY_TREES,
        "A single tree with random cut points: rarely used alone.",
        EVERY,
        TREE_FILLED,  # unlike the forest, a single extra tree refuses NaN
        _tree_settings,
        _tree_build(True),
        badges=lambda c: ["Fast"],
    ),
    ModelSpec(
        "random_forest",
        "Random forest",
        FAMILY_BAGGING,
        "Hundreds of trees on random rows and features, averaged: robust and accurate.",
        EVERY,
        TREE,
        _forest_settings,
        _forest_build(False),
        badges=lambda c: ["Robust", "Missing ok"],
    ),
    ModelSpec(
        "extra_trees",
        "Extra trees",
        FAMILY_BAGGING,
        "A random forest with random cut points: faster, often as accurate.",
        EVERY,
        TREE,
        _forest_settings,
        _forest_build(True),
        badges=lambda c: ["Fast", "Missing ok"],
    ),
    ModelSpec(
        "bagging",
        "Bagging",
        FAMILY_BAGGING,
        "Many full trees, each on a random draw of rows, averaged.",
        EVERY,
        TREE_FILLED,
        _bagging_settings,
        _bagging_build,
    ),
    ModelSpec(
        "hist_gradient_boosting",
        "Histogram gradient boosting",
        FAMILY_BOOSTING,
        "Many small trees, each correcting the previous ones, on binned features: "
        "fast on large data.",
        EVERY,
        TREE,
        _hist_settings,
        _hist_build,
        badges=_boosting_badges,
    ),
    ModelSpec(
        "gradient_boosting",
        "Gradient boosting",
        FAMILY_BOOSTING,
        "The classic boosting of shallow trees: accurate, slower than the histogram version.",
        EVERY,
        TREE_FILLED,
        _gb_settings,
        _gb_build,
        badges=lambda c: ["Accurate"] + ([f"Slow on {c.rows:,} rows"] if c.rows > 50_000 else []),
    ),
    ModelSpec(
        "adaboost",
        "AdaBoost",
        FAMILY_BOOSTING,
        "Each new tree focuses on the rows the previous ones got wrong.",
        EVERY,
        TREE_FILLED,
        _ada_settings,
        _ada_build,
    ),
    ModelSpec(
        "xgboost",
        "XGBoost",
        FAMILY_BOOSTING,
        "Many small trees correcting each other, with regularization: "
        "the reference on tabular data.",
        EVERY,
        TREE,
        _xgb_settings,
        _xgb_build,
        badges=_boosting_badges,
        package="xgboost",
    ),
    ModelSpec(
        "lightgbm",
        "LightGBM",
        FAMILY_BOOSTING,
        "Boosting that grows trees leaf by leaf: very fast on large data.",
        EVERY,
        TREE,
        _lgbm_settings,
        _lgbm_build,
        badges=_boosting_badges,
        package="lightgbm",
    ),
    ModelSpec(
        "catboost",
        "CatBoost",
        FAMILY_BOOSTING,
        "Boosting with symmetric trees, robust with little tuning.",
        EVERY,
        TREE,
        _cat_settings,
        _cat_build,
        badges=_boosting_badges,
        package="catboost",
    ),
]
