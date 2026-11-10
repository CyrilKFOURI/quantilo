"""What a model of the catalogue declares (019 spec): its tasks, profile, settings with their level,
compatibility rules, badges and limits. The interface is built from these declarations."""

import importlib.util
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from quantilo.ml import browser
from quantilo.ml.preparation import ModelProfile
from quantilo.ml.target import Task

Level = Literal["essential", "advanced", "expert"]
Kind = Literal["select", "number", "int", "bool", "text"]
SEED = 42


@dataclass
class Setting:
    key: str
    label: str
    kind: Kind
    default: Any
    level: Level = "essential"
    options: list[str] = field(default_factory=list)
    minimum: float | None = None
    maximum: float | None = None
    log: bool = False  # a slider on a log scale
    why: str = ""  # why the default
    shown_when: dict[str, list[Any]] | None = None  # hidden unless another setting has one of these
    group: str = "Main"  # Structure, Regularization, Sampling, Class balance, Optimizer, Stopping…
    param: str = ""  # the library's name for it
    option_labels: dict[str, str] = field(default_factory=dict)


@dataclass
class Rule:
    """When the settings in `when` have one of the listed values, `setting` must be one of
    `allowed` (the other options are disabled with `reason`)."""

    when: dict[str, list[Any]]
    setting: str
    allowed: list[Any]
    reason: str


@dataclass
class Context:
    """What defaults, badges and limits may depend on."""

    rows: int  # training rows
    task: Task
    minority_share: float | None = None  # classification: share of the rarest class
    features: int = 10  # usable columns
    categorical_share: float = 0.0  # share of category columns among them
    missing_share: float = 0.0  # share of empty cells among them


Build = Callable[[dict[str, Any], Context], Any]


@dataclass
class ModelSpec:
    key: str
    label: str
    family: str
    description: str
    tasks: tuple[Task, ...]
    profile: ModelProfile
    settings: Callable[[Context], list[Setting]]
    build: Build
    rules: list[Rule] = field(default_factory=list)
    badges: Callable[[Context], list[str]] = lambda context: []
    limit: Callable[[Context], str | None] = lambda context: None  # why it cannot run here
    package: str | None = None  # optional package it needs
    desktop_only: bool = False  # R-WEB-04: never offered in the browser version (102)
    probabilities: bool | Callable[[dict[str, Any]], bool] = True  # False: calibrated
    linear: bool = False  # coefficients make sense
    profile_for: Callable[[dict[str, Any]], ModelProfile] | None = None

    def has_probabilities(self, values: dict[str, Any]) -> bool:
        return self.probabilities(values) if callable(self.probabilities) else self.probabilities

    def model_profile(self, values: dict[str, Any]) -> ModelProfile:
        return self.profile_for(values) if self.profile_for else self.profile

    @property
    def installed(self) -> bool:
        return self.package is None or importlib.util.find_spec(self.package) is not None

    @property
    def elsewhere(self) -> bool:
        """Left to the desktop app here: the engine runs in the browser (R-WEB-04)."""
        return self.desktop_only and browser.active()

    @property
    def available(self) -> bool:
        """It can be trained in this build: its package is there, and it is not left to the
        desktop app (R-WEB-04)."""
        return self.installed and not self.elsewhere

    def missing(self) -> str:
        """Why it cannot be trained in this build (when it is not available)."""
        if self.elsewhere:
            return browser.desktop_only(self.label)
        return f"{self.label} is not installed in this build."


class SettingsError(ValueError):
    pass


def _matches(values: dict[str, Any], when: dict[str, list[Any]]) -> bool:
    return all(values.get(key) in allowed for key, allowed in when.items())


def visible(setting: Setting, values: dict[str, Any]) -> bool:
    return setting.shown_when is None or _matches(values, setting.shown_when)


def _coerce(setting: Setting, value: Any) -> Any:
    label = setting.label
    if setting.kind == "select":
        if value not in setting.options:
            raise SettingsError(f"{label}: choose one of {', '.join(map(str, setting.options))}.")
        return value
    if setting.kind == "bool":
        if not isinstance(value, bool):
            raise SettingsError(f"{label} must be yes or no.")
        return value
    if setting.kind == "text":
        if not isinstance(value, str) or len(value) > 100:
            raise SettingsError(f"{label} must be a short text.")
        return value.strip()
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise SettingsError(f"{label} must be a number.") from None
    if not math.isfinite(number):
        raise SettingsError(f"{label} must be a finite number.")
    if setting.kind == "int":
        if number != int(number):
            raise SettingsError(f"{label} must be a whole number.")
        number = int(number)
    if (setting.minimum is not None and number < setting.minimum) or (
        setting.maximum is not None and number > setting.maximum
    ):
        raise SettingsError(f"{label} must be between {setting.minimum:g} and {setting.maximum:g}.")
    return number


def resolve(spec: ModelSpec, context: Context, given: dict[str, Any] | None) -> dict[str, Any]:
    """Settings with their defaults, the given values validated, the rules enforced: a default
    that breaks a rule takes the first allowed value; a value the user gave that breaks one is
    refused."""
    given = given or {}
    settings = spec.settings(context)
    known = {s.key for s in settings}
    unknown = set(given) - known
    if unknown:
        raise SettingsError(f"Unknown setting(s): {', '.join(sorted(unknown))}.")
    values = {s.key: _coerce(s, given.get(s.key, s.default)) for s in settings}
    for rule in spec.rules:
        if not _matches(values, rule.when) or values.get(rule.setting) in rule.allowed:
            continue
        if rule.setting in given:
            raise SettingsError(rule.reason)
        values[rule.setting] = rule.allowed[0]
    return values


def layers(text: str) -> tuple[int, ...]:
    """ "64, 32" → (64, 32)."""
    try:
        sizes = tuple(int(part) for part in text.replace(";", ",").split(",") if part.strip())
    except ValueError:
        raise SettingsError(
            "Hidden layers: whole numbers separated by commas, e.g. 64, 32."
        ) from None
    if not sizes or len(sizes) > 5 or any(s < 1 or s > 1024 for s in sizes):
        raise SettingsError("Hidden layers: 1 to 5 layers of 1 to 1 024 neurons.")
    return sizes


def classifier(context: Context) -> bool:
    return context.task != "regression"


def class_weight_setting(context: Context) -> Setting:
    imbalanced = context.minority_share is not None and context.minority_share < 0.2
    return Setting(
        "class_weight",
        "Balance classes",
        "select",
        "balanced" if imbalanced else "none",
        options=["none", "balanced"],
        why=(
            f"The rarest class is {context.minority_share:.0%} of rows: weights balance it."
            if imbalanced and context.minority_share is not None
            else "Classes are balanced enough."
        ),
        group="Class balance",
        param="class_weight",
    )


def weight(values: dict[str, Any]) -> str | None:
    return "balanced" if values.get("class_weight") == "balanced" else None


def seed_setting() -> Setting:
    return Setting(
        "random_state",
        "Seed",
        "int",
        SEED,
        level="expert",
        minimum=0,
        maximum=2**31 - 1,
        why="Fixed so that training again gives the same model.",
        group="Optimizer",
        param="random_state",
    )
