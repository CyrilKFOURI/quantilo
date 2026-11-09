"""The variable page's data: one section filled according to the column's analysis type."""

from dataclasses import dataclass

from quantilo.variable import profiles
from quantilo.variable.numeric import (
    QQ,
    Curve,
    Describe,
    Normality,
    describe,
    ecdf,
    kde,
    normality,
    qq,
)
from quantilo.variable.source import VariableSource

NUMERIC = ("numeric_continuous", "numeric_discrete")
CATEGORICAL = ("categorical_nominal", "categorical_ordinal", "boolean", "numeric_discrete")
VIOLIN_SAMPLE = 5_000


@dataclass
class NumericProfile:
    describe: Describe
    normality: Normality
    kde: Curve | None
    ecdf: Curve | None
    qq: QQ | None
    sample: list[float]  # for the violin plot


@dataclass
class VariableProfile:
    name: str
    type: str
    numeric: NumericProfile | None = None
    categorical: profiles.CategoricalProfile | None = None
    date: profiles.DateProfile | None = None
    text: profiles.TextProfile | None = None
    identifier: profiles.IdentifierProfile | None = None


def numeric_profile(source: VariableSource) -> NumericProfile:
    summary = describe(source)
    return NumericProfile(
        summary,
        normality(source, summary),
        kde(source, summary),
        ecdf(source, summary),
        qq(source, summary),
        [float(v) for v in source.sample(VIOLIN_SAMPLE)],
    )


def analyze(source: VariableSource) -> VariableProfile:
    column = source.column
    profile = VariableProfile(column.name, column.type)
    if column.type in NUMERIC:
        profile.numeric = numeric_profile(source)
    if column.type in CATEGORICAL:
        profile.categorical = profiles.categorical(source)
    if column.type == "datetime":
        profile.date = profiles.dates(source, column.parser.kind == "year")
    if column.type == "text":
        profile.text = profiles.text(source)
    if column.type == "identifier":
        profile.identifier = profiles.identifier(source)
    return profile
