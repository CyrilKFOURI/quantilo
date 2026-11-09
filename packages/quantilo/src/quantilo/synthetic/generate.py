"""From typed training rows to a typed synthetic frame: learn the columns, run a method, decode,
then add the regenerated identifiers and fake values."""

from dataclasses import dataclass, field, replace
from typing import cast

import pandas as pd

from quantilo.synthetic import encode, identifiers
from quantilo.synthetic.methods import Fitted, Options, StopCheck, generate
from quantilo.synthetic.plan import Method, Treatment
from quantilo.typing.types import AnalysisType


@dataclass
class Spec:
    """One column of the synthetic data and how it is produced."""

    name: str
    type: AnalysisType
    treatment: Treatment  # "model", "unique" or "fake" ("leave" columns are not passed)
    order: list[str] | None = None  # ordinal levels
    grouped: bool = False  # R-SYN-08
    minimum: float | None = None  # user bounds (numbers)
    maximum: float | None = None


@dataclass
class Synthesis:
    frame: pd.DataFrame  # typed synthetic rows, columns in the order of the specs
    train: pd.DataFrame  # the training rows as learned (rare levels grouped, R-SYN-08)
    learned: list[encode.Learned]
    fitted: Fitted
    bounds: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)


def prepare(train: pd.DataFrame, specs: list[Spec]) -> tuple[pd.DataFrame, list[encode.Learned]]:
    """Training rows with rare levels grouped (R-SYN-08), and what was learned of each column."""
    frame = train.copy()
    learned = []
    for spec in specs:
        if spec.treatment != "model":
            continue
        if spec.grouped:
            frame[spec.name] = encode.group_rare(encode.column_of(frame, spec.name))
        learned.append(
            encode.learn_column(
                spec.name, encode.column_of(frame, spec.name), spec.type, spec.order
            )
        )
    return frame, learned


def group_like(frame: pd.DataFrame, learned: list[encode.Learned]) -> pd.DataFrame:
    """Other real rows (held out) with the same grouping of rare levels as the training rows,
    so the evaluation compares like with like."""
    out = frame.copy()
    for item in learned:
        if item.grouped and item.name in out:
            known = set(item.levels)
            text = encode.labels(encode.column_of(out, item.name))
            out[item.name] = pd.Series(
                [v if v is None or v in known else encode.OTHER for v in text],
                index=text.index,
                dtype=object,
            )
    return out


def _row_hashes(frame: pd.DataFrame) -> pd.Series:
    """One hash per row, the same for equal values whatever their storage (dates in µs or ns)."""
    columns = {}
    for name in frame:
        values = encode.column_of(frame, name)
        if pd.api.types.is_datetime64_any_dtype(values):
            columns[name] = pd.Series(encode.seconds(values))
        elif pd.api.types.is_numeric_dtype(values):
            columns[name] = pd.Series(values.to_numpy(dtype=float))
        else:
            columns[name] = encode.labels(values).reset_index(drop=True)
    # pandas.util loads lazily, so its stubs hide the function
    return pd.util.hash_pandas_object(pd.DataFrame(columns), index=False)  # pyright: ignore[reportAttributeAccessIssue]


def drop_copies(
    synthetic: pd.DataFrame, train: pd.DataFrame, rows: int, seed: int
) -> tuple[pd.DataFrame, int]:
    """Keep `rows` synthetic rows that do not copy a training row unique in the data (synthpop's
    "replicated uniques" rule); the spare rows generated take their place. A combination that
    many real rows share reveals nobody, so it stays."""
    real = _row_hashes(cast(pd.DataFrame, train[list(synthetic.columns)])).value_counts()
    copied = _row_hashes(synthetic).isin(set(cast(pd.Series, real[real == 1]).index))
    kept = cast(pd.DataFrame, synthetic[~copied.to_numpy()])
    dropped = int(copied.iloc[:rows].sum())
    if len(kept) < rows and len(kept):
        kept = pd.concat([kept, kept.sample(rows - len(kept), replace=True, random_state=seed)])
    return kept.head(rows).reset_index(drop=True), dropped


def synthesize(
    train: pd.DataFrame,
    specs: list[Spec],
    method: Method,
    options: Options,
    existing: dict[str, pd.Series] | None = None,
    should_stop: StopCheck = lambda: False,
) -> Synthesis:
    """`existing` holds every real value of the "unique" and "fake" columns (never reused)."""
    frame, learned = prepare(train, specs)
    bounds = {s.name: (s.minimum, s.maximum) for s in specs if s.treatment == "model"}
    columns: dict[str, pd.Series] = {}
    fitted = Fitted(method)
    entity_ids: pd.Series | None = None
    if method == "timegan":
        from quantilo.synthetic.methods import (
            timegan,  # needs PyTorch: imported on use
        )

        sequence = options.sequence
        assert sequence is not None and sequence.time is not None
        made = timegan.generate(frame, learned, options, should_stop)
        fitted = made.fitted
        options = replace(options, rows=len(made.times))
        for item in learned:
            if item.name == sequence.time:
                columns[item.name] = encode.decode_column(made.times, item, clip=False)
            elif item.name != sequence.entity:
                columns[item.name] = encode.decode_column(
                    made.coded[item.name], item, bounds[item.name]
                )
        if sequence.entity is not None:
            real = (existing or {}).get(sequence.entity, encode.column_of(train, sequence.entity))
            names = identifiers.new_identifiers(real, sequence.sequences, options.seed)
            entity_ids = pd.Series(names.to_numpy()[made.sequence])
    elif learned:
        extra = max(10, options.rows // 20) if options.drop_copies else 0
        wanted = replace(options, rows=options.rows + extra)
        coded, fitted = generate(
            method, encode.encode(frame, learned), learned, wanted, should_stop
        )
        decoded = pd.DataFrame(
            {i.name: encode.decode_column(coded[i.name], i, bounds[i.name]) for i in learned}
        )
        if options.drop_copies:
            decoded, fitted.dropped = drop_copies(decoded, frame, options.rows, options.seed)
            if fitted.dropped:
                fitted.notes.append(f"{fitted.dropped} rows identical to a real row were replaced.")
        else:
            decoded = decoded.head(options.rows)
        columns.update({name: decoded[name].reset_index(drop=True) for name in decoded})
    existing = existing or {}
    for i, spec in enumerate(specs):
        real = existing.get(
            spec.name, encode.column_of(train, spec.name) if spec.name in train else pd.Series()
        )
        if spec.treatment == "unique":
            columns[spec.name] = identifiers.new_identifiers(real, options.rows, options.seed + i)
        elif spec.treatment == "fake":
            columns[spec.name] = identifiers.fake_values(real, options.rows, options.seed + i)
    if entity_ids is not None and options.sequence and options.sequence.entity:
        columns[options.sequence.entity] = entity_ids
    result = pd.DataFrame({s.name: columns[s.name].reset_index(drop=True) for s in specs})
    return Synthesis(result, frame, learned, fitted, bounds)
