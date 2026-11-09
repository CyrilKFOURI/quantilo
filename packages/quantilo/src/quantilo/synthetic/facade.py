"""Synthetic data for the library: the app's Synthetic data page as methods.

synth = ds.synthetic()
synth.plan()                          # what each column becomes, and why
made = synth.generate(method="cart")  # synthetic rows + their report
made.report.scores                    # diagnostic, quality, detection, privacy
made.to_dataset()                     # explore the synthetic rows like any dataset
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from quantilo import gpu
from quantilo.synthetic import compare as side_by_side
from quantilo.synthetic.engine import (
    Bound,
    Generation,
    RunRequest,
    SequenceRequest,
    SyntheticEngine,
    SyntheticError,
    export_frame,
)
from quantilo.synthetic.plan import Method, Plan, Treatment
from quantilo.synthetic.privacy import Disclosure, PrivacyError, disclosure
from quantilo.synthetic.report import Report
from quantilo.synthetic.utility import Utility, UtilityError, utility

if TYPE_CHECKING:
    from quantilo.dataset import Dataset

SAMPLE = 20_000  # rows compared column by column, as in the app


def gpu_compute(use_gpu: bool | None = None) -> tuple[str | None, bool]:
    """The card PyTorch can use (or None), and whether the deep methods may use it (`None`:
    whenever there is one)."""
    status = gpu.detect()
    card = status.name if status.state == "ready" and status.torch else None
    return card, card is not None if use_gpu is None else use_gpu


class SyntheticData:
    """Synthetic rows made from a dataset, with the real rows they were learned from and the
    report judging them."""

    def __init__(self, source: "Dataset", made: Generation, request: RunRequest) -> None:
        self._source = source
        self.generation = made
        self.request = request

    def __repr__(self) -> str:
        scores = self.report.scores
        quality = f"{scores.quality:.0%}" if scores.quality is not None else "n/a"
        return f"<SyntheticData {self.generation.method}: {len(self.data)} rows, quality {quality}>"

    @property
    def data(self) -> pd.DataFrame:
        """The synthetic rows, typed as a dataset stores them."""
        return export_frame(self.generation.data, self.generation.columns)

    @property
    def report(self) -> Report:
        return self.generation.report

    def utility(self, target: str) -> Utility:
        """Train on synthetic, test on real (TSTR) against train on real (TRTR)."""
        made = self.generation
        try:
            found = utility(
                made.train, made.data, made.holdout, made.report.learned, target, self.request.seed
            )
        except UtilityError as error:
            raise SyntheticError(str(error), "bad_target") from None
        made.report.utility = found
        made.report.scores.utility = found.ratio
        return found

    def disclosure(self, known: list[str], sensitive: str) -> Disclosure:
        """How well knowing `known` columns of a real person lets one guess `sensitive` (CAP)."""
        made = self.generation
        try:
            found = disclosure(
                made.train, made.data, made.report.learned, known, sensitive, self.request.seed
            )
        except PrivacyError as error:
            raise SyntheticError(str(error), "bad_disclosure") from None
        made.report.disclosure = found
        return found

    def _learned(self, name: str) -> Any:
        item = next((i for i in self.report.learned if i.name == name), None)
        if item is None:
            raise SyntheticError(f"{name!r} is not a modelled column.", "unknown_column", 404)
        return item

    def column(
        self, name: str, bins: int | None = None, target: str | None = None
    ) -> side_by_side.ColumnDetail:
        """One column, real against synthetic: distributions, statistics and score."""
        item = self._learned(name)
        target_item = self._learned(target) if target else None
        made = self.generation
        score = next((s.score for s in made.report.quality.shape_scores if s.column == name), None)
        real = made.train.sample(min(len(made.train), SAMPLE), random_state=self.request.seed)
        fake = made.data.sample(min(len(made.data), SAMPLE), random_state=self.request.seed)
        return side_by_side.column_detail(real, fake, item, score, bins, target_item)

    def pair(self, a: str, b: str) -> side_by_side.PairDetail:
        """Two columns together, real against synthetic."""
        if a == b:
            raise SyntheticError("Pick two different columns.", "same_column")
        made = self.generation
        first, second = self._learned(a), self._learned(b)
        return side_by_side.pair_detail(made.train, made.data, first, second, self.request.seed)

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write the synthetic rows to CSV or Parquet (by the extension)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() == ".parquet":
            self.data.to_parquet(target, index=False)
        elif target.suffix.lower() == ".csv":
            self.data.to_csv(target, index=False)
        else:
            raise SyntheticError("Save as .csv or .parquet.", "bad_format")
        return target

    def to_dataset(self, name: str | None = None) -> "Dataset":
        """The synthetic rows as a dataset, with the source's column types."""
        from quantilo.dataset import from_pandas

        made = from_pandas(self.data, name or f"{self._source.name} (synthetic)")
        types = {column.name: column for column in self._source.types}
        for column in made.types:
            original = types.get(column.name)
            if original and original.type != column.type:
                made.retype(column.name, original.type)
            if original and original.order and original.order != column.order:
                made.set_order(column.name, original.order)
        return made


class Synthesizer:
    """Synthetic copies of a dataset: no real row, the same shapes and links."""

    def __init__(self, source: "Dataset", engine: SyntheticEngine, dataset_id: str) -> None:
        self._source = source
        self._engine = engine
        self._id = dataset_id

    def plan(self, sequence: SequenceRequest | None = None) -> Plan:
        """Per column: model it, regenerate it, fake it or leave it out, and why; the methods
        and which ones can run; the default number of rows; TimeGAN's histories."""
        return self._engine.plan(self._id, sequence)

    def generate(
        self,
        method: Method = "cart",
        rows: int | None = None,
        *,
        seed: int = 42,
        smooth: bool = True,
        drop_copies: bool = True,
        treatments: dict[str, Treatment] | None = None,
        bounds: dict[str, tuple[float | None, float | None]] | None = None,
        epochs: int | None = None,
        device: Literal["auto", "cpu", "gpu"] = "auto",
        sequence: SequenceRequest | None = None,
    ) -> SyntheticData:
        """Generate synthetic rows (CART, Gaussian copula, Bayesian network, independent
        columns; CTGAN, TVAE, TimeGAN with the `deep` extra) and evaluate them."""
        request = RunRequest(
            method=method,
            rows=rows,
            seed=seed,
            smooth=smooth,
            drop_copies=drop_copies,
            treatments=treatments or {},
            bounds={k: Bound(minimum=lo, maximum=hi) for k, (lo, hi) in (bounds or {}).items()},
            epochs=epochs,
            device=device,
            sequence=sequence,
        )
        prepared = self._engine._prepare(self._id, request)
        made = self._engine._generate(self._id, request, prepared)
        return SyntheticData(self._source, made, request)
