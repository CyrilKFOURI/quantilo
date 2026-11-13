"""Use a model on new data from the library (037): the same check, predictions and export as the
app's "Predict on new data", on a trained model or on a model exported as a zip.

model.check(new)                     # Ready, Check or Blocked, column by column
model.predict(new, keep=["id"])      # predicted (+ p(class) or low / high), reliability
model.prediction(new)                # the rows, the check, the scores when `new` has the answer
model.export("churn_lightgbm.zip")   # model.joblib + model_card.json + predict.py, no data rows

model = dx.load_model("churn_lightgbm.zip")   # reads the card first; warns when versions differ
"""

import hashlib
import io
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import joblib
import pandas as pd
from pydantic import ValidationError

from quantilo.filters.scope import scoped_source, where_for
from quantilo.ml import browser
from quantilo.ml import predict as use
from quantilo.ml.engine import MlError

if TYPE_CHECKING:
    from quantilo.dataset import Dataset

MAX_PIPELINE_BYTES = 2_000_000_000  # an exported pipeline larger than this is refused


@dataclass
class Prediction:
    """Predictions on new data: the rows, the check made first, and the summary (distribution
    against the validation, scores when the data has the answer)."""

    rows: pd.DataFrame
    check: use.ModelCheck
    summary: use.PredictionReport


class Predicting:
    """What a trained or a reloaded model does with new rows. Subclasses give `card` and
    `pipeline`."""

    @property
    def card(self) -> use.ModelCard:
        raise NotImplementedError

    @property
    def pipeline(self) -> Any:
        raise NotImplementedError

    @staticmethod
    def _scope(data: "Dataset | pd.DataFrame") -> tuple[str, list[Any]]:
        from quantilo.dataset import Dataset, from_pandas

        dataset = data if isinstance(data, Dataset) else from_pandas(data, "predict")
        types = dataset.types
        source = scoped_source(
            str(dataset.path), where_for(types, dataset.conditions), numbered=True
        )
        return source, types

    def check(self, data: "Dataset | pd.DataFrame") -> use.ModelCheck:
        """Can the model predict this data? Missing columns, types, unknown categories and
        values out of the training range, column by column; level ready, check or blocked."""
        source, views = self._scope(data)
        connection = duckdb.connect()
        try:
            return use.check(connection, source, self.card, views)
        finally:
            connection.close()

    def prediction(
        self, data: "Dataset | pd.DataFrame", keep: list[str] | None = None
    ) -> Prediction:
        """Check, then predict every row (in blocks of 100 000): see `predict`. Refused when the
        check is blocked."""
        source, views = self._scope(data)
        card = self.card
        connection = duckdb.connect()
        try:
            checked = use.check(connection, source, card, views)
            if checked.level == "blocked":
                raise MlError(checked.message, "blocked")
            totals = use.Accumulator(card)
            blocks = []
            for block in use.predict_blocks(connection, source, self.pipeline, card, views, keep):
                totals.add(block.drop(columns="__row"))
                blocks.append(block)
        finally:
            connection.close()
        rows = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
        return Prediction(rows.drop(columns="__row", errors="ignore"), checked, totals.summary())

    def predict(
        self, data: "Dataset | pd.DataFrame", keep: list[str] | None = None
    ) -> pd.DataFrame:
        """Predictions for new rows, in their order: the `keep` columns, `predicted` (for a
        classifier one `p(<class>)` per class, the class chosen with the run's threshold; for a
        number `low` / `high`, a 90 % conformal interval), `less_reliable` and `why` (an important
        value the model never saw), and `actual` when the data has the target."""
        return self.prediction(data, keep).rows

    def explain_row(self, data: "Dataset | pd.DataFrame", row: int) -> use.RowExplanation:
        """Why the model predicts what it does for one row (0-based row of the data)."""
        source, views = self._scope(data)
        connection = duckdb.connect()
        try:
            return use.explain_row(connection, source, self.pipeline, self.card, views, row)
        finally:
            connection.close()

    def what_would_change(
        self,
        data: "Dataset | pd.DataFrame",
        row: int,
        editable: list[str] | None = None,
        target_class: str | None = None,
        target_value: float | None = None,
        max_changes: int = 3,
    ) -> Any:
        """125 bis (R-XAI-21): at most `max_changes` changes to one row of new data that give
        `target_class` (or bring a number to `target_value`), judged plausible against the rows
        of that same data."""
        from quantilo.ml.counterfactual import CounterfactualError

        source, views = self._scope(data)
        connection = duckdb.connect()
        try:
            return use.counterfactual_row(
                connection, source, self.pipeline, self.card, views, row, editable,
                target_class, target_value, max_changes,
            )  # fmt: skip
        except CounterfactualError as error:
            raise MlError(str(error), "no_counterfactual") from None
        finally:
            connection.close()

    def export(self, path: str | os.PathLike[str]) -> Path:
        """Write the model as a zip: model.joblib, model_card.json (no data rows) and predict.py,
        which reloads it with `dx.load_model`."""
        target = Path(path)
        buffer = io.BytesIO()
        joblib.dump(self.pipeline, buffer)
        target.write_bytes(use.export_zip(self.card, buffer.getvalue(), target.name))
        return target


class LoadedModel(Predicting):
    """A model reloaded from an exported zip: its card and its pipeline."""

    def __init__(self, card: use.ModelCard, pipeline: Any, path: Path) -> None:
        self._card = card
        self._pipeline = pipeline
        self.path = path

    @property
    def card(self) -> use.ModelCard:
        return self._card

    @property
    def pipeline(self) -> Any:
        return self._pipeline

    def __repr__(self) -> str:
        card = self._card
        where = self.path.name
        return f"<LoadedModel {card.model} -> {card.target.name!r} ({card.task}) from {where}>"


def load_model(path: str | os.PathLike[str]) -> LoadedModel:
    """Reload a model exported by the app or by `model.export`. The card is read and checked
    first (format, the pipeline's SHA-256); a zip without one is refused. Loading the pipeline
    runs a pickle: only load models you trust. A warning names every library whose version
    differs from the training one."""
    import zipfile

    found = Path(path)
    try:
        archive = zipfile.ZipFile(found)
    except (OSError, zipfile.BadZipFile) as error:
        raise MlError(f"{found.name} is not a model zip: {error}", "bad_model") from None
    with archive:
        names = set(archive.namelist())
        if "model_card.json" not in names:
            raise MlError(
                f"{found.name} has no model_card.json: refused. A model file can run code when it "
                "is loaded; only zips exported by Quantilo, with their card, are read.",
                "no_model_card",
            )
        if "model.joblib" not in names:
            raise MlError(f"{found.name} has no model.joblib.", "bad_model")
        try:
            card = use.ModelCard.model_validate_json(archive.read("model_card.json"))
        except ValidationError as error:
            message = error.errors()[0].get("msg", "invalid")
            raise MlError(f"Invalid model card: {message}.", "bad_model_card") from None
        if archive.getinfo("model.joblib").file_size > MAX_PIPELINE_BYTES:
            raise MlError("model.joblib is larger than 2 GB: refused.", "bad_model")
        blob = archive.read("model.joblib")
    if card.pipeline_sha256 != hashlib.sha256(blob).hexdigest():
        raise MlError("model.joblib does not match its card (SHA-256): refused.", "bad_model")
    for sentence in use.version_warnings(card):
        warnings.warn(sentence, UserWarning, stacklevel=2)
    # the card was checked above: from here the pipeline is unpickled (runs code)
    pipeline = browser.single_core(joblib.load(io.BytesIO(blob)))  # R-WEB-02
    return LoadedModel(card, pipeline, found)
