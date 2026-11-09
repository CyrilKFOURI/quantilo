"""The Auto of the machine-learning page (038): the engine's AutoML within a time budget, as a
background job followed step by step, stoppable at any time, the model it keeps saved as a normal
run (with the Auto's journal and leaderboard in the run's JSON)."""

import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

from quantilo.filters.model import Condition
from quantilo.ml import auto as automl
from quantilo.ml import classes as binning
from quantilo.ml import evaluate
from quantilo.ml import models as catalogue
from quantilo.ml import train as trainer
from quantilo.ml import tune as tuner
from quantilo.ml.data import Field

from app.modules.ml.service import KEPT_JOBS, MlError, MlService

AutoRequest = automl.AutoRequest


@dataclass
class AutoJob:
    id: str
    status: Literal["running", "done", "stopped", "failed"]
    target: str
    progress: automl.AutoProgress | None = None
    run_id: str | None = None  # the kept model, saved as a run
    error: str | None = None


class AutoService:
    """Auto jobs on top of the ML service (its rows, validation and saved runs)."""

    def __init__(self, ml: MlService) -> None:
        self.ml = ml
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ml-auto")
        self._jobs: OrderedDict[str, AutoJob] = OrderedDict()
        self._stops: set[str] = set()
        self._lock = threading.Lock()

    def start(
        self,
        dataset_id: str,
        target: str,
        request: AutoRequest,
        conditions: list[Condition],
        classes: binning.Classes | None = None,
        validation: evaluate.Validation | None = None,
    ) -> AutoJob:
        unknown = [k for k in request.models or [] if k not in catalogue.CATALOGUE]
        if unknown:
            raise MlError(f"Unknown model(s): {', '.join(unknown)}.", "unknown_model")
        fields, field_, source = self.ml._request(dataset_id, target, None, conditions, classes)
        job = AutoJob(uuid.uuid4().hex[:12], "running", target)
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > KEPT_JOBS:
                self._jobs.popitem(last=False)
        self._executor.submit(
            self._run,
            job,
            dataset_id,
            fields,
            field_,
            source,
            request,
            conditions,
            classes,
            validation or evaluate.Validation(),
        )
        return job

    def _run(
        self,
        job: AutoJob,
        dataset_id: str,
        fields: list[Field],
        target: Field,
        source: str,
        request: AutoRequest,
        conditions: list[Condition],
        classes: binning.Classes | None,
        validation: evaluate.Validation,
    ) -> None:
        def started(progress: automl.AutoProgress) -> None:
            job.progress = progress

        try:
            outcome = automl.run(
                self.ml,
                fields,
                target,
                source,
                request,
                classes,
                validation,
                should_stop=lambda: job.id in self._stops,
                started=started,
            )
            if outcome.trained is not None:
                job.run_id = self.ml._save(
                    dataset_id, target.name, outcome.trained, conditions, outcome.progress
                )
            if outcome.progress.stopped:
                job.status = "stopped"
            elif outcome.trained is None:
                job.status, job.error = "failed", "No model could be trained on every row."
            else:
                job.status = "done"
        except (
            MlError,
            automl.AutoError,
            trainer.TrainingError,
            tuner.TuneError,
            catalogue.SettingsError,
            ValueError,
        ) as error:
            job.status, job.error = "failed", str(error)
        except Exception as error:
            job.status, job.error = "failed", f"The Auto failed: {error}"
        finally:
            self._stops.discard(job.id)

    def job(self, job_id: str) -> AutoJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise MlError("This Auto is no longer followed: open the runs.", "unknown_job")
        return job

    def stop(self, job_id: str) -> AutoJob:
        """Stop at the next checkpoint; the best model found so far is saved as a run."""
        job = self.job(job_id)
        if job.status == "running":
            self._stops.add(job_id)
        return job
