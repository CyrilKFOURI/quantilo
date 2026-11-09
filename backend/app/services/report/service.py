"""Generate dataset reports in the background (R-REP-13); the last few stay in memory."""

import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from quantilo.compare.service import CompareService
from quantilo.correlations.service import CorrelationService
from quantilo.filters.model import Condition
from quantilo.profiling.service import ProfilingService
from quantilo.report.generate import ReportServices, generate
from quantilo.variable.service import VariableService

from app.core.errors import AppError
from app.core.jobs import JobRunner
from app.services.columns.service import ColumnService

KEPT_JOBS = 8


class ReportError(AppError):
    def __init__(self, message: str, code: str = "report_error", status: int = 400) -> None:
        super().__init__(message, code=code, status_code=status)


@dataclass
class ReportJob:
    id: str
    status: Literal["running", "done", "failed", "cancelled"]
    progress: float = 0.0  # 0..1
    message: str | None = None  # "Variables 12 / 40"
    error: str | None = None
    file_name: str | None = None


class Cancelled(Exception):
    pass


class ReportService:
    def __init__(
        self,
        columns: ColumnService,
        profiling: ProfilingService,
        variable: VariableService,
        correlations: CorrelationService,
        compare: CompareService,
        version: Callable[[], str],
        runner: JobRunner,
    ) -> None:
        self.columns = columns
        self.profiling = profiling
        self.variable = variable
        self.correlations = correlations
        self.compare = compare
        self.version = version
        self.runner = runner
        self._jobs: OrderedDict[str, ReportJob] = OrderedDict()
        self._files: dict[str, str] = {}
        self._stops: set[str] = set()
        self._lock = threading.Lock()

    def start(self, dataset_id: str, conditions: list[Condition]) -> ReportJob:
        self.columns.ready(dataset_id)  # unknown or unready dataset: an error now, not in the job
        job = ReportJob(uuid.uuid4().hex[:12], "running", message="Overview")
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > KEPT_JOBS:
                old, _ = self._jobs.popitem(last=False)
                self._files.pop(old, None)
        self.runner.submit(lambda: self._run(job, dataset_id, conditions))
        return job

    def job(self, job_id: str) -> ReportJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise ReportError(
                "This report is no longer available: export it again.", "unknown_job", 404
            )
        return job

    def cancel(self, job_id: str) -> ReportJob:
        job = self.job(job_id)
        if job.status == "running":
            self._stops.add(job_id)
        return job

    def file(self, job_id: str) -> tuple[str, str]:
        """(file name, HTML) of a finished report."""
        job = self.job(job_id)
        content = self._files.get(job_id)
        if job.status != "done" or content is None or job.file_name is None:
            raise ReportError("The report is not ready.", "report_not_ready", 409)
        return job.file_name, content

    def _check(self, job: ReportJob) -> None:
        if job.id in self._stops:
            raise Cancelled

    def _run(self, job: ReportJob, dataset_id: str, conditions: list[Condition]) -> None:
        try:
            content, name = self.build(dataset_id, conditions, job)
            self._files[job.id] = content
            job.file_name = name
            job.progress, job.message, job.status = 1.0, None, "done"
        except Cancelled:
            job.status, job.message = "cancelled", None
        except Exception as error:  # reported to the user through the job
            job.status, job.error = "failed", str(error) or type(error).__name__
        finally:
            self._stops.discard(job.id)

    def build(
        self, dataset_id: str, conditions: list[Condition], job: ReportJob | None = None
    ) -> tuple[str, str]:
        """(HTML, file name). Steps: overview, one per column, correlations, pairs."""

        def progress(fraction: float, message: str) -> None:
            if job is not None:
                self._check(job)
                job.progress, job.message = fraction, message

        services = ReportServices(
            self.columns, self.profiling, self.variable, self.correlations, self.compare
        )
        return generate(services, dataset_id, conditions, self.version(), progress)
