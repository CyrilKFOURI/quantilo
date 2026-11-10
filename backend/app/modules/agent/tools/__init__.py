"""The agent's own tools: the app's V1 analyses (spec 026, § 5.2), its memory and method."""

from dataclasses import dataclass
from typing import Any

from quantilo.compare.service import CompareService
from quantilo.correlations.service import CorrelationService
from quantilo.profiling.service import ProfilingService
from quantilo.regression.service import RegressionService
from quantilo.variable.service import VariableService

from app.modules.agent.engine.tools import Tool, ToolRegistry
from app.modules.agent.repository import NoteRepository
from app.services.columns.service import ColumnService
from app.services.report.service import ReportService
from app.services.versions.service import VersionService


@dataclass(frozen=True)
class AppServices:
    """The app's services the tools call, injected at setup (no import of other modules)."""

    columns: ColumnService
    profiling: ProfilingService
    variable: VariableService
    compare: CompareService
    correlations: CorrelationService
    reports: ReportService
    versions: VersionService | None = None  # what changed between two versions (031)
    regression: RegressionService | None = None  # the Regression tab (068)


def default_registry(
    services: AppServices, notes: NoteRepository, extra: list[Tool[Any]] | None = None
) -> ToolRegistry:
    """The agent's own tools, then the tools the modules offer."""
    from app.modules.agent.tools.changes import WhatChanged
    from app.modules.agent.tools.dataset import (
        CountRows,
        GetDatasetOverview,
        ListColumns,
        ProfileColumn,
    )
    from app.modules.agent.tools.memory import GetPlaybook, Remember
    from app.modules.agent.tools.regression import FitRegression
    from app.modules.agent.tools.relations import (
        CompareColumns,
        GetCorrelations,
        GetStructure,
        RankAssociationsWith,
    )
    from app.modules.agent.tools.report import ExportReport

    return ToolRegistry(
        [
            GetDatasetOverview(services),
            ListColumns(services),
            ProfileColumn(services),
            CountRows(services),
            CompareColumns(services),
            GetCorrelations(services),
            RankAssociationsWith(services),
            GetStructure(services),
            FitRegression(services),
            ExportReport(services),
            WhatChanged(services),
            GetPlaybook(),
            Remember(notes),
            *(extra or []),
        ]
    )
