"""Topics of a text column (059): found on demand, kept as a derived column."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, Response
from pydantic import BaseModel, Field
from quantilo.text.service import TextService, topics_summary
from quantilo.text.topics import HIGHEST_K, LOWEST_K, Topics

from app.api.filters import Filters
from app.services.columns.service import ColumnService

router = APIRouter(tags=["text"])


def get_text_service(request: Request) -> TextService:
    return request.app.state.text


def get_column_service(request: Request) -> ColumnService:
    return request.app.state.columns


Text = Annotated[TextService, Depends(get_text_service)]
Columns = Annotated[ColumnService, Depends(get_column_service)]
ColumnName = Annotated[str, Query(min_length=1)]
TopicCount = Annotated[int | None, Query(ge=LOWEST_K, le=HIGHEST_K)]


class KeepTopics(BaseModel):
    column: str = Field(min_length=1)
    k: int | None = Field(default=None, ge=LOWEST_K, le=HIGHEST_K)


@router.get("/datasets/{dataset_id}/text/topics")
def topics(
    dataset_id: str, column: ColumnName, service: Text, filters: Filters, k: TopicCount = None
) -> Topics:
    """TF-IDF then NMF on at most 20,000 texts (R-TOP-01 to 05); `k` from 3 to 12, else by
    R-TOP-03. Slow: the page asks for it only when the user opens Advanced, More."""
    return service.topics(dataset_id, column, k, filters)


@router.post("/datasets/{dataset_id}/text/topics/keep")
def keep_topics(
    dataset_id: str, body: KeepTopics, service: Text, columns: Columns, filters: Filters
) -> Topics:
    """Keep these topics on the column (R-TOP-06): its Topic column becomes a derived column."""
    found = service.topics(dataset_id, body.column, body.k, filters)
    columns.keep_topics(dataset_id, body.column, topics_summary(found))
    found.kept = True
    return found


@router.delete("/datasets/{dataset_id}/text/topics", status_code=204)
def forget_topics(dataset_id: str, column: ColumnName, columns: Columns) -> Response:
    """Forget the kept topics: the Topic column is no longer offered."""
    columns.keep_topics(dataset_id, column, None)
    return Response(status_code=204)
