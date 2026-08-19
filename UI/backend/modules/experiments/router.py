"""HTTP routes for browsing, filtering, and comparing experiment reports."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from experiment.report_service import (
    DEFAULT_COLUMNS,
    PERIODS,
    SUPPORTED_AGGREGATIONS,
    SUPPORTED_OPERATORS,
    aggregate_records,
    equity_series,
    query_records,
    record_detail,
)
from UI.backend.modules.experiments import service


router = APIRouter(prefix="/api/experiments", tags=["experiments"])


class FilterInput(BaseModel):
    field: str
    operator: str
    value: Any = None


class LoadInput(BaseModel):
    paths: list[str] = Field(min_length=1)
    deduplicate: bool = True


class QueryInput(BaseModel):
    dataset_id: str
    period: str = "long"
    filters: list[FilterInput] = Field(default_factory=list)
    columns: list[str] | None = None
    sort_by: str | None = "performance.cagr"
    descending: bool = True
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=500)


class AggregateInput(BaseModel):
    dataset_id: str
    period: str = "long"
    filters: list[FilterInput] = Field(default_factory=list)
    group_by: list[str] = Field(min_length=1, max_length=2)
    metric: str
    aggregations: list[str] = Field(
        default_factory=lambda: ["count", "median", "mean"]
    )


class EquityInput(BaseModel):
    dataset_id: str
    period: str = "long"
    record_ids: list[str] = Field(min_length=1, max_length=20)


@router.get("/health")
def health() -> dict[str, Any]:
    root = service.get_reports_root()
    return {
        "status": "ok",
        "reports_root": str(root),
        "root_exists": root.is_dir(),
    }


@router.get("/browse")
def browse(path: str = Query(default="")) -> dict[str, Any]:
    try:
        return service.browse_report_directory(path)
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/load")
def load_reports(payload: LoadInput) -> dict[str, Any]:
    try:
        dataset_id, dataset = service.load_reports(
            payload.paths,
            deduplicate=payload.deduplicate,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "dataset_id": dataset_id,
        "record_count": len(dataset.records),
        "report_files": dataset.report_files,
        "malformed_lines": dataset.malformed_lines,
        "duplicate_records": dataset.duplicate_records,
        "periods": list(PERIODS),
        "default_columns": list(DEFAULT_COLUMNS),
    }


@router.get("/schema")
def schema(dataset_id: str, period: str = "long") -> dict[str, Any]:
    entry = _entry(dataset_id)
    try:
        fields = entry.schema(period)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "period": period,
        "fields": fields,
        "operators": list(SUPPORTED_OPERATORS),
        "aggregations": list(SUPPORTED_AGGREGATIONS),
    }


@router.post("/query")
def query(payload: QueryInput) -> dict[str, Any]:
    entry = _entry(payload.dataset_id)
    try:
        return query_records(
            entry.dataset.records,
            payload.period,
            filters=_filters(payload.filters),
            columns=payload.columns,
            sort_by=payload.sort_by,
            descending=payload.descending,
            page=payload.page,
            page_size=payload.page_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/aggregate")
def aggregate(payload: AggregateInput) -> dict[str, Any]:
    entry = _entry(payload.dataset_id)
    try:
        return aggregate_records(
            entry.dataset.records,
            payload.period,
            filters=_filters(payload.filters),
            group_by=payload.group_by,
            metric=payload.metric,
            aggregations=payload.aggregations,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/equity")
def equity(payload: EquityInput) -> dict[str, Any]:
    entry = _entry(payload.dataset_id)
    try:
        series = equity_series(
            entry.dataset.records,
            payload.record_ids,
            payload.period,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"period": payload.period, "series": series}


@router.get("/record/{dataset_id}/{record_id}")
def detail(dataset_id: str, record_id: str) -> dict[str, Any]:
    entry = _entry(dataset_id)
    try:
        return record_detail(entry.dataset.records, record_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _entry(dataset_id: str) -> service.DatasetEntry:
    try:
        return service.registry.get(dataset_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Unknown or expired dataset_id; load the report paths again",
        ) from exc


def _filters(filters: list[FilterInput]) -> list[dict[str, Any]]:
    return [item.model_dump() for item in filters]
