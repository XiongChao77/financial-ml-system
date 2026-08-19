"""FastAPI routes for label configuration, generation, and visualization."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from UI.backend.modules.labels.service import (
    LabelConfig,
    LabelConfigurationError,
    LabelDataError,
    LabelGenerationError,
    current_labels,
    generate_labels,
    label_schema,
)


router = APIRouter(prefix="/api/labels", tags=["Labels"])


@router.get("/schema")
def schema() -> dict[str, Any]:
    return label_schema()


@router.get("/current")
def current() -> dict[str, Any]:
    try:
        return current_labels()
    except LabelDataError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/latest")
def latest() -> dict[str, Any]:
    """Compatibility alias for the canonical ``/current`` route."""

    return current()


@router.post("/generate")
def generate(config: LabelConfig) -> dict[str, Any]:
    try:
        return generate_labels(config)
    except LabelConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (LabelGenerationError, LabelDataError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


__all__ = ["router"]
