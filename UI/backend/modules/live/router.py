"""HTTP ingestion and read APIs for live strategy monitoring."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from UI.backend.modules.live.models import RunnerSnapshot
from UI.backend.modules.live.store import (
    LiveSnapshotConflict,
    live_snapshot_store,
)


router = APIRouter(tags=["live"])


@router.post("/internal/live/snapshots")
async def publish_snapshot(payload: RunnerSnapshot) -> dict[str, Any]:
    try:
        return live_snapshot_store.update(payload)
    except LiveSnapshotConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/api/live/strategies")
async def strategies() -> dict[str, Any]:
    return live_snapshot_store.strategies()


@router.get("/api/live/strategies/{strategy_id}")
async def strategy_detail(strategy_id: str) -> dict[str, Any]:
    strategy = live_snapshot_store.strategy(strategy_id)
    if strategy is None:
        raise HTTPException(status_code=404, detail="Live strategy not found")
    return strategy
