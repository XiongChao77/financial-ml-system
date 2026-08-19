"""State and filesystem services for experiment report analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import os
from pathlib import Path
from threading import RLock
from typing import Any, Sequence

from experiment.report_service import (
    ReportDataset,
    ReportRecord,
    discover_schema,
    load_report_dataset,
    resolve_under_root,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REPORTS_ROOT = PROJECT_ROOT.parent / "quant_output" / "batch_experiments"
MAX_DATASETS = 6


def get_reports_root() -> Path:
    """Return the configured root from which report folders may be selected."""

    return Path(
        os.environ.get("REPORTS_ROOT", DEFAULT_REPORTS_ROOT)
    ).expanduser().resolve()


@dataclass
class DatasetEntry:
    """One loaded report dataset plus its lazily discovered schemas."""

    dataset: ReportDataset
    schemas: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    lock: RLock = field(default_factory=RLock, repr=False)

    def schema(self, period: str) -> list[dict[str, Any]]:
        with self.lock:
            fields = self.schemas.get(period)
            if fields is None:
                fields = discover_schema(self.dataset.records, period)
                self.schemas[period] = fields
            return fields


class DatasetRegistry:
    """Small thread-safe LRU registry shared by experiment and backtest APIs."""

    def __init__(self, max_entries: int = MAX_DATASETS):
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._entries: dict[str, DatasetEntry] = {}
        self._order: list[str] = []
        self._lock = RLock()

    def put(self, dataset_id: str, dataset: ReportDataset) -> DatasetEntry:
        entry = DatasetEntry(dataset=dataset)
        with self._lock:
            self._entries[dataset_id] = entry
            if dataset_id in self._order:
                self._order.remove(dataset_id)
            self._order.append(dataset_id)
            while len(self._order) > self.max_entries:
                expired = self._order.pop(0)
                self._entries.pop(expired, None)
        return entry

    def get(self, dataset_id: str) -> DatasetEntry:
        with self._lock:
            entry = self._entries.get(dataset_id)
            if entry is None:
                raise KeyError(dataset_id)
            self._order.remove(dataset_id)
            self._order.append(dataset_id)
            return entry

    def record(self, dataset_id: str, record_id: str) -> ReportRecord:
        entry = self.get(dataset_id)
        for record in entry.dataset.records:
            if record.record_id == record_id:
                return record
        raise KeyError(f"Unknown record id: {record_id}")

    def clear(self) -> None:
        """Remove disposable datasets, primarily for process lifecycle tests."""

        with self._lock:
            self._entries.clear()
            self._order.clear()


registry = DatasetRegistry()


def browse_report_directory(path: str = "") -> dict[str, Any]:
    """List selectable child folders below the configured report root."""

    root = get_reports_root()
    current = resolve_under_root(root, path)
    if not current.is_dir():
        raise ValueError(f"Not a directory: {path}")

    children = []
    for child in sorted(current.iterdir(), key=lambda item: item.name.casefold()):
        if not child.is_dir():
            continue
        resolved_child = resolve_under_root(root, child)
        children.append(
            {
                "name": child.name,
                "path": relative_to_reports_root(resolved_child),
                "has_report": (resolved_child / "reports.jsonl").is_file(),
            }
        )

    return {
        "root": str(root),
        "current": relative_to_reports_root(current),
        "parent": (
            None
            if current == root
            else relative_to_reports_root(current.parent)
        ),
        "direct_report": (current / "reports.jsonl").is_file(),
        "recursive_report_count": sum(1 for _ in current.rglob("reports.jsonl")),
        "children": children,
    }


def load_reports(paths: Sequence[str], *, deduplicate: bool = True) -> tuple[str, ReportDataset]:
    """Load one or more selected folders and register the resulting dataset."""

    dataset = load_report_dataset(
        paths,
        allowed_root=get_reports_root(),
        deduplicate=deduplicate,
    )
    dataset_id = dataset_identity(dataset)
    registry.put(dataset_id, dataset)
    return dataset_id, dataset


def dataset_identity(dataset: ReportDataset) -> str:
    """Build a stable identifier for one loaded set of report files."""

    material = "\0".join(dataset.report_files)
    material += f"\0{len(dataset.records)}\0{dataset.duplicate_records}"
    return sha256(material.encode("utf-8")).hexdigest()[:16]


def relative_to_reports_root(path: Path) -> str:
    relative = path.resolve().relative_to(get_reports_root())
    return "" if str(relative) == "." else str(relative)
