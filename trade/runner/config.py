"""Runner-owned configuration.

Strategy decision parameters deliberately live in ``trade.strategy``.  This
module contains only orchestration, data selection and execution concerns.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Union

from data_process import common


@dataclass(frozen=True)
class ExperimentContext:
    """Immutable source revision captured once by the process orchestrator."""

    git_commit: str

    def __post_init__(self):
        if not self.git_commit:
            raise ValueError("git_commit must not be empty")


@dataclass
class BrokerConfig:
    initial_equity: float = 10_000.0
    commission_pct: float = 0.05
    leverage: float = 10.0
    margin_warn_pct: float = 0.95


@dataclass(frozen=True)
class BacktestEngineConfig:
    runonce: bool = False
    cheat_on_open: bool = False
    # Fill Market orders submitted from next() at that bar's close instead of
    # the following bar's open. This is Backtrader broker cheat-on-close mode.
    cheat_on_close: bool = True
    max_cpus: int = 1


@dataclass(frozen=True)
class ModelDataConfig:
    atr_ref_bars: int = 80
    prep_output_dir: str = common.DATA_OUT_DIR
    train_output_dir: str = common.TRAIN_OUT_DIR
    period: str = "long"  # long | forward | all
    device: str = "auto"  # 'auto'/'cuda'/'cpu'
    use_prediction_cache: bool = False

    def __post_init__(self):
        if self.atr_ref_bars <= 0:
            raise ValueError("atr_ref_bars must be a positive integer")
        if self.period not in {"long", "forward", "all"}:
            raise ValueError(
                "period must be one of: 'long', 'forward', 'all'"
            )

@dataclass(kw_only=True)
class CsvDataConfig(common.MarketDataSourceConfig):
    atr_ref_bars: int
    from_date: Optional[str] = None
    to_date: Optional[str] = None

    def __post_init__(self):
        if self.atr_ref_bars <= 0:
            raise ValueError("atr_ref_bars must be a positive integer")


RunnerDataConfig = Union[ModelDataConfig, CsvDataConfig]


@dataclass(frozen=True)
class RunnerConfig:
    """Complete input for the standalone backtest runner.

    The concrete ``data_config`` type selects the internal data pipeline, while
    ``strategy_config`` selects the strategy/venue adapter.
    """

    strategy_config: Any
    data_config: RunnerDataConfig
    save_dir: str
    experiment_context: ExperimentContext
    broker_config: BrokerConfig = field(default_factory=BrokerConfig)
    engine_config: BacktestEngineConfig = field(default_factory=BacktestEngineConfig)
    # Batch workers leave this disabled so concurrent runs never overwrite the
    # user-facing single-strategy report.
    publish_frontend_report: bool = False
    # Standalone runs keep a self-contained detail artifact. Batch experiments
    # use reports.jsonl as their canonical output and disable this duplicate.
    persist_full_report: bool = True
