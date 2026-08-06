"""Runner-owned configuration.

Strategy decision parameters deliberately live in ``trade.strategy``.  This
module contains only orchestration, data selection and execution concerns.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Union

from data_process import common


@dataclass
class BrokerConfig:
    initial_equity: float = 10_000.0
    commission_pct: float = 0.05
    leverage: float = 10.0


@dataclass(frozen=True)
class BacktestEngineConfig:
    runonce: bool = False
    cheat_on_open: bool = False
    max_cpus: int = 1


@dataclass(frozen=True)
class ModelDataConfig:
    atr_ref_bars: int
    prep_output_dir: str = common.DATA_OUT_DIR
    train_output_dir: str = common.TRAIN_OUT_DIR
    period: str = "long"  # short | forward | long
    recent_months: int = 2
    device: str = "auto"
    train_config: Any = None

    def __post_init__(self):
        if self.atr_ref_bars <= 0:
            raise ValueError("atr_ref_bars must be a positive integer")

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
class ReportConfig:
    save_path: Optional[str] = None


@dataclass(frozen=True)
class RunnerConfig:
    """Complete input for the standalone backtest runner.

    The concrete ``data_config`` type selects the internal data pipeline, while
    ``strategy_config`` selects the strategy/venue adapter.
    """

    strategy_config: Any
    data_config: RunnerDataConfig
    broker_config: BrokerConfig = field(default_factory=BrokerConfig)
    engine_config: BacktestEngineConfig = field(default_factory=BacktestEngineConfig)
    report_config: ReportConfig = field(default_factory=ReportConfig)
