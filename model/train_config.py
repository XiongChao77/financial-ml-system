import hashlib
from dataclasses import dataclass, field
from typing import List, Optional

from data_process import feature
# ==============================================================================
# 1. Configuration
# ==============================================================================
@dataclass
class DataConfig:
    label_col: str = "label"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    purge_overlap: bool = True

    def __post_init__(self):
        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")
        if not 0 < self.val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1")
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must leave a non-empty test split")

@dataclass
class BaseModelConfig:
    seq_len: int = 96
    model_type: str = "base"
    model_version: int = 1

class TrainTask:
    DIRECT_3CLASS = "DIRECT_3CLASS"
    DUAL_HEAD_3CLASS = "DUAL_HEAD_3CLASS"
    TRIGGER_DIRECTION = "TRIGGER_DIRECTION"
    LONG_SHORT_OVR = "LONG_SHORT_OVR"

    TRIGGER = "TRIGGER"
    DIRECTION = "DIRECTION"
    LONG_OVR = "LONG_OVR"
    SHORT_OVR = "SHORT_OVR"

    BINARY_TASKS = {TRIGGER, DIRECTION, LONG_OVR, SHORT_OVR}
    COMBO_TASKS = {TRIGGER_DIRECTION, LONG_SHORT_OVR}
    
@dataclass
class LSTMConfig(BaseModelConfig):
    seq_len: int = 216     # 160 best for LSTM
    model_type: str = "lstm"
    model_version: int = 1
    hidden_size: int = 64
    num_layers: int = 2
    bidirectional: bool = True
    lstm_dropout: float = 0.4
    head_dropout: float = 0.2
    p_drop: float = 0.3
    readout: str = ['last' , 'meanmax' , 'attn', 'mix'][3]
    head: str = ['linear' , 'mlp'][0]
    in_locked_p: float = 0.05               # V4 locked dropout on inputs
    out_locked_p: float = 0              # V4 locked dropout on LSTM outputs (before pooling)
    input_norm: bool = True                # V4 LayerNorm on input features
    input_proj_dim: int | None = None      # V4 optional projection before LSTM: a linear layer mapping raw feature dim (e.g. 48) to a new dim D (dim reduction)
    logit_clip: float | None = None        # V4 

@dataclass
class TransformerConfig(BaseModelConfig):
    seq_len: int = 216     # 160 best for LSTM
    model_type: str = "transformer"
    model_version: int = 3
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.3
    attn_dropout: float = 0.1
    drop_path: float = 0
    in_locked_p: float = 0
    use_alibi: bool = True
    pos_encoding: str = "none"
    cls_token: bool = False
    readout: str = "cls" #"cls" | "meanmax" | "attn" | "mix"
    head: str = "linear"
    ffn_type: str = "swiglu"
    use_feature_weighting: bool = False

@dataclass
class ConvLSTMConfig(BaseModelConfig):
    seq_len: int = 160     # 160 best for LSTM
    model_type: str = "conv_lstm"
    model_version: int = 1
    d_model: int = 64
    hidden_size = 64
    conv_layers: int = 5
    conv_kernel: int = 5
    conv_dropout: float = 0.10
    conv_dilations: str = ""
    bidirectional: bool = True
    lstm_dropout: float = 0.2
    input_norm: bool = True
    in_locked_p: float = 0.05
    out_locked_p: float = 0.05
    head_dropout: float = 0.2
    readout: str = "mix"    # 'last'|'meanmax'|'attn'|'mix'
    head: str = "linear"    # 'linear'|'mlp'
    logit_clip: Optional[float] = None
    p_drop: Optional[float] = None
    task_proj_dim: int = 64
    use_feature_weighting: bool = False

@dataclass
class TCNConfig(BaseModelConfig):
    seq_len: int = 216     # 160 best for LSTM
    model_type: str = "tcn"
    model_version: int = 1
    num_channels: list = field(default_factory=lambda: [64, 128, 256])
    kernel_size: int = 3
    dropout: float = 0.2
    readout: str = "mix"
    logit_clip: Optional[float] = None

@dataclass
class MambaConfig(BaseModelConfig):
    seq_len: int = 216     # 160 best for LSTM
    model_type: str = "mamba"
    model_version: int = 1
    d_model: int = 128
    n_layers: int = 4
    d_state: int = 16
    expand: int = 2
    dropout: float = 0.1
    readout: str = "mix"  # 'last' | 'meanmax' | 'mix'
    logit_clip: Optional[float] = None

@dataclass
class XGBoostConfig(BaseModelConfig):
    seq_len: int = 96
    model_type: str = "xgboost"
    model_version: int = 2

    xgb_depth: int = 4
    xgb_estimators: int = 300
    learning_rate: float = 0.03

    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0

    tree_method: str = "hist"           # "hist"/"gpu_hist"  can be "gpu_hist" with a GPU, depending on the xgboost version
    eval_metric: str = "logloss"
    use_scaler: bool = False            # tree models need no standardization

@dataclass
class SVCConfig(BaseModelConfig):
    seq_len: int = 96
    model_type: str = "svc"
    model_version: int = 1

    C: float = 1.0
    kernel: str = "rbf"
    gamma: str = "scale"
    use_scaler: bool = True

@dataclass
class LogisticRegressionSklearnConfig(BaseModelConfig):
    seq_len: int = 96
    model_type: str = "logistic_regression_sklearn"
    model_version: int = 1

    C: float = 1.0
    penalty: str = "l2"
    solver: str = "lbfgs"
    max_iter: int = 1000
    use_scaler: bool = True

@dataclass
class CNNConfig(BaseModelConfig):
    seq_len: int = 216     # 160 best for LSTM
    model_type: str = "cnn"
    model_version: int = 1
    p_drop: float = 0.3
    tau: float = 16.0
    use_tpool: bool = False

@dataclass
class TrainConfig:
    model_cfg: BaseModelConfig = field(
        default_factory=lambda: ConvLSTMConfig(model_version=3)
    )
    data_cfg: DataConfig = field(default_factory=DataConfig)
    feature_conf_list: List[str] = field(default_factory=lambda: feature_conf_list)
    epochs: int = 100
    batch_size: int = 256#256
    lr: float = 3e-4
    gate_lr: float = 3e-4
    weight_decay: float = 5e-4
    patience: int = 8
    seed: int = 42
    stride: int = 8
    use_cache: bool = False
    lambda_dir: float = 0.1  # Importance of long/short direction
    lambda_main:float = 0.7
    lambda_cost: float = 1  # Flip / missed trend / noisy trades
    flip_penalty: float = 1
    miss_penalty: float = 1
    false_trade_penalty: float = 1
    minority_sampling_ratio: Optional[float] = None
    mag_warmup_epochs:int = 8
    temperature:float = 2.0
    selection_metric: str = "macro_f1"  # "macro_f1" or "loss"
    label_smoothing :float = 0.02
    train_compatibility:str = ''
    train_task: str = TrainTask.DIRECT_3CLASS

    def __post_init__(self):
        if self.selection_metric not in {"macro_f1", "loss"}:
            raise ValueError("selection_metric must be 'macro_f1' or 'loss'")
        if self.epochs <= 0 or self.batch_size <= 0 or self.patience <= 0:
            raise ValueError("epochs, batch_size and patience must be positive")
        if (
            self.minority_sampling_ratio is not None
            and not 0 < self.minority_sampling_ratio < 0.5
        ):
            raise ValueError(
                "minority_sampling_ratio must be between 0 and 0.5"
            )
        feature_hash = hashlib.sha256(
            "\0".join(self.feature_conf_list).encode("utf-8")
        ).hexdigest()[:12]
        self.train_compatibility = (
            f"{self.model_cfg.seq_len}_{self.stride}_{feature_hash}"
        )

@dataclass
class LogisticConfig(BaseModelConfig):
    seq_len: int = 96     # 160 best for LSTM
    model_type: str = "logistic_regression"
    model_version: int = 1


COMBO_SUB_TASKS = {
    TrainTask.TRIGGER_DIRECTION: (TrainTask.TRIGGER, TrainTask.DIRECTION),
    TrainTask.LONG_SHORT_OVR: (TrainTask.LONG_OVR, TrainTask.SHORT_OVR),
}

# feature_direction_map: feature name -> ic_direction (1 positive / -1 negative)
# Features with direction=-1 are multiplied by -1 before training, to correlate positively with the return
feature_direction_map = {
    "PVT": -1,
    "BOLL_PB_25": -1,
    "RSI_14": -1,
    "close": -1,
    "KELTNER_MIDDLE_14": -1,
    "low": -1,
    "MOM_20_RV20": -1,
    "DONCHIAN_POS_20": -1,
    "high": -1,
    "OBV": -1,
    "MOM_20_SKIP1": -1,
    "KELTNER_UPPER_14": -1,
    "open": -1,
    "DONCHIAN_DIST_L_20": -1,
    "DONCHIAN_DIST_U_20": 1,
    "MACD_12_26_DIF_PCT": -1,
    "MOM_20": -1,
    "KELTNER_LOWER_14": -1,
    "DONCHIAN_MIDDLE_20": -1,
    "DONCHIAN_LOWER_20": -1,
    "DONCHIAN_UPPER_20": -1,
    "VWAP_7": -1,
    "MFI_25": -1,
    "MOM_10": -1,
    "BOLL_MIDDLE_25": -1,
    "dist_to_high_100": 1,
    "KDJ_K": -1,
    "MA_BAR_S_L": -1,
    "KDJ_D": -1,
    "BOLL_LOWER_25": -1,
    "KDJ_J": -1,
    "vpin_49": 1,
    "MA_BAR_M_L": -1,
    "MACD_12_26_HIST_PCT": -1,
    "BOLL_BW_25": 1,
    "poc_bias_49": -1,
    "BOLL_UPPER_25": -1,
    "MACD_12_26_DIF": -1,
    "vpin_14": 1,
    "MOM_60": -1,
    "D_MA_DAY_S_L": -1,
    "id_factor_20": -1,
    "MACD_12_26_DEA": -1,
    "MACD_12_26_SIG_DIST": -1,
    "VWAP_Bias_7": -1,
    "close_pos": -1,
    "vol_parkinson_100": 1,
    "vol_gk_100": 1,
    "id_factor_100": -1,
    "dist_to_high_20": 1,
    "skew_20": 1,
    "D_MA_BAR_S_L": -1,
    "er_126": 1,
    "imbalance_14": -1,
    "CMF_25": 1,
    "VWAP_BIAS": -1,
    "MACD_12_26_HIST_ACCEL": -1,
    "vol_gk_14": 1,
    "hurst_126": 1,
    "atr_14": 1,
    "vol_parkinson_14": 1,
    "body": 1,
    "body_pct": 1,
    "doji_score": 1,
    "body_mom": 1,
    "imbalance_49": 1,
    "max_range": 1,
    "kurt_100": 1,
    "MACD_12_26_HIST": 1,
    "skew_100": 1,
    "Vol_Trend": 1,
    "poc_bias_14": 1,
    "upper_wick_pct": 1,
    "VOL_ratio_14": 1,
    "kurt_20": 1,
    "ATS": 1,
    "DONCHIAN_BW_20": 1,
    "QAV_SLOPE_49": 1,
    "lower_wick_pct": 1,
    "hurst_14": 1,
    "QAV_SURGE_49": 1,
    "lower_wick": 1,
    "vol_regime_14": 1,
    "wick_bias": 1,
    "trade_density_14": 1,
    "er_14": 1,
    "quote_asset_volume": 1,
    "MA_DAY_S_L": 1,
    "number_of_trades": 1,
    "taker_buy_quote_volume": 1,
    "MA_WEEK_M_L": -1,
    "VOL_MA_14": 1,
    "trade_density_49": 1,
    "upper_wick": 1,
    "volume": 1,
    "taker_buy_base_volume": 1,
}

feature_conf_list = [

    # =========================
    # Raw Market State
    # =========================
    "open",
    "high",
    "low",
    "close",
    "volume",
    "number_of_trades",
    "quote_asset_volume",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    # =========================
    # 1) Trend / directional persistence: whether price has continuation
    # =========================
    "MA_WEEK_M_L",        # Long-term regime direction (core)
    "PVT",                # Price-volume enhanced momentum
    "dist_to_high_100",   # Breakout-style trend structure
    "id_factor_100",
    "id_factor_20",
    "MFI_999",            # Extreme money flow
    "MFI_99",
    # =========================
    # 2) Volatility regime: amplitude and risk environment
    # =========================
    "vol_gk_100",          # Long-term volatility
    "vol_gk_14",           # Short-term volatility
    "skew_100",
    "kurt_100",            # Tail structure (extreme risk)
    # "BOLL_BW_25",         # Decide after uplift testing
    "RSI_14",              # Relative Strength Index (momentum/overheat; indirectly reflects volatility)
    # =========================
    # 3) Efficiency / market structure: trending vs ranging
    # =========================
    "er_126",              # Trend efficiency ratio (high-quality structure factor)
    # =========================
    # 4) Participation / liquidity: market activity
    # =========================
    "trade_density_14",    # Continuous participation intensity
    "vol_event_flag_500",  # Extreme volume event (regime trigger)
    # =========================
    # 5) Order flow / imbalance: buy-vs-sell dominance
    # =========================
    "vpin_49",             # Mid-term order-flow imbalance
    "vpin_14",             # Needs uplift testing
    # =========================
    # 6) Spatial / price position: where price sits within ranges/cost
    # =========================
    "poc_bias_600",        # Deviation from high-volume node (strong structural anchor)
    "poc_bias_99",
    "close_pos",           # Relative position within range
    # =========================
    # 7) Candlestick / path microstructure
    # =========================
    "upper_wick_pct",
    "lower_wick_pct",
]

seq_len = 216
SingleModelConfig = TrainConfig(
    model_cfg=ConvLSTMConfig(model_version=1, seq_len=seq_len),
    feature_conf_list=feature.FEATURE_LIST_COMMODITY,
    train_task=TrainTask.DIRECT_3CLASS,
)
