from dataclasses import dataclass, field, asdict
from typing import Optional, Union, List, Dict
from enum import IntEnum,Enum
# ==============================================================================
# 1. 配置定义 (Configuration)
# ==============================================================================
@dataclass
class DataConfig:
    label_col: str = "label"
    train_ratio: float = 0.7
    val_ratio: float = 0.15

@dataclass
class BaseModelConfig:
    seq_len: int = 96
    model_type: str = "base"
    model_version: int = 1

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
    seq_len: int = 216     # 160 best for LSTM
    model_type: str = "xgboost"
    model_version: int = 1
    xgb_depth: int = 6
    xgb_estimators: int = 100
    learning_rate: float = 3e-4

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
    model_cfg: BaseModelConfig = field(default_factory=ConvLSTMConfig)
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
    lambda_trig: float = 0.5
    lambda_dir: float = 0.1  # Importance of long/short direction
    lambda_main:float = 0.7
    lambda_cost: float = 0.4  # Flip / missed trend / noisy trades
    lambda_gate: float = 1e-3
    mag_alpha: float = 0
    mag_limit: float = 4.0
    bias_lambda: float = 0.5
    flip_penalty: float = 1.2
    miss_penalty: float = 0.6
    false_trade: float = 1
    mag_warmup_epochs:int = 8
    temperature:float = 2.0
    best_f1 : bool = True
    label_smoothing :float = 0.02
    loss_fun_version : int = 4
    train_compatibility:str = ''

    def __post_init__(self):
        self.train_compatibility =f"{str(self.model_cfg.seq_len)}_{str(self.stride)}_{str(hash(tuple(self.feature_conf_list)))}"

@dataclass
class LogisticConfig(BaseModelConfig):
    seq_len: int = 96     # 160 best for LSTM
    model_type: str = "logistic_regression"
    model_version: int = 1

class TrainTask:
    SINGLE_MODEL_3CLASS = "SINGLE_MODEL_3CLASS"
    TRIGGER_DIR = "TRIGGER_DIR"
    LONG_SHORT_OVR = "LONG_SHORT_OVR"

    SINGLE_MODEL_TRIGGER = "SINGLE_MODEL_TRIGGER"
    SINGLE_MODEL_DIR = "SINGLE_MODEL_DIR"

    SINGLE_MODEL_LONG_OVR = "SINGLE_MODEL_LONG_OVR"
    SINGLE_MODEL_SHORT_OVR = "SINGLE_MODEL_SHORT_OVR"

# feature_direction_map: 特征名 -> ic_direction (1 正向 / -1 负向)
# 训练前会对 direction=-1 的特征乘以 -1，使其与收益正相关
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

# SingleModelTrainConfig = TrainConfig(model_cfg = ConvLSTMConfig(model_version= 3))
# train_task_config = TrainTask.SINGLE_MODEL_3CLASS
# SingleModelTrainConfig = TrainConfig(model_cfg = LogisticConfig(model_version= 1))
# SingleModelTrainConfig = TrainConfig(model_cfg = TransformerConfig(model_version= 1))
seq_len = 128
SingleModelTrigger = TrainConfig(model_cfg = TransformerConfig(model_version= 1,seq_len=seq_len))
SingleModelDirection = TrainConfig(model_cfg = TransformerConfig(model_version= 1,seq_len=seq_len))
train_task_config = TrainTask.SINGLE_MODEL_TRIGGER
