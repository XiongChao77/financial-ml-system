import json
from typing import Type, Tuple, Dict
from model.models.model_base import BaseTimeSeriesModel

# =============================
# 显式 import 所有模型
# =============================
from model.models.xgboost_model import XGBoostAdapter
from model.models.tcn_v1 import TCN1D_V1
from model.models.mamba_v1 import Mamba1D_V1
from model.models.lstm import LSTM1D_V1          
from model.models.lstm_v2 import LSTM1D_V2      # V2
from model.models.lstm_v3 import LSTM1D_V3      # V3  > V1 > V2
from model.models.lstm_v4 import LSTM1D_V4      #
from model.models.conv_lstm_v1 import ConvLSTM1D_V1
from model.models.conv_lstm_v2 import ConvLSTM1D_V2
from model.models.cnn import CNN1D_V1
from model.models.transformer_v1 import Transformer1D_V1
from model.models.transformer_v2 import Transformer1D_V2
from model.models.transformer_v3 import Transformer1D_V3
# 后续新模型直接在这里加 import

class ModelFactory:
    """
    Centralized model factory.

    - All available models are explicitly listed here
    - Selection is based on (model_type, model_version)
    - Models must inherit BaseTimeSeriesModel
    """

    model_list = [
        XGBoostAdapter,
        TCN1D_V1,
        Mamba1D_V1,
        LSTM1D_V1,
        LSTM1D_V2,
        LSTM1D_V3,  #good
        LSTM1D_V4,  #good
        ConvLSTM1D_V1,
        ConvLSTM1D_V2,
        CNN1D_V1,
        Transformer1D_V1,
        Transformer1D_V2,
        Transformer1D_V3,
    ]

    # 内部索引
    _index: Dict[Tuple[str, int], Type[BaseTimeSeriesModel]] = {}

    # =============================
    # 初始化索引（只做一次）
    # =============================
    @classmethod
    def _build_index(cls):
        if cls._index:
            return

        for model_cls in cls.model_list:
            if not issubclass(model_cls, BaseTimeSeriesModel):
                raise TypeError(
                    f"{model_cls.__name__} must inherit BaseTimeSeriesModel"
                )

            key = (model_cls.MODEL_TYPE, model_cls.MODEL_VERSION)

            if key in cls._index:
                raise ValueError(f"Duplicate model registered: {key}")

            cls._index[key] = model_cls

    # =============================
    # 查询模型类
    # =============================
    @classmethod
    def get_model_class(cls, model_type: str, model_version: int):
        cls._build_index()

        key = (model_type, model_version)
        if key not in cls._index:
            available = sorted(cls._index.keys())
            raise KeyError(
                f"Model not found: {key}. "
                f"Available models: {available}"
            )
        return cls._index[key]

    # =============================
    # 训练阶段构建
    # =============================
    @classmethod
    def build_for_training(
        cls,
        model_type: str,
        model_version: int,
        device,
        **model_kwargs,
    ):
        model_cls = cls.get_model_class(model_type, model_version)
        model = model_cls(**model_kwargs)
        return model.to(device)

    # =============================
    # 加载 checkpoint
    # =============================
    @classmethod
    def load_from_checkpoint(
        cls,
        model_path: str,
        meta_path: str,
        device,
    ):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        model_cls = cls.get_model_class(
            meta["model_type"],
            meta["model_version"],
        )

        model, meta = model_cls.load_checkpoint(
            model_path=model_path,
            meta_path=meta_path,
            device=device,
        )
        return model, meta

    # =============================
    # 调试 / 可视化
    # =============================
    @classmethod
    def list_models(cls):
        cls._build_index()
        return sorted(cls._index.keys())
