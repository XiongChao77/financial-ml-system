import json, os
import hashlib
from dataclasses import asdict, is_dataclass,dataclass
import numpy as np
from typing import Any, Dict
import pandas as pd

def stop_loss_atr_pct(df: pd.DataFrame, holdbar: int) -> pd.Series:
    length = max(10, round(0.8 * holdbar))
    length = int(length)
    length = max(length, 2)

    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)

    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()

    return atr / close

def safe_get(d, keys, default=0):
    """Safely get nested dict values (avoid KeyError)."""
    cur = d
    for k in keys:
        cur = cur.get(k, {})
    return cur if cur != {} else default

def json_safe(x):
    """Recursively convert objects into JSON-serializable structures."""
    # numpy scalar -> python scalar
    if isinstance(x, np.generic):
        return x.item()

    # numpy array -> list
    if isinstance(x, np.ndarray):
        return x.tolist()

    # dict: keys must be JSON-compatible; safest is str
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}

    # list/tuple
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]

    return x

TASK_HASH_LENGTH = 12


def param_hash(d, length=TASK_HASH_LENGTH):
    """Compute a stable hash for a parameter dict (used to identify parameter combinations)."""
    s = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True)
class TaskIdentity:
    """Canonical identity for every stage of one experiment task.

    Component hashes identify reusable artifacts. ``full_hash`` identifies the
    complete prep/train/simulation combination and is the hash written into the
    backtest report. All hashes use the same algorithm and length.
    """

    prep_hash: str
    train_hash: str
    sim_hash: str
    full_hash: str

    @staticmethod
    def prep_hash_for(params: Dict[str, Any]) -> str:
        return param_hash(json_safe(params))

    @staticmethod
    def train_hash_for(params: Dict[str, Any]) -> str:
        return param_hash(json_safe(params))

    @staticmethod
    def sim_hash_for(params: Dict[str, Any]) -> str:
        return param_hash(json_safe(params))

    @classmethod
    def from_params(
        cls,
        *,
        prep: Dict[str, Any],
        train: Dict[str, Any],
        sim: Dict[str, Any],
    ) -> "TaskIdentity":
        prep_params = json_safe(prep)
        train_params = json_safe(train)
        sim_params = json_safe(sim)
        return cls(
            prep_hash=cls.prep_hash_for(prep_params),
            train_hash=cls.train_hash_for(train_params),
            sim_hash=cls.sim_hash_for(sim_params),
            full_hash=param_hash(
                {
                    "prep": prep_params,
                    "train": train_params,
                    "sim": sim_params,
                }
            ),
        )

    @classmethod
    def from_configs(cls, *, strategy_config, broker_config, common, train) -> "TaskIdentity":
        return cls.from_params(
            prep=asdict(common),
            train=asdict(train),
            sim={
                "strategy_config": {
                    "config_type": type(strategy_config).__name__,
                    **asdict(strategy_config),
                },
                "broker_config": asdict(broker_config),
            },
        )

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)


def load_selected_configs(path):
    """
    Read selected_configs.jsonl.
    Returns: list[dict], each element is a full report.
    """
    records = []
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines
                continue
    return records

def recursive_get(data, target_key, default=None):
    """
    Supports two modes:

    1. Dot path:
       recursive_get(data, "long.params.common.predict_num")

    2. Recursive key search:
       recursive_get(data, "predict_num")
    """

    def get_by_path(obj, path):
        cur = obj
        for part in path.split("."):
            if isinstance(cur, dict):
                if part not in cur:
                    return default
                cur = cur[part]

            elif isinstance(cur, list):
                # support numeric index in path, e.g. "items.0.value"
                if not part.isdigit():
                    return default
                idx = int(part)
                if idx < 0 or idx >= len(cur):
                    return default
                cur = cur[idx]

            else:
                return default

        return cur

    # 1. If target_key is dot path, try exact path first
    if isinstance(target_key, str) and "." in target_key:
        value = get_by_path(data, target_key)
        if value is not default:
            return value

    # 2. Original recursive key search
    if isinstance(data, dict):
        if target_key in data:
            return data[target_key]

        for _, v in data.items():
            res = recursive_get(v, target_key, default=default)
            if res is not default:
                return res

    elif isinstance(data, list):
        for item in data:
            res = recursive_get(item, target_key, default=default)
            if res is not default:
                return res

    return default

def dump_params_json(obj, logger):
    if is_dataclass(obj):
        data = asdict(obj)
    elif isinstance(obj, dict):
        data = obj
    else:
        raise TypeError(f"Unsupported config type: {type(obj)}")

    logger.info("Params | " + json.dumps(data, indent=2, ensure_ascii=False))

def load_reports(path):
    """
    Read jsonl file line by line and skip malformed lines.
    """
    reports = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                reports.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return reports

def make_model_cfg(d):
    from model import train_config
    from dataclasses import fields
    model_type = d.get("model_type")

    for cls in train_config.BaseModelConfig.__subclasses__():
        obj = cls()
        if obj.model_type == model_type:
            valid_keys = {f.name for f in fields(cls)}
            kwargs = {k: v for k, v in d.items() if k in valid_keys}
            return cls(**kwargs)

    raise ValueError(f"Unknown model_type: {model_type}")

def config_from_dict_train(train_params: Dict):
    """
    Restore TrainConfig from dict stored in task spec.
    Intentionally ignores nested model_cfg/data_cfg dicts in spec (those fields are dataclasses).
    """
    import model.train as train

    t_cfg = train.TrainConfig()
    for k, v in (train_params or {}).items():
        if k == "model_cfg" and isinstance(v, dict):
            t_cfg.model_cfg = make_model_cfg(v)
        elif k == "data_cfg" and isinstance(v, dict):
            t_cfg.data_cfg = train.DataConfig(**v)
        elif hasattr(t_cfg, k):
            setattr(t_cfg, k, v)
    return t_cfg
