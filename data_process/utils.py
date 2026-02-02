import json,os
import hashlib
from dataclasses import asdict
import numpy as np

def safe_get(d, keys, default=0):
    """从多层 dict 中取值，避免 KeyError"""
    cur = d
    for k in keys:
        cur = cur.get(k, {})
    return cur if cur != {} else default

def json_safe(x):
    """递归把对象转换为可 JSON 序列化的结构"""
    # numpy scalar -> python scalar
    if isinstance(x, np.generic):
        return x.item()

    # numpy array -> list
    if isinstance(x, np.ndarray):
        return x.tolist()

    # dict: key 必须是 str/int/float/bool/None，最稳是 str
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}

    # list/tuple
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]

    return x

def param_hash(d, length=12):
    """对参数字典计算稳定 hash，用于区分不同参数组合。"""
    s = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:length]
    
def calc_params_hash(*, strategy, common, train, algo="sha1", length=8):
    """
    对参数快照计算稳定 hash
    """
    payload = {
        "strategy": asdict(strategy),
        "common": asdict(common),
        "train": asdict(train),
    }

    # canonical JSON：key 排序 + 无空格
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    h = hashlib.new(algo, s.encode("utf-8")).hexdigest()

    return h[:length] if length else h

def load_selected_configs(path):
    """
    读取 selected_configs.jsonl
    返回：list[dict]，每个元素是一条完整 report
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
                # 坏行直接跳过，符合你一贯的设计哲学
                continue
    return records


