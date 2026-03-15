import json,os,subprocess
import hashlib
from dataclasses import asdict, is_dataclass,dataclass
import numpy as np
from datetime import datetime
from typing import Dict, Iterable, List
import pandas as pd

def stop_loss_atr(df: pd.DataFrame, holdbar: int) -> pd.Series:
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

def auto_git_commit(logger):
    """
    检查是否有未提交的修改，如果有，自动执行 git add 和 commit
    """
    try:
        # 1. 检查工作区状态
        status = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
        
        if not status:
            logger.info("✅ Git workspace is clean. No auto-commit needed.")
            return

        # 2. 发现修改，执行自动提交
        logger.info("📝 Detected uncommitted changes. Performing auto-commit...")
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        commit_msg = f"Auto-commit before experiment"
        
        # 执行 git add .
        subprocess.run(["git", "add", "."], check=True)
        # 执行 git commit
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        
        # 获取最新的 commit hash 记录在 log 中方便追溯
        new_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        logger.info(f"🚀 Auto-commit successful. Commit hash: {new_hash}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Auto-commit failed: {e}")
    except Exception as e:
        logger.error(f"⚠️ Unexpected error during git operation: {e}")

def get_data_metadata_hash(data_files: List[str]) -> str:
    """
    根据文件名、大小和修改时间生成元数据哈希
    """
    m = hashlib.md5()
    for f_path in sorted(data_files):
        if not os.path.exists(f_path):
            continue
        stat = os.stat(f_path)
        # 按照用户要求：文件名 + 大小 + 修改时间
        info = f"{os.path.basename(f_path)}|{stat.st_size}|{stat.st_mtime}"
        m.update(info.encode('utf-8'))
    return m.hexdigest()

def create_git_tag(tag_prefix="exp"):
    """
    为当前提交打上实验标签
    """
    tag_name = f"{tag_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        subprocess.run(["git", "tag", "-a", tag_name, "-m", f"Experiment started at {tag_name}"], check=True)
        return tag_name
    except Exception as e:
        return f"tag_failed_{str(e)}"

def isolate_and_relaunch(exp_dir: str, ignore_patterns: List[str]):
    """
    将项目拷贝到实验目录并从新位置重新启动
    """
    src_dir = os.path.abspath(os.path.join(current_work_dir, ".."))
    # 在实验目录下创建一个 'code' 文件夹存放代码
    dst_dir = os.path.join(exp_dir, "source_code")
    
    if os.path.exists(dst_dir):
        return # 已经处于隔离环境，不再重复拷贝
    
    print(f"📦 Isolating codebase to: {dst_dir}")
    shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns(*ignore_patterns))
    
    # 构建重新启动的命令
    new_script_path = os.path.join(dst_dir, "experiment", os.path.basename(__file__))
    new_args = [sys.executable, new_script_path] + sys.argv[1:] + ["--is_isolated"]
    
    # 重新启动并退出当前进程
    subprocess.run(new_args)
    sys.exit(0)

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

def recursive_get(data, target_key):
    # 1. 如果直接就是字典，先看当前层有没有
    if isinstance(data, dict):
        if target_key in data:
            return data[target_key]
        
        # 2. 当前层没有，则“展开”字典，递归进入每一个 Value 查找
        for k, v in data.items():
            res = recursive_get(v, target_key)
            if res is not None:
                return res
                
    # 3. 如果遇到列表（量化回测中常见的参数组合列表），也“展开”它
    elif isinstance(data, list):
        for item in data:
            res = recursive_get(item, target_key)
            if res is not None:
                return res
                
    return None

def dump_params_json(obj, logger):
    if is_dataclass(obj):
        data = asdict(obj)
    elif isinstance(obj, dict):
        data = obj
    else:
        raise TypeError(f"Unsupported config type: {type(obj)}")

    logger.info("Params | " + json.dumps(data, indent=2, ensure_ascii=False))