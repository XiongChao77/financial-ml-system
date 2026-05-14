import json, os, subprocess
import hashlib
import shutil
import sys
from dataclasses import asdict, is_dataclass,dataclass
import numpy as np
from datetime import datetime
from typing import Dict, Iterable, List
import pandas as pd

current_work_dir = os.path.dirname(__file__)

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

def auto_git_commit(logger):
    """
    Check for uncommitted changes and auto-run git add/commit if any exist.
    """
    try:
        # 1. Check workspace status
        status = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
        
        if not status:
            logger.info("✅ Git workspace is clean. No auto-commit needed.")
            return

        # 2. Changes found -> auto commit
        logger.info("📝 Detected uncommitted changes. Performing auto-commit...")
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        commit_msg = f"Auto-commit before experiment"
        
        # Run: git add .
        subprocess.run(["git", "add", "."], check=True)
        # Run: git commit
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        
        # Log the latest commit hash for traceability
        new_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        logger.info(f"🚀 Auto-commit successful. Commit hash: {new_hash}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Auto-commit failed: {e}")
    except Exception as e:
        logger.error(f"⚠️ Unexpected error during git operation: {e}")

def get_data_metadata_hash(data_files: List[str]) -> str:
    """
    Generate a metadata hash from file name, size, and mtime.
    """
    m = hashlib.md5()
    for f_path in sorted(data_files):
        if not os.path.exists(f_path):
            continue
        stat = os.stat(f_path)
        # file name + size + modified time
        info = f"{os.path.basename(f_path)}|{stat.st_size}|{stat.st_mtime}"
        m.update(info.encode('utf-8'))
    return m.hexdigest()

def create_git_tag(tag_prefix="exp"):
    """
    Create an annotated git tag for the current commit (experiment marker).
    """
    tag_name = f"{tag_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        subprocess.run(["git", "tag", "-a", tag_name, "-m", f"Experiment started at {tag_name}"], check=True)
        return tag_name
    except Exception as e:
        return f"tag_failed_{str(e)}"

def isolate_and_relaunch(exp_dir: str, ignore_patterns: List[str]):
    """
    Copy the project into an experiment directory and relaunch from the new location.
    """
    src_dir = os.path.abspath(os.path.join(current_work_dir, ".."))
    # Create a subfolder under exp_dir to store source code
    dst_dir = os.path.join(exp_dir, "source_code")
    
    if os.path.exists(dst_dir):
        return # Already in an isolated environment; do not copy again
    
    print(f"📦 Isolating codebase to: {dst_dir}")
    shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns(*ignore_patterns))
    
    # Build relaunch command
    new_script_path = os.path.join(dst_dir, "experiment", os.path.basename(__file__))
    new_args = [sys.executable, new_script_path] + sys.argv[1:] + ["--is_isolated"]
    
    # Relaunch and exit the current process
    subprocess.run(new_args)
    sys.exit(0)

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

def param_hash(d, length=12):
    """Compute a stable hash for a parameter dict (used to identify parameter combinations)."""
    s = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:length]
    
def calc_params_hash(*, strategy, common, train, algo="sha1", length=8):
    """
    Compute a stable hash for a parameter snapshot.
    """
    payload = {
        "strategy": asdict(strategy),
        "common": asdict(common),
        "train": asdict(train),
    }

    # canonical JSON: sorted keys, no extra whitespace
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    h = hashlib.new(algo, s.encode("utf-8")).hexdigest()

    return h[:length] if length else h

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

def recursive_get(data, target_key):
    # 1. If current node is a dict, check the current level first
    if isinstance(data, dict):
        if target_key in data:
            return data[target_key]
        
        # 2. Not found -> search each value recursively
        for k, v in data.items():
            res = recursive_get(v, target_key)
            if res is not None:
                return res
                
    # 3. If current node is a list, search each item recursively
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