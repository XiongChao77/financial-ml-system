import pandas as pd
import numpy as np
import os, sys, datetime, time, logging
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import concurrent.futures
from typing import Dict, Any, List, Set

# 引入项目路径
current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir, '..'))

import data_process.common as common
from model import data_loader
from sklearn.feature_selection import mutual_info_regression
from scipy.cluster.hierarchy import dendrogram, linkage
import torch

# 尝试导入 HSIC，若无 ignite 则静默跳过
try:
    from ignite.metrics import HSIC
except ImportError:
    HSIC = None

import model.train_2head as train

# --- 全局配置 ---
HIGH_CORR_THRESHOLD = 0.90  
LEAKAGE_THRESHOLD = 0.99    

# --- 绘图字体设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 1. 核心计算工具
# ==============================================================================

def compute_hsic_ignite(x_data, y_data, max_samples=2000) -> float:
    """使用 PyTorch Ignite 计算 HSIC (非线性依赖)"""
    if HSIC is None: return 0.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        if hasattr(x_data, 'values'): x_data = x_data.values
        if hasattr(y_data, 'values'): y_data = y_data.values
        
        # 采样保护，HSIC 计算复杂度为 O(N^2)
        N = len(x_data)
        if N > max_samples:
            indices = np.random.choice(N, max_samples, replace=False)
            x_data, y_data = x_data[indices], y_data[indices]

        t_x = torch.as_tensor(x_data, dtype=torch.float32, device=device).view(-1, 1)
        t_y = torch.as_tensor(y_data, dtype=torch.float32, device=device).view(-1, 1)

        metric = HSIC(sigma_x=-1, sigma_y=-1)
        metric.update((t_x, t_y))
        score = metric.compute()
        return float(score)
    except Exception:
        return 0.0
    finally:
        if device.type == 'cuda': torch.cuda.empty_cache()

def get_smart_redundancy_filter(df: pd.DataFrame, target_col: str, threshold: float = 0.90):
    """剔除冗余特征：保留与目标相关性更高的一方"""
    feature_cols = [c for c in df.columns if c != target_col]
    if not feature_cols: return []
    
    corr_matrix = df[feature_cols].corr(method='pearson').abs()
    target_corr = df[feature_cols].corrwith(df[target_col], method='spearman').abs()
    
    drop_suggestions = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            if val >= threshold:
                feat_a, feat_b = corr_matrix.columns[i], corr_matrix.columns[j]
                drop, keep = (feat_a, feat_b) if target_corr.get(feat_a, 0) < target_corr.get(feat_b, 0) else (feat_b, feat_a)
                
                drop_suggestions.append({
                    'Feature_Drop': drop, 'Feature_Keep': keep, 'Inter_Corr': val,
                    'Drop_Target_Corr': target_corr.get(drop, 0), 'Keep_Target_Corr': target_corr.get(keep, 0)
                })
    return drop_suggestions

# ==============================================================================
# 2. 核心分析逻辑
# ==============================================================================

def single_run_analysis(pre_task: common.CommonDefine, train_cfg: train.TrainConfig, df: pd.DataFrame, output_dir: str):
    """单次分析流程：特征生成 -> 归一化 -> 相关性度量"""
    try:
        # 1. 基础准备
        common.attach_label(df, pre_task)
        label_counts = df['label'].value_counts(normalize=True)
        ratios = {f'ratio_{k}': label_counts.get(k, 0.0) for k in [0, 1, 2]}

        # 2. 提取归一化后的窗口数据 (取最后一帧)
        feature_cols = train_cfg.data_cfg.feature_cols if train_cfg.data_cfg.feature_cols else list(df.columns)
        print(f"Features num:{len(feature_cols)},: {feature_cols}")
        # 使用 train_2head 中定义的特征列表
        ds = data_loader.TimeSeriesWindowDataset(
            df=df, 
            feature_config_list=common.FEATURE_CONFIG_LIST,
            kline_interval_ms=common.get_interval_ms(pre_task.interval),
            feature_cols=feature_cols,
            label_col='label',
            window=train_cfg.data_cfg.window,
            stride=train_cfg.stride,
            use_cache=False
        )
        
        X_last_step = ds.X[:, -1, :].numpy()
        df_final = pd.DataFrame(X_last_step, columns=ds.feature_names)
        df_final['label'] = ds.y.numpy()
        
        # 3. 相关性计算
        feature_names = ds.feature_names
        target = df_final['label']
        
        res_corr = {}
        leakage = []
        
        # 批量计算 MI
        mi_scores = mutual_info_regression(df_final[feature_names], target, random_state=42)
        
        for idx, feat in enumerate(feature_names):
            p_val = df_final[feat].corr(target, method='pearson')
            s_val = df_final[feat].corr(target, method='spearman')
            
            if abs(p_val) > LEAKAGE_THRESHOLD: leakage.append(feat)
            
            res_corr[f"{feat}_pearson"] = abs(p_val)
            res_corr[f"{feat}_spearman"] = abs(s_val)
            res_corr[f"{feat}_mi"] = mi_scores[idx]
            res_corr[f"{feat}_hsic"] = compute_hsic_ignite(df_final[feat], target)

        # 4. 绘图
        plot_visualizations(df_final, 'label', output_dir, pre_task.interval)

        return {
            'period': pre_task.interval,
            'candlestick_num': pre_task.candlestick_num,
            'predict_num': pre_task.predict_num,
            'ratios': ratios,
            'corr_res': res_corr,
            'redundancy_res': get_smart_redundancy_filter(df_final, 'label', HIGH_CORR_THRESHOLD),
            'leakage': leakage
        }
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        return {'error': str(e)}

def plot_visualizations(df, target_col, output_dir, tag):
    """生成特征聚类图与重要性排行"""
    feats = [c for c in df.columns if c != target_col]
    if len(feats) < 5: return
    
    # Dendrogram
    plt.figure(figsize=(12, 6))
    dist = 1 - df[feats].corr().abs().fillna(0)
    dendrogram(linkage(dist, 'ward'), labels=feats, leaf_rotation=90)
    plt.title(f"Feature Clustering - {tag}")
    plt.savefig(os.path.join(output_dir, f"cluster_{tag}.png"))
    plt.close()

def _unpack_and_run(args):
    return single_run_analysis(*args)

# ==============================================================================
# 3. Main Entry
# ==============================================================================

def main():
    logger, _ = common.setup_session_logger(sub_folder='correlation_analyze')
    output_dir = os.path.join(common.PERSISTENCE_DIR, 'correlation_result')
    os.makedirs(output_dir, exist_ok=True)
    
    pre_task = common.CommonDefine()
    pre_task.interval = '15m'
    train_cfg = train.TrainConfig()
    
    csv_path = common.origin_data_path
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing data: {csv_path}")
    
    df_raw = pd.read_csv(csv_path)
    df_raw = common.clean_data_quality_auto(df_raw, logger)
    common.attach_attr(df_raw, common.FEATURE_CONFIG_LIST, para=pre_task)
    
    # 构造任务元组
    tasks = [(pre_task, train_cfg, df_raw.copy(), output_dir)]
    
    all_rows = []
    redundancy_rows = []

    # 考虑到 DataFrame 较大且只需运行一次，直接运行或少量并行
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        for res in executor.map(_unpack_and_run, tasks):
            if 'error' in res: continue
            
            # 整合主结果
            row = {
                'period': res['period'],
                'candlestick': res['candlestick_num'],
                'predict': res['predict_num'],
                **res['ratios'],
                **res['corr_res']
            }
            all_rows.append(row)
            
            # 整合冗余建议
            if res['redundancy_res']:
                for r in res['redundancy_res']:
                    r.update({'period': res['period']})
                    redundancy_rows.append(r)

    # 保存 CSV
    if all_rows:
        pd.DataFrame(all_rows).to_csv(os.path.join(output_dir, 'analysis_summary.csv'), index=False)
        # 打印 Top 10
        mi_cols = [c for c in all_rows[0].keys() if c.endswith('_mi')]
        top_mi = pd.Series({c.replace('_mi',''): all_rows[0][c] for c in mi_cols}).sort_values(ascending=False).head(10)
        print("\n🏆 Top 10 Features (Mutual Information):\n", top_mi)

    if redundancy_rows:
        pd.DataFrame(redundancy_rows).to_csv(os.path.join(output_dir, 'redundancy_suggestions.csv'), index=False)

    print(f"🏁 Analysis complete. Results in: {output_dir}")

if __name__ == "__main__":
    main()