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
HIGH_CORR_THRESHOLD = 0.80     

# --- 绘图字体与清晰度设置 ---
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 100
# 保存时使用更高 DPI
FIG_SAVE_DPI = 150

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

def single_run_analysis(pre_task: common.BaseDefine, train_cfg: train.TrainConfig, df: pd.DataFrame, output_dir: str):
    """单次分析流程：特征生成 -> 归一化 -> 针对 Label 和 Return_Rate 的双重相关性度量"""
    try:
        # 1. 基础准备
        df =common.attach_label(df, pre_task)
        # 确保 return_rate 存在
        if 'return_rate' not in df.columns:
            logging.warning("df 中未找到 return_rate，将尝试计算。")
            df['return_rate'] = df['close'].pct_change(pre_task.predict_num).shift(-pre_task.predict_num)

        # 2. 构造特征列表
        # 核心技巧：将 return_rate 暂时放入 feat_cols 参与 Dataset 窗口对齐，这样能拿到与特征同步的截面数据
        all_cols = [c for c in df.columns if c not in data_loader.DROP_FEATURES]
        if 'return_rate' not in all_cols: all_cols.append('return_rate')
        
        ds = data_loader.TimeSeriesWindowDataset(
            df=df, 
            kline_interval_ms=common.get_interval_ms(pre_task.interval),
            feature_cols=all_cols,
            label_col='label',
            window=train_cfg.data_cfg.window,
            stride=train_cfg.stride,
            use_cache=False
        )
        
        # 3. 提取最后一帧截面数据
        X_last_step = ds.X[:, -1, :].numpy()
        df_final = pd.DataFrame(X_last_step, columns=ds.feature_names)
        df_final['label'] = ds.y.numpy()
        
        # 将 return_rate 从特征列中分离出来作为目标
        target_label = df_final['label']
        target_return = pd.Series(ds.returns.numpy(), index=df_final.index)
        feature_names = [c for c in ds.feature_names if c != 'return_rate']
        
        res_corr = {}
        
        # 4. 计算相关性
        # 批量计算 MI (对 Label 用分类/回归均可，对 Return 必须用回归)
        mi_label = mutual_info_regression(df_final[feature_names], target_label, random_state=42)
        mi_return = mutual_info_regression(df_final[feature_names], target_return, random_state=42)
        
        for idx, feat in enumerate(feature_names):
            # --- 针对 Label 的分析 ---
            res_corr[f"{feat}_L_pearson"] = abs(df_final[feat].corr(target_label, method='pearson'))
            res_corr[f"{feat}_L_mi"] = mi_label[idx]
            
            # --- 针对 Return Rate 的分析 ---
            res_corr[f"{feat}_R_pearson"] = abs(df_final[feat].corr(target_return, method='pearson'))
            res_corr[f"{feat}_R_spearman"] = abs(df_final[feat].corr(target_return, method='spearman'))
            res_corr[f"{feat}_R_mi"] = mi_return[idx]
            # 如果安装了 HSIC，计算非线性相关性
            if HSIC:
                res_corr[f"{feat}_R_hsic"] = compute_hsic_ignite(df_final[feat], target_return)

        # 5. 绘图（保持原有逻辑）
        plot_visualizations(df_final[feature_names + ['label']], 'label', output_dir, f"{pre_task.interval}_dual")
    
        return {
            'period': pre_task.interval,
            'corr_res': res_corr,
            'redundancy_res': get_smart_redundancy_filter(df_final[feature_names + ['label']], 'label', HIGH_CORR_THRESHOLD)
        }
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {'error': str(e)}

def plot_visualizations(df, target_col, output_dir, tag):
    """生成特征聚类图与重要性排行（高清晰度）"""
    feats = [c for c in df.columns if c != target_col]
    if len(feats) < 5:
        return

    n_feats = len(feats)
    # 按特征数量动态调整画布：特征多时放大，保证标签可读
    fig_h = max(6, n_feats * 0.35)
    fig_w = max(14, n_feats * 0.4)
    label_fontsize = max(8, min(11, 120 // max(1, n_feats // 10)))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    dist = 1 - df[feats].corr().abs().fillna(0)
    link = linkage(dist, 'ward')
    dendrogram(
        link,
        labels=feats,
        leaf_rotation=90,
        leaf_font_size=label_fontsize,
        ax=ax,
    )
    ax.set_title(f"Feature Clustering - {tag}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Feature", fontsize=11)
    ax.tick_params(axis='both', labelsize=label_fontsize)
    fig.tight_layout(pad=1.5)
    out_path = os.path.join(output_dir, f"cluster_{tag}.png")
    fig.savefig(out_path, dpi=FIG_SAVE_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def _unpack_and_run(args):
    return single_run_analysis(*args)

# ==============================================================================
# 3. Main Entry
# ==============================================================================

def main():
    logger, _ = common.setup_session_logger(sub_folder='correlation_result')
    output_dir = os.path.join(common.PERSISTENCE_DIR, 'correlation_result')
    os.makedirs(output_dir, exist_ok=True)
    
    pre_task = common.BaseDefine()
    pre_task.interval = '15m'
    train_cfg = train.TrainConfig()
    
    csv_path = os.path.join(common.PROJECT_DATA_DIR, f"{pre_task.symbol}_{pre_task.interval}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing data: {csv_path}")
    
    df_raw = pd.read_csv(csv_path)
    df_raw = common.clean_data_quality_auto(df_raw, logger)
    df_raw = common.attach_attr(df_raw, common.FEATURE_GROUP_LIST, para=pre_task)
    
    # 构造任务元组
    tasks = [(pre_task, train_cfg, df_raw.copy(), output_dir)]
    
    all_rows = []
    redundancy_rows = []

    # 运行分析
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        for res in executor.map(_unpack_and_run, tasks):
            if not res or 'error' in res: continue
            
            # 1. 整合主结果：仅保留存在于 res 中的键
            row = {
                'period': res.get('period', pre_task.interval),
                **res.get('corr_res', {})
            }
            all_rows.append(row)
            
            # 2. 整合冗余建议
            if res.get('redundancy_res'):
                for r in res['redundancy_res']:
                    r.update({'period': res.get('period', pre_task.interval)})
                    redundancy_rows.append(r)

    # 保存并展示结果
    if all_rows:
        df_summary = pd.DataFrame(all_rows)
        df_summary.to_csv(os.path.join(output_dir, 'analysis_summary.csv'), index=False)
        
        # --- 新增：转换为 Feature x Metric 矩阵 ---
        # 提取相关性列
        metric_types = ['pearson', 'spearman', 'mi', 'hsic']
        metric_cols = [c for c in df_summary.columns if any(m in c for m in metric_types)]
        
        # 假设我们只关心第一个周期的数据（如果是多周期，可在此循环）
        first_row = df_summary.iloc[0][metric_cols]
        
        matrix_data = []
        for col_name, value in first_row.items():
            # 识别是 Label 还是 Return 的指标
            if '_L_' in col_name:
                parts = col_name.rsplit('_L_', 1)
                target_prefix = "L_"
            elif '_R_' in col_name:
                parts = col_name.rsplit('_R_', 1)
                target_prefix = "R_"
            else:
                continue # 跳过非指标列
                
            if len(parts) == 2:
                feature_name = parts[0]
                metric_name = target_prefix + parts[1] # 组合成 L_mi, R_mi 等
                matrix_data.append({
                    'Feature': feature_name, 
                    'Metric': metric_name, 
                    'Value': value
                })
        
        # 转换为 2D 矩阵
        df_matrix = pd.DataFrame(matrix_data).pivot(index='Feature', columns='Metric', values='Value')
        
        # 按照 Return 的互信息（R_mi）降序排列，这是量化里最有含金量的指标
        if 'R_mi' in df_matrix.columns:
            df_matrix = df_matrix.sort_values(by='R_mi', ascending=False)
        
        # 保存并打印
        matrix_path = os.path.join(output_dir, 'feature_comparison_matrix.csv')
        df_matrix.to_csv(matrix_path)
        
        print("\n📊 特征全维度对比矩阵 (Top 15 sorted by R_mi):")
        # 选取你最关心的几列展示
        show_cols = [c for c in ['L_mi', 'R_mi', 'L_pearson', 'R_pearson'] if c in df_matrix.columns]
        print(df_matrix[show_cols].head(15).to_string())

    if redundancy_rows:
        pd.DataFrame(redundancy_rows).to_csv(os.path.join(output_dir, 'redundancy_suggestions.csv'), index=False)

    print(f"🏁 Analysis complete. Results in: {output_dir}")

if __name__ == "__main__":
    main()