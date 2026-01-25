import pandas as pd
import numpy as np
import os, sys, datetime, time, html
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
from sklearn.ensemble import RandomForestRegressor
from scipy.cluster.hierarchy import dendrogram, linkage
import torch
from ignite.metrics import HSIC

# --- 全局配置 ---
HIGH_CORR_THRESHOLD = 0.90  # 冗余阈值
ANALYZE_TARGET = 'REL'      # 'ORIGIN' (原始数据) 或 'REL' (归一化后的数据)
LEAKAGE_THRESHOLD = 0.99    # 判定为数据泄露的相关性阈值

# --- 绘图字体设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False
try:
    fm._rebuild()
except:
    pass

# ==============================================================================
# 1. 核心计算工具 (HSIC, 智能冗余, 可视化)
# ==============================================================================

def compute_hsic_ignite(x_data, y_data, max_samples=3000) -> float:
    """使用 PyTorch Ignite 计算 HSIC (非线性依赖)，带采样保护"""
    if HSIC is None: return 0.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        if hasattr(x_data, 'values'): x_data = x_data.values
        if hasattr(y_data, 'values'): y_data = y_data.values
        
        # 强制降采样，防止 OOM
        N = len(x_data)
        if N > max_samples:
            indices = np.random.choice(N, max_samples, replace=False)
            x_data = x_data[indices]
            y_data = y_data[indices]

        t_x = torch.as_tensor(x_data, dtype=torch.float32, device=device).view(-1, 1)
        t_y = torch.as_tensor(y_data, dtype=torch.float32, device=device).view(-1, 1)

        metric = HSIC(sigma_x=-1, sigma_y=-1)
        metric.update((t_x, t_y))
        score = metric.compute()
        metric.reset()
        return float(score)
    except Exception:
        return 0.0
    finally:
        if device.type == 'cuda': torch.cuda.empty_cache()

def get_smart_redundancy_filter(df: pd.DataFrame, target_col: str, threshold: float = 0.90):
    """
    智能冗余剔除逻辑：
    当特征 A 和 B 高度相关时，保留与 Target 相关性更高那个，建议剔除另一个。
    """
    feature_cols = [c for c in df.columns if c != target_col]
    if not feature_cols: return []
    
    # 计算特征间相关性矩阵
    corr_matrix = df[feature_cols].corr(method='pearson').abs()
    # 计算特征与目标的相关性 (作为裁判)
    target_corr = df[feature_cols].corrwith(df[target_col], method='spearman').abs()
    
    drop_suggestions = []
    processed_pairs = set()

    # 遍历上三角矩阵
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            if val >= threshold:
                feat_a = corr_matrix.columns[i]
                feat_b = corr_matrix.columns[j]
                
                # 裁判：谁与目标相关性低，谁就被建议剔除
                score_a = target_corr.get(feat_a, 0)
                score_b = target_corr.get(feat_b, 0)
                
                if score_a < score_b:
                    drop = feat_a
                    keep = feat_b
                else:
                    drop = feat_b
                    keep = feat_a
                
                drop_suggestions.append({
                    'Feature_Drop': drop,
                    'Feature_Keep': keep,
                    'Inter_Corr': val,
                    'Drop_Target_Corr': min(score_a, score_b),
                    'Keep_Target_Corr': max(score_a, score_b)
                })
    return drop_suggestions

# --- 可视化函数组 ---

def plot_visualizations(df: pd.DataFrame, target_col: str, output_dir: str, tag: str):
    """生成三种关键图表：聚类图、重要性排行、热力图"""
    feature_cols = [c for c in df.columns if c != target_col]
    if len(feature_cols) < 2: return

    # 1. 特征聚类树状图 (Dendrogram) - 识别同质化严重的特征群
    try:
        corr = df[feature_cols].corr(method='spearman').fillna(0)
        dist = 1 - corr.abs() # 距离定义为 1 - 相关性
        linked = linkage(dist, 'ward')
        
        plt.figure(figsize=(12, 6))
        dendrogram(linked, orientation='top', labels=feature_cols, leaf_rotation=90)
        plt.title(f"Feature Clustering Hierarchy ({tag})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"viz_{tag}_dendrogram.png"))
        plt.close()
    except Exception as e:
        print(f"[Viz Error] Dendrogram: {e}")

    # 2. 特征重要性条形图 (Top 20 MI)
    try:
        # 简单采样计算 MI 用于绘图
        sample_df = df.sample(min(5000, len(df)))
        mi = mutual_info_regression(sample_df[feature_cols], sample_df[target_col])
        mi_series = pd.Series(mi, index=feature_cols).sort_values(ascending=False).head(20)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x=mi_series.values, y=mi_series.index, palette='viridis')
        plt.title(f"Top 20 Features by Mutual Information ({tag})")
        plt.xlabel("MI Score")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"viz_{tag}_importance.png"))
        plt.close()
    except Exception as e:
        print(f"[Viz Error] Importance: {e}")

# ==============================================================================
# 2. 单次分析逻辑
# ==============================================================================

def analyze_correlation(df: pd.DataFrame, target_col: str = 'return_rate', use_hsic: bool = True):
    """
    计算相关性矩阵、MI、HSIC，并进行泄露检测。
    """
    if target_col not in df.columns:
        print(f"[Warning] Target {target_col} not found, skipping analysis.")
        return {}, []

    # 1. 基础清理
    drop_cols = data_loader.DROP_FEATURES
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore').dropna()
    
    # 如果是 REL 模式，通常 target_col 是 label，需要确保它是数值型
    if df_clean[target_col].dtype == 'object':
         df_clean[target_col] = df_clean[target_col].astype(float)

    if len(df_clean) < 50: return {}, []

    feature_cols = [c for c in df_clean.columns if c != target_col]
    
    # 2. 计算各项指标
    # Pearson & Spearman
    corr_pearson = df_clean[feature_cols].corrwith(df_clean[target_col], method='pearson').abs()
    corr_spearman = df_clean[feature_cols].corrwith(df_clean[target_col], method='spearman').abs()
    
    # Mutual Information (MI)
    MAX_MI_SAMPLES = 10000
    df_mi = df_clean.iloc[-MAX_MI_SAMPLES:]
    try:
        mi_scores = mutual_info_regression(df_mi[feature_cols], df_mi[target_col], random_state=42)
        mi_series = pd.Series(mi_scores, index=feature_cols)
    except:
        mi_series = pd.Series(0, index=feature_cols)
        
    # HSIC
    hsic_series = pd.Series(0.0, index=feature_cols)
    if use_hsic:
        y_val = df_mi[target_col].values
        for feat in feature_cols:
            hsic_series[feat] = compute_hsic_ignite(df_mi[feat].values, y_val)

    # 3. 组装结果
    corr_result = {}
    leakage_warnings = []

    for feat in feature_cols:
        p_val = corr_pearson.get(feat, 0)
        s_val = corr_spearman.get(feat, 0)
        m_val = mi_series.get(feat, 0)
        
        # 🚨 泄露检测
        if p_val > LEAKAGE_THRESHOLD or s_val > LEAKAGE_THRESHOLD:
            leakage_warnings.append(feat)

        corr_result[f"{feat}_pearson"] = p_val
        corr_result[f"{feat}_spearman"] = s_val
        corr_result[f"{feat}_mi"] = m_val
        corr_result[f"{feat}_hsic"] = hsic_series.get(feat, 0)

    # 4. 获取智能冗余建议
    redundancy_suggestions = get_smart_redundancy_filter(df_clean, target_col, threshold=HIGH_CORR_THRESHOLD)

    return corr_result, redundancy_suggestions, leakage_warnings

def single_run_analysis(candlestick_num: int, predict_num: int, vm: float, smr: float, df: pd.DataFrame, output_dir: str):
    """
    单次任务执行入口：生成标签 -> (归一化) -> 分析 -> 绘图
    """
    # A. 生成标签
    try:
        common.attach_triple_barrier_label(df, candlestick_num, predict_num, 
                            vol_multiplier=vm, stop_multiplier_rate=smr) # keep_rate=True 保留 return_rate 列
    except Exception as e:
        return {'error': str(e)}

    # 统计标签比例
    label_counts = df['label'].value_counts(normalize=True)
    ratios = {f'ratio_{k}': label_counts.get(k, 0.0) for k in [0, 1, 2]}

    # B. 数据准备 (ORIGIN vs REL)
    df_to_analyze = None
    target_col = 'return_rate' # 默认针对连续收益率分析

    if ANALYZE_TARGET == 'ORIGIN':
        df_to_analyze = df.copy()
    
    elif ANALYZE_TARGET == 'REL':
        # 核心：使用 TimeSeriesWindowDataset 获取归一化后的数据
        try:
            # 使用全量特征
            feat_cols = [c for c in df.columns if c not in data_loader.DROP_FEATURES]
            full_ds = data_loader.TimeSeriesWindowDataset(
                df, feature_cols=feat_cols, label_col='label', window=candlestick_num, kline_interval_ms=common.load_interval_ms(),
                stride = 2, use_cache = True
            )
            # 提取最后一帧
            X_np = full_ds.X[:, -1, :].numpy()
            df_to_analyze = pd.DataFrame(X_np, columns=full_ds.feature_names)
            
            # 💡 对于归一化数据，我们通常分析其与 Label (0,1,2) 或 return_rate 的关系
            # 既然是 REL，我们尽量尝试用 return_rate (如果 dataset 对齐支持)，
            # 这里简单起见使用 label (离散) 作为 Target，或者如果能对齐 indices 则取 return_rate
            # 简单实现：使用 label
            df_to_analyze['label'] = full_ds.y.numpy()
            target_col = 'label' 
            
        except Exception as e:
            print(f"Dataset error: {e}")
            return {'error': str(e)}

    # C. 执行分析
    corr_res, redundancy_res, leakage = analyze_correlation(df_to_analyze, target_col=target_col)
    
    # D. 绘图 (仅对第一个任务或特定参数绘图，避免文件爆炸)
    # 这里我们生成 unique tag，每次都覆盖或者保存
    viz_tag = f"C{candlestick_num}_P{predict_num}_{ANALYZE_TARGET}"
    # 简单的频率控制：如果文件不存在则画图
    viz_check_path = os.path.join(output_dir, f"viz_{viz_tag}_importance.png")
    if not os.path.exists(viz_check_path):
        plot_visualizations(df_to_analyze, target_col, output_dir, viz_tag)

    if leakage:
        print(f"🚨 LEAKAGE DETECTED in {viz_tag}: {leakage}")

    return {
        'candlestick_num': candlestick_num,
        'predict_num': predict_num,
        'vm': vm,
        'smr': smr,
        'ratios': ratios,
        'corr_res': corr_res,
        'redundancy_res': redundancy_res,
        'leakage': leakage
    }

def _unpack_and_run(args):
    """并行任务解包器"""
    c_num, p_num, vm, smr, df, out_dir = args
    return single_run_analysis(c_num, p_num, vm, smr, df, out_dir)

# ==============================================================================
# 3. 主流程与结果汇总
# ==============================================================================

def main():
    start_time = time.time()
    output_dir = os.path.join(common.PROJECT_DIR, 'correlation_result')
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    print(f"🚀 Start Analysis | Mode: {ANALYZE_TARGET} | Output: {output_dir}")

    # --- 参数网格 ---
    # 示例简化网格，可按需恢复全量 range
    candlestick_num_range = range(64, 192, 32) # [64, 96, 128...]
    period_list = ["5m", "15m", "1h", "4h"] 
    
    all_final_results = []
    redundancy_dfs = []

    for period in period_list:
        csv_path = os.path.join(os.path.dirname(common.origin_data_path), f"BTCUSDT_{period}.csv")
        if not os.path.exists(csv_path): continue
        
        print(f"\n📂 Processing Period: {period}")
        df_base = pd.read_csv(csv_path)
        # 预计算特征 (只做一次)
        common.attach_attr(df_base, common.FEATURE_CONFIG_LIST, common.load_interval_ms())
        
        tasks = []
        # 构建任务队列
        for c_num in candlestick_num_range:
            for p_num in [c_num//8, c_num//12]: # Predict window logic
                # 使用默认 vm/mt，如需网格搜索可在此嵌套循环
                tasks.append((c_num, p_num, 1.3, 0.4, df_base.copy(), output_dir))
                # _unpack_and_run((c_num, p_num, 1.3, 0.4, df_base.copy(), output_dir))
                # exit()
        
        # 并行执行
        results_buffer = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            for res in executor.map(_unpack_and_run, tasks):
                if 'error' not in res:
                    results_buffer.append(res)
        
        # 整理单周期结果
        for res in results_buffer:
            # 1. 扁平化相关性数据
            base_info = {
                'period': period,
                'candlestick_num': res['candlestick_num'],
                'predict_num': res['predict_num'],
                **res['ratios']
            }
            # 合并相关性分数
            row = {**base_info, **res['corr_res']}
            all_final_results.append(row)
            
            # 2. 收集冗余建议
            if res['redundancy_res']:
                df_red = pd.DataFrame(res['redundancy_res'])
                for k, v in base_info.items():
                    df_red[k] = v
                redundancy_dfs.append(df_red)

    # --- 结果汇总与保存 ---
    if all_final_results:
        df_master = pd.DataFrame(all_final_results)
        
        # 1. 保存主分析结果
        main_csv = os.path.join(output_dir, f'analysis_summary_{ANALYZE_TARGET}.csv')
        df_master.to_csv(main_csv, index=False)
        print(f"\n✅ Main analysis saved to: {main_csv}")
        
        # 2. 打印 Top Features (基于 MI 均值)
        mi_cols = [c for c in df_master.columns if c.endswith('_mi')]
        if mi_cols:
            avg_mi = df_master[mi_cols].mean().sort_values(ascending=False).head(10)
            print("\n🏆 Top 10 Features (Avg Mutual Information):")
            print(avg_mi)

    if redundancy_dfs:
        df_redundancy = pd.concat(redundancy_dfs, ignore_index=True)
        red_csv = os.path.join(output_dir, f'redundancy_suggestions_{ANALYZE_TARGET}.csv')
        df_redundancy.to_csv(red_csv, index=False, float_format='%.4f')
        print(f"✅ Smart redundancy suggestions saved to: {red_csv}")
        print("   (Check 'Feature_Drop' vs 'Feature_Keep' columns)")

    time_taken = time.time() - start_time
    print(f"\n🏁 All Done in {time_taken:.1f}s")

if __name__ == "__main__":
    main()