#!/usr/bin/env python3
import argparse
import json,os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.ndimage import uniform_filter

# --- 配置 ---
EPS = 1e-9

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

def load_and_process(path: Path, period: str = 'short') -> pd.DataFrame:
    """带调试输出的数据加载函数"""
    try:
        # 1. 检查文件是否存在
        if not path.exists():
            print(f"❌ Error: File not found at {path}")
            return pd.DataFrame()

        df_raw = pd.read_json(path, lines=True)
        
        # 2. 调试：打印原始列名
        print(f"\n--- Debug: {period.upper()} Period Parsing ---")
        print(f"Raw Columns Found: {df_raw.columns.tolist()}")

        if period not in df_raw.columns:
            print(f"❌ Error: Period '{period}' not in JSON columns!")
            return pd.DataFrame()

        # 3. 调试：采样第一行数据结构
        sample_row = df_raw[period].iloc[0]
        if isinstance(sample_row, dict):
            print(f"Sample keys in '{period}': {list(sample_row.keys())}")
        else:
            print(f"⚠️ Warning: Data in '{period}' is not a dictionary, it is a {type(sample_row)}")

        def extract_data(row):
            try:
                p = row[period]
                if p is None: return None
                
                res = {
                    "predict_num": recursive_get(p, "predict_num"),
                    "holdbar": recursive_get(p, "holdbar"),
                    "cagr": recursive_get(p, "cagr"),
                    "calmar": recursive_get(p, "calmar"),
                    "long_pnl": recursive_get(p, "long_pnl"),
                    "long_win_rate": recursive_get(p, "long_win_rate"),
                    "short_pnl": recursive_get(p, "short_pnl"),
                    "short_win_rate": recursive_get(p, "short_win_rate"),
                }
                return res
            except Exception as e:
                # 只有在真正报错时才打印，避免刷屏
                return None

        # 4. 转换并过滤
        processed_list = [item for item in df_raw.apply(extract_data, axis=1) if item]
        df_final = pd.DataFrame(processed_list)

        # 5. 调试：检查最终 DataFrame 内容
        if not df_final.empty:
            print(f"✅ Success: Loaded {len(df_final)} rows for {period}")
            # 检查关键字段是否有全是 NaN 的情况
            nan_counts = df_final[['predict_num', 'holdbar', 'cagr']].isna().sum()
            if nan_counts.any():
                print(f"⚠️ Warning: Missing values found:\n{nan_counts}")
            print(f"Data Preview:\n{df_final[['predict_num', 'holdbar', 'cagr']].head(3)}")
        else:
            print(f"❌ Error: DataFrame is empty after extraction. Check your key paths in recursive_get.")

        return df_final

    except Exception as e:
        print(f"🔥 Critical Loading Error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def calculate_robustness(pivot_df: pd.DataFrame):
    """
    计算鲁棒性得分：当前格子与周围邻域的均值。
    能够有效识别“参数高原”，避开“参数孤岛”。
    """
    if pivot_df.empty:
        print("⚠️ Warning: Empty DataFrame passed to calculate_robustness.")
        return pd.DataFrame()

    data = pivot_df.values
    # 使用 3x3 窗口进行平滑处理
    try:
        smoothed = uniform_filter(data, size=3, mode='constant', cval=np.nanmin(data))
        return pd.DataFrame(smoothed, index=pivot_df.index, columns=pivot_df.columns)
    except ValueError as e:
        print(f"⚠️ Error in calculate_robustness: {e}")
        return pd.DataFrame()

def plot_enhanced_heatmaps(df: pd.DataFrame, metric: str, outdir: Path, period: str = 'short'):
    """
    为每个指标生成三个维度的热力图：Mean, Max, Robustness
    
    Args:
        df: 数据 DataFrame
        metric: 指标名称 ('cagr', 'calmar', 'long_pnl', 'long_win_rate', 'short_pnl', 'short_win_rate')
        outdir: 输出目录
        period: 'short' 或 'long'
    """
    # 准备三种聚合数据
    p_mean = df.groupby(["predict_num", "holdbar"])[metric].mean().unstack()
    p_max  = df.groupby(["predict_num", "holdbar"])[metric].max().unstack()
    p_rob  = calculate_robustness(p_mean)

    titles = {
        "mean": f"{metric.upper()} - Mean (Average Performance)",
        "max": f"{metric.upper()} - Max (Best Case Scenario)",
        "robust": f"{metric.upper()} - Robustness (Neighborhood Mean)"
    }
    
    # 创建三栏大图
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    datasets = [p_mean, p_max, p_rob]
    sub_types = ["mean", "max", "robust"]

    for ax, data, stype in zip(axes, datasets, sub_types):
        sns.heatmap(data, cmap="RdYlGn", annot=True, fmt=".2f", ax=ax, 
                    linewidths=0.5, cbar_kws={'shrink': 0.8})
        ax.set_title(titles[stype], fontsize=14, fontweight='bold')
        
        # 标注该图中的最大值位置
        max_idx = np.unravel_index(np.nanargmax(data.values), data.shape)
        ax.add_patch(plt.Rectangle((max_idx[1], max_idx[0]), 1, 1, 
                                   fill=False, edgecolor='blue', lw=3, ls='--'))

    plt.suptitle(f"Detailed Sensitivity Analysis ({period.upper()}): {metric.upper()}", fontsize=20, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"detailed_analysis_{period}_{metric}.png"), dpi=150, bbox_inches="tight")
    plt.close()

def save_detailed_stats(df: pd.DataFrame, outdir: Path, period: str = 'short'):
    """生成更详细的统计 CSV
    
    Args:
        df: 数据 DataFrame
        outdir: 输出目录
        period: 'short' 或 'long'
    """
    for metric in ["cagr", "calmar"]:
        pivot = df.groupby(["predict_num", "holdbar"])[metric].mean().unstack()
        
        stats = pd.DataFrame(index=pivot.index)
        stats["mean"] = pivot.mean(axis=1)
        stats["max"]  = pivot.max(axis=1)
        stats["min"]  = pivot.min(axis=1)
        stats["std"]  = pivot.std(axis=1)
        
        # 稳定性系数：变异系数的倒数，越高越稳定
        stats["stability_score"] = (stats["mean"].abs() / (stats["std"] + EPS))
        
        # 找到最佳参数配套
        stats["best_candle_config"] = pivot.idxmax(axis=1)
        
        stats.sort_values("mean", ascending=False).to_csv(os.path.join(outdir, f"comprehensive_stats_{period}_{metric}.csv"))

def main():
    report_path = r"/home/chao/work/quant_output/batch_experiments/2026-02-12/DOGEUSDT_15m/10_29_58/reports.jsonl"

    path = Path(report_path).resolve()
    outdir = os.path.join(path.parent, 'heatmaps')
    os.makedirs(outdir, exist_ok=True)

    # 处理 'short' 和 'long' 两个周期
    for period in ['short', 'long']:
        print(f"\n📊 Processing {period.upper()} period...")
        df = load_and_process(path, period=period)
        if df.empty:
            print(f"⚠️  No data found for {period} period")
            continue

        # 1. 绘制增强型热力图
        for m in ["cagr", "calmar", "long_pnl", "long_win_rate", "short_pnl", "short_win_rate"]:
            plot_enhanced_heatmaps(df, m, outdir, period=period)
        
        # 2. 保存详细统计
        save_detailed_stats(df, outdir, period=period)
        
        print(f"✨ {period.upper()} 期间的深度分析完成！")
    
    print(f"\n✨ 所有的深度分析图表已生成完毕！")
    print(f"📂 报告路径: {outdir}")
    print(f"💡 建议优先观察 'robust' 热力图中被蓝色虚线框出的区域，那是真正的参数‘高原’。")

if __name__ == "__main__":
    main()