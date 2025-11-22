import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any

# --- 1. 配置 ---
# 请确保在运行此脚本前，已执行 preparation.py 生成 train_data.csv
# 假设此脚本从项目根目录运行
RELATIVE_DATA_PATH = 'data_process/output/train_data.csv'
HIGH_CORR_THRESHOLD = 0.90 # 设定冗余阈值

# --- 2. 分析函数 ---
def analyze_correlation(data_path: str, target_column: str = 'label'):
    """计算特征相关性，打印重要性，并生成热力图。"""
    
    if not os.path.exists(data_path):
        print(f"错误: 训练数据文件未找到: {data_path}. 请先运行 preparation.py。")
        return

    # 加载数据
    df = pd.read_csv(data_path)
    
    # --- A. 特征-目标相关性 (Signal Quality Check) ---
    print("\n--- 1. 特征与目标标签的相关性 (Label Importance) ---")
    
    if target_column not in df.columns:
        print(f"错误: 目标列 '{target_column}' 不存在。")
        return

    # 计算与 Label 的相关性（取绝对值），并降序排列
    target_corr = df.corr()[target_column].abs().sort_values(ascending=False)
    
    # 排除 Label 自身
    if target_column in target_corr:
        target_corr = target_corr.drop(target_column)

    print("Top 10 最相关特征（绝对值）：")
    print(target_corr.head(10).to_string())
    
    # --- B. 特征-特征相关性 (Redundancy Check) ---
    print("\n--- 2. 特征冗余分析 (Redundancy Check) ---")
    
    # 排除 Target 列后计算特征间的相关性矩阵
    corr_matrix = df.drop(columns=[target_column], errors='ignore').corr()
    
    redundant_pairs = []
    
    # 遍历矩阵的上三角部分，查找高相关性特征对
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= HIGH_CORR_THRESHOLD:
                redundant_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })

    if redundant_pairs:
        print(f"\n高度冗余特征对 (相关系数 >= {HIGH_CORR_THRESHOLD:.2f}):")
        df_redundant = pd.DataFrame(redundant_pairs)
        print(df_redundant.sort_values(by='Correlation', ascending=False).to_string())
    else:
        print(f"未找到相关系数 >= {HIGH_CORR_THRESHOLD:.2f} 的特征对。")

    # --- C. 可视化 (Heatmap) ---
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", cbar_kws={'label': '相关系数'})
    plt.title('特征相关性矩阵 (排除 Target)')
    plt.tight_layout()
    plt.savefig('feature_correlation_heatmap.png')
    print("\n--- 3. 可视化 ---")
    print("相关性热力图已保存为 'feature_correlation_heatmap.png'。")

# --- 3. 执行 ---
if __name__ == "__main__":
    
    print("--- Correlation Analysis Utility ---")
    
    try:
        # 假设脚本在项目根目录运行
        TRAIN_DATA_FILE = os.path.join(os.getcwd(), RELATIVE_DATA_PATH)
        analyze_correlation(TRAIN_DATA_FILE)
    except Exception as e:
        print(f"在运行相关性分析时发生错误: {e}")
        print("请检查 RELATIVE_DATA_PATH 是否正确指向 train_data.csv 文件。")