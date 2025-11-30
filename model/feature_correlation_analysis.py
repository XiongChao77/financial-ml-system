import pandas as pd
import numpy as np
import os,sys,datetime
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any
current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir,'..'))
import data_process.common as common
import data_loader
import data_process.preparation as preparation
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import concurrent.futures

# 1. 设置中文字体 (优先使用常见的黑体)
# 在 Windows/Anaconda 环境中，'SimHei' (黑体) 或 'Microsoft YaHei' (微软雅黑) 通常是可用的。
# 如果您在 Linux/Mac 上，可以使用 'Heiti TC' 或 'WenQuanYi Zen Hei'。
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False # 解决负号 '-' 显示为方块的问题

# 2. 清除 Matplotlib 字体缓存 (非常关键！)
# Matplotlib 会缓存字体设置，需要手动删除缓存才能识别新的配置。
try:
    fm._rebuild()
except:
    pass

# --- 1. 配置 ---
# 请确保在运行此脚本前，已执行 preparation.py 生成 train_data.csv
# 假设此脚本从项目根目录运行
RELATIVE_DATA_PATH = 'data_process/output/train_data.csv'
HIGH_CORR_THRESHOLD = 0.90 # 设定冗余阈值
LOG_FILE = 'correlation_analysis_log.txt'
REL_LOG_FILE = 'rel_correlation_analysis_log.txt'
# --- 2. 分析函数 (修改为返回字符串) ---
def analyze_correlation(df: pd.DataFrame, target_column: str = 'label',visualization = True) -> str:
    """计算特征相关性，将结果格式化为字符串返回。"""

    df.drop(columns=[f for f in data_loader.DROP_FEATURES if f != target_column], inplace=True, errors='ignore')

    if target_column not in df.columns:
        return f"错误: 目标列 '{target_column}' 不存在。"

    output_str = f"Analysis Run Time: {datetime.datetime.now()}\n"
    output_str += "=" * 50 + "\n"
    
    # --- A. 特征-目标相关性 (Label Importance) ---
    output_str += "\n--- 1. 特征与目标标签的相关性 (Label Importance) ---\n"
    target_corr = df.corr()[target_column].abs().sort_values(ascending=False)
    target_corr = target_corr.drop(target_column, errors='ignore')

    output_str += "最相关特征（绝对值）：\n"
    output_str += target_corr.to_string() + "\n"
    
    # --- B. 特征-特征相关性 (Redundancy Check) ---
    output_str += "\n--- 2. 特征冗余分析 (Redundancy Check) ---\n"
    
    corr_matrix = df.drop(columns=[target_column], errors='ignore').corr()
    
    redundant_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= HIGH_CORR_THRESHOLD:
                redundant_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })

    if redundant_pairs:
        output_str += f"\n高度冗余特征对 (相关系数 >= {HIGH_CORR_THRESHOLD:.2f}):\n"
        df_redundant = pd.DataFrame(redundant_pairs)
        output_str += df_redundant.sort_values(by='Correlation', ascending=False).to_string() + "\n"
    else:
        output_str += f"未找到相关系数 >= {HIGH_CORR_THRESHOLD:.2f} 的特征对。\n"

    if visualization:
        # --- C. 可视化 (Heatmap) ---
        plt.figure(figsize=(16, 12))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", cbar_kws={'label': '相关系数'})
        plt.title('特征相关性矩阵 (排除 Target)')
        plt.tight_layout()
        plt.savefig('feature_correlation_heatmap.png')
        output_str += "\n--- 3. 可视化 ---\n"
        output_str += "相关性热力图已保存为 'feature_correlation_heatmap.png'。\n"
    
    return output_str

# --- 3. 新增日志写入函数 ---
def log_analysis_to_file(analysis_content: str, path):
    """将分析结果和配置参数追加写入文件。"""
    
    config_info = "\n" + "="*20 + " Configuration " + "="*20 + "\n"
    config_info += "=" * 50 + "\n"
    
    # 将配置信息添加到分析内容前面
    full_log = config_info + analysis_content
    
    try:
        with open(path , 'w', encoding='utf-8') as f:
            f.write(full_log)
        print(f"\n[Success] Analysis results appended to {LOG_FILE}")
        
    except Exception as e:
        print(f"[Error] Failed to write to log file: {e}")

# =======================================================
# 🌟 新增：并行执行函数
# =======================================================
def single_run_analysis(vm: float, mt: float, df_ta_only: pd.DataFrame):
    """
    针对一组 (vm, mt) 参数执行完整的标签、相关性、归一化和分析流程。
    返回包含参数和两个分析结果字符串的字典。
    """
    
    # --- 1. 标签计算 (使用传入的 vm, mt) ---
    # 假设 common.attach_label(df, vol_multiplier, min_threshold) 已修改
    try:
        df_label = common.attach_label(df_ta_only.copy(), vol_multiplier=vm, min_threshold=mt)
    except AssertionError as e:
        # 如果 attach_label 内部断言失败（如数据太少），记录错误
        error_msg = f"ERROR in attach_label for vm={vm}, mt={mt}: {e}"
        return {'vm': vm, 'mt': mt, 'raw': error_msg, 'reg': error_msg}


    # --- 2. 原始特征分析 (Case 1) ---
    analysis_result_raw = f"Labeling Vol Multiplier (vol_multiplier): {vm},Min Threshold (min_threshold): {mt}\n"
    # df_label 包含了标签，可以直接进行相关性分析
    analysis_result_raw += analyze_correlation(df_label.copy(), visualization=False)


    # --- 3. 归一化和归一化特征分析 (Case 2) ---
    analysis_result_reg = f"Labeling Vol Multiplier (vol_multiplier): {vm},Min Threshold (min_threshold): {mt}\n"
    
    feat_cols = [col for col in df_label.columns]
    # **注意：TimeSeriesWindowDataset 内部处理 NaN，这里使用 df_label 的副本**
    full_ds = data_loader.TimeSeriesWindowDataset(
        df=df_label.copy(), 
        feature_cols=feat_cols, 
        label_col='label', 
        window=common.candlestick_num
    )
    
    # 提取最后一个时间步 X3d[M, T, F] -> X_np[M, F]
    X_np = full_ds.X[:, -1, :].numpy() 
    
    # 创建新的 DataFrame
    df_scaled = pd.DataFrame(X_np, columns=full_ds.feature_names)
    df_scaled['label'] = full_ds.y.numpy()
    
    analysis_result_reg += analyze_correlation(df_scaled, visualization=False)
    
    # 返回结构化结果
    return {
        'vm': vm,
        'mt': mt,
        'raw': analysis_result_raw,
        'reg': analysis_result_reg
    }

def _unpack_and_run(task_tuple):
    """
    辅助函数，用于 ProcessPoolExecutor.map()。
    它接收一个任务元组 (vm, mt, df_ta_only)，并将其解包传递给 single_run_analysis。
    """
    vm, mt, df_ta_only = task_tuple
    return single_run_analysis(vm, mt, df_ta_only)

def main():
    print("--- Correlation Analysis Utility ---")
    df_base = pd.read_csv(common.origin_data_path)
    vm_range = np.arange(0.4, 1.51, 0.1)
    mt_range = np.arange(0.003, 0.0121, 0.001)
    
    # 🌟 优化点 1: 预计算技术指标 (TA) - 只执行一次
    # attach_attr 假定只计算 TA，不计算标签，且不依赖全局 vm/mt 变量
    df_ta_only = common.attach_attr(df_base.copy())

    # --- 1. 准备任务参数列表 ---
    tasks = []
    for vm in vm_range:
        for mt in mt_range:
            # 任务参数：(vm, mt, df_ta_only)
            tasks.append((vm, mt, df_ta_only))
    
    all_results = [] # 存储所有任务返回的字典
    
    # --- 2. 使用 ProcessPoolExecutor 进行并行计算 ---
    num_processes = os.cpu_count() if os.cpu_count() else 4 # 安全获取核心数
    print(f"Starting parallel analysis using {num_processes} processes for {len(tasks)} tasks...")

    # ProcessPoolExecutor 适用于计算密集型任务，实现真正的并行
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        # executor.map 传递参数，并返回一个结果迭代器
        results_iterator = executor.map(_unpack_and_run, tasks)
        
        # 收集结果
        for result_dict in results_iterator:
            all_results.append(result_dict)

    # --- 3. 排序是关键！确保日志顺序正确 ---
    # 按照 vm (主排序键) 和 mt (次排序键) 对结果进行排序
    sorted_results = sorted(all_results, key=lambda x: (x['vm'], x['mt']))

    # --- 4. 合并结果 ---
    final_analysis_result = ''.join([r['raw'] for r in sorted_results])
    final_analysis_result_reg = ''.join([r['reg'] for r in sorted_results])

    # --- 5. 写入文件 ---
    log_analysis_to_file(final_analysis_result, os.path.join(common.PROJECT_DIR, LOG_FILE))
    log_analysis_to_file(final_analysis_result_reg, os.path.join(common.PROJECT_DIR, REL_LOG_FILE))


# --- 4. 执行 ---
if __name__ == "__main__":
    main()