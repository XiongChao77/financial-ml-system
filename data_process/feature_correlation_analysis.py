import pandas as pd
import numpy as np
import os,sys,datetime
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any
current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir,'..'))
import data_process.common as common
from model import data_loader
from data_process import preparation
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import concurrent.futures
import html,time
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import torch
from ignite.metrics import HSIC
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
HIGH_CORR_THRESHOLD = 0.90 # 设定冗余阈值
ANALYZE_TARGET = 'REL'  # ORIGIN: only analyze origin data, REL:only analyze rel data

def compute_hsic_ignite(x_data, y_data, device=None, max_samples=4000) -> float:
    """
    使用 PyTorch Ignite 计算 HSIC。
    包含强制降采样保护，防止 OOM 或 Tensor 过大错误。
    
    参数:
        max_samples: 最大样本数限制 (默认 5000)。
                     HSIC 是 O(N^2) 复杂度，超过 10000 会极慢且易爆显存。
    """
    if HSIC is None:
        print("[Warning] pytorch-ignite not installed.")
        return 0.0

    # 1. 自动确定设备
    device = torch.device('cuda')   #only run on GPU

    try:
        # --- 🛡️ 安全检查 1: 确保输入是 Numpy 数组以便处理 ---
        if hasattr(x_data, 'values'): x_data = x_data.values
        if hasattr(y_data, 'values'): y_data = y_data.values
        
        # 确保是 numpy array (处理 list 等情况)
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)

        # --- 🛡️ 安全检查 2: 强制降采样 (关键修复) ---
        # 如果数据量超过 max_samples，随机采样或取最后 N 个
        N = len(x_data)
        if N > max_samples:
            # 方案 A: 取最后 max_samples 个 (保留时间序列特性)
            # x_data = x_data[-max_samples:]
            # y_data = y_data[-max_samples:]
            
            # 方案 B: 随机采样 (更能代表整体分布，推荐)
            indices = np.random.choice(N, max_samples, replace=False)
            x_data = x_data[indices]
            y_data = y_data[indices]

        # 2. 转 Tensor 并移至设备
        t_x = torch.as_tensor(x_data, dtype=torch.float32, device=device).view(-1, 1)
        t_y = torch.as_tensor(y_data, dtype=torch.float32, device=device).view(-1, 1)

        # 3. 初始化与计算
        # sigma=-1 启用启发式搜索(自动计算核宽)
        metric = HSIC(sigma_x=-1, sigma_y=-1)
        metric.update((t_x, t_y))
        score = metric.compute()
        metric.reset()
        
        return float(score)

    except Exception as e:
        # 捕捉所有错误 (包括 OOM, Tensor too large 等)
        print(f"[HSIC Error] {str(e)[:10]}...") # 只打印前100字符避免刷屏
        return 0.0
    
    finally:
        # 4. 显存清理
        if device.type == 'cuda':
            torch.cuda.empty_cache()

# --- 2. 分析函数 (修改为返回字符串) ---
def analyze_correlation(
    df: pd.DataFrame,
    target_col: str = 'return_rate',
    use_hsic: bool = True,
    use_rf: bool = False
):
    """
    专业版因子相关性分析：
    - Pearson   衡量线性关系
    - Spearman  衡量单调关系
    - MI (regression)   Mutual Information - 非线性依赖强度.通常强特征的 MI 可能会达到 0.05 或 0.1 以上（视具体数据噪声而定）
    - HSIC (pyHSICLasso)    统计学意义上的非线性依赖检测”
    - RF importance (optional)      用于“判断哪些信息模型能吃掉”
    """

    # ==== 目标必须是 return_rate（连续） ====
    if target_col not in df.columns:
        raise RuntimeError(f"缺少连续收益列 {target_col}")

    # ==== 删除无关列 ====
    drop_cols = data_loader.DROP_FEATURES
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    df = df.dropna()
    if len(df) < 50:
        return {}, {}

    feature_cols = [c for c in df.columns if c != target_col]

    # ===================================================
    # 1. Pearson / Spearman
    # ===================================================
    corr_pearson = df[feature_cols + [target_col]].corr(method='pearson')[target_col].abs() # feature_cols + [target_col] for determinate order
    corr_spearman = df[feature_cols + [target_col]].corr(method='spearman')[target_col].abs()

    # ===================================================
    # 2. Mutual Information（reg）
    # ===================================================
    MAX_SAMPLES = 15000
    df_mi = df.iloc[-MAX_SAMPLES:]

    X_mi = df_mi[feature_cols].values
    y_mi = df_mi[target_col].values

    try:
        mi_scores = mutual_info_regression(X_mi, y_mi, random_state=42)
        mi_series = pd.Series(mi_scores, index=feature_cols)
    except Exception as e:
        print(f"[MI Warning] {e}")
        mi_series = pd.Series(0, index=feature_cols)

    # ===================================================
    # 3. HSIC with PyTorch Ignite (Fixed)
    # ===================================================
    # 【关键修改】使用与 MI 相同的采样数据 y_mi (来自 df.iloc[-MAX_SAMPLES:])
    # 这里的 y_mi 已经在上面定义了: y_mi = df_mi[target_col].values
    
    hsic_series = pd.Series(0.0, index=feature_cols)

    if use_hsic and torch.cuda.is_available():
        # 预先将 y 转为 tensor 并不是必须的，封装函数里会处理，
        # 但为了避免循环中重复转换 y，可以保持传入 numpy array 即可。
        
        for feat in feature_cols:
            # 【关键修改】使用采样后的 df_mi，而不是全量 df
            x_values = df_mi[feat].values 
            
            # 使用采样后的 y_mi
            score = compute_hsic_ignite(x_values, y_mi)
            
            hsic_series[feat] = score

    # ===================================================
    # 4. RandomForest Importance（optional）
    # ===================================================
    rf_series = pd.Series(0.0, index=feature_cols)

    if use_rf:
        try:
            rf = RandomForestRegressor(
                n_estimators=80,
                max_depth=5,
                random_state=123
            )
            rf.fit(X_mi, y_mi)
            rf_series = pd.Series(rf.feature_importances_, index=feature_cols)
        except Exception as e:
            print(f"[RF Warning] {e}")

    # ===================================================
    # 5. Assemble result dict
    # ===================================================
    corr_result = {}

    for feat in feature_cols:
        corr_result[f"{feat}_pearson"]  = corr_pearson.get(feat, 0)
        corr_result[f"{feat}_spearman"] = corr_spearman.get(feat, 0)
        corr_result[f"{feat}_mi"]       = mi_series.get(feat, 0)
        corr_result[f"{feat}_hsic"]     = hsic_series.get(feat, 0)
        corr_result[f"{feat}_rf"]       = rf_series.get(feat, 0)

    # ===================================================
    # 6. 冗余因子（Pearson 上三角）
    # ===================================================
    corr_matrix = df[feature_cols].corr(method='pearson')
    redundant_pairs = []

    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            val = corr_matrix.iloc[i, j]
            if abs(val) >= 0.90:
                redundant_pairs.append({
                    'Feature 1': feature_cols[i],
                    'Feature 2': feature_cols[j],
                    'Correlation': val
                })

    redundant_result = pd.DataFrame(redundant_pairs).to_dict() if redundant_pairs else {}

    return corr_result, redundant_result

# =======================================================
# 🌟 新增：并行执行函数
# =======================================================
def single_run_analysis(candlestick_num:int, predict_num:int, vm: float, mt: float, df: pd.DataFrame):
    """
    针对一组 (vm, mt) 参数执行完整的标签、相关性、归一化和分析流程。
    返回包含参数和两个分析结果字符串的字典。
    """
    print(f"candlestick_num:{candlestick_num},predict_num:{predict_num},single_run_analysis")
    # --- 1. 标签计算 (使用传入的 vm, mt) ---
    try:
        common.attach_label(df, candlestick_num, predict_num, 
                                       vol_multiplier=vm, min_threshold=mt, keep_rate = True)
        
        # 🌟 新增：计算标签比例 🌟
        # normalize=True 会返回百分比 (0.0 ~ 1.0)
        label_counts = df['label'].value_counts(normalize=True)
        
        # 使用 .get() 防止某些极端情况下某一类完全没出现
        ratios = {
            'ratio_0': label_counts.get(0, 0.0), # 跌/空
            'ratio_1': label_counts.get(1, 0.0), # 盘整/无操作
            'ratio_2': label_counts.get(2, 0.0)  # 涨/多
        }

    except AssertionError as e:
        raise RuntimeError(f"ERROR in attach_label for vm={vm}, mt={mt}: {e}")
    
    # --- 2. 原始特征分析 (Case 1) ---
    analysis_result_raw =  {}
    if ANALYZE_TARGET == 'ORIGIN':
        # df 包含了标签，可以直接进行相关性分析
        analysis_result_raw = analyze_correlation(df.copy())

    # --- 3. 归一化和归一化特征分析 (Case 2) ---
    analysis_result_reg = {}
    if ANALYZE_TARGET == 'REL':
        print(f"candlestick_num:{candlestick_num},predict_num:{predict_num},REL")
        feat_cols = [col for col in df.columns]
        # **注意：TimeSeriesWindowDataset 内部处理 NaN，这里使用 df 的副本**
        full_ds = data_loader.TimeSeriesWindowDataset(
            df, 
            feature_cols=feat_cols, 
            label_col='label', 
            window=candlestick_num
        )
        print(f"candlestick_num:{candlestick_num},predict_num:{predict_num},TimeSeriesWindowDataset")
        # 提取最后一个时间步 X3d[M, T, F] -> X_np[M, F]
        X_np = full_ds.X[:, -1, :].numpy() 
        
        # 创建新的 DataFrame
        df_scaled = pd.DataFrame(X_np, columns=full_ds.feature_names)
        df_scaled['label'] = full_ds.y.numpy()
        
        analysis_result_reg = analyze_correlation(df_scaled)
        print(f"candlestick_num:{candlestick_num},predict_num:{predict_num},analyze_correlation")

    # 返回结构化结果，包含比例
    return {
        'candlestick_num': candlestick_num,
        'predict_num': predict_num,
        'vm': vm,
        'mt': mt,
        'raw': analysis_result_raw,
        'reg': analysis_result_reg,
        'ratio_0': ratios['ratio_0'],  # <--- 新增
        'ratio_1': ratios['ratio_1'],  # <--- 新增
        'ratio_2': ratios['ratio_2']   # <--- 新增
    }

def _unpack_and_run(task_tuple):
    """
    辅助函数，用于 ProcessPoolExecutor.map()。
    它接收一个任务元组 (vm, mt, df)，并将其解包传递给 single_run_analysis。
    """
    candlestick_num, predict_num, vm, mt, df = task_tuple
    return single_run_analysis(candlestick_num, predict_num, vm, mt, df)

def main():
    print("--- Correlation Analysis Utility ---")
    start_time = time.time()
    output_dir= os.path.join(common.PROJECT_DIR, 'correlation_result')
    if not os.path.exists(output_dir):  os.makedirs(output_dir)
    vm_range = np.arange(0.4, 1.51, 0.1)
    mt_range = np.arange(0.003, 0.0121, 0.001)
    candlestick_num_range = np.arange(64, 160, 16)
    period_list = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"] #['12h', '1d']
    # 🌟 优化点 1: 预计算技术指标 (TA) - 只执行一次
    all_period_results = {}
    sorted_results_list = []
    index = 0
    for period in period_list:
        evaluate_file = os.path.join(os.path.dirname(common.origin_data_path), f"BTCUSDT_{period}.csv" )
        df_base = pd.read_csv(evaluate_file)
        common.attach_attr(df_base,common.FEATURE_CONFIG_LIST, common.load_interval_ms())
        tasks = []
        for candlestick_num in candlestick_num_range:
            for predict_num in [candlestick_num//8, candlestick_num//6, candlestick_num//4]:
            # --- 1. 准备任务参数列表 ---
            # for vm in vm_range:
            #     for mt in mt_range:
            #         # 任务参数：(candlestick_num,vm, mt, df)
            #         # tasks.append((candlestick_num,vm, mt, df))
            #         pass
                result =  _unpack_and_run((candlestick_num, predict_num, 0, 0, df_base.copy()))
                tasks.append((candlestick_num, predict_num, 0, 0, df_base.copy()))
        all_results = [] # 存储所有任务返回的字典
        
        # --- 2. 使用 ProcessPoolExecutor 进行并行计算 ---
        num_processes = 8 #avoid cuda error #os.cpu_count() if os.cpu_count() else 4 # 安全获取核心数
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
        sorted_results = sorted(all_results, key=lambda x: (x['candlestick_num']))
        sorted_results_list.append([period,sorted_results])
    perform_statistical_analysis(sorted_results_list)
    # -----------------------------------------------------
    # 🌟 计时结束并打印 🌟
    # -----------------------------------------------------
    end_time = time.time()
    end_datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_taken = end_time - start_time
    
    print("-" * 50)
    print(f"Analysis Completed!")
    print(f"End Time: {end_datetime_str}")
    # 使用 round() 格式化时间
    print(f"Total Time Taken: {time_taken:.2f} seconds ({round(time_taken / 60, 2)} minutes)")


#pea (Pearson Correlation - 皮尔逊相关系数)  线性关系
#spe (Spearman Correlation - 斯皮尔曼等级相关) 单调关系（有方向性关联）
#mi (Mutual Information - 互信息)  通常强特征的 MI 可能会达到 0.05 或 0.1 以上（视具体数据噪声而定）
def perform_statistical_analysis(results_list: list[str, Dict]):
    print("\n" + "="*50)
    print("🚀 开始统计分析 (Statistical Analysis)")
    print("="*50)
    data_key = 'raw' if ANALYZE_TARGET == 'ORIGIN' else 'reg'
    # --- 1. 数据扁平化 ---
    rows = []
    all_redundant_data = []
    for period,results in results_list:
        for res in results:
            # 跳过无结果或出错的项
            if not res[data_key] or isinstance(res[data_key], str): 
                continue 
            
            # 获取 correlation_dict (假设 analyze_correlation 返回 (corr_dict, redundant_dict))
            # 注意：请确保 analyze_correlation 返回的是元组，且第一项是字典
            corr_data = res[data_key][0]
            redundant_data = res[data_key][1]
            
            row = {
                'candlestick_num': res['candlestick_num'],
                'predict_num': res['predict_num'],
                'vm': res['vm'],
                'mt': res['mt'],
                # 🌟 新增：提取标签比例
                'ratio_0': res.get('ratio_0', 0.0),
                'ratio_1': res.get('ratio_1', 0.0),
                'ratio_2': res.get('ratio_2', 0.0),
            }

            config = {
                'period': period,
                'candlestick_num': res['candlestick_num'],
                'predict_num': res['predict_num'],
                'vm': res['vm'],
                'mt': res['mt'],
            }
            # --- 🌟 [新增] 冗余数据处理 ---
            if redundant_data:
                # 将字典形式 {'Feature 1': {0: 'A', 1: 'C'}, ...} 转换回 DataFrame
                df_redundant = pd.DataFrame(redundant_data) 
                
                # 为每一行冗余数据添加配置参数
                for key, value in config.items():
                    df_redundant[key] = value
                    
                all_redundant_data.append(df_redundant)

            row.update(corr_data)
            rows.append(row)

        if not rows:
            print("❌ 没有有效的结果数据。")
            return

        df_master = pd.DataFrame(rows)
        
        # 定义列组
        param_cols = ['candlestick_num', 'predict_num','vm', 'mt']
        ratio_cols = ['ratio_0', 'ratio_1', 'ratio_2']
        # 特征列是排除参数和比例列之后的列
        feature_cols = [c for c in df_master.columns if c not in param_cols and c not in ratio_cols]

        # --- 2. 特征重要性分析 (不变) ---
        print(f"\n📊 [1] 特征表现排行 (Top Features)")
        feature_stats = df_master[feature_cols].agg(['mean', 'max', 'std']).T
        feature_stats = feature_stats.sort_values(by='max', ascending=False)
        print(feature_stats.head(5).to_string())

        # --- 3. 最佳参数组合 (包含标签分布) ---
        print(f"\n🏆 [2] 最佳参数组合 (按 Top 3 特征相关性排序)")
        print("-" * 50)
        
        # 计算得分为前3个特征相关性的均值
        # 建议修改：只基于 MI (互信息) 计算得分，因为这是最“真实”的信息量
        mi_cols = [c for c in df_master.columns if c.endswith('_mi')]
        if mi_cols:
            df_master['score_mean_mi_top3'] = df_master[mi_cols].apply(lambda x: x.nlargest(3).mean(), axis=1)
            # 按 MI 得分排序
            df_master = df_master.sort_values(by='score_mean_mi_top3', ascending=False)
        df_master['score_max'] = df_master[feature_cols].max(axis=1)
        
        # 保存文件 (现在包含了 ratio 列)
        output_path = os.path.join(common.PROJECT_DIR, 'correlation_result', f'analysis_data_{period}_{data_key}.csv')
        df_master.to_csv(output_path, index=False)

    # --- 🌟 [新增] 保存冗余因子数据 ---
    if all_redundant_data:
        # 使用 pd.concat 合并所有任务的冗余数据
        df_redundant_master = pd.concat(all_redundant_data, ignore_index=True)
        redundant_output_path = os.path.join(common.PROJECT_DIR, 'correlation_result', f'redundancy_analysis_{data_key}.csv')
        
        # 使用 float_format 确保相关性数值精度
        df_redundant_master.to_csv(redundant_output_path, index=False, float_format='%.6f')
        print(f"\n✅ 冗余因子对数据已保存至: {redundant_output_path}")
    else:
        print("\nℹ️ 未发现高冗余度特征对 (相关系数 >= 0.90)。")
    print(f"\n✅ 完整数据 (含标签分布) 已保存至: {output_path}")

# --- 4. 执行 ---
if __name__ == "__main__":
    main()