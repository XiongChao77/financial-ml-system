import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from scipy.stats import norm

# 确保 common.py 在路径中
import common 

class LabelRegimeAnalyzer:
    def __init__(self, df, interval_ms, symbol="ETHUSDT", interval = '5m'):
        self.df = df
        self.interval_ms = interval_ms
        self.symbol = symbol
        self.interval = interval
        self.results_df = None

    def run_parameter_sweep(self, vol_range, stop_range):
        """
        第一阶段：扫描参数空间，记录原始诱导分布向量 [P_short, P_neutral, P_long]
        """
        sweep_data = []
        print(f"🚀 正在探测参数空间分布 (Symbol: {self.symbol})...")
        
        for vol in tqdm(vol_range, desc="Vol Steps"):
            for stop in stop_range:
                temp_df = self.df.copy()
                # 调用 common.py 中的三重屏障逻辑
                temp_df = common.attach_triple_barrier_label(
                    temp_df, 
                    interval_ms=self.interval_ms,
                    vol_mult_long=vol,
                    stop_rate_long=stop,
                    vol_mult_short=vol,
                    stop_rate_short=stop,
                )
                
                counts = temp_df['label'].value_counts(normalize=True).to_dict()
                # 显式映射：0=Short, 1=Neutral, 2=Long
                dist_vec = [counts.get(0, 0.0), counts.get(1, 0.0), counts.get(2, 0.0)]
                
                sweep_data.append({
                    "vol_multiplier": vol,
                    "stop_rate": stop,
                    "p_short": dist_vec[0],
                    "p_neutral": dist_vec[1],
                    "p_long": dist_vec[2]
                })
        
        self.results_df = pd.DataFrame(sweep_data)
        return self.results_df

    def analyze_and_plot(self, output_dir="regime_discovery_detailed"):
        """
        核心逻辑：在对数空间同时计算并输出一阶（灵敏度）与二阶（加速度/符号）分布分析。
        一阶展示变化的剧烈程度，二阶展示变化的趋势方向与非均匀性。
        """
        output_dir = os.path.join(output_dir, f"{self.symbol}_{self.interval}")
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        vols = np.array(sorted(self.results_df['vol_multiplier'].unique()))
        stops = np.array(sorted(self.results_df['stop_rate'].unique()))
        
        # 1. 构建分布张量 [V, S, Class]
        dist_tensor = np.zeros((len(vols), len(stops), 3))
        for i, v in enumerate(vols):
            for j, s in enumerate(stops):
                row = self.results_df[(self.results_df['vol_multiplier'] == v) & 
                                      (self.results_df['stop_rate'] == s)].iloc[0]
                dist_tensor[i, j] = [row['p_short'], row['p_neutral'], row['p_long']]

        # --- 核心计算：对数空间的梯度 ---
        log_vols = np.log(vols)
        log_stops = np.log(stops)

        # A. 计算一阶偏导与灵敏度 (1st Order - Jacobian)
        grad_v = np.gradient(dist_tensor, log_vols, axis=0) 
        grad_s = np.gradient(dist_tensor, log_stops, axis=1)
        
        sens_short   = np.sqrt(grad_v[..., 0]**2 + grad_s[..., 0]**2)
        sens_neutral = np.sqrt(grad_v[..., 1]**2 + grad_s[..., 1]**2)
        sens_long    = np.sqrt(grad_v[..., 2]**2 + grad_s[..., 2]**2)
        total_sens   = np.sqrt(np.sum(grad_v**2 + grad_s**2, axis=2))

        # B. 计算二阶偏导与拉普拉斯算子 (2nd Order - Laplacian Signed)
        grad_vv = np.gradient(grad_v, log_vols, axis=0)
        grad_ss = np.gradient(grad_s, log_stops, axis=1)
        
        # 每一类的带符号拉普拉斯算子: $\nabla^2 P = \frac{\partial^2 P}{\partial (\ln v)^2} + \frac{\partial^2 P}{\partial (\ln s)^2}$
        lap_short   = grad_vv[..., 0] + grad_ss[..., 0]
        lap_neutral = grad_vv[..., 1] + grad_ss[..., 1]
        lap_long    = grad_vv[..., 2] + grad_ss[..., 2]
        # 总体曲率模长 (用于识别非均匀性总强度)
        total_curvature = np.sqrt(np.sum(grad_vv**2 + grad_ss**2, axis=2))

        # C. 偏移偏斜度 (Skewness)
        eps = 1e-9
        skewness = sens_long / (sens_long + sens_short + eps)

        # --- 绘图 1: 三类标签的一阶灵敏度 (dV/dS 综合变化) ---
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        titles_1st = [(sens_short, "Short Sensitivity (1st Order)", "Reds"),
                      (sens_neutral, "Neutral Sensitivity (1st Order)", "Greys"),
                      (sens_long, "Long Sensitivity (1st Order)", "Greens")]
        for idx, (data, title, cmap) in enumerate(titles_1st):
            sns.heatmap(pd.DataFrame(data, index=vols, columns=stops), 
                        ax=axes[idx], annot=True, cmap=cmap, fmt=".3f")
            axes[idx].set_title(title)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/split_variations_1st.png")
        plt.close()

        # --- 绘图 2: 三类标签的二阶带符号加速度 (Laplacian) ---
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        titles_2nd = [(lap_short, "Short Acceleration (Signed 2nd)", "RdBu_r"),
                      (lap_neutral, "Neutral Acceleration (Signed 2nd)", "RdBu_r"),
                      (lap_long, "Long Acceleration (Signed 2nd)", "RdBu_r")]
        for idx, (data, title, cmap) in enumerate(titles_2nd):
            # center=0 确保红色代表正(加速)，蓝色代表负(减速)
            sns.heatmap(pd.DataFrame(data, index=vols, columns=stops), 
                        ax=axes[idx], annot=True, cmap=cmap, center=0, fmt=".3f")
            axes[idx].set_title(title)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/split_variations_signed_2nd.png")
        plt.close()

        # --- 绘图 3: 总体一阶 vs 二阶对比 ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        sns.heatmap(pd.DataFrame(total_sens, index=vols, columns=stops), ax=ax1, annot=True, cmap="magma", fmt=".3f")
        ax1.set_title("Total 1st Order Sensitivity (The Rate of Change)")
        sns.heatmap(pd.DataFrame(total_curvature, index=vols, columns=stops), ax=ax2, annot=True, cmap="viridis", fmt=".3f")
        ax2.set_title("Total 2nd Order Curvature (The Non-uniformity)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/overall_1st_vs_2nd.png")
        plt.close()

        print(f"✅ 深度梯度分析已完成！")
        print(f"📍 一阶分类变化 (1x3): {output_dir}/split_variations_1st.png")
        print(f"📍 二阶带符号变化 (1x3): {output_dir}/split_variations_signed_2nd.png")
        print(f"📍 总体一二阶对比: {output_dir}/overall_1st_vs_2nd.png")

    def plot_null_hypothesis_comparison(self, output_dir="regime_discovery_detailed"):
        """
        新增：将实际分布与均匀分布、高斯分布零假设进行对比。
        """
        output_dir = os.path.join(output_dir, f"{self.symbol}_{self.interval}")
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        vols = sorted(self.results_df['vol_multiplier'].unique())
        # 我们选取 Stop Rate 的中位数进行横向对比
        mid_stop = sorted(self.results_df['stop_rate'].unique())[len(self.results_df['stop_rate'].unique())//2]
        
        subset = self.results_df[self.results_df['stop_rate'] == mid_stop].sort_values('vol_multiplier')

        # 1. 计算高斯零假设比例
        # 在随机高斯世界中，阈值 k 对应的概率为：
        # P(Long) = 1 - Phi(k), P(Short) = Phi(-k), P(Neutral) = Phi(k) - Phi(-k)
        gaussian_nulls = []
        for v in vols:
            p_long = 1 - norm.cdf(v)
            p_short = norm.cdf(-v)
            p_neutral = 1 - (p_long + p_short)
            gaussian_nulls.append([p_short, p_neutral, p_long])
        gaussian_nulls = np.array(gaussian_nulls)

        # 2. 绘图对比
        fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
        classes = ['p_short', 'p_neutral', 'p_long']
        titles = ['Short Proportion', 'Neutral Proportion', 'Long Proportion']
        colors = ['#ff4d4d', '#7f8c8d', '#2ecc71']

        for i, (cls, title) in enumerate(zip(classes, titles)):
            ax = axes[i]
            # 实际分布
            ax.plot(vols, subset[cls], 'o-', label='Actual Distribution', color=colors[i], linewidth=3)
            # 高斯零假设
            ax.plot(vols, gaussian_nulls[:, i], '--', label='Gaussian Null ($r \sim N(0,\sigma^2)$)', color='black', alpha=0.6)
            # 均匀零假设
            ax.axhline(1/3, color='blue', linestyle=':', label='Uniform Null (1/3)', alpha=0.4)
            
            ax.set_title(f"{title} vs Null Hypotheses\n(at Stop Rate: {mid_stop})")
            ax.set_xlabel("Volatility Multiplier")
            if i == 0: ax.set_ylabel("Probability")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/null_hypothesis_comparison.png")
        plt.close()

        # 3. 计算“偏离结构”：KL 散度 (Kullback-Leibler Divergence)
        # 衡量实际分布偏离高斯随机世界的程度
        kl_divs = []
        for i, v in enumerate(vols):
            actual = subset.iloc[i][classes].values.astype(float) + 1e-9
            target = gaussian_nulls[i] + 1e-9
            kl = np.sum(actual * np.log(actual / target))
            kl_divs.append(kl)

        plt.figure(figsize=(10, 6))
        plt.plot(vols, kl_divs, 's-', color='purple', linewidth=2)
        plt.title(f"{self.symbol} - Distribution 'Information Gain' over Gaussian Null\n(Higher = More Non-random Structure)")
        plt.xlabel("Volatility Multiplier")
        plt.ylabel("KL Divergence")
        plt.savefig(f"{output_dir}/information_structure_gain.png")
        plt.close()

        print(f"✅ 零假设对比完成！")
        print(f"📍 趋势对比图: {output_dir}/null_hypothesis_comparison.png")
        print(f"📍 结构增益图: {output_dir}/information_structure_gain.png")

# --- 运行示例 ---
if __name__ == "__main__":
    import logging
    # 强制读取数据并清理
    df_raw = pd.read_csv(common.origin_data_path)
    df_clean = common.clean_data_quality_auto(df_raw, logging.getLogger('dummy'))
    
    # 自动获取周期毫秒数
    try:
        from common import load_interval_ms
        interval_ms = load_interval_ms()
    except:
        interval_ms = 300000 # 兜底 5m
    
    analyzer = LabelRegimeAnalyzer(df_clean, interval_ms, symbol=common.symbol, interval=common.interval)
    
    # 定义研究步长
    vols = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
    stops = np.array([0.3, 0.5, 0.6, 0.7, 0.8, 1.0])
    
    analyzer.run_parameter_sweep(vols, stops)
    analyzer.analyze_and_plot()