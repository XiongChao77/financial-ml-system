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
    def __init__(self, df, interval_ms, symbol="ETHUSDT", interval='5m', output_dir=common.PERSISTENCE_DIR):
        self.df = df
        self.interval_ms = interval_ms
        self.symbol = symbol
        self.interval = interval
        # 统一输出路径
        self.output_dir = os.path.join(output_dir, "regime_discovery_output", f"{symbol}_{interval}")
        if not os.path.exists(self.output_dir): 
            os.makedirs(self.output_dir)
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
                para = common.CommonDefine()
                # 显式进行 round 处理，防止浮点数精度问题导致的 1.49999
                para.vol_multiplier_long = round(float(vol), 2)
                para.stop_multiplier_rate_long = round(float(stop), 4)
                para.vol_multiplier_short = round(float(vol), 2)
                para.stop_multiplier_rate_short = round(float(stop), 4)
                
                temp_df = common.attach_label(temp_df, para=para)
                
                counts = temp_df['label'].value_counts(normalize=True).to_dict()
                # 显式映射：0=Short, 1=Neutral, 2=Long
                sweep_data.append({
                    'vol_multiplier': para.vol_multiplier_long,
                    'stop_rate': para.stop_multiplier_rate_long,
                    'p_short': counts.get(0, 0.0),
                    'p_neutral': counts.get(1, 0.0),
                    'p_long': counts.get(2, 0.0)
                })
        
        self.results_df = pd.DataFrame(sweep_data)
        print(f"✅ 扫描完成，得到 {len(self.results_df)} 组分布样本。")

    def _get_adaptive_params(self, rows, cols, num_subplots=1):
        """
        [自适应逻辑] 根据行列数量计算最佳的 figsize 和字体大小，防止文字重叠
        """
        cell_w = 0.85  # 每个格子的宽度 (inch)
        cell_h = 0.55  # 每个格子的高度 (inch)
        
        # 计算单张子图所需的宽高
        plot_w = max(8, cols * cell_w)
        plot_h = max(6, rows * cell_h)
        
        # 总宽度 = 单图宽 * 图表数量 + 间隙
        total_w = plot_w * num_subplots + (num_subplots - 1) * 2
        
        # 字体大小随格子密度动态调整 (在 6 到 10 之间缩放)
        font_size = min(10, max(6, 40 / max(rows, cols)))
        
        return (total_w, plot_h), font_size

    def analyze_and_plot(self, output_dir=None):
        """
        第二阶段：计算分布的梯度和曲率，并生成自适应热力图
        """
        if output_dir is None: output_dir = self.output_dir
            
        vols = np.array(sorted(self.results_df['vol_multiplier'].unique()))
        stops = np.array(sorted(self.results_df['stop_rate'].unique()))
        
        # 构造分布张量用于梯度计算
        dist_tensor = np.zeros((len(vols), len(stops), 3))
        for i, v in enumerate(vols):
            for j, s in enumerate(stops):
                # 使用 np.isclose 避免浮点数匹配失败
                mask = (np.isclose(self.results_df['vol_multiplier'], v)) & \
                       (np.isclose(self.results_df['stop_rate'], s))
                row = self.results_df[mask].iloc[0]
                dist_tensor[i, j] = [row['p_short'], row['p_neutral'], row['p_long']]

        log_vols = np.log(vols)
        log_stops = np.log(stops)

        # 1. 计算一阶导数（灵敏度）
        grad_v = np.gradient(dist_tensor, log_vols, axis=0) 
        grad_s = np.gradient(dist_tensor, log_stops, axis=1)
        
        sens_short   = np.sqrt(grad_v[..., 0]**2 + grad_s[..., 0]**2)
        sens_neutral = np.sqrt(grad_v[..., 1]**2 + grad_s[..., 1]**2)
        sens_long    = np.sqrt(grad_v[..., 2]**2 + grad_s[..., 2]**2)
        total_sens   = np.sqrt(np.sum(grad_v**2 + grad_s**2, axis=2))

        # 2. 计算二阶导数（曲率/加速度）
        grad_vv = np.gradient(grad_v, log_vols, axis=0)
        grad_ss = np.gradient(grad_s, log_stops, axis=1)
        
        lap_short   = grad_vv[..., 0] + grad_ss[..., 0]
        lap_neutral = grad_vv[..., 1] + grad_ss[..., 1]
        lap_long    = grad_vv[..., 2] + grad_ss[..., 2]
        total_curvature = np.sqrt(np.sum(grad_vv**2 + grad_ss**2, axis=2))

        # 获取自适应绘图参数
        figsize_3, fs = self._get_adaptive_params(len(vols), len(stops), num_subplots=3)
        figsize_2, _ = self._get_adaptive_params(len(vols), len(stops), num_subplots=2)

        # --- 绘图 1: 三类标签的一阶灵敏度 ---
        fig, axes = plt.subplots(1, 3, figsize=figsize_3)
        titles_1st = [(sens_short, "Short Sensitivity (1st Order)", "Reds"),
                      (sens_neutral, "Neutral Sensitivity (1st Order)", "Greys"),
                      (sens_long, "Long Sensitivity (1st Order)", "Greens")]
        
        for idx, (data, title, cmap) in enumerate(titles_1st):
            df_plot = pd.DataFrame(data, index=vols, columns=np.round(stops, 3))
            sns.heatmap(df_plot, ax=axes[idx], annot=True, cmap=cmap, fmt=".2f",
                        annot_kws={"size": fs}, cbar_kws={"shrink": 0.8})
            axes[idx].set_title(title, fontsize=fs+4)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/split_variations_1st.png", bbox_inches='tight')
        plt.close()

        # --- 绘图 2: 三类标签的二阶加速度 ---
        fig, axes = plt.subplots(1, 3, figsize=figsize_3)
        titles_2nd = [(lap_short, "Short Acceleration", "RdBu_r"),
                      (lap_neutral, "Neutral Acceleration", "RdBu_r"),
                      (lap_long, "Long Acceleration", "RdBu_r")]
        
        for idx, (data, title, cmap) in enumerate(titles_2nd):
            df_plot = pd.DataFrame(data, index=vols, columns=np.round(stops, 3))
            sns.heatmap(df_plot, ax=axes[idx], annot=True, cmap=cmap, center=0, fmt=".2f",
                        annot_kws={"size": fs}, cbar_kws={"shrink": 0.8})
            axes[idx].set_title(title, fontsize=fs+4)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/split_variations_signed_2nd.png", bbox_inches='tight')
        plt.close()

        # --- 绘图 3: 总体一阶 vs 二阶对比 ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_2)
        sns.heatmap(pd.DataFrame(total_sens, index=vols, columns=np.round(stops, 3)), 
                    ax=ax1, annot=True, cmap="magma", fmt=".2f", annot_kws={"size": fs})
        ax1.set_title("Total 1st Order Sensitivity", fontsize=fs+6)
        
        sns.heatmap(pd.DataFrame(total_curvature, index=vols, columns=np.round(stops, 3)), 
                    ax=ax2, annot=True, cmap="viridis", fmt=".2f", annot_kws={"size": fs})
        ax2.set_title("Total 2nd Order Curvature", fontsize=fs+6)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/overall_1st_vs_2nd.png", bbox_inches='tight')
        plt.close()

    def plot_null_hypothesis_comparison(self, output_dir=None):
        """
        对比实际分布与高斯零假设
        """
        if output_dir is None: output_dir = self.output_dir
        
        vols = sorted(self.results_df['vol_multiplier'].unique())
        stops = sorted(self.results_df['stop_rate'].unique())
        mid_stop = stops[len(stops)//2]
        
        # 提取切片数据
        mask = np.isclose(self.results_df['stop_rate'], mid_stop)
        subset = self.results_df[mask].sort_values('vol_multiplier')
        
        # 计算高斯分布下的理论概率
        gaussian_nulls = []
        for v in vols:
            threshold = v * mid_stop
            p_not_neutral = 2 * (1 - norm.cdf(threshold))
            gaussian_nulls.append([p_not_neutral/2, 1-p_not_neutral, p_not_neutral/2])
        gaussian_nulls = np.array(gaussian_nulls)

        # 计算 KL 散度
        kl_divs = []
        classes = ['p_short', 'p_neutral', 'p_long']
        for i, v in enumerate(vols):
            actual = subset.iloc[i][classes].values.astype(float) + 1e-9
            target = gaussian_nulls[i] + 1e-9
            kl = np.sum(actual * np.log(actual / target))
            kl_divs.append(kl)

        # 自适应调整线图宽度
        fig_w = max(15, len(vols) * 0.4)
        fig, axes = plt.subplots(1, 3, figsize=(fig_w, 7), sharey=True)
        colors = ['#ff4d4d', '#7f8c8d', '#2ecc71']
        
        for i, (cls, title) in enumerate(zip(classes, ['Short', 'Neutral', 'Long'])):
            axes[i].plot(vols, subset[cls], 'o-', label='Actual', color=colors[i], lw=2)
            axes[i].plot(vols, gaussian_nulls[:, i], '--', label='Gaussian Null', color='black', alpha=0.6)
            axes[i].set_title(f"{title} Proportion\n(Stop Rate: {mid_stop:.2f})")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(f"{output_dir}/null_hypothesis_comparison.png")
        plt.close()

        # KL 散度图
        plt.figure(figsize=(max(12, len(vols)*0.35), 7))
        plt.plot(vols, kl_divs, 's-', color='purple', lw=2, markersize=8)
        plt.title("Information Gain over Gaussian Null (KL Divergence)")
        plt.xlabel("Volatility Multiplier")
        plt.ylabel("KL Divergence")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/information_structure_gain.png")
        plt.close()

        print(f"✅ 所有图表已生成至: {output_dir}")

# --- 运行示例 ---
if __name__ == "__main__":
    import logging
    # 加载数据逻辑 (根据你的项目路径调整)
    df_raw = pd.read_csv(common.origin_data_path)
    df_clean = common.clean_data_quality_auto(df_raw, logging.getLogger('dummy'))
    
    # 获取周期毫秒数
    interval_ms = common.get_interval_ms('5m')
    
    analyzer = LabelRegimeAnalyzer(df_clean, interval_ms, symbol="ETHUSDT", interval='5m')
    
    # 按照你要求的间隔生成参数范围
    vol_range = np.round(np.linspace(0.5, 3.0, 26), 1) # 0.5 到 3.0, 间隔 0.1
    stop_range = [0.5, 1.0, 1.5, 2.0] # 示例 stop range
    
    analyzer.run_parameter_sweep(vol_range, stop_range)
    analyzer.analyze_and_plot()
    analyzer.plot_null_hypothesis_comparison()