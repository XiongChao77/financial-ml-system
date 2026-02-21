import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from scipy.stats import norm

# Ensure common.py is in the path
import common 

class LabelRegimeAnalyzer:
    def __init__(self, df, interval_ms, para = common.BaseDefine(), output_dir=common.PERSISTENCE_DIR):
        self.df = df
        self.interval_ms = interval_ms
        self.symbol = para.symbol
        self.interval = para.interval
        self.para = para
        # Unified output path
        self.output_dir = os.path.join(output_dir, "regime_discovery_output", f"{para.symbol}_{para.interval}")
        if not os.path.exists(self.output_dir): 
            os.makedirs(self.output_dir)
        self.results_df = None

    def run_parameter_sweep(self, vol_range, stop_range, fun):
        """
        Phase 1: Scan parameter space, record original induced distribution vector [P_short, P_neutral, P_long]
        """
        sweep_data = []
        print(f"🚀 Scanning parameter space distribution (Symbol: {self.symbol})...")
        
        for vol in tqdm(vol_range, desc="Vol Steps"):
            for stop in stop_range:
                temp_df = self.df.copy()
                para = self.para
                # Explicitly round to prevent floating point precision issues causing 1.49999
                para.vol_multiplier_long = round(float(vol), 2)
                para.stop_multiplier_rate_long = stop
                para.vol_multiplier_short = round(float(vol), 2)
                para.stop_multiplier_rate_short = stop
                
                temp_df = fun(temp_df, para=para)
                
                # Exclude invalid samples (label == -1) before calculating proportions
                valid_df = temp_df[temp_df['label'] != common.Signal.INVALID]
                
                if len(valid_df) > 0:
                    counts = valid_df['label'].value_counts(normalize=True).to_dict()
                else:
                    counts = {}
                
                # Explicit mapping: 0=Short, 1=Neutral, 2=Long
                sweep_data.append({
                    'vol_multiplier': para.vol_multiplier_long,
                    'stop_rate': para.stop_multiplier_rate_long,
                    'p_short': counts.get(common.Signal.NEGATIVE, 0.0),
                    'p_neutral': counts.get(common.Signal.NEUTRAL, 0.0),
                    'p_long': counts.get(common.Signal.POSITIVE, 0.0)
                })
        
        self.results_df = pd.DataFrame(sweep_data)
        print(f"✅ Scan completed, obtained {len(self.results_df)} distribution samples.")

    def _get_adaptive_params(self, rows, cols, num_subplots=1):
        """
        [Adaptive logic] Calculate optimal figsize and font size based on row/column count to prevent text overlap
        """
        cell_w = 0.85  # Width of each cell (inch)
        cell_h = 0.55  # Height of each cell (inch)
        
        # Calculate width and height required for a single subplot
        plot_w = max(8, cols * cell_w)
        plot_h = max(6, rows * cell_h)
        
        # Total width = single plot width * number of plots + gaps
        total_w = plot_w * num_subplots + (num_subplots - 1) * 2
        
        # Font size dynamically adjusted based on cell density (scaled between 6 and 10)
        font_size = min(10, max(6, 40 / max(rows, cols)))
        
        return (total_w, plot_h), font_size

    def _build_dist_tensor(self):
        """Construct (vols, stops, dist_tensor) from results_df, shared by log/linear derivative calculations."""
        vols = np.array(sorted(self.results_df['vol_multiplier'].unique()))
        stops = np.array(sorted(self.results_df['stop_rate'].unique()))
        dist_tensor = np.zeros((len(vols), len(stops), 3))
        for i, v in enumerate(vols):
            for j, s in enumerate(stops):
                mask = (np.isclose(self.results_df['vol_multiplier'], v)) & \
                       (np.isclose(self.results_df['stop_rate'], s))
                row = self.results_df[mask].iloc[0]
                dist_tensor[i, j] = [row['p_short'], row['p_neutral'], row['p_long']]
        return vols, stops, dist_tensor

    def _compute_log_derivatives(self, vols, stops, dist_tensor):
        """Only compute first, second, third-order derivatives and divergence gradient in log scale, return log_data and plotting parameters."""
        log_vols = np.log(vols)
        log_stops = np.log(stops)
        grad_v = np.gradient(dist_tensor, log_vols, axis=0)
        grad_s = np.gradient(dist_tensor, log_stops, axis=1)
        sens_short   = np.sqrt(grad_v[..., 0]**2 + grad_s[..., 0]**2)
        sens_neutral = np.sqrt(grad_v[..., 1]**2 + grad_s[..., 1]**2)
        sens_long    = np.sqrt(grad_v[..., 2]**2 + grad_s[..., 2]**2)
        total_sens   = np.sqrt(np.sum(grad_v**2 + grad_s**2, axis=2))
        grad_vv = np.gradient(grad_v, log_vols, axis=0)
        grad_ss = np.gradient(grad_s, log_stops, axis=1)
        lap_short   = grad_vv[..., 0] + grad_ss[..., 0]
        lap_neutral = grad_vv[..., 1] + grad_ss[..., 1]
        lap_long    = grad_vv[..., 2] + grad_ss[..., 2]
        total_curvature = np.sqrt(np.sum(grad_vv**2 + grad_ss**2, axis=2))
        grad_vvv = np.gradient(grad_vv, log_vols, axis=0)
        grad_vvs = np.gradient(grad_vv, log_stops, axis=1)
        grad_vss = np.gradient(grad_ss, log_vols, axis=0)
        grad_sss = np.gradient(grad_ss, log_stops, axis=1)
        jerk_short   = np.sqrt(grad_vvv[..., 0]**2 + grad_vvs[..., 0]**2 + grad_vss[..., 0]**2 + grad_sss[..., 0]**2)
        jerk_neutral = np.sqrt(grad_vvv[..., 1]**2 + grad_vvs[..., 1]**2 + grad_vss[..., 1]**2 + grad_sss[..., 1]**2)
        jerk_long    = np.sqrt(grad_vvv[..., 2]**2 + grad_vvs[..., 2]**2 + grad_vss[..., 2]**2 + grad_sss[..., 2]**2)
        total_jerk   = np.sqrt(np.sum(grad_vvv**2 + grad_vvs**2 + grad_vss**2 + grad_sss**2, axis=2))
        lap_tensor = np.stack([lap_short, lap_neutral, lap_long], axis=2)
        grad_div_v = np.gradient(lap_tensor, log_vols, axis=0)
        grad_div_s = np.gradient(lap_tensor, log_stops, axis=1)
        div_grad_short   = np.sqrt(grad_div_v[..., 0]**2 + grad_div_s[..., 0]**2)
        div_grad_neutral = np.sqrt(grad_div_v[..., 1]**2 + grad_div_s[..., 1]**2)
        div_grad_long    = np.sqrt(grad_div_v[..., 2]**2 + grad_div_s[..., 2]**2)
        total_div_grad   = np.sqrt(np.sum(grad_div_v**2 + grad_div_s**2, axis=2))
        log_data = dict(
            sens_short=sens_short, sens_neutral=sens_neutral, sens_long=sens_long, total_sens=total_sens,
            lap_short=lap_short, lap_neutral=lap_neutral, lap_long=lap_long, total_curvature=total_curvature,
            jerk_short=jerk_short, jerk_neutral=jerk_neutral, jerk_long=jerk_long, total_jerk=total_jerk,
            div_grad_short=div_grad_short, div_grad_neutral=div_grad_neutral, div_grad_long=div_grad_long,
            total_div_grad=total_div_grad,
        )
        figsize_3, fs = self._get_adaptive_params(len(vols), len(stops), num_subplots=3)
        figsize_2, _ = self._get_adaptive_params(len(vols), len(stops), num_subplots=2)
        return log_data, figsize_3, figsize_2, fs

    def _compute_linear_derivatives(self, vols, stops, dist_tensor):
        """Only compute first, second, third-order derivatives and divergence gradient in linear scale, return linear_data and plotting parameters."""
        grad_v_lin = np.gradient(dist_tensor, vols, axis=0)
        grad_s_lin = np.gradient(dist_tensor, stops, axis=1)
        sens_short_lin   = np.sqrt(grad_v_lin[..., 0]**2 + grad_s_lin[..., 0]**2)
        sens_neutral_lin = np.sqrt(grad_v_lin[..., 1]**2 + grad_s_lin[..., 1]**2)
        sens_long_lin    = np.sqrt(grad_v_lin[..., 2]**2 + grad_s_lin[..., 2]**2)
        total_sens_lin   = np.sqrt(np.sum(grad_v_lin**2 + grad_s_lin**2, axis=2))
        grad_vv_lin = np.gradient(grad_v_lin, vols, axis=0)
        grad_ss_lin = np.gradient(grad_s_lin, stops, axis=1)
        lap_short_lin   = grad_vv_lin[..., 0] + grad_ss_lin[..., 0]
        lap_neutral_lin = grad_vv_lin[..., 1] + grad_ss_lin[..., 1]
        lap_long_lin    = grad_vv_lin[..., 2] + grad_ss_lin[..., 2]
        total_curvature_lin = np.sqrt(np.sum(grad_vv_lin**2 + grad_ss_lin**2, axis=2))
        grad_vvv_lin = np.gradient(grad_vv_lin, vols, axis=0)
        grad_vvs_lin = np.gradient(grad_vv_lin, stops, axis=1)
        grad_vss_lin = np.gradient(grad_ss_lin, vols, axis=0)
        grad_sss_lin = np.gradient(grad_ss_lin, stops, axis=1)
        jerk_short_lin   = np.sqrt(grad_vvv_lin[..., 0]**2 + grad_vvs_lin[..., 0]**2 + grad_vss_lin[..., 0]**2 + grad_sss_lin[..., 0]**2)
        jerk_neutral_lin = np.sqrt(grad_vvv_lin[..., 1]**2 + grad_vvs_lin[..., 1]**2 + grad_vss_lin[..., 1]**2 + grad_sss_lin[..., 1]**2)
        jerk_long_lin    = np.sqrt(grad_vvv_lin[..., 2]**2 + grad_vvs_lin[..., 2]**2 + grad_vss_lin[..., 2]**2 + grad_sss_lin[..., 2]**2)
        total_jerk_lin   = np.sqrt(np.sum(grad_vvv_lin**2 + grad_vvs_lin**2 + grad_vss_lin**2 + grad_sss_lin**2, axis=2))
        lap_tensor_lin = np.stack([lap_short_lin, lap_neutral_lin, lap_long_lin], axis=2)
        grad_div_v_lin = np.gradient(lap_tensor_lin, vols, axis=0)
        grad_div_s_lin = np.gradient(lap_tensor_lin, stops, axis=1)
        div_grad_short_lin   = np.sqrt(grad_div_v_lin[..., 0]**2 + grad_div_s_lin[..., 0]**2)
        div_grad_neutral_lin = np.sqrt(grad_div_v_lin[..., 1]**2 + grad_div_s_lin[..., 1]**2)
        div_grad_long_lin    = np.sqrt(grad_div_v_lin[..., 2]**2 + grad_div_s_lin[..., 2]**2)
        total_div_grad_lin   = np.sqrt(np.sum(grad_div_v_lin**2 + grad_div_s_lin**2, axis=2))
        linear_data = dict(
            sens_short=sens_short_lin, sens_neutral=sens_neutral_lin, sens_long=sens_long_lin, total_sens=total_sens_lin,
            lap_short=lap_short_lin, lap_neutral=lap_neutral_lin, lap_long=lap_long_lin, total_curvature=total_curvature_lin,
            jerk_short=jerk_short_lin, jerk_neutral=jerk_neutral_lin, jerk_long=jerk_long_lin, total_jerk=total_jerk_lin,
            div_grad_short=div_grad_short_lin, div_grad_neutral=div_grad_neutral_lin, div_grad_long=div_grad_long_lin,
            total_div_grad=total_div_grad_lin,
        )
        figsize_3, fs = self._get_adaptive_params(len(vols), len(stops), num_subplots=3)
        figsize_2, _ = self._get_adaptive_params(len(vols), len(stops), num_subplots=2)
        return linear_data, figsize_3, figsize_2, fs

    def analyze_and_plot_log(self, output_dir=None):
        """Only responsible for log-type derivatives: compute and plot log-scale heatmaps."""
        if output_dir is None:
            output_dir = self.output_dir
        vols, stops, dist_tensor = self._build_dist_tensor()
        log_data, figsize_3, figsize_2, fs = self._compute_log_derivatives(vols, stops, dist_tensor)
        self._plot_heatmaps_log(output_dir, vols, stops, log_data, figsize_3, figsize_2, fs)

    def analyze_and_plot_linear(self, output_dir=None):
        """Only responsible for linear-type derivatives: compute and plot linear-scale heatmaps."""
        if output_dir is None:
            output_dir = self.output_dir
        vols, stops, dist_tensor = self._build_dist_tensor()
        linear_data, figsize_3, figsize_2, fs = self._compute_linear_derivatives(vols, stops, dist_tensor)
        self._plot_heatmaps_linear(output_dir, vols, stops, linear_data, figsize_3, figsize_2, fs)

    def analyze_and_plot(self, output_dir=None):
        """Compute log/linear derivatives and plot heatmaps separately. Internally builds dist_tensor only once and calls two sub-functions."""
        if output_dir is None:
            output_dir = self.output_dir
        vols, stops, dist_tensor = self._build_dist_tensor()
        log_data, figsize_3, figsize_2, fs = self._compute_log_derivatives(vols, stops, dist_tensor)
        linear_data, figsize_3_lin, figsize_2_lin, fs_lin = self._compute_linear_derivatives(vols, stops, dist_tensor)
        self._plot_heatmaps_log(output_dir, vols, stops, log_data, figsize_3, figsize_2, fs)
        self._plot_heatmaps_linear(output_dir, vols, stops, linear_data, figsize_3_lin, figsize_2_lin, fs_lin)

    def _plot_heatmaps_log(self, output_dir, vols, stops, data, figsize_3, figsize_2, fs):
        """Plot heatmaps using log-scale derivative data (currently unused, kept as backup)."""
        d = data
        # First order
        fig, axes = plt.subplots(1, 3, figsize=figsize_3)
        for idx, (arr, title, cmap) in enumerate([
            (d["sens_short"], "Short Sensitivity (1st Order)", "Reds"),
            (d["sens_neutral"], "Neutral Sensitivity (1st Order)", "Greys"),
            (d["sens_long"], "Long Sensitivity (1st Order)", "Greens"),
        ]):
            df_plot = pd.DataFrame(arr, index=vols, columns=np.round(stops, 3))
            sns.heatmap(df_plot, ax=axes[idx], annot=True, cmap=cmap, fmt=".2f",
                        annot_kws={"size": fs}, cbar_kws={"shrink": 0.8})
            axes[idx].set_title(title, fontsize=fs+4)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/split_variations_1st.png", bbox_inches='tight')
        plt.close()

        # Second order
        fig, axes = plt.subplots(1, 3, figsize=figsize_3)
        for idx, (arr, title, cmap) in enumerate([
            (d["lap_short"], "Short Acceleration", "RdBu_r"),
            (d["lap_neutral"], "Neutral Acceleration", "RdBu_r"),
            (d["lap_long"], "Long Acceleration", "RdBu_r"),
        ]):
            df_plot = pd.DataFrame(arr, index=vols, columns=np.round(stops, 3))
            sns.heatmap(df_plot, ax=axes[idx], annot=True, cmap=cmap, center=0, fmt=".2f",
                        annot_kws={"size": fs}, cbar_kws={"shrink": 0.8})
            axes[idx].set_title(title, fontsize=fs+4)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/split_variations_signed_2nd.png", bbox_inches='tight')
        plt.close()

        # Total first order vs second order
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_2)
        sns.heatmap(pd.DataFrame(d["total_sens"], index=vols, columns=np.round(stops, 3)),
                    ax=ax1, annot=True, cmap="magma", fmt=".2f", annot_kws={"size": fs})
        ax1.set_title("Total 1st Order Sensitivity", fontsize=fs+6)
        sns.heatmap(pd.DataFrame(d["total_curvature"], index=vols, columns=np.round(stops, 3)),
                    ax=ax2, annot=True, cmap="viridis", fmt=".2f", annot_kws={"size": fs})
        ax2.set_title("Total 2nd Order Curvature", fontsize=fs+6)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/overall_1st_vs_2nd.png", bbox_inches='tight')
        plt.close()

        # Third order
        fig, axes = plt.subplots(1, 3, figsize=figsize_3)
        for idx, (arr, title, cmap) in enumerate([
            (d["jerk_short"], "Short 3rd Order (Jerk)", "Reds"),
            (d["jerk_neutral"], "Neutral 3rd Order (Jerk)", "Greys"),
            (d["jerk_long"], "Long 3rd Order (Jerk)", "Greens"),
        ]):
            df_plot = pd.DataFrame(arr, index=vols, columns=np.round(stops, 3))
            sns.heatmap(df_plot, ax=axes[idx], annot=True, cmap=cmap, fmt=".2f",
                        annot_kws={"size": fs}, cbar_kws={"shrink": 0.8})
            axes[idx].set_title(title, fontsize=fs+4)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/split_variations_3rd.png", bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(figsize_2[0] // 2, figsize_2[1]))
        sns.heatmap(pd.DataFrame(d["total_jerk"], index=vols, columns=np.round(stops, 3)),
                    ax=ax, annot=True, cmap="plasma", fmt=".2f", annot_kws={"size": fs})
        ax.set_title("Total 3rd Order (Jerk)", fontsize=fs+6)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/overall_3rd.png", bbox_inches='tight')
        plt.close()

        # Divergence gradient
        fig, axes = plt.subplots(1, 3, figsize=figsize_3)
        for idx, (arr, title, cmap) in enumerate([
            (d["div_grad_short"], "Short |∇(div)|", "Reds"),
            (d["div_grad_neutral"], "Neutral |∇(div)|", "Greys"),
            (d["div_grad_long"], "Long |∇(div)|", "Greens"),
        ]):
            df_plot = pd.DataFrame(arr, index=vols, columns=np.round(stops, 3))
            sns.heatmap(df_plot, ax=axes[idx], annot=True, cmap=cmap, fmt=".2f",
                        annot_kws={"size": fs}, cbar_kws={"shrink": 0.8})
            axes[idx].set_title(title, fontsize=fs+4)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/split_divergence_gradient.png", bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(figsize_2[0] // 2, figsize_2[1]))
        sns.heatmap(pd.DataFrame(d["total_div_grad"], index=vols, columns=np.round(stops, 3)),
                    ax=ax, annot=True, cmap="cividis", fmt=".2f", annot_kws={"size": fs})
        ax.set_title("Total |∇(div)| (log)", fontsize=fs+6)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/overall_divergence_gradient.png", bbox_inches='tight')
        plt.close()

    def _plot_heatmaps_linear(self, output_dir, vols, stops, data, figsize_3, figsize_2, fs):
        """Plot heatmaps using linear-scale derivative data."""
        d = data
        suffix = "_linear"

        # First order
        fig, axes = plt.subplots(1, 3, figsize=figsize_3)
        for idx, (arr, title, cmap) in enumerate([
            (d["sens_short"], "Short Sensitivity (1st, linear)", "Reds"),
            (d["sens_neutral"], "Neutral Sensitivity (1st, linear)", "Greys"),
            (d["sens_long"], "Long Sensitivity (1st, linear)", "Greens"),
        ]):
            df_plot = pd.DataFrame(arr, index=vols, columns=np.round(stops, 3))
            sns.heatmap(df_plot, ax=axes[idx], annot=True, cmap=cmap, fmt=".2f",
                        annot_kws={"size": fs}, cbar_kws={"shrink": 0.8})
            axes[idx].set_title(title, fontsize=fs+4)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/split_variations_1st{suffix}.png", bbox_inches='tight')
        plt.close()

        # Second order
        fig, axes = plt.subplots(1, 3, figsize=figsize_3)
        for idx, (arr, title, cmap) in enumerate([
            (d["lap_short"], "Short Acceleration (linear)", "RdBu_r"),
            (d["lap_neutral"], "Neutral Acceleration (linear)", "RdBu_r"),
            (d["lap_long"], "Long Acceleration (linear)", "RdBu_r"),
        ]):
            df_plot = pd.DataFrame(arr, index=vols, columns=np.round(stops, 3))
            sns.heatmap(df_plot, ax=axes[idx], annot=True, cmap=cmap, center=0, fmt=".2f",
                        annot_kws={"size": fs}, cbar_kws={"shrink": 0.8})
            axes[idx].set_title(title, fontsize=fs+4)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/split_variations_signed_2nd{suffix}.png", bbox_inches='tight')
        plt.close()

        # Total first order vs second order
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_2)
        sns.heatmap(pd.DataFrame(d["total_sens"], index=vols, columns=np.round(stops, 3)),
                    ax=ax1, annot=True, cmap="magma", fmt=".2f", annot_kws={"size": fs})
        ax1.set_title("Total 1st Order Sensitivity (linear)", fontsize=fs+6)
        sns.heatmap(pd.DataFrame(d["total_curvature"], index=vols, columns=np.round(stops, 3)),
                    ax=ax2, annot=True, cmap="viridis", fmt=".2f", annot_kws={"size": fs})
        ax2.set_title("Total 2nd Order Curvature (linear)", fontsize=fs+6)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/overall_1st_vs_2nd{suffix}.png", bbox_inches='tight')
        plt.close()

        # Third order
        fig, axes = plt.subplots(1, 3, figsize=figsize_3)
        for idx, (arr, title, cmap) in enumerate([
            (d["jerk_short"], "Short 3rd Order (linear)", "Reds"),
            (d["jerk_neutral"], "Neutral 3rd Order (linear)", "Greys"),
            (d["jerk_long"], "Long 3rd Order (linear)", "Greens"),
        ]):
            df_plot = pd.DataFrame(arr, index=vols, columns=np.round(stops, 3))
            sns.heatmap(df_plot, ax=axes[idx], annot=True, cmap=cmap, fmt=".2f",
                        annot_kws={"size": fs}, cbar_kws={"shrink": 0.8})
            axes[idx].set_title(title, fontsize=fs+4)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/split_variations_3rd{suffix}.png", bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(figsize_2[0] // 2, figsize_2[1]))
        sns.heatmap(pd.DataFrame(d["total_jerk"], index=vols, columns=np.round(stops, 3)),
                    ax=ax, annot=True, cmap="plasma", fmt=".2f", annot_kws={"size": fs})
        ax.set_title("Total 3rd Order (Jerk, linear)", fontsize=fs+6)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/overall_3rd{suffix}.png", bbox_inches='tight')
        plt.close()

        # Divergence gradient
        fig, axes = plt.subplots(1, 3, figsize=figsize_3)
        for idx, (arr, title, cmap) in enumerate([
            (d["div_grad_short"], "Short |∇(div)| (linear)", "Reds"),
            (d["div_grad_neutral"], "Neutral |∇(div)| (linear)", "Greys"),
            (d["div_grad_long"], "Long |∇(div)| (linear)", "Greens"),
        ]):
            df_plot = pd.DataFrame(arr, index=vols, columns=np.round(stops, 3))
            sns.heatmap(df_plot, ax=axes[idx], annot=True, cmap=cmap, fmt=".2f",
                        annot_kws={"size": fs}, cbar_kws={"shrink": 0.8})
            axes[idx].set_title(title, fontsize=fs+4)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/split_divergence_gradient{suffix}.png", bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(figsize_2[0] // 2, figsize_2[1]))
        sns.heatmap(pd.DataFrame(d["total_div_grad"], index=vols, columns=np.round(stops, 3)),
                    ax=ax, annot=True, cmap="cividis", fmt=".2f", annot_kws={"size": fs})
        ax.set_title("Total |∇(div)| (linear)", fontsize=fs+6)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/overall_divergence_gradient{suffix}.png", bbox_inches='tight')
        plt.close()

    def plot_null_hypothesis_comparison(self, output_dir=None):
        """
        Compare actual distribution with Gaussian null hypothesis
        """
        if output_dir is None: output_dir = self.output_dir
        
        vols = sorted(self.results_df['vol_multiplier'].unique())
        stops = sorted(self.results_df['stop_rate'].unique())
        mid_stop = stops[len(stops)//2]
        
        # Extract slice data
        mask = np.isclose(self.results_df['stop_rate'], mid_stop)
        subset = self.results_df[mask].sort_values('vol_multiplier')
        
        # Calculate theoretical probability under Gaussian distribution
        gaussian_nulls = []
        for v in vols:
            threshold = v * mid_stop
            p_not_neutral = 2 * (1 - norm.cdf(threshold))
            gaussian_nulls.append([p_not_neutral/2, 1-p_not_neutral, p_not_neutral/2])
        gaussian_nulls = np.array(gaussian_nulls)

        # Calculate KL divergence
        kl_divs = []
        classes = ['p_short', 'p_neutral', 'p_long']
        for i, v in enumerate(vols):
            actual = subset.iloc[i][classes].values.astype(float) + 1e-9
            target = gaussian_nulls[i] + 1e-9
            kl = np.sum(actual * np.log(actual / target))
            kl_divs.append(kl)

        # Adaptively adjust line plot width
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

        # KL divergence plot
        plt.figure(figsize=(max(12, len(vols)*0.35), 7))
        plt.plot(vols, kl_divs, 's-', color='purple', lw=2, markersize=8)
        plt.title("Information Gain over Gaussian Null (KL Divergence)")
        plt.xlabel("Volatility Multiplier")
        plt.ylabel("KL Divergence")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/information_structure_gain.png")
        plt.close()

        print(f"✅ All plots generated to: {output_dir}")

    def plot_long_ratio_vs_vol_multiplier(self):
        """
        Plot long label ratio vs vol_multiplier_long.
        X-axis: vol_multiplier_long
        Y-axis: p_long (long label ratio)
        """
        output_dir = self.output_dir
        if self.results_df is None:
            print("⚠️ Please run run_parameter_sweep() first")
            return
        
        vols = sorted(self.results_df['vol_multiplier'].unique())
        stops = sorted(self.results_df['stop_rate'].unique())
        
        # Adaptive chart width
        fig_w = max(12, len(vols) * 0.5)
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, 7))
        
        # Draw a line for each stop_rate
        colors = plt.cm.viridis(np.linspace(0, 1, len(stops)))
        for idx, stop in enumerate(stops):
            mask = np.isclose(self.results_df['stop_rate'], stop)
            subset = self.results_df[mask].sort_values('vol_multiplier')
            ax.plot(subset['vol_multiplier'], subset['p_long'], 
                   'o-', label=f'Stop Rate: {stop:.2f}', 
                   color=colors[idx], lw=2, markersize=6)
        
        ax.set_xlabel('vol_multiplier_long', fontsize=12)
        ax.set_ylabel('Long Label Ratio (p_long)', fontsize=12)
        ax.set_title('Long Label Ratio vs vol_multiplier_long', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        y_min = self.results_df['p_long'].min()
        y_max = self.results_df['p_long'].max()
        margin = max(0.05, (y_max - y_min) * 0.1) if y_max > y_min else 0.05
        ax.set_ylim([max(0, y_min - margin), min(1, y_max + margin)])
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/long_ratio_vs_vol_multiplier.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Long ratio plot saved: {output_dir}/long_ratio_vs_vol_multiplier.png")

    def plot_vol_vs_distribution(self, output_dir=None):
        """
        Plot X-Y curves of vol_range (X-axis) vs sample distribution (Y-axis).
        Three subplots correspond to p_short / p_neutral / p_long; different stop_rate values are different curves within the same subplot.
        Overlay Gaussian theoretical curves (dashed lines) to compare sample distribution with Gaussian approximation.

        Additional output: first derivative plot dp/d(vol_multiplier)
        """
        if output_dir is None:
            output_dir = self.output_dir
        if self.results_df is None:
            print("⚠️ Please run run_parameter_sweep() first")
            return

        vols = np.array(sorted(self.results_df['vol_multiplier'].unique()))
        stops = sorted(self.results_df['stop_rate'].unique())

        # =========================
        # 1) Original proportion curves p(v)
        # =========================
        fig_w = max(14, len(vols) * 0.5)
        fig, axes = plt.subplots(1, 3, figsize=(fig_w, 6), sharex=True)
        colors = plt.cm.viridis(np.linspace(0, 1, len(stops)))

        for idx, stop in enumerate(stops):
            mask = np.isclose(self.results_df['stop_rate'], stop)
            subset = self.results_df[mask].sort_values('vol_multiplier')
            label = f"stop_rate: {stop:.2f}"

            axes[0].plot(subset['vol_multiplier'], subset['p_short'],   'o-', label=label, color=colors[idx], lw=2, markersize=5)
            axes[1].plot(subset['vol_multiplier'], subset['p_neutral'], 'o-', label=label, color=colors[idx], lw=2, markersize=5)
            axes[2].plot(subset['vol_multiplier'], subset['p_long'],    'o-', label=label, color=colors[idx], lw=2, markersize=5)

        # Gaussian reference: ignore stop_rate, only plot vol -> distribution
        threshold = vols
        p_not_neutral = 2 * (1 - norm.cdf(threshold))
        p_short_g   = p_not_neutral / 2
        p_neutral_g = 1 - p_not_neutral
        p_long_g    = p_not_neutral / 2

        axes[0].plot(vols, p_short_g,   '--', label='Gaussian (vol-only)', color='black', lw=2, alpha=0.8)
        axes[1].plot(vols, p_neutral_g, '--', label='Gaussian (vol-only)', color='black', lw=2, alpha=0.8)
        axes[2].plot(vols, p_long_g,    '--', label='Gaussian (vol-only)', color='black', lw=2, alpha=0.8)

        for ax_i, title in enumerate(['Short (p_short)', 'Neutral (p_neutral)', 'Long (p_long)']):
            axes[ax_i].set_xlabel('vol_multiplier', fontsize=11)
            axes[ax_i].set_ylabel('Proportion', fontsize=11)
            axes[ax_i].set_title(title, fontsize=12, fontweight='bold')
            axes[ax_i].legend(loc='best', fontsize=8)
            axes[ax_i].grid(True, alpha=0.3)
            axes[ax_i].set_ylim(-0.02, 1.02)

        fig.suptitle('Sample Distribution vs vol_multiplier (solid: empirical, dashed: Gaussian)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/vol_vs_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Vol vs distribution plot saved: {output_dir}/vol_vs_distribution.png")

        # =========================
        # 1.5) Zoomed view for vol >= 1.5: Long/Short only
        # =========================
        vol_threshold = 1.5
        vols_filtered = vols[vols >= vol_threshold]
        
        if len(vols_filtered) > 0:
            fig_zoom_w = max(10, len(vols_filtered) * 0.5)
            fig_zoom, axes_zoom = plt.subplots(1, 2, figsize=(fig_zoom_w, 6), sharex=True)
            colors_zoom = plt.cm.viridis(np.linspace(0, 1, len(stops)))
            
            # Track max Y values for auto-scaling
            max_y_short = 0
            max_y_long = 0
            
            for idx, stop in enumerate(stops):
                mask = np.isclose(self.results_df['stop_rate'], stop)
                subset = self.results_df[mask].sort_values('vol_multiplier')
                
                # Filter to vol >= 1.5
                subset_filtered = subset[subset['vol_multiplier'] >= vol_threshold]
                
                if len(subset_filtered) > 0:
                    v_filtered = subset_filtered['vol_multiplier'].to_numpy(dtype=float)
                    ps_filtered = subset_filtered['p_short'].to_numpy(dtype=float)
                    pl_filtered = subset_filtered['p_long'].to_numpy(dtype=float)
                    
                    label = f"stop_rate: {stop:.2f}"
                    axes_zoom[0].plot(v_filtered, ps_filtered, 'o-', label=label, color=colors_zoom[idx], lw=2, markersize=5)
                    axes_zoom[1].plot(v_filtered, pl_filtered, 'o-', label=label, color=colors_zoom[idx], lw=2, markersize=5)
                    
                    max_y_short = max(max_y_short, ps_filtered.max())
                    max_y_long = max(max_y_long, pl_filtered.max())
            
            # Gaussian reference for filtered range
            p_not_neutral_g_filtered = 2 * (1 - norm.cdf(vols_filtered))
            p_short_g_filtered = p_not_neutral_g_filtered / 2
            p_long_g_filtered = p_not_neutral_g_filtered / 2
            
            axes_zoom[0].plot(vols_filtered, p_short_g_filtered, '--', label='Gaussian (vol-only)', color='black', lw=2, alpha=0.8)
            axes_zoom[1].plot(vols_filtered, p_long_g_filtered, '--', label='Gaussian (vol-only)', color='black', lw=2, alpha=0.8)
            
            # Update max Y considering Gaussian curve
            max_y_short = max(max_y_short, p_short_g_filtered.max())
            max_y_long = max(max_y_long, p_long_g_filtered.max())
            
            # Set Y-axis limits with small margin
            margin_short = max_y_short * 0.05 if max_y_short > 0 else 0.01
            margin_long = max_y_long * 0.05 if max_y_long > 0 else 0.01
            
            axes_zoom[0].set_ylim(-margin_short, max_y_short + margin_short)
            axes_zoom[1].set_ylim(-margin_long, max_y_long + margin_long)
            
            axes_zoom[0].set_xlabel('vol_multiplier', fontsize=11)
            axes_zoom[0].set_ylabel('Proportion', fontsize=11)
            axes_zoom[0].set_title('Short (p_short) - Zoomed (vol >= 1.5)', fontsize=12, fontweight='bold')
            axes_zoom[0].legend(loc='best', fontsize=8)
            axes_zoom[0].grid(True, alpha=0.3)
            
            axes_zoom[1].set_xlabel('vol_multiplier', fontsize=11)
            axes_zoom[1].set_ylabel('Proportion', fontsize=11)
            axes_zoom[1].set_title('Long (p_long) - Zoomed (vol >= 1.5)', fontsize=12, fontweight='bold')
            axes_zoom[1].legend(loc='best', fontsize=8)
            axes_zoom[1].grid(True, alpha=0.3)
            
            fig_zoom.suptitle('Sample Distribution vs vol_multiplier (Zoomed: vol >= 1.5, Y-axis auto-scaled)', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/vol_vs_distribution_zoomed_1.5plus.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Vol vs distribution zoomed plot (vol >= 1.5) saved: {output_dir}/vol_vs_distribution_zoomed_1.5plus.png")

        # =========================
        # 2) First derivative dp/dv
        # =========================
        fig_w2 = max(14, len(vols) * 0.5)
        fig2, axes2 = plt.subplots(1, 3, figsize=(fig_w2, 6), sharex=True)
        colors2 = plt.cm.viridis(np.linspace(0, 1, len(stops)))

        # empirical: numerical derivative for each stop_rate curve
        for idx, stop in enumerate(stops):
            mask = np.isclose(self.results_df['stop_rate'], stop)
            subset = self.results_df[mask].sort_values('vol_multiplier')

            v = subset['vol_multiplier'].to_numpy(dtype=float)
            ps = subset['p_short'].to_numpy(dtype=float)
            pn = subset['p_neutral'].to_numpy(dtype=float)
            pl = subset['p_long'].to_numpy(dtype=float)

            # np.gradient supports non-uniform x: dp/dv
            dps = np.gradient(ps, v)
            dpn = np.gradient(pn, v)
            dpl = np.gradient(pl, v)

            label = f"stop_rate: {stop:.2f}"
            axes2[0].plot(v, dps, 'o-', label=label, color=colors2[idx], lw=2, markersize=5)
            axes2[1].plot(v, dpn, 'o-', label=label, color=colors2[idx], lw=2, markersize=5)
            axes2[2].plot(v, dpl, 'o-', label=label, color=colors2[idx], lw=2, markersize=5)

        # gaussian: also take derivative of p_g(v) (vol-only)
        dps_g = np.gradient(p_short_g, vols)
        dpn_g = np.gradient(p_neutral_g, vols)
        dpl_g = np.gradient(p_long_g, vols)

        axes2[0].plot(vols, dps_g, '--', label='Gaussian d/dv (vol-only)', color='black', lw=2, alpha=0.8)
        axes2[1].plot(vols, dpn_g, '--', label='Gaussian d/dv (vol-only)', color='black', lw=2, alpha=0.8)
        axes2[2].plot(vols, dpl_g, '--', label='Gaussian d/dv (vol-only)', color='black', lw=2, alpha=0.8)

        for ax_i, title in enumerate(['d(p_short)/dv', 'd(p_neutral)/dv', 'd(p_long)/dv']):
            axes2[ax_i].set_xlabel('vol_multiplier', fontsize=11)
            axes2[ax_i].set_ylabel('1st derivative', fontsize=11)
            axes2[ax_i].set_title(title, fontsize=12, fontweight='bold')
            axes2[ax_i].legend(loc='best', fontsize=8)
            axes2[ax_i].grid(True, alpha=0.3)

        fig2.suptitle('1st Derivative vs vol_multiplier (solid: empirical, dashed: Gaussian)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/vol_vs_distribution_1st_derivative.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 1st derivative plot saved: {output_dir}/vol_vs_distribution_1st_derivative.png")

        # =========================
        # 3) Second derivative d2p/dv2
        # =========================
        fig_w3 = max(14, len(vols) * 0.5)
        fig3, axes3 = plt.subplots(1, 3, figsize=(fig_w3, 6), sharex=True)
        colors3 = plt.cm.viridis(np.linspace(0, 1, len(stops)))

        # empirical: second-order numerical derivative for each stop_rate curve
        for idx, stop in enumerate(stops):
            mask = np.isclose(self.results_df['stop_rate'], stop)
            subset = self.results_df[mask].sort_values('vol_multiplier')

            v = subset['vol_multiplier'].to_numpy(dtype=float)
            ps = subset['p_short'].to_numpy(dtype=float)
            pn = subset['p_neutral'].to_numpy(dtype=float)
            pl = subset['p_long'].to_numpy(dtype=float)

            # First order
            dps = np.gradient(ps, v)
            dpn = np.gradient(pn, v)
            dpl = np.gradient(pl, v)
            # Second order
            ddps = np.gradient(dps, v)
            ddpn = np.gradient(dpn, v)
            ddpl = np.gradient(dpl, v)

            label = f"stop_rate: {stop:.2f}"
            axes3[0].plot(v, ddps, 'o-', label=label, color=colors3[idx], lw=2, markersize=5)
            axes3[1].plot(v, ddpn, 'o-', label=label, color=colors3[idx], lw=2, markersize=5)
            axes3[2].plot(v, ddpl, 'o-', label=label, color=colors3[idx], lw=2, markersize=5)

        # gaussian: second derivative of p_g(v) (vol-only)
        dps_g = np.gradient(p_short_g, vols)
        dpn_g = np.gradient(p_neutral_g, vols)
        dpl_g = np.gradient(p_long_g, vols)
        ddps_g = np.gradient(dps_g, vols)
        ddpn_g = np.gradient(dpn_g, vols)
        ddpl_g = np.gradient(dpl_g, vols)

        axes3[0].plot(vols, ddps_g, '--', label='Gaussian d2/dv2 (vol-only)', color='black', lw=2, alpha=0.8)
        axes3[1].plot(vols, ddpn_g, '--', label='Gaussian d2/dv2 (vol-only)', color='black', lw=2, alpha=0.8)
        axes3[2].plot(vols, ddpl_g, '--', label='Gaussian d2/dv2 (vol-only)', color='black', lw=2, alpha=0.8)

        for ax_i, title in enumerate(['d²(p_short)/dv²', 'd²(p_neutral)/dv²', 'd²(p_long)/dv²']):
            axes3[ax_i].set_xlabel('vol_multiplier', fontsize=11)
            axes3[ax_i].set_ylabel('2nd derivative', fontsize=11)
            axes3[ax_i].set_title(title, fontsize=12, fontweight='bold')
            axes3[ax_i].legend(loc='best', fontsize=8)
            axes3[ax_i].grid(True, alpha=0.3)

        fig3.suptitle('2nd Derivative vs vol_multiplier (solid: empirical, dashed: Gaussian)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/vol_vs_distribution_2nd_derivative.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 2nd derivative plot saved: {output_dir}/vol_vs_distribution_2nd_derivative.png")


# --- Running example ---
if __name__ == "__main__":
    import logging
    # Load data logic (adjust according to your project path)
    df_raw = pd.read_csv(common.origin_data_path)
    df_clean = common.clean_data_quality_auto(df_raw, logging.getLogger('dummy'))
    
    # Get period milliseconds
    interval_ms = common.get_interval_ms('5m')
    
    analyzer = LabelRegimeAnalyzer(df_clean, interval_ms, symbol="ETHUSDT", interval='5m')
    
    # Generate parameter range according to your required interval
    vol_range = np.round(np.linspace(0.5, 3.0, 26), 1) # 0.5 to 3.0, interval 0.1
    stop_range = [0.5, 1.0, 1.5, 2.0] # Example stop range
    
    analyzer.run_parameter_sweep(vol_range, stop_range)
    analyzer.analyze_and_plot()
    analyzer.plot_null_hypothesis_comparison()