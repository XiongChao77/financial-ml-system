import numpy as np
import pandas as pd
import torch,os,sys
import multiprocessing as mp  # 🌟 引入多进程模块
import logging
import random
from copy import deepcopy
# 路径设置
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))

from data_process import common,preparation
from model import train

# ==============================================================================
# GA 核心参数
# ==============================================================================
POP_SIZE = 40#40       # 种群大小
GENERATIONS = 15# 15     # 进化代数
MUTATION_RATE = 0.2 
FULL_CONFIG = common.FEATURE_CONFIG_LIST # 原始全量组列表
# 1. 识别并提取必选组和可选组
# 假设 FeatureOrigin 在列表末尾，我们通过类名进行过滤
MANDATORY_CONFIG = [item for item in common.FEATURE_CONFIG_LIST if item[0].__name__ == "FeatureOrigin"]
EVOLVABLE_CONFIG = [item for item in common.FEATURE_CONFIG_LIST if item[0].__name__ != "FeatureOrigin"]

# 2. 重新定义基因长度
GENE_LENGTH = len(EVOLVABLE_CONFIG)

# 🌟 新增：子进程的工作函数，用于运行训练并返回结果
def train_worker(result_queue, sub_config_list, data_cfg, train_cfg, model_cfg):
    """
    在子进程中运行。
    由于 logging 对象通常不能跨进程传递，我们在子进程里简单配置或不配置。
    """
    try:
        # 子进程内部可以重新简单配置 logger，或者传 None (如果 train.py 处理了 None)
        # 这里建议重新初始化一个临时 logger 以免崩溃
        worker_logger = logging.getLogger("worker")
        
        # 运行训练
        metrics = train.run_training(
            sub_config_list, 
            worker_logger, 
            data_cfg, 
            train_cfg, 
            model_cfg
        )
        # 将结果放入队列传回父进程
        result_queue.put(metrics)
    except Exception as e:
        # 如果崩溃，传回错误信息
        result_queue.put(f"ERROR: {str(e)}")

class GroupGAOptimizer:
    def __init__(self, logger):
        self.logger = logger
        self.population = [np.random.randint(0, 2, GENE_LENGTH) for _ in range(POP_SIZE)]
        self.best_f1 = -1.0
        self.best_mask = None
        
        # 🌟 新增：用于记录进化的详细历史
        self.history = [] 
        
        self.data_cfg = train.DataConfig()
        self.train_cfg = train.TrainConfig()
        self.train_cfg.stride = 8
        self.train_cfg.epochs = 5
        self.train_cfg.use_cache = False
        self.model_cfg = train.ConvLSTMConfig()

    def calculate_fitness(self, mask):
        sub_config_list = [EVOLVABLE_CONFIG[i] for i, bit in enumerate(mask) if bit == 1]
        sub_config_list += MANDATORY_CONFIG
        
        # 🌟 核心修改：使用多进程启动训练
        result_queue = mp.Queue()
        
        # 创建子进程
        # 注意：这里我们不再把 self.logger 传进去，因为它不可序列化 (Pickle Error)
        p = mp.Process(
            target=train_worker, 
            args=(result_queue, sub_config_list, self.data_cfg, self.train_cfg, self.model_cfg)
        )
        
        try:
            p.start()
            # 等待结果，设置一个超时时间（可选，比如 30 分钟）
            metrics = result_queue.get() 
            p.join() # 彻底回收子进程资源
            
            # 检查子进程是否报错
            if isinstance(metrics, str) and metrics.startswith("ERROR"):
                self.logger.error(f"❌ Subprocess failed: {metrics}")
                return 0.0

            # --- 以下逻辑保持不变 (提取指标、计算分数) ---
            val_f1 = metrics.get('val_f1', 0) if isinstance(metrics, dict) else 0
            test_f1 = metrics.get('test_f1', 0) if isinstance(metrics, dict) else 0
            overfit_gap = metrics.get('overfit_gap', 0) if isinstance(metrics, dict) else 0
            
            p_long = metrics.get('precision_long', 0)
            r_long = metrics.get('recall_long', 0)
            p_short = metrics.get('precision_short', 0)
            r_short = metrics.get('recall_short', 0)

            is_dead_model = (r_long < 0.005) and (r_short < 0.005)
            
            score = val_f1
            if p_long > 0.45: score += 0.05 
            if p_short > 0.45: score += 0.05
            score -= (overfit_gap * 0.3) 
            score -= (len(sub_config_list) * 0.0005)
            
            if is_dead_model:
                score = score * 0.1

            record = {
                "mask": "".join(map(str, mask)),
                "f1": test_f1,
                "val_f1": val_f1,
                "fitness": score,
                "p_long": p_long,
                "r_long": r_long,
                "num_groups": len(sub_config_list)
            }
            if isinstance(metrics, dict): record.update(metrics)
            self.history.append(record)
        
            return max(0.001, score)
            
        except Exception as e:
            self.logger.error(f"❌ Process management failed: {e}")
            if p.is_alive():
                p.terminate()
            return 0.0

    def log_diversity(self):
        """Calculates how different the individuals are from each other."""
        distances = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                dist = np.sum(self.population[i] != self.population[j])
                distances.append(dist)
        avg_dist = np.mean(distances)
        self.logger.info(f"🧬 Population Diversity (Avg Hamming Distance): {avg_dist:.2f} / {GENE_LENGTH}")
        return avg_dist

    def analyze_importance(self, top_n=10):
        if not self.history: return
        
        df_hist = pd.DataFrame(self.history)
        # Convert mask string back to columns
        mask_cols = df_hist['mask'].apply(lambda x: pd.Series(list(map(int, x))))
        
        self.logger.info("\n📊 === Alpha Factor Contribution Analysis ===")
        
        results = []
        for i in range(GENE_LENGTH):
            group_name = EVOLVABLE_CONFIG[i][0].__name__
            # Correlation between this feature being 'ON' and the F1 score
            correlation = mask_cols[i].corr(df_hist['f1'])
            selection_rate = mask_cols[i].mean()
            
            results.append({
                "Group": group_name,
                "Corr_with_F1": correlation,
                "Selection_Rate": selection_rate
            })
            
        res_df = pd.DataFrame(results).sort_values("Corr_with_F1", ascending=False)
        for _, row in res_df.iterrows():
            star = "⭐" if row['Corr_with_F1'] > 0.3 else "  "
            self.logger.info(f"{star} {row['Group']:<25} | Corr: {row['Corr_with_F1']:>6.2f} | Rate: {row['Selection_Rate']:>6.1%}")

    def save_history(self):
        """🌟 将所有尝试过的组合保存到 CSV"""
        df_hist = pd.DataFrame(self.history)
        path = os.path.join(common.TEMPORARY_DIR, "ga_evolution_history.csv")
        df_hist.to_csv(path, index=False)
        self.logger.warning(f"💾 进化历史已保存至: {path}")

    def evolve(self):
        for gen in range(GENERATIONS):
            self.logger.info(f"\n" + "🌀" * 10 + f" Generation {gen} " + "🌀" * 10)
            
            # 计算当前代所有个体的适应度
            fitness_scores = []
            for ind in self.population:
                score = self.calculate_fitness(ind)
                fitness_scores.append(score)
            
            # 记录历史最优
            max_idx = np.argmax(fitness_scores)
            if fitness_scores[max_idx] > self.best_f1:
                self.best_f1 = fitness_scores[max_idx]
                self.best_mask = self.population[max_idx].copy()
                self.logger.warning(f"🏆 New Champion Found! F1: {self.best_f1:.4f} | Mask: {self.best_mask}")

            # 选择、交叉、变异逻辑 (简化版)
            self.population = self._create_next_generation(fitness_scores)

            # 🌟 NEW: Generation Summary Table
            gen_history = self.history[-POP_SIZE:]
            df_gen = pd.DataFrame(gen_history)
            
            self.logger.info(f"\n--- Gen {gen} Summary ---")
            self.logger.info(f"Avg F1: {df_gen['f1'].mean():.4f} | Max F1: {df_gen['f1'].max():.4f}")
            self.logger.info(f"Avg Groups: {df_gen['num_groups'].mean():.1f}")
            self.log_diversity() # Check if we are stuck
            
            if gen % 2 == 0: # Analyze importance every 2 generations
                self.analyze_importance()
        # 🌟 新增：在循环结束后强制执行最后一次分析
        self.logger.info("\n🏁 Evolution Finished. Final Alpha Analysis:")
        self.analyze_importance()
        self.save_history() # 顺便保存历史到 CSV

    def _create_next_generation(self, scores):
        # 轮盘赌选择
        idx = np.argsort(scores)[-int(POP_SIZE/2):] # 保留表现最好的前一半 (精英策略)
        parents = [self.population[i] for i in idx]
        
        next_gen = list(parents) # 精英直接晋级
        while len(next_gen) < POP_SIZE:
            # 随机交叉
            p1, p2 = random.sample(parents, 2)
            cp = random.randint(1, GENE_LENGTH - 1)
            child = np.concatenate([p1[:cp], p2[cp:]])
            # 随机变异
            if random.random() < MUTATION_RATE:
                m_idx = random.randint(0, GENE_LENGTH - 1)
                child[m_idx] = 1 - child[m_idx]
            next_gen.append(child)
        return next_gen

if __name__ == "__main__":
    logger, _ = common.setup_session_logger(sub_folder='ga_group_select')
    mp.set_start_method("spawn", force=True)
    # preparation.main(common.FEATURE_CONFIG_LIST, logger)
    optimizer = GroupGAOptimizer(logger)
    optimizer.evolve()