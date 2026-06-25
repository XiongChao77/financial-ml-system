import numpy as np
import pandas as pd
import torch, os, sys
import multiprocessing as mp
import logging
import random
import pickle
import traceback
import time
from datetime import datetime

# 路径设置
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))

from data_process import common, preparation
from model import train

# ==============================================================================
# 🧬 GA 核心参数配置 (语义化调优版)
# ==============================================================================
POP_SIZE = 40          # 种群大小：40 (保持多样性)
GENERATIONS = 30       # 进化代数：25 (给算法足够时间做减法)
MUTATION_RATE = 0.1    # 变异率：0.1 (稳定积木，微调为主)
ELITISM_COUNT = 2      # 精英保留：每代最强的2个直接晋级

# 惩罚系数
PENALTY_FEATURE_COUNT = 0.0005  # 每多一个特征，F1 扣 0.0005 (奥卡姆剃刀)
PENALTY_OVERFIT = 0.3           # 过拟合惩罚权重

# 1. 识别并提取必选组和可选组
# 强制保留 Origin 和 Candle 作为语义底座
FULL_CONFIG = common.FEATURE_GROUP_LIST
MANDATORY_CONFIG = [item for item in FULL_CONFIG if item[0].__name__ in ["FeatureOrigin"]]
EVOLVABLE_CONFIG = [item for item in FULL_CONFIG if item[0].__name__ not in ["FeatureOrigin"]]

# 2. 定义基因长度 (只进化可选部分)
GENE_LENGTH = len(EVOLVABLE_CONFIG)

# ==============================================================================
# 🛠️ 子进程工作函数 (带异常熔断)
# ==============================================================================
def train_worker(result_queue, sub_config_list, data_cfg, train_cfg, model_cfg):
    """
    在隔离的子进程中运行训练，防止 GPU 显存泄漏或模型崩溃影响主进程。
    """
    try:
        # 重新初始化简单的 logger
        worker_logger = logging.getLogger(f"worker_{os.getpid()}")
        worker_logger.setLevel(logging.WARNING)
        
        # 运行训练
        metrics = train.run_training(
            sub_config_list, 
            worker_logger, 
            data_cfg, 
            train_cfg, 
            model_cfg
        )
        # 成功则返回字典
        result_queue.put(metrics)
        
    except Exception:
        # 💥 捕获所有崩溃，并返回完整的错误堆栈
        error_msg = traceback.format_exc()
        result_queue.put(f"ERROR: {error_msg}")

# ==============================================================================
# 🧠 GA 优化器主类
# ==============================================================================
class GroupGAOptimizer:
    def __init__(self, logger):
        self.logger = logger
        
        # 初始化状态
        self.population = [np.random.randint(0, 2, GENE_LENGTH) for _ in range(POP_SIZE)]
        self.best_f1 = -1.0
        self.best_mask = None
        self.history = []
        self.start_gen = 0
        
        # 配置文件初始化
        self.data_cfg = train.DataConfig()
        self.train_cfg = train.TrainConfig()
        self.train_cfg.stride = 16
        self.train_cfg.epochs = 5      # GA 搜索时用较少 Epoch 快速验证
        self.train_cfg.use_cache = False # 必须关闭缓存，因为特征组合变了
        self.model_cfg = train.ConvLSTMConfig()
        
        # 存档路径
        self.checkpoint_path = os.path.join(common.TEMPORARY_DIR, "ga_checkpoint.pkl")
        
        # 🚀 启动时尝试加载存档
        self.load_checkpoint()

    def save_checkpoint(self, gen):
        """保存当前进化状态，防止断电/崩溃"""
        try:
            checkpoint = {
                'gen': gen,
                'population': self.population,
                'best_f1': self.best_f1,
                'best_mask': self.best_mask,
                'history': self.history,
                'random_state': random.getstate(),
                'np_state': np.random.get_state()
            }
            with open(self.checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            self.logger.warning(f"💾 Checkpoint saved at Generation {gen}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save checkpoint: {e}")

    def load_checkpoint(self):
        """尝试加载历史存档"""
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'rb') as f:
                    cp = pickle.load(f)
                
                self.start_gen = cp['gen'] + 1
                self.population = cp['population']
                self.best_f1 = cp['best_f1']
                self.best_mask = cp['best_mask']
                self.history = cp['history']
                
                # 恢复随机数状态，保证复现
                random.setstate(cp['random_state'])
                np.random.set_state(cp['np_state'])
                
                self.logger.warning(f"♻️  Resuming from Checkpoint: Generation {self.start_gen}")
                self.logger.warning(f"🏆 Current Best F1: {self.best_f1:.4f}")
            except Exception as e:
                self.logger.error(f"❌ Failed to load checkpoint (will start fresh): {e}")

    def calculate_fitness(self, mask):
        """计算个体的适应度分数"""
        # 1. 拼接积木：必选组 + 基因选中的可选组
        sub_config_list = [EVOLVABLE_CONFIG[i] for i, bit in enumerate(mask) if bit == 1]
        sub_config_list += MANDATORY_CONFIG  # 加上 Origin 和 Candle
        
        # 安全检查：如果变异导致所有可选组都没选，只剩必选组，也可以跑
        
        result_queue = mp.Queue()
        p = mp.Process(
            target=train_worker, 
            args=(result_queue, sub_config_list, self.data_cfg, self.train_cfg, self.model_cfg)
        )
        
        try:
            p.start()
            # 设置超时，防止死锁 (例如 30分钟)
            metrics = result_queue.get(timeout=1800) 
            p.join()
            
            # 2. 检查报错
            if isinstance(metrics, str) and metrics.startswith("ERROR"):
                self.logger.error(f"❌ Subprocess CRASHED:\n{metrics}")
                return 0.001

            # 3. 提取指标
            val_f1 = metrics.get('val_f1', 0)
            test_f1 = metrics.get('test_f1', 0) # 仅记录，不参与 fitness
            overfit_gap = metrics.get('overfit_gap', 0)
            p_long = metrics.get('precision_long', 0)
            p_short = metrics.get('precision_short', 0)
            r_long = metrics.get('recall_long', 0)
            r_short = metrics.get('recall_short', 0)

            # 4. 💀 死模判定 (Dead Model Check)
            if r_long < 0.005 and r_short < 0.005:
                return 0.001 # 极低分，淘汰

            # 5.  适应度公式 (Score Calculation)
            score = val_f1
            
            # 奖励高精度
            if p_long > 0.45: score += 0.05 
            if p_short > 0.45: score += 0.05
            
            # 惩罚过拟合
            score -= (overfit_gap * PENALTY_OVERFIT) 
            
            # 惩罚特征数量 (奥卡姆剃刀)
            score -= (len(sub_config_list) * PENALTY_FEATURE_COUNT)
            
            # 记录历史
            record = {
                "mask": "".join(map(str, mask)),
                "f1": test_f1,
                "val_f1": val_f1,
                "fitness": score,
                "num_groups": len(sub_config_list),
                "active_features": [cfg[0].__name__ for cfg in sub_config_list]
            }
            self.history.append(record)
        
            return max(0.001, score)
            
        except Exception as e:
            self.logger.error(f"❌ Process failed: {e}")
            if p.is_alive(): p.terminate()
            return 0.001

    def log_diversity(self):
        """计算种群多样性 (汉明距离)"""
        if POP_SIZE < 2: return 0
        distances = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                dist = np.sum(self.population[i] != self.population[j])
                distances.append(dist)
        avg_dist = np.mean(distances)
        self.logger.info(f"🧬 Diversity (Avg Hamming): {avg_dist:.2f} / {GENE_LENGTH}")

    def analyze_importance(self):
        """分析哪些特征是'天选积木'"""
        if not self.history: return
        
        df_hist = pd.DataFrame(self.history)
        if 'mask' not in df_hist.columns: return

        # 转换 mask 字符串为矩阵
        mask_cols = df_hist['mask'].apply(lambda x: pd.Series(list(map(int, x))))
        
        self.logger.info("\n📊 === Feature Importance Analysis ===")
        results = []
        
        for i in range(GENE_LENGTH):
            group_name = EVOLVABLE_CONFIG[i][0].__name__
            # 计算该特征开启与否与 F1 的相关性
            if df_hist['f1'].std() > 0: # 防止全0报错
                corr = mask_cols[i].corr(df_hist['f1'])
            else:
                corr = 0
            select_rate = mask_cols[i].mean()
            
            results.append({
                "Feature": group_name,
                "Corr": corr,
                "Rate": select_rate
            })
            
        res_df = pd.DataFrame(results).sort_values("Corr", ascending=False)
        
        for _, row in res_df.iterrows():
            mark = "⭐" if row['Corr'] > 0.2 else "  "
            mark = "🔥" if row['Rate'] > 0.8 else mark
            self.logger.info(f"{mark} {row['Feature']:<20} | Corr: {row['Corr']:>6.2f} | Rate: {row['Rate']:>6.1%}")

    def evolve(self):
        """主进化循环"""
        self.logger.info(f"🚀 Starting Evolution: {GENERATIONS} gens, Pop {POP_SIZE}")
        
        for gen in range(self.start_gen, GENERATIONS):
            self.logger.info(f"\n" + "="*40)
            self.logger.info(f"🌀 GENERATION {gen} / {GENERATIONS-1}")
            self.logger.info(f"="*40)
            
            fitness_scores = []
            
            # 1. 计算适应度
            for i, ind in enumerate(self.population):
                start_t = time.time()
                score = self.calculate_fitness(ind)
                elapsed = time.time() - start_t
                fitness_scores.append(score)
                self.logger.info(f"  > Ind {i+1:02d}/{POP_SIZE} | Fit: {score:.4f} | Time: {elapsed:.1f}s")
            
            # 2. 记录最优
            fitness_scores = np.array(fitness_scores)
            max_idx = np.argmax(fitness_scores)
            
            if fitness_scores[max_idx] > self.best_f1:
                self.best_f1 = fitness_scores[max_idx]
                self.best_mask = self.population[max_idx].copy()
                
                # 打印当前最强积木
                best_names = [EVOLVABLE_CONFIG[i][0].__name__ for i, bit in enumerate(self.best_mask) if bit == 1]
                best_names += [m[0].__name__ for m in MANDATORY_CONFIG]
                self.logger.warning(f"🏆 NEW RECORD! Score: {self.best_f1:.4f}")
                self.logger.warning(f"🧱 Best Blocks: {best_names}")

            # 3. 生成下一代
            self.population = self._create_next_generation(fitness_scores)
            
            # 4. 统计与分析
            self.log_diversity()
            if gen % 2 == 0: self.analyze_importance()
            
            # 5. 💾 关键步骤：每代存档
            self.save_checkpoint(gen)
            
            # 6. 保存完整历史CSV
            pd.DataFrame(self.history).to_csv(
                os.path.join(common.TEMPORARY_DIR, "ga_history_full.csv"), index=False
            )

        self.logger.info("\n🏁 Evolution Completed.")
        self.analyze_importance()

    def _create_next_generation(self, scores):
        """选择、交叉、变异"""
        # 精英策略
        sorted_indices = np.argsort(scores)[::-1] # 降序
        elites = [self.population[i] for i in sorted_indices[:ELITISM_COUNT]]
        
        next_gen = list(elites)
        
        # 锦标赛选择父代
        def tournament_select():
            candidates = random.sample(range(POP_SIZE), 3)
            best = candidates[0]
            for c in candidates[1:]:
                if scores[c] > scores[best]: best = c
            return self.population[best]

        while len(next_gen) < POP_SIZE:
            p1 = tournament_select()
            p2 = tournament_select()
            
            # 交叉
            child = p1.copy()
            if GENE_LENGTH > 1:
                cx_point = random.randint(1, GENE_LENGTH - 1)
                child = np.concatenate([p1[:cx_point], p2[cx_point:]])
            
            # 变异
            for k in range(GENE_LENGTH):
                if random.random() < MUTATION_RATE:
                    child[k] = 1 - child[k]
            
            next_gen.append(child)
            
        return next_gen

if __name__ == "__main__":
    # 必须设置 spawn 启动方式，兼容 PyTorch 多进程
    mp.set_start_method("spawn", force=True)
    
    logger, _ = common.setup_session_logger(sub_folder='ga_group_select')
    
    # 打印必选配置，确认语义底座
    mandatory_names = [item[0].__name__ for item in MANDATORY_CONFIG]
    logger.info(f"🔒 Mandatory Semantic Base: {mandatory_names}")
    
    optimizer = GroupGAOptimizer(logger)
    optimizer.evolve()