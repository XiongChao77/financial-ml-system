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

# Path setup
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))

from data_process import common, preparation
from model import train

# ==============================================================================
# 🧬 GA core parameters (semantic tuning version)
# ==============================================================================
POP_SIZE = 40          # population size: 40 (keeps the diversity)
GENERATIONS = 30       # generations: 25 (enough time for the algorithm to prune)
MUTATION_RATE = 0.1    # mutation rate: 0.1 (stable building blocks, mostly fine tuning)
ELITISM_COUNT = 2      # elitism: the 2 strongest of each generation pass on directly

# Penalty coefficients
PENALTY_FEATURE_COUNT = 0.0005  # every extra feature costs 0.0005 F1 (Occam's razor)
PENALTY_OVERFIT = 0.3           # overfitting penalty weight

# 1. Identify and extract the mandatory and the optional groups
# Origin and Candle are always kept as the semantic base
FULL_CONFIG = common.FEATURE_GROUP_LIST
MANDATORY_CONFIG = [item for item in FULL_CONFIG if item[0].__name__ in ["FeatureOrigin"]]
EVOLVABLE_CONFIG = [item for item in FULL_CONFIG if item[0].__name__ not in ["FeatureOrigin"]]

# 2. Gene length (only the optional part evolves)
GENE_LENGTH = len(EVOLVABLE_CONFIG)

# ==============================================================================
# 🛠️ Subprocess worker (with crash isolation)
# ==============================================================================
def train_worker(result_queue, sub_config_list, data_cfg, train_cfg, model_cfg):
    """
    Runs the training in an isolated subprocess, so a GPU memory leak or a model crash cannot hit the main process.
    """
    try:
        # Re-initialize a simple logger
        worker_logger = logging.getLogger(f"worker_{os.getpid()}")
        worker_logger.setLevel(logging.WARNING)
        
        # Run the training
        metrics = train.run_training(
            sub_config_list, 
            worker_logger, 
            data_cfg, 
            train_cfg, 
            model_cfg
        )
        # Return a dict on success
        result_queue.put(metrics)
        
    except Exception:
        # 💥 Catch every crash and return the full stack trace
        error_msg = traceback.format_exc()
        result_queue.put(f"ERROR: {error_msg}")

# ==============================================================================
# 🧠 GA optimizer main class
# ==============================================================================
class GroupGAOptimizer:
    def __init__(self, logger):
        self.logger = logger
        
        # Initialize the state
        self.population = [np.random.randint(0, 2, GENE_LENGTH) for _ in range(POP_SIZE)]
        self.best_f1 = -1.0
        self.best_mask = None
        self.history = []
        self.start_gen = 0
        
        # Configuration initialization
        self.data_cfg = train.DataConfig()
        self.train_cfg = train.TrainConfig()
        self.train_cfg.stride = 16
        self.train_cfg.epochs = 5      # fewer epochs for a fast check during the GA search
        self.train_cfg.use_cache = False # the cache must be off, the feature combination changes
        self.model_cfg = train.ConvLSTMConfig()
        
        # Checkpoint path
        self.checkpoint_path = os.path.join(common.TEMPORARY_DIR, "ga_checkpoint.pkl")
        
        # 🚀 Try to load a checkpoint on start-up
        self.load_checkpoint()

    def save_checkpoint(self, gen):
        """Save the current evolution state, to survive a power cut / crash"""
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
        """Try to load a previous checkpoint"""
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'rb') as f:
                    cp = pickle.load(f)
                
                self.start_gen = cp['gen'] + 1
                self.population = cp['population']
                self.best_f1 = cp['best_f1']
                self.best_mask = cp['best_mask']
                self.history = cp['history']
                
                # Restore the RNG state so the run stays reproducible
                random.setstate(cp['random_state'])
                np.random.set_state(cp['np_state'])
                
                self.logger.warning(f"♻️  Resuming from Checkpoint: Generation {self.start_gen}")
                self.logger.warning(f"🏆 Current Best F1: {self.best_f1:.4f}")
            except Exception as e:
                self.logger.error(f"❌ Failed to load checkpoint (will start fresh): {e}")

    def calculate_fitness(self, mask):
        """Fitness score of one individual"""
        # 1. Assemble the blocks: mandatory groups + the optional groups the gene selected
        sub_config_list = [EVOLVABLE_CONFIG[i] for i, bit in enumerate(mask) if bit == 1]
        sub_config_list += MANDATORY_CONFIG  # add Origin and Candle
        
        # Safety check: a mutation may deselect every optional group, running with the mandatory ones is fine
        
        result_queue = mp.Queue()
        p = mp.Process(
            target=train_worker, 
            args=(result_queue, sub_config_list, self.data_cfg, self.train_cfg, self.model_cfg)
        )
        
        try:
            p.start()
            # Timeout, so a deadlock cannot block us (e.g. 30 minutes)
            metrics = result_queue.get(timeout=1800) 
            p.join()
            
            # 2. Check for errors
            if isinstance(metrics, str) and metrics.startswith("ERROR"):
                self.logger.error(f"❌ Subprocess CRASHED:\n{metrics}")
                return 0.001

            # 3. Extract the metrics
            val_f1 = metrics.get('val_f1', 0)
            test_f1 = metrics.get('test_f1', 0) # recorded only, not part of the fitness
            overfit_gap = metrics.get('overfit_gap', 0)
            p_long = metrics.get('precision_long', 0)
            p_short = metrics.get('precision_short', 0)
            r_long = metrics.get('recall_long', 0)
            r_short = metrics.get('recall_short', 0)

            # 4. 💀 dead model check
            if r_long < 0.005 and r_short < 0.005:
                return 0.001 # very low score, eliminated

            # 5.  Score calculation
            score = val_f1
            
            # Reward accuracy
            if p_long > 0.45: score += 0.05 
            if p_short > 0.45: score += 0.05
            
            # Penalize overfitting
            score -= (overfit_gap * PENALTY_OVERFIT) 
            
            # Penalize the feature count (Occam's razor)
            score -= (len(sub_config_list) * PENALTY_FEATURE_COUNT)
            
            # Record the history
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
        """Population diversity (Hamming distance)"""
        if POP_SIZE < 2: return 0
        distances = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                dist = np.sum(self.population[i] != self.population[j])
                distances.append(dist)
        avg_dist = np.mean(distances)
        self.logger.info(f"🧬 Diversity (Avg Hamming): {avg_dist:.2f} / {GENE_LENGTH}")

    def analyze_importance(self):
        """Find out which features are the 'chosen blocks'"""
        if not self.history: return
        
        df_hist = pd.DataFrame(self.history)
        if 'mask' not in df_hist.columns: return

        # Turn the mask strings into a matrix
        mask_cols = df_hist['mask'].apply(lambda x: pd.Series(list(map(int, x))))
        
        self.logger.info("\n📊 === Feature Importance Analysis ===")
        results = []
        
        for i in range(GENE_LENGTH):
            group_name = EVOLVABLE_CONFIG[i][0].__name__
            # Correlation between this feature being on and the F1
            if df_hist['f1'].std() > 0: # guard against an all-zero column
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
        """Main evolution loop"""
        self.logger.info(f"🚀 Starting Evolution: {GENERATIONS} gens, Pop {POP_SIZE}")
        
        for gen in range(self.start_gen, GENERATIONS):
            self.logger.info(f"\n" + "="*40)
            self.logger.info(f"🌀 GENERATION {gen} / {GENERATIONS-1}")
            self.logger.info(f"="*40)
            
            fitness_scores = []
            
            # 1. Fitness
            for i, ind in enumerate(self.population):
                start_t = time.time()
                score = self.calculate_fitness(ind)
                elapsed = time.time() - start_t
                fitness_scores.append(score)
                self.logger.info(f"  > Ind {i+1:02d}/{POP_SIZE} | Fit: {score:.4f} | Time: {elapsed:.1f}s")
            
            # 2. Record the best
            fitness_scores = np.array(fitness_scores)
            max_idx = np.argmax(fitness_scores)
            
            if fitness_scores[max_idx] > self.best_f1:
                self.best_f1 = fitness_scores[max_idx]
                self.best_mask = self.population[max_idx].copy()
                
                # Print the strongest blocks so far
                best_names = [EVOLVABLE_CONFIG[i][0].__name__ for i, bit in enumerate(self.best_mask) if bit == 1]
                best_names += [m[0].__name__ for m in MANDATORY_CONFIG]
                self.logger.warning(f"🏆 NEW RECORD! Score: {self.best_f1:.4f}")
                self.logger.warning(f"🧱 Best Blocks: {best_names}")

            # 3. Build the next generation
            self.population = self._create_next_generation(fitness_scores)
            
            # 4. Statistics and analysis
            self.log_diversity()
            if gen % 2 == 0: self.analyze_importance()
            
            # 5. 💾 key step: checkpoint every generation
            self.save_checkpoint(gen)
            
            # 6. Save the full history CSV
            pd.DataFrame(self.history).to_csv(
                os.path.join(common.TEMPORARY_DIR, "ga_history_full.csv"), index=False
            )

        self.logger.info("\n🏁 Evolution Completed.")
        self.analyze_importance()

    def _create_next_generation(self, scores):
        """Selection, crossover, mutation"""
        # Elitism
        sorted_indices = np.argsort(scores)[::-1] # descending
        elites = [self.population[i] for i in sorted_indices[:ELITISM_COUNT]]
        
        next_gen = list(elites)
        
        # Tournament selection of the parents
        def tournament_select():
            candidates = random.sample(range(POP_SIZE), 3)
            best = candidates[0]
            for c in candidates[1:]:
                if scores[c] > scores[best]: best = c
            return self.population[best]

        while len(next_gen) < POP_SIZE:
            p1 = tournament_select()
            p2 = tournament_select()
            
            # Crossover
            child = p1.copy()
            if GENE_LENGTH > 1:
                cx_point = random.randint(1, GENE_LENGTH - 1)
                child = np.concatenate([p1[:cx_point], p2[cx_point:]])
            
            # Mutation
            for k in range(GENE_LENGTH):
                if random.random() < MUTATION_RATE:
                    child[k] = 1 - child[k]
            
            next_gen.append(child)
            
        return next_gen

if __name__ == "__main__":
    # The spawn start method is required for PyTorch multiprocessing
    mp.set_start_method("spawn", force=True)
    
    logger, _ = common.setup_session_logger(sub_folder='ga_group_select')
    
    # Print the mandatory configuration to confirm the semantic base
    mandatory_names = [item[0].__name__ for item in MANDATORY_CONFIG]
    logger.info(f"🔒 Mandatory Semantic Base: {mandatory_names}")
    
    optimizer = GroupGAOptimizer(logger)
    optimizer.evolve()