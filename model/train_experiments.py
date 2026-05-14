import subprocess
import os,sys
import re
import pandas as pd
import time
from typing import List, Dict, Any

# Training script path
TRAIN_SCRIPT = 'model/cnn_timeseries_torch.py'

# =======================================================
# 1. Define experiment configs
# =======================================================
EXPERIMENTS = [
    # Experiment A: current stable config (Batch Size 256)
    {'name': 'Exp_A_BS256_D03', 'batch_size': 256, 'dropout': 0.3, 'lr': 3e-4, 'epochs': 25, 'comment': 'Base Config'},
    
    # Experiment B: smaller batch size (more regularization/generalization)
    {'name': 'Exp_B_BS128_D03', 'batch_size': 128, 'dropout': 0.3, 'lr': 3e-4, 'epochs': 30, 'comment': 'Smaller Batch Size'},
    
    # Experiment C: higher regularization (higher dropout)
    {'name': 'Exp_C_BS256_D05', 'batch_size': 256, 'dropout': 0.5, 'lr': 3e-4, 'epochs': 30, 'comment': 'Higher Dropout'},
]

# =======================================================
# 2. Helper: parse training metrics
# =======================================================
def parse_metrics(output: str) -> Dict[str, Any]:
    """Extract key metrics from training output (e.g., Macro F1 and Accuracy)."""
    metrics = {'macro_f1': None, 'accuracy': None}

    # 1. Extract Macro F1
    f1_match = re.search(r'Test macro-F1:([\d\.]+)', output)
    if f1_match:
        metrics['macro_f1'] = float(f1_match.group(1))

    # 2. Extract Accuracy
    acc_match = re.search(r'accuracy\s+([\d\.]+)', output)
    if acc_match:
        metrics['accuracy'] = float(acc_match.group(1))

    # 3. Extract final va_loss (used as early-stopping indicator)
    va_loss_match = re.search(r'va_loss\s+([\d\.]+)\s+\|\s+va_macroF1\s+([\d\.]+)', output.split('Early stopping.')[0].split('\n')[-2])
    if va_loss_match:
        metrics['best_val_loss'] = float(va_loss_match.group(1))
    
    return metrics

# =======================================================
# 3. Core runner
# =======================================================
def run_experiment_suite(experiments: List[Dict[str, Any]]):
    results = []
    
    for i, config in enumerate(experiments):
        print(f"\n========================================================")
        print(f"| Starting Experiment {i+1}/{len(experiments)}: {config['name']}")
        print(f"| Config: BS={config['batch_size']}, DO={config['dropout']}, LR={config['lr']}")
        print(f"========================================================\n")
        
        start_time = time.time()
        
        # Build CLI args
        cmd = [
            'python', TRAIN_SCRIPT,
            f"--batch_size={config['batch_size']}",
            f"--dropout={config['dropout']}",
            f"--lr={config['lr']}",
            f"--epochs={config['epochs']}",
        ]
        
        try:
            # Run training script and capture output
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout
            
            # Parse results
            metrics = parse_metrics(output)
            
            # Record results
            run_time = time.time() - start_time
            result_entry = {
                'Experiment': config['name'],
                'Batch_Size': config['batch_size'],
                'Dropout': config['dropout'],
                'Macro_F1': metrics['macro_f1'],
                'Accuracy': metrics['accuracy'],
                'Best_Val_Loss': metrics.get('best_val_loss'),
                'Run_Time_s': round(run_time, 2),
                'Comment': config['comment'],
            }
            results.append(result_entry)
            
            print(f"\n[SUCCESS] {config['name']} completed in {run_time:.2f}s. Macro F1: {metrics['macro_f1']:.4f}")
            
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Experiment {config['name']} FAILED.")
            print(f"Stdout:\n{e.stdout[-1000:]}")  # Print tail stdout for debugging
            print(f"Stderr:\n{e.stderr}")
            # Record failure
            results.append({'Experiment': config['name'], 'Status': 'FAILED', 'Batch_Size': config['batch_size'], 'Dropout': config['dropout']})
            
        except FileNotFoundError:
             print(f"\n[FATAL ERROR] Training script not found at {TRAIN_SCRIPT}. Please check the path.")
             break

    # 4. Save final results
    if results:
        df_results = pd.DataFrame(results)
        output_file = f"experiment_results_{int(time.time())}.csv"
        df_results.to_csv(output_file, index=False)
        print(f"\n[DONE] All experiments finished. Results saved to {output_file}")


if __name__ == '__main__':
    # Ensure dependency is installed (pandas)
    try:
        import pandas as pd
    except ImportError:
        print("Error: Pandas is required for saving results. Please run 'pip install pandas'")
        sys.exit(1)
        
    run_experiment_suite(EXPERIMENTS)