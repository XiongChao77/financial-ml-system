import subprocess
import os,sys
import re
import pandas as pd
import time
from typing import List, Dict, Any

# 假设您的训练脚本路径
TRAIN_SCRIPT = 'model/cnn_timeseries_torch.py'

# =======================================================
# 1. 定义实验配置列表
# =======================================================
EXPERIMENTS = [
    # 实验 A: 当前稳定配置 (Batch Size 256)
    {'name': 'Exp_A_BS256_D03', 'batch_size': 256, 'dropout': 0.3, 'lr': 3e-4, 'epochs': 25, 'comment': 'Base Config'},
    
    # 实验 B: 减小 Batch Size (增加正则化/泛化能力)
    {'name': 'Exp_B_BS128_D03', 'batch_size': 128, 'dropout': 0.3, 'lr': 3e-4, 'epochs': 30, 'comment': 'Smaller Batch Size'},
    
    # 实验 C: 增加正则化 (高 Dropout)
    {'name': 'Exp_C_BS256_D05', 'batch_size': 256, 'dropout': 0.5, 'lr': 3e-4, 'epochs': 30, 'comment': 'Higher Dropout'},
]

# =======================================================
# 2. 辅助函数：解析训练结果
# =======================================================
def parse_metrics(output: str) -> Dict[str, Any]:
    """从训练输出中提取关键指标，如 Macro F1 和 Accuracy"""
    metrics = {'macro_f1': None, 'accuracy': None}

    # 1. 提取 Macro F1 Score
    f1_match = re.search(r'Test macro-F1:([\d\.]+)', output)
    if f1_match:
        metrics['macro_f1'] = float(f1_match.group(1))

    # 2. 提取 Accuracy
    acc_match = re.search(r'accuracy\s+([\d\.]+)', output)
    if acc_match:
        metrics['accuracy'] = float(acc_match.group(1))

    # 3. 提取最终的 va_loss (作为早停性能指标)
    va_loss_match = re.search(r'va_loss\s+([\d\.]+)\s+\|\s+va_macroF1\s+([\d\.]+)', output.split('Early stopping.')[0].split('\n')[-2])
    if va_loss_match:
        metrics['best_val_loss'] = float(va_loss_match.group(1))
    
    return metrics

# =======================================================
# 3. 核心执行函数
# =======================================================
def run_experiment_suite(experiments: List[Dict[str, Any]]):
    results = []
    
    for i, config in enumerate(experiments):
        print(f"\n========================================================")
        print(f"| Starting Experiment {i+1}/{len(experiments)}: {config['name']}")
        print(f"| Config: BS={config['batch_size']}, DO={config['dropout']}, LR={config['lr']}")
        print(f"========================================================\n")
        
        start_time = time.time()
        
        # 构造命令行参数
        cmd = [
            'python', TRAIN_SCRIPT,
            f"--batch_size={config['batch_size']}",
            f"--dropout={config['dropout']}",
            f"--lr={config['lr']}",
            f"--epochs={config['epochs']}",
        ]
        
        try:
            # 执行训练脚本并捕获输出
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout
            
            # 解析结果
            metrics = parse_metrics(output)
            
            # 记录结果
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
            print(f"Stdout:\n{e.stdout[-1000:]}") # 打印最后的 stdout 帮助调试
            print(f"Stderr:\n{e.stderr}")
            # 记录失败结果
            results.append({'Experiment': config['name'], 'Status': 'FAILED', 'Batch_Size': config['batch_size'], 'Dropout': config['dropout']})
            
        except FileNotFoundError:
             print(f"\n[FATAL ERROR] Training script not found at {TRAIN_SCRIPT}. Please check the path.")
             break

    # 4. 保存最终结果
    if results:
        df_results = pd.DataFrame(results)
        output_file = f"experiment_results_{int(time.time())}.csv"
        df_results.to_csv(output_file, index=False)
        print(f"\n[DONE] All experiments finished. Results saved to {output_file}")


if __name__ == '__main__':
    # 确保依赖库已安装 (pandas)
    try:
        import pandas as pd
    except ImportError:
        print("Error: Pandas is required for saving results. Please run 'pip install pandas'")
        sys.exit(1)
        
    run_experiment_suite(EXPERIMENTS)