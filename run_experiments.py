import os, sys, time, logging, argparse
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))
from data_process import preparation, common
import model.train_old as train
from trade.bt import simulation

def main():
    # 1. 配置参数解析器：支持长短选项
    parser = argparse.ArgumentParser(description="Quant Trading Pipeline Control")
    
    parser.add_argument('-p', '--prep', action='store_true', 
                        help='Execute data preparation stage')
    parser.add_argument('-t', '--train', action='store_true', 
                        help='Execute model training stage')
    parser.add_argument('-s', '--sim', action='store_true', 
                        help='Execute backtest simulation stage')
    parser.add_argument('-a', '--all', action='store_true', 
                        help='Execute all stages')

    args = parser.parse_args()

    # 2. 补全类型提示
    logger: logging.Logger
    logger, _ = common.setup_session_logger(sub_folder='experiment', file_level=logging.DEBUG)
    
    # 是否开启全流程
    run_all = args.all
    
    begin_time = time.time()
    stats = {"preparation": 0.0, "train": 0.0, "simulation": 0.0}

    # --- 阶段 1: Preparation ---
    if args.prep or run_all:
        logger.info(">>> [1/3] Starting Preparation...")
        start = time.time()
        preparation.main(logger)
        stats["preparation"] = time.time() - start

    # --- 阶段 2: Train ---
    if args.train or run_all:
        logger.info(">>> [2/3] Starting Training...")
        start = time.time()
        train.main(logger)
        stats["train"] = time.time() - start

    # --- 阶段 3: Simulation ---
    if args.sim or run_all:
        logger.info(">>> [3/3] Starting Simulation...")
        start = time.time()
        args = simulation.Parameters()
        simulation.main(logger,args)
        args.thresh = 0.4
        simulation.main(logger,args)
        stats["simulation"] = time.time() - start

    end_time = time.time()
    total_time = end_time - begin_time

    # 3. 打印精简的耗时报告
    msg = (f"Run Summary | "
           f"Prep: {stats['preparation']:.2f}s | "
           f"Train: {stats['train']:.2f}s | "
           f"Sim: {stats['simulation']:.2f}s | "
           f"Total: {total_time:.2f}s")
    print("\n" + "="*len(msg))
    print(msg)
    print("="*len(msg))

if __name__ == "__main__":
    main()