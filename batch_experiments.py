import os, sys, time, logging, argparse, copy, json
from datetime import datetime, timedelta  # 新增用于时间计算
import numpy as np
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))
from data_process import preparation, common
import model.train_2head as train
from trade.bt import simulation

def load_checkpoint(exp_dir):
    """加载进度游标: [p_idx, t_idx, s_idx]"""
    path = os.path.join(exp_dir, "checkpoint_cursor.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return [-1, -1, -1] # 默认全部未开始

def save_checkpoint(exp_dir, cursor):
    """保存当前完成的游标"""
    path = os.path.join(exp_dir, "checkpoint_cursor.json")
    with open(path, "w") as f:
        json.dump(cursor, f)

def finish_time_evaluate(stats):
    """
    仅使用 stats 字典中的数据预估剩余时间。
    """
    # 提取平均耗时 (处理分母为 0 的情况)
    avg_p = stats["preparation"]['time'] / stats["preparation"]['count'] if stats["preparation"]['count'] > 0 else 0
    avg_t = stats["train"]['time'] / stats["train"]['count'] if stats["train"]['count'] > 0 else 0
    avg_s = stats["simulation"]['time'] / stats["simulation"]['count'] if stats["simulation"]['count'] > 0 else 0

    # 如果刚开始运行，没有统计数据，返回占位符
    if avg_p == 0 and avg_t == 0 and avg_s == 0:
        return "[ETA: Calculating...]"

    # 获取配置总量
    max_t = stats['train']['max_count']
    max_s = stats['simulation']['max_count']

    # 获取剩余数量
    left_p = stats['preparation']['left_count']
    left_t = stats['train']['left_count']
    left_s = stats['simulation']['left_count']

    # --- 嵌套剩余时间计算逻辑 ---
    
    # 1. 当前正在运行的 T 块中，剩余的 S 任务时间
    current_t_remaining_s = left_s * avg_s
    
    # 2. 当前正在运行的 P 块中，剩余的 T 块总时间
    # 每个 T 块包含：1次 Training + max_s 次 Simulation
    current_p_remaining_t = left_t * (avg_t + max_s * avg_s)
    
    # 3. 剩余的全部 P 块总时间
    # 每个 P 块包含：1次 Prep + max_t 次 Training + (max_t * max_s) 次 Simulation
    remaining_p_blocks = (left_p - 1) if left_p > 0 else 0
    remaining_full_p_time = remaining_p_blocks * (avg_p + max_t * avg_t + max_t * max_s * avg_s)

    # 总计剩余秒数
    total_left_seconds = current_t_remaining_s + current_p_remaining_t + remaining_full_p_time

    # 格式化输出
    eta_delta = timedelta(seconds=int(total_left_seconds))
    finish_time = datetime.now() + eta_delta
    
    return f" [ETA: {str(eta_delta)} | Finish: {finish_time.strftime('%H:%M:%S')}]"

def main():
    # 1. 配置参数解析器
    parser = argparse.ArgumentParser(description="Quant Trading Pipeline Control")
    parser.add_argument('-p', '--prep', action='store_true', help='Execute data preparation stage')
    parser.add_argument('-t', '--train', action='store_true', help='Execute model training stage')
    parser.add_argument('-s', '--sim', action='store_true', help='Execute backtest simulation stage')
    parser.add_argument('-a', '--all', action='store_true', help='Execute all stages')
    parser.add_argument('-r', '--resume', type=str, help='Resume experiment from specified directory') # 新增
    
    args = parser.parse_args()

    # 2. 实验目录处理 (支持续跑)
    if args.resume:
        exp_dir = args.resume
        if not os.path.exists(exp_dir):
            print(f"❌ Error: Resume directory {exp_dir} not found.")
            return
    else:
        exp_dir = common.create_experiment_dir(
            os.path.join(common.PERSISTENCE_DIR,'batch_experiments'),
            common.CommonDefine.symbol, 
            common.CommonDefine.interval
        )

    # 初始化日志
    logger, _ = common.setup_session_logger(
        log_file_path=os.path.join(exp_dir, 'experiment.log'), 
        console_level=logging.INFO, 
        file_level=logging.INFO
    )
    common.get_git_info(logger)
    
    # 3. 加载断点 (如果是新任务，游标为 [-1, -1, -1])
    start_p, start_t, start_s = load_checkpoint(exp_dir)
    if args.resume and start_p != -1:
        logger.info(f"🔄 Resuming from P_idx={start_p}, T_idx={start_t}, S_idx={start_s}")

    run_all = args.all
    begin_time = time.time()
    stats = {"preparation": {'time': 0,'count':0,'max_count': 0,'left_count': 0,'avg_t':None}, 
            "train": {'time': 0,'count':0,'left_count': 0,'max_count': 0,'avg_t':None},
            "simulation": {'time': 0,'count':0,'left_count': 0,'max_count': 0,'avg_t':None}}

    # ==========================
    #  任务列表生成 (保持原样)
    # ==========================
    
    # --- 阶段 1: Preparation ---
    preparation_task = []
 
    if args.prep or run_all:
        for cn in [96]:    #[96, 128]
            for pn in [16]:     #[12, 16]
                item = common.CommonDefine() 
                item.candlestick_num = cn
                item.predict_num = pn
                preparation_task.append(item)
    else:
        preparation_task.append(common.CommonDefine() )
    stats['preparation']['max_count'] = len(preparation_task)
    stats['preparation']['left_count'] = stats['preparation']['max_count']

    # --- 阶段 2: Train ---
    training_task = []
    if args.train or run_all:
        for flip_penalty in np.linspace(0.5, 2.5, num=20): # 示例减少数量方便测试
            for miss_penalty in np.linspace(flip_penalty*0.2, flip_penalty*1.5, num=10):
                t_cfg = train.TrainConfig()
                t_cfg.use_cache = True # 批量实验通常关闭 Train 内部缓存以响应参数变化
                t_cfg.flip_penalty = float(flip_penalty)
                t_cfg.miss_penalty = float(miss_penalty)
                training_task.append(t_cfg)
    else:
        training_task.append(train.TrainConfig())
    stats['train']['max_count'] = len(training_task)
    stats['train']['left_count'] = stats['train']['max_count']

    # --- 阶段 3: Simulation ---
    simulation_task = []
    if args.sim or run_all:
        for hb in [16, 20]:#[16, 20, 24, 28, 32]
            s_cfg = simulation.StrategyPara() # 假设这是你的策略配置类
            s_cfg.holdbar = hb
            simulation_task.append(s_cfg)
    else:
        simulation_task.append(simulation.StrategyPara()) # 默认值
    stats['simulation']['max_count'] = len(simulation_task)
    stats['simulation']['left_count'] = stats['simulation']['max_count']

    total_task_num = len(preparation_task) * len(training_task) * len(simulation_task)
    global_idx = 0

    # ==========================
    #  执行循环 (增加索引控制)
    # ==========================
    
    # [Level 1] Preparation Loop
    for i, p_task in enumerate(preparation_task):
        # 1. 之前完成的大组直接跳过
        if i < start_p: 
            global_idx += len(training_task) * len(simulation_task)
            continue
        
        # 2. 如果是当前断点组或新组，必须执行 Preparation 
        # (即使 i == start_p，也需要重新生成数据，因为 train 依赖它)count
        logger.info(f">>> [Group P-{stats['preparation']['count']}/{stats['preparation']['left_count']},T-{stats['train']['count']}/{stats['train']['left_count']},S-{stats['simulation']['count']}/{stats['simulation']['left_count']}]"
                    f" Preparation {finish_time_evaluate(stats)}...")
        start = time.time()
        preparation.main(logger, para=p_task)
        stats["preparation"]['time'] += time.time() - start
        stats["preparation"]['count'] += 1
        stats['preparation']['left_count'] = stats['preparation']['max_count'] - i

        # [Level 2] Training Loop
        for j, t_task in enumerate(training_task):
            # 1. 在当前 P 组内，跳过之前完成的 T 组
            if i == start_p and j < start_t:
                global_idx += len(simulation_task)
                continue
            
            # 2. 执行 Training 
            # (即使 j == start_t，也需要重新训练模型，因为 simulation 依赖内存中的模型或临时文件)
            logger.info(f">>> [Group P-{i}/{stats['preparation']['max_count']},T-{j}/{stats['train']['max_count']}]...")
            start = time.time()
            # 确保传递给 train 的配置是独立的
            train.main(logger, train_cfg=t_task) 
            stats["train"]['time'] += time.time() - start
            stats["train"]['count'] += 1
            stats['train']['left_count'] = stats['train']['max_count'] - j

            # [Level 3] Simulation Loop
            for k, s_task in enumerate(simulation_task):
                global_idx += 1
                
                # 1. 跳过完全匹配的已完成任务 (断点位置)
                # 注意：这里是 <= start_s，因为 saved_s 代表"已完成"的索引
                if i == start_p and j == start_t and k <= start_s:
                    continue

                logger.info(f">>> [{global_idx}/{total_task_num}] Simulation ...")
                start = time.time()
                
                # 执行回测
                report = simulation.main(logger, para=s_task, pre_para=p_task, train_cfg=t_task)
                
                # 保存结果
                common.append_jsonl(
                    os.path.join(exp_dir, "reports.jsonl"),
                    report["statistics"]
                )
                
                # 关键：每做完一个原子任务，立即保存游标
                save_checkpoint(exp_dir, [i, j, k])
                
                stats["simulation"]['time'] += time.time() - start
                stats["simulation"]['count'] += 1
                stats['simulation']['left_count'] = stats['simulation']['max_count'] - k

    end_time = time.time()
    logger.info("\n" + "="*40)
    logger.info(f"✅ All tasks completed in {end_time - begin_time:.2f}s")
    logger.info("="*40)

if __name__ == "__main__":
    main()