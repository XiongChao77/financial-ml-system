import os
import shutil
from pathlib import Path

# --- 配置区域 ---

# 导出的目标文件夹名称
OUTPUT_DIR = "exported_project_files"

# 需要导出的文件列表 (使用原始字符串 r"" 处理反斜杠)
FILES_TO_EXPORT = [
    r"data_process/common.py",
    r"data_process/download_binance_klines.py",
    r"data_process/preparation.py",
    r"data_process/ta_calculation.py",
    r"model/cnn_timeseries_torch.py",
    r"model/data_loader.py",
    r"trade_simulation/cus_analyzer.py",
    r"trade_simulation/simulation.py",
    r"export_files.py"
    # 注意：你提供的列表中 trade_simulation\cus_analyzer.py 出现了两次，这里已自动去重
]

# --- 主逻辑 ---

def main():
    # 获取当前脚本所在的绝对路径 (QUANT 目录)
    base_path = Path(__file__).parent.resolve()
    target_base = base_path / OUTPUT_DIR

    print(f"Start exporting files to: {target_base}\n")

    # 使用 set 去重，防止重复复制
    files_set = set(FILES_TO_EXPORT)
    
    success_count = 0
    fail_count = 0

    for rel_path_str in files_set:
        # 构造源文件和目标文件的完整路径
        src_file = base_path / rel_path_str
        dst_file = target_base / rel_path_str

        if src_file.exists() and src_file.is_file():
            try:
                # 确保目标文件的父目录存在
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                
                # 复制文件 (copy2 保留文件元数据，如修改时间)
                shutil.copy2(src_file, dst_file)
                print(f"[OK] {rel_path_str}")
                success_count += 1
            except Exception as e:
                print(f"[ERROR] Failed to copy {rel_path_str}: {e}")
                fail_count += 1
        else:
            print(f"[MISSING] File not found: {rel_path_str}")
            fail_count += 1

    print("-" * 30)
    print(f"Export complete. Success: {success_count}, Failed: {fail_count}")
    print(f"Output directory: {target_base}")

if __name__ == "__main__":
    main()