import os
import shutil

# 1. 定义丢失文件存放路径和恢复目标路径
lost_found_path = ".git/lost-found/other/"
output_dir = "recovered_files_all"

# 2. 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 3. 检查 lost-found 目录是否存在
if not os.path.exists(lost_found_path):
    print(f"❌ 找不到目录: {lost_found_path}")
    print("请先在终端运行命令: git fsck --lost-found")
else:
    files_found = os.listdir(lost_found_path)
    print(f"🔍 发现 {len(files_found)} 个潜在的丢失文件，正在恢复中...")

    for filename in files_found:
        src_path = os.path.join(lost_found_path, filename)
        # 我们给恢复出的文件加上 .py 后缀，方便你在编辑器里看代码高亮
        dst_path = os.path.join(output_dir, f"restored_{filename[:8]}.py")
        
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"⚠️ 无法复制 {filename}: {e}")

    print(f"✅ 恢复完成！请查看目录: {output_dir}")