import os
import sys
import ast
import shutil
import argparse

# 尝试获取默认路径配置
try:
    current_work_dir = os.path.dirname(__file__) 
    sys.path.append(os.path.join(current_work_dir, '..'))
    from data_process import common
    DEFAULT_ROOT = common.PROJECT_DIR
    DEFAULT_OUTPUT = os.path.join(common.TEMPORARY_DIR, "pull_deps")
except ImportError:
    DEFAULT_ROOT = os.getcwd()
    DEFAULT_OUTPUT = os.path.join(os.getcwd(), 'output')

class ImportUsageVisitor(ast.NodeVisitor):
    """
    AST 访问器：
    1. 记录所有的 Import 语句及其对应的模块名。
    2. 记录代码中所有被“加载”（使用）的变量名。
    """
    def __init__(self):
        # 存储导入映射: { 变量名: 模块路径 }
        # 例如: import utils -> { 'utils': 'utils' }
        # 例如: import utils as u -> { 'u': 'utils' }
        self.imports_map = {} 
        
        # 存储代码中实际出现过的名字集合
        self.used_names = set()
        
        # 特殊标记：是否有 from xxx import *
        self.star_imports = set()

    def visit_Import(self, node):
        for alias in node.names:
            # 记录 变量名 -> 模块名
            var_name = alias.asname if alias.asname else alias.name.split('.')[0]
            self.imports_map[var_name] = alias.name

    def visit_ImportFrom(self, node):
        if not node.module and node.level == 0:
            return # 忽略异常导入

        # 处理 from module import *
        if any(alias.name == '*' for alias in node.names):
            if node.module:
                self.star_imports.add(node.module)
            return

        # 处理普通 from 导入
        module_name = node.module if node.module else ""
        # 处理相对导入前缀 (from .utils import x)
        if node.level > 0:
            module_name = "." * node.level + module_name

        for alias in node.names:
            var_name = alias.asname if alias.asname else alias.name
            # 这里我们记录模块源。
            # 注意：from module import func，依赖的是 module
            self.imports_map[var_name] = module_name

    def visit_Name(self, node):
        # 记录所有作为 Load (读取/使用) 的变量名
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
    
    def visit_Attribute(self, node):
        # 处理 obj.prop，我们需要确保 obj 被记录为使用
        self.generic_visit(node)

def analyze_file_imports(file_path):
    """
    解析文件，返回 { '模块名' } 集合。
    仅返回那些在代码中被“实际使用”的模块。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=file_path)
    except Exception as e:
        print(f"⚠️ 警告: 语法错误跳过 {file_path}: {e}")
        return set()

    visitor = ImportUsageVisitor()
    visitor.visit(tree)

    active_modules = set()

    # 1. 总是包含星号导入的模块 (无法静态判断是否使用)
    active_modules.update(visitor.star_imports)

    # 2. 检查 __init__.py 特例
    # __init__.py 的导入通常是为了暴露给外部，即使内部没使用也应该保留
    is_init = os.path.basename(file_path) == '__init__.py'

    # 3. 对比 导入表 和 使用表
    for var_name, module_name in visitor.imports_map.items():
        # 如果是 __init__.py 或者 变量名在代码中被使用了
        if is_init or var_name in visitor.used_names:
            active_modules.add(module_name)
        # else:
        #     print(f"  [未使用] 在 {os.path.basename(file_path)} 中忽略: {module_name} (别名: {var_name})")

    return active_modules

def index_project_files(root_dir):
    """建立文件索引: module.name -> file_path"""
    file_map = {}
    abs_root = os.path.abspath(root_dir)
    print(f"正在索引项目文件: {abs_root} ...")
    
    for dirpath, _, filenames in os.walk(abs_root):
        for fname in filenames:
            if not fname.endswith('.py'): continue
            full_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(full_path, abs_root)
            
            # 转换为模块名
            base_name = os.path.splitext(rel_path)[0]
            mod_name = base_name.replace(os.sep, '.')
            
            file_map[mod_name] = full_path
            
            # 处理包 (folder/__init__.py -> folder)
            if fname == '__init__.py':
                pkg_name = os.path.dirname(rel_path).replace(os.sep, '.')
                file_map[pkg_name] = full_path
                
    return file_map

def find_project_dependencies(script_path, project_root):
    abs_script = os.path.abspath(script_path)
    if not os.path.exists(abs_script):
        print(f"❌ 错误: 文件不存在 {abs_script}")
        return []

    module_map = index_project_files(project_root)
    
    found_files = {abs_script}
    queue = [abs_script]
    processed_modules = set()

    print(f"开始智能分析 (剔除未使用依赖): {abs_script}")

    while queue:
        current_file = queue.pop(0)
        
        # === 核心变化：使用智能分析函数 ===
        imported_modules = analyze_file_imports(current_file)
        
        # 解析相对导入所需的当前包路径
        # 计算当前文件相对于 root 的包路径
        rel_dir = os.path.dirname(os.path.relpath(current_file, project_root))
        current_pkg = rel_dir.replace(os.sep, '.')
        if current_pkg == '.': current_pkg = ''

        for mod in imported_modules:
            # 处理相对导入逻辑 (如 from . import x)
            if mod.startswith('.'):
                if current_pkg:
                    # 简单的相对路径解析
                    # 注意：严谨的相对导入解析很复杂，这里做简化处理
                    # .utils -> pkg.utils, ..utils -> parent.utils
                    dots = len(mod) - len(mod.lstrip('.'))
                    base_mod = mod.lstrip('.')
                    # 这里的解析可能不完美，但在大多数扁平结构中有效
                    full_mod_name = f"{current_pkg}.{base_mod}" if base_mod else current_pkg
                else:
                    full_mod_name = mod.lstrip('.')
            else:
                full_mod_name = mod

            # 在索引中查找
            target_file = None
            
            # 1. 尝试直接匹配
            if full_mod_name in module_map:
                target_file = module_map[full_mod_name]
            
            # 2. 尝试匹配父级 (from a.b import c -> 可能是 a/b.py)
            if not target_file:
                parts = full_mod_name.split('.')
                for i in range(len(parts), 0, -1):
                    sub = '.'.join(parts[:i])
                    if sub in module_map:
                        target_file = module_map[sub]
                        break
            
            if target_file and target_file not in found_files:
                found_files.add(target_file)
                queue.append(target_file)

    return sorted(list(found_files))

def copy_files(file_list, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建输出目录: {target_dir}")

    count = 0
    for src in file_list:
        filename = os.path.basename(src)
        dst = os.path.join(target_dir, filename)
        
        if os.path.exists(dst) and not os.path.samefile(src, dst):
            base, ext = os.path.splitext(filename)
            parent = os.path.basename(os.path.dirname(src))
            new_name = f"{base}_{parent}{ext}"
            dst = os.path.join(target_dir, new_name)
            # print(f"⚠️  重名处理: {filename} -> {new_name}")

        try:
            shutil.copy2(src, dst)
            count += 1
        except Exception as e:
            print(f"❌ 拷贝失败 {src}: {e}")
    
    print(f"\n✅ 成功提取 {count} 个文件到: {target_dir}")

def main():
    parser = argparse.ArgumentParser(description="Python 智能依赖提取 (忽略未使用的引用)")
    parser.add_argument("-s", "--script", required=False, help="入口文件")
    parser.add_argument("-r", "--root_dir", required=False, help="项目根目录")
    parser.add_argument("-o", "--output", required=False, help="输出目录")

    args = parser.parse_args()

    # 默认值处理
    if not args.script:
        args.script = r"C:\Users\xc176\Desktop\Project\Quant\trade\market\ftmo\market_ftmo.py"
        print(f"使用默认脚本: {args.script}")
    if not args.root_dir:
        if 'DEFAULT_ROOT' in globals() and DEFAULT_ROOT:
            args.root_dir = DEFAULT_ROOT
        else:
            args.root_dir = os.path.dirname(os.path.dirname(args.script))
        print(f"使用默认根目录: {args.root_dir}")

    files = find_project_dependencies(args.script, args.root_dir)

    if not files:
        print("❌ 未找到依赖文件")
        return

    print(f"\n🔍 智能分析结果: 共 {len(files)} 个活跃文件")
    for f in files:
        # 打印相对路径，看起来更清晰
        print(f" - {os.path.relpath(f, args.root_dir)}")

    if args.output:
        out_path = args.output
    else:
        out_path = DEFAULT_OUTPUT
    
    print("-" * 30)
    copy_files(files, out_path)

if __name__ == "__main__":
    main()