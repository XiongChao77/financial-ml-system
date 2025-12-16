import os, colorlog , logging
from datetime import datetime

# 1. 定义新的日志级别整数值
RECORD_LEVEL_NUM = 25
logging.addLevelName(RECORD_LEVEL_NUM, "RECORD")
class CustomLogger(logging.Logger):
    """
    自定义 Logger 类，添加了一个 'important' 级别的日志方法。
    """
    def record(self, message, *args, **kws):
        # 检查 Logger 是否实际处理这个级别，以避免不必要的开销
        if self.isEnabledFor(RECORD_LEVEL_NUM):
            self._log(RECORD_LEVEL_NUM, message, args, **kws)

# 2. 【核心修改】注册自定义 Logger 类
# 这一步必须在任何 logging.getLogger() 调用之前执行
logging.setLoggerClass(CustomLogger)

# ====================================================================
# B. 配置 Logger 函数
# ====================================================================

def setup_logger(log_name:str, log_path,console_level:int =logging.INFO,record_level:int = RECORD_LEVEL_NUM ) -> CustomLogger:
    """
    设置日志记录器：控制台彩色输出 (DEBUG+)，文件输出 (INFO+)。
    :param log_name: 日志记录器的名称。
    :param log_dir: 日志文件存储的目录。
    :return: 配置好的 CustomLogger 对象。
    """
    # 确保日志目录存在
    os.makedirs(log_path, exist_ok=True)
    
    # 因为 CustomLogger 已经注册，这里返回的就是 CustomLogger 实例
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG) # 确保所有消息都能进入处理流程

    # 避免重复添加 handlers (重要，防止多次调用函数时重复记录)
    if logger.handlers:
        logger.handlers = []

    # 统一格式字符串
    # log_format = "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
    log_format = "%(levelname)s - %(message)s"

    # --- 1. 控制台处理程序 (StreamHandler) ---
    ch = logging.StreamHandler()
    ch.setLevel(console_level) # 您的要求是 console: All (即 DEBUG+)
    
    # 彩色格式化器
    # 【核心修改】添加 RECORD 级别的颜色
    log_colors: dict[str, str] = {
        'DEBUG':    'cyan',
        'INFO':     'green',
        'RECORD':   'blue',       # 为您的新级别定义颜色
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'bold_red,bg_yellow',
    }
    
    color_formatter = colorlog.ColoredFormatter(
        "%(log_color)s" + log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors=log_colors
    )
    ch.setFormatter(color_formatter)
    logger.addHandler(ch)

    # --- 2. 文件处理程序 (FileHandler) ---
    log_file = os.path.join(log_path, f"{log_name}.log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    
    # 设置文件最低输出等级：INFO
    # INFO (20), RECORD (25), WARNING (30)... 都会写入文件
    fh.setLevel(record_level) 
    
    # 普通格式化器
    file_formatter = logging.Formatter(log_format) # 使用包含文件名和行号的格式
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    # 打印注册的日志级别，方便确认
    logger.record(f"Logger initialized. File level: {logging.getLevelName(fh.level)}, Console level: {logging.getLevelName(ch.level)}")

    return logger

def setup_session_logger(log_root: str, sub_folder: str, symbol: str = "", console_level: int = logging.INFO):
    """
    配置实盘会话日志：
    1. 生成带时间戳的文件名 (session_SYMBOL_Time.log)
    2. 将 FileHandler 挂载到 Root Logger (捕获所有模块日志)
    3. 将 StreamHandler (彩色) 挂载到 Root Logger (控制台显示所有日志)
    
    :param log_root: 日志根目录 (例如 common.TEMPORARY_DIR)
    :param sub_folder: 子文件夹名 (例如 'market_ftmo_sessions')
    :param symbol: 交易品种
    :return: (root_logger, log_file_path)
    """
    # 1. 准备目录
    log_dir = os.path.join(log_root, sub_folder)
    os.makedirs(log_dir, exist_ok=True)

    # 2. 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sym_str = f"_{symbol}" if symbol else ""
    log_filename = f"session{sym_str}_{timestamp}.log"
    log_file_path = os.path.join(log_dir, log_filename)

    # 3. 获取 Root Logger
    # 注意：我们配置 Root，这样 self.logger = logging.getLogger("AnyName") 都会被捕获
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # 开启所有级别，具体过滤交给 Handler

    # 清除旧的 Handlers (防止重复打印，特别是 Jupyter 或多次调用时)
    if root_logger.handlers:
        root_logger.handlers = []

    # 4. 配置控制台输出 (复用之前的彩色格式)
    log_format_console = "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    
    color_formatter = colorlog.ColoredFormatter(
        log_format_console,
        datefmt="%H:%M:%S", # 控制台时间短一点，方便看
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'RECORD':   'blue',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red,bg_yellow',
        }
    )
    ch.setFormatter(color_formatter)
    root_logger.addHandler(ch)

    # 5. 配置文件输出 (全量记录)
    fh = logging.FileHandler(log_file_path, encoding='utf-8')
    fh.setLevel(logging.INFO) # 文件通常记录 INFO 以上
    
    # 文件格式包含完整日期
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(file_formatter)
    root_logger.addHandler(fh)

    # 打印一条初始化消息
    root_logger.info(f"Session Logger Initialized. Log file: {log_file_path}")

    return root_logger, log_file_path