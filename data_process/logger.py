import os, colorlog , logging


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
    log_format = "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"

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