import logging
import os
from logging.handlers import RotatingFileHandler

class LoggerManager:
    """
    日志管理器类，用于创建和管理日志记录器实例
    """
    
    def __init__(self, name: str = "app_logger", log_file: str = "logs/llm_app.log", level: int = logging.INFO):
        """
        初始化LoggerManager实例
        
        Args:
            name: logger名称
            log_file: 日志文件路径
            level: 日志级别
        """
        # 创建logs目录（如果不存在）
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 创建logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # 避免重复添加handler
        if not self.logger.handlers:
            # 创建文件处理器（使用轮转日志）
            file_handler = RotatingFileHandler(
                log_file, 
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(level)
            
            # 创建控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            
            # 创建格式器并添加到处理器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # 将处理器添加到logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def get_logger(self) -> logging.Logger:
        """
        返回配置好的logger实例
        
        Returns:
            配置好的logger实例
        """
        return self.logger