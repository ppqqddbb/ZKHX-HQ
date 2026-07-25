"""
企业级日志管理模块
提供统一的日志配置和格式化
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler


class LoggerManager:
    """日志管理器"""

    _loggers = {}

    @classmethod
    def get_logger(
        cls,
        name: str,
        log_dir: Optional[Path] = None,
        level: int = logging.INFO,
        console: bool = True,
        file: bool = True,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5
    ) -> logging.Logger:
        """
        获取或创建日志记录器

        Args:
            name: 日志记录器名称
            log_dir: 日志目录
            level: 日志级别
            console: 是否输出到控制台
            file: 是否输出到文件
            max_bytes: 日志文件最大大小
            backup_count: 保留的备份文件数量

        Returns:
            配置好的日志记录器
        """
        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False

        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        if file and log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            log_file = log_dir / f"{name}.log"
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        cls._loggers[name] = logger
        return logger


def get_logger(name: str, log_dir: Optional[Path] = None) -> logging.Logger:
    """便捷函数：获取日志记录器"""
    return LoggerManager.get_logger(name, log_dir)
