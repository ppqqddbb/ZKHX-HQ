"""
企业级异常处理模块
提供统一的异常类型和错误处理机制
"""


class AISBaseException(Exception):
    """AIS系统基础异常类"""

    def __init__(self, message: str, error_code: str = None, details: dict = None):
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        return f"[{self.error_code}] {self.message}"


class DataProcessingError(AISBaseException):
    """数据处理异常"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "DATA_PROCESSING_ERROR", details)


class DataValidationError(AISBaseException):
    """数据验证异常"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "DATA_VALIDATION_ERROR", details)


class AnalysisError(AISBaseException):
    """分析异常"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "ANALYSIS_ERROR", details)


class ModelError(AISBaseException):
    """模型异常"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "MODEL_ERROR", details)


class ConfigurationError(AISBaseException):
    """配置异常"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "CONFIGURATION_ERROR", details)


class FileOperationError(AISBaseException):
    """文件操作异常"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "FILE_OPERATION_ERROR", details)
