"""
AIS数据分析系统 - 核心模块
提供数据处理、分析、模型预测等核心功能
"""

# 配置管理
from core.config import config, paths, data_config, model_config, analysis_config

# 异常处理
from core.exceptions import (
    AISBaseException,
    DataValidationError,
    FileOperationError,
    DataProcessingError,
    AnalysisError,
    ModelError,
    ConfigurationError
)

# 日志管理
from core.logger import get_logger, LoggerManager

# 数据验证
from core.validators import DataValidator

# 空间索引
from core.spatial import SpatialIndexManager

# 数据处理
from core.data.loader import DataLoader
from core.data.process import DataProcessor
from core.data.filter import DataFilter
from core.data.save import DataSaver

# 分析模块
from core.analysis.hotspot import HotspotAnalyzer
from core.analysis.frequent import FrequentAnalyzer
from core.analysis.target import TargetAnalyzer
from core.analysis.association import AssociationAnalyzer
from core.analysis.clustering import ClusteringAnalyzer

# 数据库
from core.database.database import engine, SessionLocal, get_db, init_db
from core.database.models import User, AnalysisTask, ShipInfo

# 模型
from core.model.prediction import PatchTSTManager

__all__ = [
    # 配置
    'config',
    'paths',
    'data_config',
    'model_config',
    'analysis_config',

    # 异常
    'AISBaseException',
    'DataValidationError',
    'FileOperationError',
    'DataProcessingError',
    'AnalysisError',
    'ModelError',
    'ConfigurationError',

    # 日志
    'get_logger',
    'LoggerManager',

    # 验证
    'DataValidator',

    # 空间索引
    'SpatialIndexManager',

    # 数据处理
    'DataLoader',
    'DataProcessor',
    'DataFilter',
    'DataSaver',

    # 分析
    'HotspotAnalyzer',
    'FrequentAnalyzer',
    'TargetAnalyzer',
    'AssociationAnalyzer',
    'ClusteringAnalyzer',

    # 数据库
    'engine',
    'SessionLocal',
    'get_db',
    'init_db',
    'User',
    'AnalysisTask',
    'ShipInfo',

    # 模型
    'PatchTSTManager'
]
