"""
企业级配置管理模块
提供统一的配置管理和环境变量支持
"""
import os
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field
import torch


@dataclass
class PathConfig:
    """路径配置"""
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    log_dir: Path = field(init=False)
    upload_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    db_path: Path = field(init=False)

    def __post_init__(self):
        self.log_dir = self.base_dir / "log"
        self.upload_dir = self.base_dir / "upload"
        self.output_dir = self.base_dir / "output"
        self.db_path = self.base_dir / "ais_system.db"

        # 自动创建必要目录
        for dir_path in [self.log_dir, self.upload_dir, self.output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """数据处理配置"""
    # 列名定义
    MMSI_COL: str = "MMSI"
    TIMESTAMP_COL: str = "Timestamp"
    LATITUDE_COL: str = "Latitude"
    LONGITUDE_COL: str = "Longitude"
    SOG_COL: str = "SOG"
    COG_COL: str = "COG"
    ROT_COL: str = "ROT"
    HEADING_COL: str = "Heading"
    SHIPTYPE_COL: str = "ShipType"
    STATUS_COL: str = "Status"
    LENGTH_COL: str = "Length"
    WIDTH_COL: str = "Width"
    DRAFT_COL: str = "Draft"
    CARGO_COL: str = "Cargo"
    GROUP_COL: str = "group_id"
    PREDICT_GROUP_COL: str = "predict_cluster"

    # 时间格式
    TIME_FORMAT: str = "%d/%m/%Y %H:%M:%S"
    US_TIME_FORMAT: str = "%Y-%m-%dT%H:%M:%S.%f"
    OUT_TIME_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    # 列名重命名映射
    RENAME_MAP: dict = field(default_factory=lambda: {
        "# Timestamp": "Timestamp",
        "MMSI": "MMSI",
        "Latitude": "Latitude",
        "Longitude": "Longitude",
        "SOG": "SOG",
        "COG": "COG",
        "Heading": "Heading",
        "Ship type": "ShipType",
        "Navigational status": "Status",
        "Length": "Length",
        "Width": "Width",
        "Draught": "Draft",
        "Cargo type": "Cargo",
        "ROT": "ROT"
    })

    # 数值列
    NUMERIC_COLUMNS: List[str] = field(default_factory=lambda: [
        "Longitude", "Latitude", "SOG", "COG", "Heading"
    ])

    # 必需列
    REQUIRED_COLUMNS: List[str] = field(default_factory=lambda: [
        "Timestamp", "MMSI", "Longitude", "Latitude", "SOG", "COG", "Heading", "ShipType"
    ])


@dataclass
class ModelConfig:
    """模型训练配置"""
    # 设备配置
    device: torch.device = field(default_factory=lambda: torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ))

    # 地球半径（米）
    EARTH_RADIUS: float = 6378137.0

    # 模型架构
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.2
    activation: str = 'gelu'
    batch_first: bool = True
    norm_first: bool = True

    # 训练参数
    lr: float = 0.0005
    weight_decay: float = 1e-4
    epochs: int = 100
    batch_size: int = 16

    # 时间序列参数
    freq: str = '5m'
    seq_len: int = 96
    pred_len: int = 60
    is_scaler: bool = False

    # 特征列
    TRAIN_COLUMNS: List[str] = field(default_factory=lambda: [
        "time_sin", "time_cos",
        "cog_sin", "cog_cos",
        "heading_sin", "heading_cos",
        "v_x", "v_y",
        "delta_x", "delta_y",
        "delta_sog", "delta_cog",
        "rel_x", "rel_y",
        "drift_angle", "curvature"
    ])

    SCALER_COLUMNS: List[str] = field(default_factory=lambda: [
        "time_sin", "time_cos",
        "cog_sin", "cog_cos",
        "heading_sin", "heading_cos",
        "v_x", "v_y",
        "delta_x", "delta_y",
        "delta_sog", "delta_cog",
        "rel_x", "rel_y",
        "drift_angle", "curvature"
    ])


@dataclass
class AnalysisConfig:
    """分析功能配置"""
    # 热点分析
    HOTSPOT_MIN_CLUSTER_SIZE: int = 50
    HOTSPOT_MIN_SAMPLES: int = 10
    HOTSPOT_H3_RESOLUTION: int = 8

    # 频繁轨迹
    FREQUENT_GRID_SIZE: float = 0.01
    FREQUENT_SIGMA: float = 2.0
    FREQUENT_PERCENTILE_THRESH: int = 90

    # 目标分析
    TARGET_EPS_KM: float = 1.0
    TARGET_MIN_SAMPLES: int = 3

    # 关联分析
    ASSOCIATION_MIN_SUPPORT: float = 0.01
    ASSOCIATION_MIN_CONFIDENCE: float = 0.5


class ConfigManager:
    """配置管理器 - 单例模式"""
    _instance: Optional['ConfigManager'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.paths = PathConfig()
        self.data = DataConfig()
        self.model = ModelConfig()
        self.analysis = AnalysisConfig()

        # 环境变量覆盖
        self._load_env_overrides()

        self._initialized = True

    def _load_env_overrides(self):
        """从环境变量加载配置覆盖"""
        # 数据库路径
        if db_path := os.getenv("AIS_DB_PATH"):
            self.paths.db_path = Path(db_path)

        # 日志目录
        if log_dir := os.getenv("AIS_LOG_DIR"):
            self.paths.log_dir = Path(log_dir)
            self.paths.log_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, name: str) -> Path:
        """获取路径配置"""
        return getattr(self.paths, name)

    def get_data_config(self, name: str):
        """获取数据配置"""
        return getattr(self.data, name)

    def get_model_config(self, name: str):
        """获取模型配置"""
        return getattr(self.model, name)

    def get_analysis_config(self, name: str):
        """获取分析配置"""
        return getattr(self.analysis, name)


# 全局配置实例
config = ConfigManager()

# 便捷访问
paths = config.paths
data_config = config.data
model_config = config.model
analysis_config = config.analysis
