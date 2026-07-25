"""
企业级热点区域分析模块
基于HDBSCAN聚类算法识别AIS数据中的热点区域
"""
import numpy as np
import polars as pl
from typing import Dict, Tuple
from hdbscan import HDBSCAN
from shapely.geometry import MultiPoint
from shapely.ops import transform
import pyproj

from core.config import data_config, model_config, analysis_config
from core.exceptions import AnalysisError, DataValidationError
from core.logger import get_logger
from core.spatial import SpatialIndexManager

logger = get_logger(__name__)

# Configuration mappings
MMSI_COL = data_config.MMSI_COL
TIMESTAMP_COL = data_config.TIMESTAMP_COL
LATITUDE_COL = data_config.LATITUDE_COL
LONGITUDE_COL = data_config.LONGITUDE_COL

# Spatial index functions (will be initialized per resolution)
def add_h3_index(df: pl.LazyFrame, resolution: int) -> pl.LazyFrame:
    """添加H3空间索引"""
    spatial_manager = SpatialIndexManager(resolution=resolution)
    return spatial_manager.add_h3_index(df, LATITUDE_COL, LONGITUDE_COL)

def h3_center(df: pl.LazyFrame) -> pl.LazyFrame:
    """转换H3索引为中心点坐标"""
    spatial_manager = SpatialIndexManager()
    return spatial_manager.h3_to_center(df, LATITUDE_COL, LONGITUDE_COL)


class HotspotAnalyzer:
    """
    热点区域分析器

    使用HDBSCAN算法识别船舶活动的热点区域，并计算时间和空间覆盖度
    """

    def __init__(
        self,
        df: pl.LazyFrame,
        min_cluster_size: int = 50,
        min_samples: int = 10,
        resolution: int = 8
    ):
        """
        初始化热点分析器

        Args:
            df: 输入AIS数据
            min_cluster_size: HDBSCAN最小聚类大小
            min_samples: HDBSCAN最小样本数
            resolution: H3空间索引分辨率

        Raises:
            DataValidationError: 输入参数无效时抛出
        """
        if df is None:
            raise DataValidationError("输入数据不能为空")

        if min_cluster_size <= 0:
            raise DataValidationError(f"min_cluster_size必须大于0，当前值: {min_cluster_size}")

        if min_samples <= 0:
            raise DataValidationError(f"min_samples必须大于0，当前值: {min_samples}")

        if not (0 <= resolution <= 15):
            raise DataValidationError(f"H3分辨率必须在[0, 15]范围内，当前值: {resolution}")

        self.df = df
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.resolution = resolution

        logger.info(
            f"热点分析器初始化完成 - "
            f"min_cluster_size={min_cluster_size}, "
            f"min_samples={min_samples}, "
            f"resolution={resolution}"
        )

    def _fit_hdbscan(self, X: np.ndarray) -> np.ndarray:
        """
        执行HDBSCAN聚类

        Args:
            X: 输入坐标数组 (n_samples, 2)

        Returns:
            聚类标签数组

        Raises:
            AnalysisError: 聚类失败时抛出
        """
        try:
            model = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples
            )
            labels = model.fit_predict(X)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            logger.info(f"HDBSCAN聚类完成 - 识别到{n_clusters}个聚类，噪声点数: {n_noise}")

            return labels

        except Exception as e:
            raise AnalysisError(
                f"HDBSCAN聚类失败: {str(e)}",
                details={"error": str(e)}
            )

    def coverage_by_mmsi(
        self,
        df_all: pl.LazyFrame,
        df_hotspot: pl.LazyFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        按MMSI计算热点区域的时间和空间覆盖度

        Args:
            df_all: 全量数据
            df_hotspot: 热点区域数据

        Returns:
            字典，键为MMSI，值为包含time_coverage和space_coverage的字典

        Raises:
            AnalysisError: 覆盖度计算失败时抛出
        """
        try:
            all_time = (
                df_all
                .group_by(MMSI_COL)
                .agg(
                    pl.col(TIMESTAMP_COL).min().alias("all_start"),
                    pl.col(TIMESTAMP_COL).max().alias("all_end"),
                )
                .with_columns(
                    (pl.col("all_end") - pl.col("all_start")).alias("all_span")
                )
                .collect()
            )

            hotspot_time = (
                df_hotspot
                .group_by(MMSI_COL)
                .agg(
                    pl.col(TIMESTAMP_COL).min().alias("hot_start"),
                    pl.col(TIMESTAMP_COL).max().alias("hot_end"),
                )
                .with_columns(
                    (pl.col("hot_end") - pl.col("hot_start")).alias("hot_span")
                )
                .collect()
            )

            all_time_map = {
                row[MMSI_COL]: row["all_span"]
                for row in all_time.iter_rows(named=True)
            }

            hotspot_time_map = {
                row[MMSI_COL]: row["hot_span"]
                for row in hotspot_time.iter_rows(named=True)
            }

            hotspot_points = (
                df_hotspot
                .group_by(MMSI_COL)
                .agg(
                    pl.struct([LONGITUDE_COL, LATITUDE_COL]).implode().alias("points")
                )
                .collect()
            )

            project = pyproj.Transformer.from_crs(
                "EPSG:4326", "EPSG:3857", always_xy=True
            ).transform

            space_map = {}

            for row in hotspot_points.iter_rows(named=True):
                mmsi = row[MMSI_COL]
                pts = row["points"]

                if pts is None or len(pts) < 3:
                    space_map[mmsi] = 0.0
                    continue

                coords = [(p[LONGITUDE_COL], p[LATITUDE_COL]) for p in pts]
                hull = MultiPoint(coords).convex_hull
                hull_proj = transform(project, hull)

                space_map[mmsi] = hull_proj.area

            result = {}

            for mmsi, all_span in all_time_map.items():
                hot_span = hotspot_time_map.get(mmsi)

                if hot_span is None or all_span.total_seconds() == 0:
                    time_cov = 0.0
                else:
                    time_cov = hot_span / all_span

                result[str(mmsi)] = {
                    "time_coverage": float(time_cov),
                    "space_coverage": float(space_map.get(mmsi, 0.0))
                }

            logger.info(f"覆盖度计算完成，共处理{len(result)}个MMSI")
            return result

        except Exception as e:
            raise AnalysisError(
                f"覆盖度计算失败: {str(e)}",
                details={"error": str(e)}
            )

    def run(self) -> Tuple[pl.LazyFrame, Dict[str, Dict[str, float]]]:
        """
        执行热点区域分析

        Returns:
            (热点区域数据, 覆盖度统计)

        Raises:
            AnalysisError: 分析失败时抛出
        """
        try:
            logger.info("开始热点区域分析")

            df = add_h3_index(self.df, self.resolution)
            h3_center_df = h3_center(df)

            X = h3_center_df.select([LATITUDE_COL, LONGITUDE_COL]).collect().to_numpy()

            if len(X) == 0:
                raise AnalysisError("H3中心点数据为空，无法进行聚类")

            labels = self._fit_hdbscan(X)

            h3_center_df = h3_center_df.with_columns(
                pl.Series("cluster", labels)
            )

            df_filtered = (
                df.join(
                    h3_center_df.select(["h3", "cluster"]),
                    on="h3",
                    how="left",
                )
                .filter(pl.col("cluster") != -1)
            )

            coverage_result = self.coverage_by_mmsi(df_all=self.df, df_hotspot=df_filtered)

            logger.info("热点区域分析完成")
            return df_filtered, coverage_result

        except Exception as e:
            if isinstance(e, AnalysisError):
                raise
            raise AnalysisError(
                f"热点区域分析失败: {str(e)}",
                details={"error": str(e)}
            )
