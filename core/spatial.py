"""
企业级空间索引模块
提供H3空间索引等地理空间处理功能
"""
import h3
import polars as pl
from typing import Union, Tuple, List
import numpy as np

from core.exceptions import DataProcessingError
from core.logger import get_logger

logger = get_logger(__name__)


class SpatialIndexManager:
    """H3空间索引管理器"""

    def __init__(self, resolution: int = 8):
        """
        初始化空间索引管理器

        Args:
            resolution: H3分辨率 (0-15)，默认8

        Raises:
            DataProcessingError: 参数无效时抛出
        """
        if not 0 <= resolution <= 15:
            raise DataProcessingError(
                f"H3分辨率必须在0-15之间，当前值: {resolution}",
                details={"resolution": resolution}
            )
        self.resolution = resolution
        logger.info(f"空间索引管理器初始化完成 - 分辨率: {resolution}")

    def add_h3_index(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        lat_col: str = "Latitude",
        lon_col: str = "Longitude"
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        为数据添加H3空间索引

        Args:
            df: 输入数据
            lat_col: 纬度列名
            lon_col: 经度列名

        Returns:
            添加了h3列的数据

        Raises:
            DataProcessingError: 处理失败时抛出
        """
        try:
            def lat_lon_to_h3(lat: float, lon: float) -> str:
                """将经纬度转换为H3索引"""
                try:
                    if lat is None or lon is None:
                        return None
                    return h3.latlng_to_cell(lat, lon, self.resolution)
                except Exception as e:
                    logger.warning(f"H3索引转换失败: lat={lat}, lon={lon}, error={e}")
                    return None

            df = df.with_columns(
                pl.struct([lat_col, lon_col])
                .map_elements(
                    lambda x: lat_lon_to_h3(x[lat_col], x[lon_col]),
                    return_dtype=pl.String
                )
                .alias("h3")
            )

            logger.info(f"H3空间索引添加完成 - 分辨率: {self.resolution}")
            return df

        except Exception as e:
            raise DataProcessingError(
                f"添加H3索引失败: {str(e)}",
                details={"lat_col": lat_col, "lon_col": lon_col, "error": str(e)}
            )

    def h3_to_center(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        lat_col: str = "Latitude",
        lon_col: str = "Longitude"
    ) -> pl.LazyFrame:
        """
        计算H3索引的中心坐标

        Args:
            df: 包含h3列的数据
            lat_col: 输出纬度列名
            lon_col: 输出经度列名

        Returns:
            H3中心点数据

        Raises:
            DataProcessingError: 处理失败时抛出
        """
        try:
            def h3_to_lat_lon(h3_index: str) -> dict:
                """将H3索引转换为经纬度"""
                try:
                    if h3_index is None:
                        return {"lat": None, "lon": None}
                    lat, lon = h3.cell_to_latlng(h3_index)
                    return {"lat": lat, "lon": lon}
                except Exception as e:
                    logger.warning(f"H3中心点转换失败: h3={h3_index}, error={e}")
                    return {"lat": None, "lon": None}

            # 获取唯一的H3索引
            h3_df = df.select("h3").unique()

            # 转换为中心点坐标
            h3_df = h3_df.with_columns(
                pl.col("h3")
                .map_elements(
                    lambda x: h3_to_lat_lon(x),
                    return_dtype=pl.Struct([
                        pl.Field("lat", pl.Float64),
                        pl.Field("lon", pl.Float64)
                    ])
                )
                .alias("center")
            )

            # 展开结构体
            h3_df = h3_df.with_columns([
                pl.col("center").struct.field("lat").alias(lat_col),
                pl.col("center").struct.field("lon").alias(lon_col)
            ]).drop("center")

            # 过滤掉空值
            h3_df = h3_df.filter(
                pl.col(lat_col).is_not_null() & pl.col(lon_col).is_not_null()
            )

            logger.info(f"H3中心点计算完成 - 唯一索引数: {h3_df.select(pl.len()).collect().item()}")
            return h3_df

        except Exception as e:
            raise DataProcessingError(
                f"计算H3中心点失败: {str(e)}",
                details={"error": str(e)}
            )

    @staticmethod
    def haversine_distance(
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """
        计算两点间的Haversine距离（米）

        Args:
            lat1: 起点纬度
            lon1: 起点经度
            lat2: 终点纬度
            lon2: 终点经度

        Returns:
            距离（米）
        """
        R = 6378137.0  # 地球半径（米）

        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)

        a = (np.sin(delta_lat / 2) ** 2 +
             np.cos(lat1_rad) * np.cos(lat2_rad) *
             np.sin(delta_lon / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c

    @staticmethod
    def mercator_projection(lat: float, lon: float) -> Tuple[float, float]:
        """
        墨卡托投影转换

        Args:
            lat: 纬度
            lon: 经度

        Returns:
            (x_meters, y_meters) 墨卡托坐标
        """
        R = 6378137.0
        x = R * np.radians(lon)
        y = R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
        return x, y
