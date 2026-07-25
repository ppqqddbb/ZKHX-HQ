"""
企业级频繁轨迹分析模块
基于密度网格和骨架提取算法识别频繁航道
"""
import numpy as np
import polars as pl
import geopandas as gpd
from typing import Tuple
from pathlib import Path
from shapely.geometry import LineString
from shapely.ops import linemerge
from scipy.ndimage import gaussian_filter, binary_closing
from scipy.spatial import KDTree, cKDTree
from skimage.morphology import skeletonize

from core.config import data_config, model_config, analysis_config
from core.exceptions import AnalysisError, DataValidationError
from core.logger import get_logger

logger = get_logger(__name__)

# Configuration mappings
MMSI_COL = data_config.MMSI_COL
TIMESTAMP_COL = data_config.TIMESTAMP_COL
LATITUDE_COL = data_config.LATITUDE_COL
LONGITUDE_COL = data_config.LONGITUDE_COL
SOG_COL = data_config.SOG_COL


class FrequentAnalyzer:
    """
    频繁轨迹分析器

    通过密度网格、高斯平滑和骨架提取识别船舶的频繁航道
    """

    def __init__(
        self,
        df: pl.LazyFrame,
        grid_size: float = 0.01,
        sigma: float = 2.0,
        percentile_thresh: int = 90,
        visualize: bool = False
    ):
        """
        初始化频繁轨迹分析器

        Args:
            df: 输入AIS数据
            grid_size: 网格大小（度）
            sigma: 高斯平滑参数
            percentile_thresh: 密度百分位阈值
            visualize: 是否可视化结果

        Raises:
            DataValidationError: 输入参数无效时抛出
        """
        if df is None:
            raise DataValidationError("输入数据不能为空")

        if grid_size <= 0:
            raise DataValidationError(f"网格大小必须大于0，当前值: {grid_size}")

        if sigma <= 0:
            raise DataValidationError(f"sigma必须大于0，当前值: {sigma}")

        if not (0 <= percentile_thresh <= 100):
            raise DataValidationError(f"百分位阈值必须在[0, 100]范围内，当前值: {percentile_thresh}")

        self.df = df
        self.grid_size = grid_size
        self.sigma = sigma
        self.percentile_thresh = percentile_thresh
        self.visualize = visualize

        logger.info(
            f"频繁轨迹分析器初始化完成 - "
            f"grid_size={grid_size}, "
            f"sigma={sigma}, "
            f"percentile_thresh={percentile_thresh}"
        )

    def _build_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        构建分析所需的数据

        Returns:
            (纬度数组, 经度数组, 纬度bins, 经度bins)

        Raises:
            AnalysisError: 数据构建失败时抛出
        """
        try:
            df_collected = (
                self.df
                .select([TIMESTAMP_COL, LATITUDE_COL, LONGITUDE_COL, SOG_COL])
                .filter(pl.col(SOG_COL) > 1.0)
                .collect()
            )

            if len(df_collected) == 0:
                raise AnalysisError("过滤后的数据为空，无法进行分析")

            lats = df_collected[LATITUDE_COL].to_numpy()
            lons = df_collected[LONGITUDE_COL].to_numpy()

            lat_bins = np.arange(lats.min(), lats.max() + self.grid_size, self.grid_size)
            lon_bins = np.arange(lons.min(), lons.max() + self.grid_size, self.grid_size)

            logger.info(f"数据构建完成 - 数据点数: {len(lats)}, 网格数: {len(lat_bins)}x{len(lon_bins)}")

            return lats, lons, lat_bins, lon_bins

        except Exception as e:
            raise AnalysisError(
                f"数据构建失败: {str(e)}",
                details={"error": str(e)}
            )

    def _build_density_grid(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        lat_bins: np.ndarray,
        lon_bins: np.ndarray
    ) -> gpd.GeoDataFrame:
        """
        构建密度网格并提取频繁航道

        Args:
            lats: 纬度数组
            lons: 经度数组
            lat_bins: 纬度bins
            lon_bins: 经度bins

        Returns:
            包含频繁航道的GeoDataFrame

        Raises:
            AnalysisError: 网格构建失败时抛出
        """
        try:
            density, _, _ = np.histogram2d(lats, lons, bins=[lat_bins, lon_bins])

            density = gaussian_filter(density, sigma=self.sigma)

            non_zero = density[density > 0]
            if len(non_zero) == 0:
                raise AnalysisError("密度网格不包含非零值")

            thresh = np.percentile(non_zero, self.percentile_thresh)
            binary = density >= thresh

            binary = binary_closing(binary, structure=np.ones((3, 3)))

            skeleton = skeletonize(binary)

            lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
            lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
            lat_grid, lon_grid = np.meshgrid(
                lat_centers, lon_centers, indexing='ij'
            )

            skel_lats = lat_grid[skeleton]
            skel_lons = lon_grid[skeleton]
            points = np.column_stack([skel_lons, skel_lats])

            if len(points) < 2:
                raise AnalysisError("骨架点过少，无法生成航道")

            tree = KDTree(points)
            max_dist = self.grid_size * 1.5
            lines = []
            used = set()

            for i, p in enumerate(points):
                if i in used:
                    continue
                neighbors = tree.query_ball_point(p, r=max_dist)
                for j in neighbors:
                    if j != i:
                        line = LineString([points[i], points[j]])
                        if line.length > 0:
                            lines.append(line)
                            used.add(j)

            if not lines:
                raise AnalysisError("未生成任何航道线段")

            merged = linemerge(lines)
            if isinstance(merged, LineString):
                geometries = [merged]
            else:
                geometries = list(merged.geoms)

            gdf = gpd.GeoDataFrame(
                geometry=geometries,
                crs="EPSG:4326"
            )

            logger.info(f"频繁航道提取完成 - 生成{len(geometries)}条航道")

            return gdf

        except Exception as e:
            if isinstance(e, AnalysisError):
                raise
            raise AnalysisError(
                f"密度网格构建失败: {str(e)}",
                details={"error": str(e)}
            )

    def run(self) -> Tuple[gpd.GeoDataFrame, pl.DataFrame]:
        """
        执行频繁轨迹分析

        Returns:
            (航道GeoDataFrame, 航道点DataFrame)

        Raises:
            AnalysisError: 分析失败时抛出
        """
        try:
            logger.info("开始频繁轨迹分析")

            lats, lons, lat_bins, lon_bins = self._build_data()

            gdf = self._build_density_grid(lats, lons, lat_bins, lon_bins)

            df_collected = self.df.collect()
            points = np.column_stack([
                df_collected[LONGITUDE_COL],
                df_collected[LATITUDE_COL]
            ])
            tree = cKDTree(points)

            records = []
            for geom in gdf.geometry:
                coords = list(geom.coords)
                for lon, lat in coords:
                    dist, idx = tree.query([lon, lat])
                    records.append({
                        LONGITUDE_COL: lon,
                        LATITUDE_COL: lat,
                        TIMESTAMP_COL: df_collected[TIMESTAMP_COL][idx]
                    })

            records_df = pl.DataFrame(records)

            logger.info("频繁轨迹分析完成")
            return gdf, records_df

        except Exception as e:
            if isinstance(e, AnalysisError):
                raise
            raise AnalysisError(
                f"频繁轨迹分析失败: {str(e)}",
                details={"error": str(e)}
            )
