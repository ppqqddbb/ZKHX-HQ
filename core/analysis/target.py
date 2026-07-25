"""
企业级目标分析模块
提供船舶活动区域、停留位置、速度统计和航程分析功能
"""
import json
import numpy as np
import polars as pl
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint, Polygon, Point, mapping
from shapely.ops import transform
import pyproj

from core.config import data_config, model_config
from core.exceptions import AnalysisError, DataValidationError
from core.logger import get_logger

logger = get_logger(__name__)

# 配置变量映射
MMSI_COL = data_config.MMSI_COL
TIMESTAMP_COL = data_config.TIMESTAMP_COL
LATITUDE_COL = data_config.LATITUDE_COL
LONGITUDE_COL = data_config.LONGITUDE_COL
SOG_COL = data_config.SOG_COL
COG_COL = data_config.COG_COL
SHIPTYPE_COL = data_config.SHIPTYPE_COL
usTimeFormat = data_config.US_TIME_FORMAT


class TargetAnalyzer:
    """
    目标分析器

    提供船舶目标的多维度分析，包括活动区域、停留位置、速度统计等
    """

    def __init__(
        self,
        df: pl.LazyFrame,
        eps_km: float = 1.0,
        min_samples: int = 3,
        zones: Optional[Union[List[Dict], str, Path]] = None
    ):
        """
        初始化目标分析器

        Args:
            df: 输入AIS数据
            eps_km: DBSCAN聚类半径（公里）
            min_samples: DBSCAN最小样本数
            zones: 自定义区域配置（列表或JSON文件路径）

        Raises:
            DataValidationError: 输入参数无效时抛出
        """
        if df is None:
            raise DataValidationError("输入数据不能为空")

        if eps_km <= 0:
            raise DataValidationError(f"eps_km必须大于0，当前值: {eps_km}")

        if min_samples <= 0:
            raise DataValidationError(f"min_samples必须大于0，当前值: {min_samples}")

        self.df = df
        self.eps_rad = eps_km / 6371.0
        self.min_samples = min_samples
        self.zones = zones

        logger.info(
            f"目标分析器初始化完成 - "
            f"eps_km={eps_km}, "
            f"min_samples={min_samples}"
        )

    def _analysis_activate_areas(self) -> List[Dict]:
        """
        分析每个MMSI的活动区域（凸包）

        Returns:
            活动区域列表

        Raises:
            AnalysisError: 分析失败时抛出
        """
        try:
            df_grouped = (
                self.df
                .group_by(MMSI_COL)
                .agg([
                    pl.struct([
                        pl.col(LATITUDE_COL),
                        pl.col(LONGITUDE_COL)
                    ]).implode().alias("points")
                ])
                .collect()
            )

            activate_areas = []
            for row in df_grouped.iter_rows():
                mmsi = row[0]
                points = row[1]
                coords = [(p[LATITUDE_COL], p[LONGITUDE_COL]) for p in points]

                if len(coords) < 3:
                    continue

                hull = MultiPoint(coords).convex_hull
                activate_areas.append({
                    MMSI_COL: mmsi,
                    "activate_area": mapping(hull)
                })

            logger.info(f"活动区域分析完成 - 处理{len(activate_areas)}个MMSI")
            return activate_areas

        except Exception as e:
            raise AnalysisError(
                f"活动区域分析失败: {str(e)}",
                details={"error": str(e)}
            )

    def _analysis_dwell_positions(self) -> pl.DataFrame:
        """
        分析停留位置（基于DBSCAN聚类）

        Returns:
            停留位置数据

        Raises:
            AnalysisError: 分析失败时抛出
        """
        try:
            df = self.df
            ts_dtype = df.collect_schema()[TIMESTAMP_COL]

            if ts_dtype == pl.Utf8:
                df = df.with_columns(
                    pl.col(TIMESTAMP_COL)
                    .str.strptime(pl.Datetime, usTimeFormat, strict=False)
                    .alias(TIMESTAMP_COL)
                )

            df_grouped = (
                df.with_columns(
                    pl.col(TIMESTAMP_COL)
                    .dt.truncate("1h")
                    .alias("hour")
                )
                .collect()
            )

            dwell_positions = []
            for (mmsi, hour), group in df_grouped.group_by([MMSI_COL, "hour"], maintain_order=True):
                if group.height < 3:
                    continue

                coords = np.radians(
                    group.select([LATITUDE_COL, LONGITUDE_COL]).to_numpy()
                )

                model = DBSCAN(
                    eps=self.eps_rad,
                    min_samples=self.min_samples,
                    metric="haversine"
                )
                labels = model.fit_predict(coords)

                group = group.with_columns(
                    pl.Series("cluster", labels)
                ).filter(pl.col("cluster") != -1)

                if group.height > 0:
                    dwell_positions.append(group)

            if not dwell_positions:
                logger.warning("未检测到停留位置")
                return pl.DataFrame()

            result = pl.concat(dwell_positions)
            logger.info(f"停留位置分析完成 - 检测到{len(result)}个停留点")

            return result

        except Exception as e:
            raise AnalysisError(
                f"停留位置分析失败: {str(e)}",
                details={"error": str(e)}
            )

    def _analysis_speed(self) -> pl.DataFrame:
        """
        分析速度统计

        Returns:
            速度统计数据

        Raises:
            AnalysisError: 分析失败时抛出
        """
        try:
            df_grouped = (
                self.df
                .group_by(MMSI_COL)
                .agg(
                    pl.mean(LONGITUDE_COL).alias("平均经度"),
                    pl.min(LONGITUDE_COL).alias("最小经度"),
                    pl.max(LONGITUDE_COL).alias("最大经度"),
                    (pl.max(LONGITUDE_COL) - pl.min(LONGITUDE_COL)).alias("经度跨度"),

                    pl.mean(LATITUDE_COL).alias("平均纬度"),
                    pl.min(LATITUDE_COL).alias("最小纬度"),
                    pl.max(LATITUDE_COL).alias("最大纬度"),
                    (pl.max(LATITUDE_COL) - pl.min(LATITUDE_COL)).alias("纬度跨度"),

                    pl.mean(SOG_COL).alias("平均速度"),
                    pl.max(SOG_COL).alias("最大速度"),
                    pl.min(SOG_COL).alias("最小速度"),
                    pl.std(SOG_COL).alias("速度标准差"),

                    pl.mean(COG_COL).alias("平均航向"),
                    pl.max(COG_COL).alias("最大航向"),
                    pl.min(COG_COL).alias("最小航向"),
                    pl.std(COG_COL).alias("航向标准差"),
                )
                .collect()
            )

            logger.info(f"速度统计分析完成 - 处理{len(df_grouped)}个MMSI")
            return df_grouped

        except Exception as e:
            raise AnalysisError(
                f"速度统计分析失败: {str(e)}",
                details={"error": str(e)}
            )

    def _analysis_total_distance(self) -> pl.DataFrame:
        """
        计算总航程（基于Haversine距离）

        Returns:
            航程数据

        Raises:
            AnalysisError: 分析失败时抛出
        """
        try:
            df_grouped = self.df.with_columns([
                pl.col(LONGITUDE_COL).shift(1).over(MMSI_COL).alias("lon_prev"),
                pl.col(LATITUDE_COL).shift(1).over(MMSI_COL).alias("lat_prev"),
            ])

            df_grouped = df_grouped.with_columns(
                pl.struct([LATITUDE_COL, LONGITUDE_COL, "lat_prev", "lon_prev"]).map_elements(
                    lambda x: 6371000.0 * 2 * np.arcsin(
                        np.sqrt(
                            np.sin(np.radians(x[LATITUDE_COL] - x["lat_prev"]) / 2) ** 2 +
                            np.cos(np.radians(x[LATITUDE_COL])) *
                            np.cos(np.radians(x["lat_prev"])) *
                            np.sin(np.radians(x[LONGITUDE_COL] - x["lon_prev"]) / 2) ** 2
                        )
                    ) if x["lat_prev"] is not None else 0.0,
                    return_dtype=pl.Float64
                ).alias("segment_dist")
            )

            total_distance = (
                df_grouped
                .group_by(MMSI_COL)
                .agg(
                    pl.sum("segment_dist").alias("总体航程_米")
                )
                .collect()
            )

            logger.info(f"航程计算完成 - 处理{len(total_distance)}个MMSI")
            return total_distance

        except Exception as e:
            raise AnalysisError(
                f"航程计算失败: {str(e)}",
                details={"error": str(e)}
            )

    def analysis_coverage_by_region(self, region_coords: List[Tuple[float, float]]) -> Dict[str, float]:
        """
        计算指定区域的时间和空间覆盖度

        Args:
            region_coords: 区域坐标列表 [(lat, lon), ...]

        Returns:
            包含时间和空间覆盖度的字典

        Raises:
            DataValidationError: 区域坐标无效时抛出
            AnalysisError: 分析失败时抛出
        """
        if len(region_coords) < 3:
            raise DataValidationError("区域至少需要3个坐标点")

        try:
            region_polygon = Polygon(
                [(lon, lat) for lat, lon in region_coords]
            )

            df_all = self.df.collect()

            ts_dtype = df_all.schema[TIMESTAMP_COL]
            if ts_dtype == pl.Utf8:
                df_all = df_all.with_columns(
                    pl.col(TIMESTAMP_COL).str.strptime(pl.Datetime, usTimeFormat, strict=False)
                )

            all_start = df_all[TIMESTAMP_COL].min()
            all_end = df_all[TIMESTAMP_COL].max()

            if all_start is None or all_end is None or all_start == all_end:
                return {"overall_time_coverage": 0.0, "overall_space_coverage": 0.0}

            all_span_min = (all_end - all_start).total_seconds() / 60

            mask = df_all.select(
                pl.struct([LATITUDE_COL, LONGITUDE_COL]).map_elements(
                    lambda x: region_polygon.contains(
                        Point(x[LONGITUDE_COL], x[LATITUDE_COL])
                    ),
                    return_dtype=pl.Boolean
                ).alias("in_region")
            )["in_region"]

            df_region = df_all.filter(mask)

            if df_region.height == 0:
                return {"overall_time_coverage": 0.0, "overall_space_coverage": 0.0}

            region_start = df_region[TIMESTAMP_COL].min()
            region_end = df_region[TIMESTAMP_COL].max()
            region_span_min = (region_end - region_start).total_seconds() / 60

            time_coverage_percent = (region_span_min / all_span_min) * 100

            coords = list(
                zip(
                    df_region[LONGITUDE_COL].to_list(),
                    df_region[LATITUDE_COL].to_list()
                )
            )

            if len(coords) < 3:
                space_coverage_percent = 0.0
            else:
                hull_region = MultiPoint(coords).convex_hull

                project = pyproj.Transformer.from_crs(
                    "EPSG:4326", "EPSG:3857", always_xy=True
                ).transform

                hull_region_proj = transform(project, hull_region)
                area_region = hull_region_proj.area

                hull_polygon_proj = transform(project, region_polygon)
                area_polygon = hull_polygon_proj.area

                space_coverage_percent = (area_region / area_polygon) * 100 if area_polygon > 0 else 0.0

            logger.info(f"区域覆盖度分析完成 - 时间覆盖: {time_coverage_percent:.2f}%, 空间覆盖: {space_coverage_percent:.2f}%")

            return {
                "overall_time_coverage": float(time_coverage_percent),
                "overall_space_coverage": float(space_coverage_percent)
            }

        except Exception as e:
            raise AnalysisError(
                f"区域覆盖度分析失败: {str(e)}",
                details={"error": str(e)}
            )

    def analysis_coverage_by_zones(
        self,
        zones: Optional[Union[List[Dict], str, Path]] = None
    ) -> List[Dict[str, Union[str, float]]]:
        """
        批量计算多个区域的覆盖度

        Args:
            zones: 区域配置列表或JSON文件路径

        Returns:
            覆盖度结果列表

        Raises:
            AnalysisError: 分析失败时抛出
        """
        if zones is None:
            zones = self.zones

        if zones is None:
            logger.warning("未提供区域配置，跳过区域覆盖度分析")
            return []

        try:
            if isinstance(zones, (str, Path)):
                with open(zones, "r", encoding="utf-8") as f:
                    zones = json.load(f)

                for zone in zones:
                    zone["coordinates"] = [[lat, lon] for lon, lat in zone["coordinates"]]

            results = []
            for zone in zones:
                zone_name = zone.get("zone_name", "Unnamed_Zone")
                coords = zone.get("coordinates", [])

                if len(coords) < 3:
                    results.append({
                        "zone_name": zone_name,
                        "overall_time_coverage": 0.0,
                        "overall_space_coverage": 0.0
                    })
                    continue

                coverage = self.analysis_coverage_by_region(region_coords=coords)
                results.append({
                    "zone_name": zone_name,
                    "overall_time_coverage": coverage.get("overall_time_coverage", 0.0),
                    "overall_space_coverage": coverage.get("overall_space_coverage", 0.0)
                })

            logger.info(f"批量区域覆盖度分析完成 - 处理{len(results)}个区域")
            return results

        except Exception as e:
            raise AnalysisError(
                f"批量区域覆盖度分析失败: {str(e)}",
                details={"error": str(e)}
            )

    def run(self) -> Tuple[List[Dict], pl.DataFrame, pl.DataFrame, pl.DataFrame, List[Dict]]:
        """
        执行完整的目标分析

        Returns:
            (活动区域, 停留位置, 速度统计, 航程数据, 区域覆盖度)

        Raises:
            AnalysisError: 分析失败时抛出
        """
        try:
            logger.info("开始目标分析")

            activate_areas = self._analysis_activate_areas()
            dwell_positions = self._analysis_dwell_positions()
            speed = self._analysis_speed()
            total_distance = self._analysis_total_distance()
            zones = self.analysis_coverage_by_zones()

            logger.info("目标分析完成")
            return activate_areas, dwell_positions, speed, total_distance, zones

        except Exception as e:
            if isinstance(e, AnalysisError):
                raise
            raise AnalysisError(
                f"目标分析失败: {str(e)}",
                details={"error": str(e)}
            )
