import json
import logging
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint, mapping
from shapely.geometry import Polygon, Point
from shapely.ops import transform
import pyproj
from utils.utils import *
from config.settings import *

class TargetAnalyzer:
    def __init__(self, df: pl.LazyFrame, eps_km: float, min_samples: int, zones):
        self.df = df
        self.eps_rad = eps_km / 6371.0
        self.min_samples = min_samples
        self.zones = zones

    def _analysis_activate_areas(self):
        df_grouped = (
            self.df
            .group_by(MMSI_COL)
            .agg([
                pl.struct([
                    pl.col(LATITUDE_COL),
                    pl.col(LONGITUDE_COL)]).implode().alias("points")
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
            activate_areas.append(
                {
                    MMSI_COL:mmsi,
                    "activate_area": mapping(hull)
                }
            )
        return activate_areas

    def _analysis_dwell_positions(self):
        df = self.df

        ts_dtype = df.schema[TIMESTAMP_COL]

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

        # df_grouped = (
        #     self.df
        #     # .with_columns(
        #     #     pl.col(TIMESTAMP_COL).str.strptime(pl.Datetime, usTimeFormat, strict=False)
        #     # )
        #     .with_columns(
        #         pl.col(TIMESTAMP_COL)
        #         .dt.truncate("1h")
        #         .alias("hour")
        #     )
        #     .collect()
        # )
        dwell_positions = []
        for (mmsi, hour), group in df_grouped.group_by([MMSI_COL, "hour"], maintain_order=True):
            if group.height < 3:
                continue
            coords = np.radians(
                group.select([LATITUDE_COL, LONGITUDE_COL]).to_numpy()
            )
            model = DBSCAN(eps=self.eps_rad, min_samples=self.min_samples, metric="haversine")
            labels = model.fit_predict(coords)
            group = group.with_columns(
                pl.Series("cluster", labels)
            ).filter(pl.col("cluster") != -1)

            if group.height > 0:
                dwell_positions.append(group)
        dwell_positions = pl.concat(dwell_positions)
        return dwell_positions

    def _analysis_speed(self):
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
        return df_grouped

    def _analysis_total_distance(self):
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
                ) if x["lat_prev"] is not None else 0.0
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
        return total_distance
    
    def analysis_coverage_by_region(self, region_coords: list[tuple]):
        """
        用户自定义空间区域，计算整体数据的时间 / 空间覆盖度（按分钟计算时间覆盖度）

        region_coords: [(lat, lon), (lat, lon), ...]，至少 3 个点
        return: { "time_coverage_percent": float, "space_coverage_percent": float }
        """

        if len(region_coords) < 3:
            raise ValueError("区域至少需要 3 个经纬度点")

        # ---------- 1. 构造用户区域 Polygon ----------
        region_polygon = Polygon(
            [(lon, lat) for lat, lon in region_coords]
        )

        # ---------- 2. collect DataFrame ----------
        df_all = self.df.collect()  # LazyFrame -> DataFrame

        # ---------- 3. 确保时间列为 Datetime ----------
        ts_dtype = df_all.schema[TIMESTAMP_COL]
        if ts_dtype == pl.Utf8:
            df_all = df_all.with_columns(
                pl.col(TIMESTAMP_COL).str.strptime(pl.Datetime, usTimeFormat, strict=False)
            )

        # ---------- 4. 整体时间跨度（分钟） ----------
        all_start = df_all[TIMESTAMP_COL].min()
        all_end = df_all[TIMESTAMP_COL].max()

        if all_start is None or all_end is None or all_start == all_end:
            return {"time_coverage_percent": 0.0, "space_coverage_percent": 0.0}

        all_span_min = (all_end - all_start).total_seconds() / 60  # 转换为分钟

        # ---------- 5. 筛选落入区域内的点 ----------
        mask = df_all.select(
            pl.struct([LATITUDE_COL, LONGITUDE_COL]).map_elements(
                lambda x: region_polygon.contains(
                    Point(x[LONGITUDE_COL], x[LATITUDE_COL])
                )
            ).alias("in_region")
        )["in_region"]

        df_region = df_all.filter(mask)

        if df_region.height == 0:
            return {"time_coverage_percent": 0.0, "space_coverage_percent": 0.0}

        # ---------- 6. 区域内时间覆盖度（按分钟计算） ----------
        region_start = df_region[TIMESTAMP_COL].min()
        region_end = df_region[TIMESTAMP_COL].max()
        region_span_min = (region_end - region_start).total_seconds() / 60

        time_coverage_percent = (region_span_min / all_span_min) * 100  # 百分比

        # ---------- 7. 区域内空间覆盖度（凸包面积百分比） ----------
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

            # 投影到米制坐标计算面积
            project = pyproj.Transformer.from_crs(
                "EPSG:4326", "EPSG:3857", always_xy=True
            ).transform

            hull_region_proj = transform(project, hull_region)
            area_region = hull_region_proj.area

            # 整体区域面积（用户自定义 Polygon）
            hull_polygon_proj = transform(project, region_polygon)
            area_polygon = hull_polygon_proj.area

            # 空间覆盖度百分比
            space_coverage_percent = (area_region / area_polygon) * 100

        return {
            "overall_time_coverage": float(time_coverage_percent),
            "overall_space_coverage": float(space_coverage_percent)
        }
    
    def analysis_coverage_by_zones(self, zones):
        """
        批量计算多个自定义区域的时间/空间覆盖度。
        
        zones: list[dict] 或 str（JSON 文件路径）
            如果是字符串，则认为是 JSON 文件路径，会自动加载
        
        return: [
            {
                "zone_name": str,
                "overall_time_coverage": float,
                "overall_space_coverage": float
            },
            ...
        ]
        """
        # 如果传入的是路径（字符串），则加载 JSON
        if isinstance(zones, str) or isinstance(zones, Path):
            with open(zones, "r", encoding="utf-8") as f:
                zones = json.load(f)
            
            # 如果 JSON 中坐标是 [lon, lat] 需要转换成 [lat, lon]
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

        return results

    def run(self):
        activate_areas = self._analysis_activate_areas()
        dwell_positions = self._analysis_dwell_positions()
        speed = self._analysis_speed()
        total_distance = self._analysis_total_distance()

        zones = self.analysis_coverage_by_zones(zones=self.zones)

        # print(results)
        return activate_areas, dwell_positions, speed, total_distance, zones