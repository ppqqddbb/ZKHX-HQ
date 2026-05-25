import logging
import polars as pl
from hdbscan import HDBSCAN
import numpy as np
from shapely.geometry import MultiPoint
from shapely.ops import transform
import pyproj
from utils.utils import *
from config.settings import *
from code.data import DataLoader

class HotspotAnalyzer:
    def __init__(
            self, 
            df, 
            min_cluster_size, 
            min_samples, 
            resolution
        ):
        self.df = df
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.resolution = resolution

    def _fit_hdbscan(self, X):
        model = HDBSCAN(
            min_cluster_size=self.min_cluster_size, 
            min_samples=self.min_samples
        )
        labels = model.fit_predict(X)
        return labels
    
    def coverage_by_mmsi(
        self,
        df_all: pl.LazyFrame,
        df_hotspot: pl.LazyFrame
    ) -> dict:
        """
        按 MMSI 分别计算：
        - 时间覆盖度（热点时间 / 总体时间）
        - 空间覆盖度（热点凸包面积，平方米）
        """

        # ---------- 时间跨度（总体） ----------
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

        # ---------- 时间跨度（热点） ----------
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

        # 转 dict，方便查
        all_time_map = {
            row[MMSI_COL]: row["all_span"]
            for row in all_time.iter_rows(named=True)
        }

        hotspot_time_map = {
            row[MMSI_COL]: row["hot_span"]
            for row in hotspot_time.iter_rows(named=True)
        }

        # ---------- 空间凸包（热点） ----------
        hotspot_points = (
            df_hotspot
            .group_by(MMSI_COL)
            .agg(
                pl.struct([LONGITUDE_COL, LATITUDE_COL]).implode().alias("points")
            )
            .collect()
        )

        # 坐标投影（WGS84 → 米）
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

        # ---------- 汇总为最终格式 ----------
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

        return result

    def run(self):
        df = add_h3_index(self.df, self.resolution)
        h3_center_df = h3_center(df)
        X = h3_center_df.select([LATITUDE_COL, LONGITUDE_COL]).collect().to_numpy()
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
        # print(coverage_result)
        return df_filtered, coverage_result
    