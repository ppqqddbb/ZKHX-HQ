import os
import logging
import numpy as np
import polars as pl
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rcParams
from shapely.geometry import LineString
from shapely.ops import linemerge
from scipy.ndimage import gaussian_filter, binary_closing
from scipy.spatial import KDTree
from skimage.morphology import skeletonize
from config.settings import *

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False


class FrequentAnalyzer:
    def __init__(self, df: pl.LazyFrame, grid_size: float, sigma: float, percentile_thresh: int, visulize: bool = False):
        self.df = df
        self.grid_size = grid_size
        self.sigma = sigma
        self.percentile_thresh = percentile_thresh
        self.visualize = visulize

    def build_data(self):
        self.df = (
            self.df
            .select([TIMESTAMP_COL, LATITUDE_COL, LONGITUDE_COL, SOG_COL])
            .filter(pl.col(SOG_COL) > 1.0)
            .collect()
        )
        lats = self.df[LATITUDE_COL].to_numpy()
        lons = self.df[LONGITUDE_COL].to_numpy()
        lat_bins, lon_bins = np.arange(lats.min(), lats.max() + self.grid_size, self.grid_size), np.arange(lons.min(), lons.max() + self.grid_size, self.grid_size)
        
        return lats, lons, lat_bins, lon_bins

    def build_density_grid(self, lats, lons, lat_bins, lon_bins):
        density, _, _ = np.histogram2d(lats, lons, bins=[lat_bins, lon_bins])
        density = gaussian_filter(density, sigma=self.sigma)
        non_zero = density[density > 0]
        if len(non_zero) == 0:
            raise ValueError('Density grid contains non-zero values.')
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
            raise ValueError("骨架点过少，无法生成航道")
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
            raise ValueError("未生成任何航道线段")
        merged = linemerge(lines)
        if isinstance(merged, LineString):
            geometries = [merged]
        else:
            geometries = list(merged.geoms)
        gdf = gpd.GeoDataFrame(
            geometry=geometries,
            crs="EPSG:4326"
        )

        return gdf

    
    def run(self):
        lats, lons, lat_bins, lon_bins = self.build_data()

        # 网格和骨架
        lat_bins = np.arange(lats.min(), lats.max() + self.grid_size, self.grid_size)
        lon_bins = np.arange(lons.min(), lons.max() + self.grid_size, self.grid_size)
        gdf = self.build_density_grid(lats, lons, lat_bins, lon_bins)

        # 将骨架点与最近的原始点匹配
        from scipy.spatial import cKDTree
        points = np.column_stack([self.df[LONGITUDE_COL], self.df[LATITUDE_COL]])
        tree = cKDTree(points)
        
        records = []
        for geom in gdf.geometry:
            coords = list(geom.coords)
            for lon, lat in coords:
                dist, idx = tree.query([lon, lat])
                records.append({
                    LONGITUDE_COL: lon,
                    LATITUDE_COL: lat,
                    TIMESTAMP_COL: self.df[TIMESTAMP_COL][idx]
                })

        records_df = pl.DataFrame(records)

        return gdf, records_df