"""
数据验证模块
提供数据完整性和合法性检查
"""
import polars as pl
from typing import List, Optional
from .exceptions import DataValidationError


class DataValidator:
    """AIS数据验证器"""

    @staticmethod
    def validate_required_columns(df: pl.LazyFrame, required_columns: List[str]) -> None:
        """验证必需列是否存在"""
        schema_columns = df.collect_schema().names()
        missing_columns = [col for col in required_columns if col not in schema_columns]

        if missing_columns:
            raise DataValidationError(
                f"缺少必需列: {', '.join(missing_columns)}",
                details={"missing_columns": missing_columns, "available_columns": schema_columns}
            )

    @staticmethod
    def validate_data_not_empty(df: pl.LazyFrame, name: str = "数据集") -> None:
        """验证数据集不为空"""
        row_count = df.select(pl.len()).collect().item()
        if row_count == 0:
            raise DataValidationError(f"{name}为空")

    @staticmethod
    def validate_latitude_range(df: pl.LazyFrame, lat_col: str) -> None:
        """验证纬度范围 [-90, 90]"""
        result = df.select(
            pl.col(lat_col).min().alias("min_lat"),
            pl.col(lat_col).max().alias("max_lat")
        ).collect()

        min_lat = result["min_lat"][0]
        max_lat = result["max_lat"][0]

        if min_lat < -90 or max_lat > 90:
            raise DataValidationError(
                f"纬度超出有效范围 [-90, 90]: min={min_lat}, max={max_lat}",
                details={"min_lat": min_lat, "max_lat": max_lat}
            )

    @staticmethod
    def validate_longitude_range(df: pl.LazyFrame, lon_col: str) -> None:
        """验证经度范围 [-180, 180]"""
        result = df.select(
            pl.col(lon_col).min().alias("min_lon"),
            pl.col(lon_col).max().alias("max_lon")
        ).collect()

        min_lon = result["min_lon"][0]
        max_lon = result["max_lon"][0]

        if min_lon < -180 or max_lon > 180:
            raise DataValidationError(
                f"经度超出有效范围 [-180, 180]: min={min_lon}, max={max_lon}",
                details={"min_lon": min_lon, "max_lon": max_lon}
            )

    @staticmethod
    def validate_sog_range(df: pl.LazyFrame, sog_col: str, max_sog: float = 100.0) -> None:
        """验证航速范围 [0, max_sog]"""
        result = df.select(
            pl.col(sog_col).min().alias("min_sog"),
            pl.col(sog_col).max().alias("max_sog")
        ).collect()

        min_sog = result["min_sog"][0]
        max_sog_val = result["max_sog"][0]

        if min_sog < 0 or max_sog_val > max_sog:
            raise DataValidationError(
                f"航速超出有效范围 [0, {max_sog}]: min={min_sog}, max={max_sog_val}",
                details={"min_sog": min_sog, "max_sog": max_sog_val}
            )

    @staticmethod
    def validate_cog_range(df: pl.LazyFrame, cog_col: str) -> None:
        """验证航向范围 [0, 360]"""
        result = df.select(
            pl.col(cog_col).min().alias("min_cog"),
            pl.col(cog_col).max().alias("max_cog")
        ).collect()

        min_cog = result["min_cog"][0]
        max_cog = result["max_cog"][0]

        if min_cog < 0 or max_cog > 360:
            raise DataValidationError(
                f"航向超出有效范围 [0, 360]: min={min_cog}, max={max_cog}",
                details={"min_cog": min_cog, "max_cog": max_cog}
            )

    @classmethod
    def validate_ais_data(
        cls,
        df: pl.LazyFrame,
        required_columns: List[str],
        lat_col: str = "Latitude",
        lon_col: str = "Longitude",
        sog_col: str = "SOG",
        cog_col: str = "COG"
    ) -> None:
        """执行完整的AIS数据验证"""
        cls.validate_required_columns(df, required_columns)
        cls.validate_data_not_empty(df)
        cls.validate_latitude_range(df, lat_col)
        cls.validate_longitude_range(df, lon_col)
        cls.validate_sog_range(df, sog_col)
        cls.validate_cog_range(df, cog_col)
