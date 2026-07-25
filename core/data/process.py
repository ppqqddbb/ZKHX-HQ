"""
企业级数据处理模块
提供AIS数据的清洗、转换和特征工程功能
"""
import numpy as np
import polars as pl
from typing import Optional, Tuple
from pathlib import Path

from core.config import data_config, model_config
from core.exceptions import DataProcessingError, DataValidationError
from core.logger import get_logger
from core.validators import DataValidator

logger = get_logger(__name__)

# 便捷访问配置
MMSI_COL = data_config.MMSI_COL
TIMESTAMP_COL = data_config.TIMESTAMP_COL
LATITUDE_COL = data_config.LATITUDE_COL
LONGITUDE_COL = data_config.LONGITUDE_COL
SOG_COL = data_config.SOG_COL
COG_COL = data_config.COG_COL
HEADING_COL = data_config.HEADING_COL
SHIPTYPE_COL = data_config.SHIPTYPE_COL
RENAME_NAME = data_config.RENAME_MAP
TimeFormat = data_config.TIME_FORMAT
NUM_COLUMNS = data_config.NUMERIC_COLUMNS
R = model_config.EARTH_RADIUS


class DataProcessor:
    """
    AIS数据处理器

    负责数据清洗、标准化、特征工程等核心数据处理任务
    """

    def __init__(self, df: pl.LazyFrame, env_df: Optional[pl.LazyFrame] = None):
        """
        初始化数据处理器

        Args:
            df: 原始AIS数据（LazyFrame）
            env_df: 可选的海洋环境数据（LazyFrame）
        """
        if df is None:
            raise DataValidationError("输入数据不能为空")

        self.df = df
        self.env_df = env_df
        self.missing_rows: Optional[pl.LazyFrame] = None
        self.outlier_rows: Optional[pl.LazyFrame] = None

        logger.info("数据处理器初始化完成")

    def rename_columns(self) -> pl.LazyFrame:
        """
        重命名列并排序数据

        Returns:
            重命名后的LazyFrame

        Raises:
            DataProcessingError: 列重命名失败时抛出
        """
        try:
            self.df = (
                self.df
                .rename(RENAME_NAME)
                .sort([MMSI_COL, TIMESTAMP_COL])
            )

            current_schema = self.df.collect_schema()
            if current_schema.get(TIMESTAMP_COL) == pl.String:
                self.df = self.df.with_columns(
                    pl.col(TIMESTAMP_COL).str.strptime(
                        pl.Datetime, TimeFormat, strict=False
                    )
                )
                logger.info(f"{TIMESTAMP_COL}列已从String解析为Datetime")
            else:
                logger.info(f"{TIMESTAMP_COL}列已经是{current_schema.get(TIMESTAMP_COL)}类型")

            logger.info("列重命名和排序完成")
            return self.df

        except Exception as e:
            raise DataProcessingError(
                f"列重命名或时间解析失败: {str(e)}",
                details={"error": str(e)}
            )

    def remove_outliers(self) -> pl.LazyFrame:
        """
        使用IQR方法移除异常值

        Returns:
            处理后的LazyFrame

        Raises:
            DataProcessingError: 异常值处理失败时抛出
        """
        try:
            outlier_conditions = [
                (
                    (pl.col(c) < (
                        pl.col(c).quantile(0.25).over(MMSI_COL) -
                        1.5 * (pl.col(c).quantile(0.75).over(MMSI_COL) -
                               pl.col(c).quantile(0.25).over(MMSI_COL))
                    )) |
                    (pl.col(c) > (
                        pl.col(c).quantile(0.75).over(MMSI_COL) +
                        1.5 * (pl.col(c).quantile(0.75).over(MMSI_COL) -
                               pl.col(c).quantile(0.25).over(MMSI_COL))
                    ))
                ).alias(f"{c}_is_outlier")
                for c in NUM_COLUMNS
            ]

            df_flagged = self.df.with_columns(outlier_conditions)

            self.outlier_rows = df_flagged.filter(
                pl.any_horizontal([pl.col(f"{c}_is_outlier") for c in NUM_COLUMNS])
            ).select(self.df.collect_schema().names())

            self.df = df_flagged.with_columns(
                [
                    pl.when(pl.col(f"{c}_is_outlier"))
                    .then(None)
                    .otherwise(pl.col(c))
                    .alias(c)
                    for c in NUM_COLUMNS
                ]
            ).drop([f"{c}_is_outlier" for c in NUM_COLUMNS])

            logger.info("异常值处理完成（已记录异常数据）")
            return self.df

        except Exception as e:
            raise DataProcessingError(
                f"异常值处理失败: {str(e)}",
                details={"error": str(e)}
            )

    def fill_missing(self) -> pl.LazyFrame:
        """
        填充缺失值

        策略：前向填充 -> 线性插值 -> 后向填充 -> 填充0

        Returns:
            处理后的LazyFrame

        Raises:
            DataProcessingError: 缺失值填充失败时抛出
        """
        try:
            self.missing_rows = self.df.filter(
                pl.any_horizontal([pl.col(c).is_null() for c in self.df.collect_schema().names()])
            )

            self.df = (
                self.df
                .with_columns(
                    pl.col(NUM_COLUMNS)
                    .fill_null(strategy="forward")
                    .over(MMSI_COL)
                    .interpolate(method="linear")
                    .over(MMSI_COL)
                    .fill_null(strategy="backward")
                    .over(MMSI_COL)
                    .fill_null(0.0)
                )
            )

            logger.info("缺失值填充完成（已记录缺失数据）")
            return self.df

        except Exception as e:
            raise DataProcessingError(
                f"缺失值填充失败: {str(e)}",
                details={"error": str(e)}
            )

    def resample(self, freq: str) -> pl.LazyFrame:
        """
        按指定频率重采样数据

        Args:
            freq: 重采样频率（如 '5m', '1h'）

        Returns:
            重采样后的LazyFrame

        Raises:
            DataProcessingError: 重采样失败时抛出
        """
        try:
            self.df = (
                self.df
                .sort([MMSI_COL, TIMESTAMP_COL])
                .with_columns(
                    pl.col(TIMESTAMP_COL).cast(pl.Datetime)
                )
                .group_by_dynamic(
                    index_column=TIMESTAMP_COL,
                    every=freq,
                    by=MMSI_COL,
                    closed='left',
                    label='left'
                )
                .agg([
                    pl.last(LONGITUDE_COL).alias(LONGITUDE_COL),
                    pl.last(LATITUDE_COL).alias(LATITUDE_COL),
                    pl.last(SOG_COL).alias(SOG_COL),
                    pl.last(COG_COL).alias(COG_COL),
                    pl.last(HEADING_COL).alias(HEADING_COL),
                    pl.last(SHIPTYPE_COL).alias(SHIPTYPE_COL)
                ])
            )

            logger.info(f"数据重采样完成，频率: {freq}")
            return self.df

        except Exception as e:
            raise DataProcessingError(
                f"数据重采样失败: {str(e)}",
                details={"freq": freq, "error": str(e)}
            )

    def complete_time_index(self, freq: str) -> pl.LazyFrame:
        """
        补全时间索引，确保每个MMSI的时间序列连续

        Args:
            freq: 时间频率

        Returns:
            补全后的LazyFrame

        Raises:
            DataProcessingError: 时间索引补全失败时抛出
        """
        try:
            time_bounds = (
                self.df
                .group_by(MMSI_COL)
                .agg(
                    pl.min(TIMESTAMP_COL).alias("start"),
                    pl.max(TIMESTAMP_COL).alias("end"),
                )
            )

            timeline = (
                time_bounds
                .with_columns(
                    pl.struct(["start", "end"])
                    .map_elements(
                        lambda x: pl.datetime_range(
                            start=x["start"],
                            end=x["end"],
                            interval=freq,
                            eager=True
                        ),
                        return_dtype=pl.List(pl.Datetime)
                    )
                    .alias(TIMESTAMP_COL)
                )
                .explode(TIMESTAMP_COL)
                .select([MMSI_COL, TIMESTAMP_COL])
            )

            self.df = (
                timeline
                .join_asof(
                    self.df,
                    on=TIMESTAMP_COL,
                    by=MMSI_COL,
                    strategy="backward"
                )
            )

            logger.info("时间索引补全完成")
            return self.df

        except Exception as e:
            raise DataProcessingError(
                f"时间索引补全失败: {str(e)}",
                details={"freq": freq, "error": str(e)}
            )

    def feature_engineering(self) -> pl.LazyFrame:
        """
        特征工程：生成衍生特征

        包括：
        - 墨卡托坐标转换
        - 三角函数特征
        - 速度分解
        - 位移特征
        - 曲率特征

        Returns:
            增强后的LazyFrame

        Raises:
            DataProcessingError: 特征工程失败时抛出
        """
        try:
            self.df = (
                self.df
                .with_columns([
                    (pl.col(LONGITUDE_COL) * np.pi / 180 * R).alias("x_meters"),
                    ((pl.col(LATITUDE_COL) * np.pi / 180 / 2 + np.pi / 4).tan().log() * R).alias("y_meters"),
                    (pl.col(COG_COL) * np.pi / 180).sin().alias("cog_sin"),
                    (pl.col(COG_COL) * np.pi / 180).cos().alias("cog_cos"),
                    (pl.col(HEADING_COL) * np.pi / 180).sin().alias("heading_sin"),
                    (pl.col(HEADING_COL) * np.pi / 180).cos().alias("heading_cos"),
                ])
                .with_columns([
                    (pl.col(SOG_COL) * pl.col("cog_sin")).alias("v_x"),
                    (pl.col(SOG_COL) * pl.col("cog_cos")).alias("v_y"),
                ])
                .with_columns([
                    pl.col("x_meters").diff().over(MMSI_COL).alias("delta_x"),
                    pl.col("y_meters").diff().over(MMSI_COL).alias("delta_y"),
                    pl.col(SOG_COL).diff().over(MMSI_COL).alias("delta_sog"),
                    pl.col(COG_COL).diff().over(MMSI_COL).alias("delta_cog")
                ])
                .with_columns([
                    (pl.col("x_meters") - pl.col("x_meters").first().over(MMSI_COL)).alias("rel_x"),
                    (pl.col("y_meters") - pl.col("y_meters").first().over(MMSI_COL)).alias("rel_y"),
                    (pl.col(COG_COL) - pl.col(HEADING_COL)).alias("drift_angle")
                ])
                .with_columns([
                    ((pl.col(TIMESTAMP_COL).dt.hour() + pl.col(TIMESTAMP_COL).dt.minute() / 60) * 2 * np.pi / 24).alias("_time_rad")
                ])
                .with_columns([
                    pl.col("_time_rad").sin().alias("time_sin"),
                    pl.col("_time_rad").cos().alias("time_cos")
                ]).drop("_time_rad")
                .fill_null(0.0)
            )

            self.df = self.df.with_columns([
                pl.col("delta_x").diff().over(MMSI_COL).alias("delta2_x"),
                pl.col("delta_y").diff().over(MMSI_COL).alias("delta2_y"),
            ]).with_columns([
                (
                    (pl.col("delta_x") * pl.col("delta2_y") - pl.col("delta_y") * pl.col("delta2_x")).abs() /
                    ((pl.col("delta_x")**2 + pl.col("delta_y")**2 + 1e-6)**1.5)
                ).alias("curvature")
            ]).fill_null(0.0)

            logger.info("特征工程完成")
            return self.df

        except Exception as e:
            raise DataProcessingError(
                f"特征工程失败: {str(e)}",
                details={"error": str(e)}
            )

    def filter_motion_data(
        self,
        min_sog: float = 0.1,
        max_sog: float = 60.0,
        min_dist_m: float = 500.0
    ) -> pl.LazyFrame:
        """
        过滤静止和异常数据

        Args:
            min_sog: 最小航速阈值（节）
            max_sog: 最大航速阈值（节）
            min_dist_m: 最小位移阈值（米）

        Returns:
            过滤后的LazyFrame

        Raises:
            DataProcessingError: 运动数据过滤失败时抛出
        """
        try:
            self.df = self.df.filter(
                (pl.col(SOG_COL) >= min_sog) &
                (pl.col(SOG_COL) <= max_sog)
            )

            self.df = self.df.with_columns([
                (pl.col(LATITUDE_COL).diff().over(MMSI_COL)).alias("_dlat"),
                (pl.col(LONGITUDE_COL).diff().over(MMSI_COL)).alias("_dlon"),
            ]).with_columns([
                (
                    (pl.col("_dlat") * 111320).pow(2) +
                    (pl.col("_dlon") * 111320 * (pl.col(LATITUDE_COL) * np.pi / 180).cos()).pow(2)
                ).sqrt().alias("displacement")
            ])

            self.df = self.df.filter(
                (pl.col("displacement").is_null()) | (pl.col("displacement") > min_dist_m)
            ).drop(["_dlat", "_dlon", "displacement"])

            logger.info(f"运动数据过滤完成: SOG [{min_sog}, {max_sog}], 位移 > {min_dist_m}m")
            return self.df

        except Exception as e:
            raise DataProcessingError(
                f"运动数据过滤失败: {str(e)}",
                details={"min_sog": min_sog, "max_sog": max_sog, "min_dist_m": min_dist_m, "error": str(e)}
            )

    def process_with_ocean(self, resolution: float = 0.1) -> pl.LazyFrame:
        """
        融合海洋环境数据

        Args:
            resolution: 网格分辨率

        Returns:
            融合后的LazyFrame

        Raises:
            DataProcessingError: 海洋数据融合失败时抛出
        """
        if self.env_df is None:
            logger.warning("未提供海洋环境数据，跳过融合步骤")
            return self.df

        try:
            self.df = self.df.with_columns([
                (pl.col(LONGITUDE_COL) / resolution).round(0).alias("grid_lon"),
                (pl.col(LATITUDE_COL) / resolution).round(0).alias("grid_lat")
            ])

            self.env_df = self.env_df.with_columns([
                (pl.col(LONGITUDE_COL) / resolution).round(0).alias("grid_lon"),
                (pl.col(LATITUDE_COL) / resolution).round(0).alias("grid_lat")
            ]).sort(TIMESTAMP_COL)

            self.df = (
                self.df
                .sort([MMSI_COL, TIMESTAMP_COL])
                .join_asof(
                    self.env_df,
                    on=TIMESTAMP_COL,
                    by=["grid_lon", "grid_lat"],
                    strategy="backward"
                )
                .fill_null(0.0)
            )

            logger.info("海洋环境数据融合完成")
            return self.df

        except Exception as e:
            raise DataProcessingError(
                f"海洋数据融合失败: {str(e)}",
                details={"resolution": resolution, "error": str(e)}
            )

    def save_quality_logs(self, missing_path: str, outlier_path: str) -> None:
        """
        保存数据质量日志

        Args:
            missing_path: 缺失数据保存路径
            outlier_path: 异常数据保存路径

        Raises:
            DataProcessingError: 保存失败时抛出
        """
        try:
            from pathlib import Path

            if self.missing_rows is not None:
                Path(missing_path).parent.mkdir(parents=True, exist_ok=True)
                self.missing_rows.sink_parquet(missing_path)
                logger.info(f"缺失数据已保存: {missing_path}")

            if self.outlier_rows is not None:
                Path(outlier_path).parent.mkdir(parents=True, exist_ok=True)
                self.outlier_rows.sink_parquet(outlier_path)
                logger.info(f"异常数据已保存: {outlier_path}")

        except Exception as e:
            raise DataProcessingError(
                f"保存质量日志失败: {str(e)}",
                details={"missing_path": missing_path, "outlier_path": outlier_path, "error": str(e)}
            )

    def run(self) -> pl.LazyFrame:
        """
        执行标准数据处理流程

        Returns:
            处理后的LazyFrame
        """
        logger.info("开始执行标准数据处理流程")

        self.rename_columns()
        self.remove_outliers()
        self.fill_missing()

        self.save_quality_logs(
            missing_path="./output/missing_data.parquet",
            outlier_path="./output/outlier_data.parquet"
        )

        logger.info("标准数据处理流程完成")
        return self.df

    def run_prediction(self, freq: str) -> pl.LazyFrame:
        """
        执行预测任务的数据处理流程

        Args:
            freq: 重采样频率

        Returns:
            处理后的LazyFrame
        """
        logger.info("开始执行预测数据处理流程")

        self.rename_columns()
        self.remove_outliers()
        self.complete_time_index(freq)
        self.fill_missing()
        self.feature_engineering()
        self.filter_motion_data()

        if self.env_df is not None:
            self.process_with_ocean()

        logger.info("预测数据处理流程完成")
        return self.df
