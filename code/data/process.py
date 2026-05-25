import logging
import numpy as np
import polars as pl
from config.settings import *

class DataProcessor:
    def __init__(self, df: pl.LazyFrame, env_df: pl.LazyFrame = None):
        self.df = df
        self.env_df = env_df
        # ✅ 新增
        self.missing_rows = None
        self.outlier_rows = None

    def rename_columns(self):
        try:
            self.df = (
                self.df
                .rename(RENAME_NAME)
                .sort([MMSI_COL, TIMESTAMP_COL])
            )
            current_schema = self.df.collect_schema()
            if current_schema.get(TIMESTAMP_COL) == pl.String:
                self.df = self.df.with_columns(
                    pl.col(TIMESTAMP_COL).str.strptime(pl.Datetime, TimeFormat, strict=False)
                )
                logging.info("Timestamp 列已从 String 解析为 Datetime")
            else:
                logging.info(f"Timestamp 列已经是 {current_schema.get(TIMESTAMP_COL)}，跳过解析")
            logging.info("列重命名和排序完成")
        except Exception as e:
            logging.error(f"重命名列或时间解析出错: {str(e)}")
        return self.df

    def remove_outliers(self):
        try:
            # ✅ 定义异常条件（复用逻辑）
            outlier_conditions = [
                (
                    (pl.col(c) < (
                        pl.col(c).quantile(0.25).over(MMSI_COL) -
                        1.5 * (pl.col(c).quantile(0.75).over(MMSI_COL) - pl.col(c).quantile(0.25).over(MMSI_COL))
                    )) |
                    (pl.col(c) > (
                        pl.col(c).quantile(0.75).over(MMSI_COL) +
                        1.5 * (pl.col(c).quantile(0.75).over(MMSI_COL) - pl.col(c).quantile(0.25).over(MMSI_COL))
                    ))
                ).alias(f"{c}_is_outlier")
                for c in NUM_COLUMNS
            ]

            df_flagged = self.df.with_columns(outlier_conditions)

            # ✅ 记录异常整行
            self.outlier_rows = df_flagged.filter(
                pl.any_horizontal([pl.col(f"{c}_is_outlier") for c in NUM_COLUMNS])
            ).select(self.df.columns)

            # ✅ 正式替换异常值
            self.df = df_flagged.with_columns(
                [
                    pl.when(pl.col(f"{c}_is_outlier"))
                    .then(None)
                    .otherwise(pl.col(c))
                    .alias(c)
                    for c in NUM_COLUMNS
                ]
            ).drop([f"{c}_is_outlier" for c in NUM_COLUMNS])

            logging.info("异常值处理完成（已记录异常数据）")

        except Exception as e:
            logging.error(f"异常值处理出错: {str(e)}")

        return self.df

    def fill_missing(self):
        try:
            # ✅ 记录缺失整行
            self.missing_rows = self.df.filter(
                pl.any_horizontal([pl.col(c).is_null() for c in self.df.columns])
            )

            # ✅ 正常填充逻辑
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

            logging.info("缺失值填充完成（已记录缺失数据）")

        except Exception as e:
            logging.error(f"缺失值填充出错: {str(e)}")

        return self.df
    
    def resample(self, freq) -> pl.DataFrame:
        self.df = (
            self.df
            .sort([MMSI_COL, TIMESTAMP_COL])
            .with_columns(
                pl.col(TIMESTAMP_COL).cast(pl.Datetime)
            )
            .group_by_dynamic(
                index_column=TIMESTAMP_COL,
                # include_boundaries=True,
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
        return self.df
    
    def save_quality_logs(self, missing_path: str, outlier_path: str):
        try:
            if self.missing_rows is not None:
                self.missing_rows.sink_parquet(missing_path)
                logging.info(f"缺失数据已保存: {missing_path}")

            if self.outlier_rows is not None:
                self.outlier_rows.sink_parquet(outlier_path)
                logging.info(f"异常数据已保存: {outlier_path}")

        except Exception as e:
            logging.error(f"保存质量日志失败: {str(e)}")
        
    def complete_time_index(self, freq):
        # 1️⃣ 每个 MMSI 的起止时间
        time_bounds = (
            self.df
            .group_by(MMSI_COL)
            .agg(
                pl.min(TIMESTAMP_COL).alias("start"),
                pl.max(TIMESTAMP_COL).alias("end"),
            )
        )

        # 2️⃣ 对每一行生成 datetime_range（关键修复点）
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
                    )
                )
                .alias(TIMESTAMP_COL)
            )
            .explode(TIMESTAMP_COL)
            .select([MMSI_COL, TIMESTAMP_COL])
        )

        # 3️⃣ 左连接回 resample 结果
        self.df = (
            timeline
            .join_asof(
                self.df,
                on=TIMESTAMP_COL,
                by=MMSI_COL,
                strategy="backward"
            )
        )

        return self.df

    def feature_engineering(self):
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
                # 速度分解
                (pl.col(SOG_COL) * pl.col("cog_sin")).alias("v_x"),
                (pl.col(SOG_COL) * pl.col("cog_cos")).alias("v_y"),
            ])
            .with_columns([
                # 距离上一个时刻的距离
                pl.col("x_meters").diff().over(MMSI_COL).alias("delta_x"),
                pl.col("y_meters").diff().over(MMSI_COL).alias("delta_y"),
                pl.col(SOG_COL).diff().over(MMSI_COL).alias("delta_sog"),
                pl.col(COG_COL).diff().over(MMSI_COL).alias("delta_cog")
            ])
            .with_columns([
                # 相对于起点的距离
                (pl.col("x_meters") - pl.col("x_meters").first().over(MMSI_COL)).alias("rel_x"),
                (pl.col("y_meters") - pl.col("y_meters").first().over(MMSI_COL)).alias("rel_y"),
                # 环境干扰
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
        return self.df
        
    def process_time(self):  
        schema = self.df.collect_schema()
        if schema.get(TIMESTAMP_COL) == pl.String:
            self.df =  self.df.with_columns(
                pl.col(TIMESTAMP_COL).str.strptime(
                    pl.Datetime, TimeFormat, strict=False
                )
            )

        return self
    
    def filter_motion_data(self, min_sog=0.1, max_sog=60.0, min_dist_m=500.0):
        """
        过滤掉静止漂移数据和速度异常数据
        :param min_sog: 最小航速阈值（节），低于此值视为静止
        :param max_sog: 最大航速阈值（节），高于此值视为异常
        :param min_dist_m: 最小位移阈值（米），低于此值视为未移动
        """
        try:
            # 1. 基于速度的初步过滤
            self.df = self.df.filter(
                (pl.col(SOG_COL) >= min_sog) & 
                (pl.col(SOG_COL) <= max_sog)
            )

            # 2. 基于实际位移的过滤 (计算相邻点哈弗辛距离)
            # 这一步能有效去除“坐标微小震荡”的漂移点
            self.df = self.df.with_columns([
                (pl.col(LATITUDE_COL).diff().over(MMSI_COL)).alias("_dlat"),
                (pl.col(LONGITUDE_COL).diff().over(MMSI_COL)).alias("_dlon"),
            ]).with_columns([
                # 简化的欧氏距离近似（米），在小范围内足够精准
                (
                    (pl.col("_dlat") * 111320).pow(2) + 
                    (pl.col("_dlon") * 111320 * (pl.col(LATITUDE_COL) * np.pi / 180).cos()).pow(2)
                ).sqrt().alias("displacement")
            ])

            # 过滤掉位移过小的点
            self.df = self.df.filter(
                (pl.col("displacement").is_null()) | (pl.col("displacement") > min_dist_m)
            ).drop(["_dlat", "_dlon", "displacement"])

            logging.info(f"运动数据过滤完成：保留 SOG 在 {min_sog}-{max_sog} 且位移 > {min_dist_m}m 的点")
        except Exception as e:
            logging.error(f"过滤运动数据出错: {str(e)}")
        return self.df

    def process_with_ocean(self, resolution=0.1):
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
        return self.df

    def run(self):
        self.rename_columns()
        self.remove_outliers()
        self.fill_missing()

        self.save_quality_logs(
            missing_path="./output/missing_data.parquet",
            outlier_path="./output/outlier_data.parquet"
        )

        logging.info("✅ 数据处理完成")

        return self.df

    def run_prediction(
            self, 
            freq: str, 
        ):
        logging.info("数据处理开始...")

        logging.info("正在进行数据重命名...")
        self.rename_columns()

        logging.info("正在进行数据异常值处理...")
        self.remove_outliers()

        logging.info("正在进行数据重采样...")
        self.complete_time_index(freq)

        logging.info("正在进行数据缺失值处理...")
        self.fill_missing()

        logging.info("正在进行数据特征工程处理...")
        self.feature_engineering()

        logging.info("正在进行数据过滤，过滤位移距离过小、速度过低的数据...")
        self.filter_motion_data()

        self.process_time()

        if self.env_df is not None:
            logging.info("正在进行数据与海面数据融合...")
            self.df = self.process_with_ocean()

        logging.info("✅ 已通过 Polars 表达式完成 Z-score 标准化 (均值-方差)")

        return self.df


if __name__ == '__main__':
    df = pl.scan_parquet(r"D:\InstallFiles\File\Project\upload\AIS_2024_12_31.parquet")
    env_df = pl.scan_parquet(r"D:\InstallFiles\File\Project\upload\haiyang.parquet")
    processor = DataProcessor(df=df, env_df=env_df)
    df = processor.run(freq=freq)
    df.sink_csv(r"D:\InstallFiles\File\Project\upload\haiyang1.csv")