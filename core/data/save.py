"""
企业级数据保存模块
提供多种格式的数据导出功能
"""
import json
import polars as pl
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict

from core.config import data_config, model_config, analysis_config
from core.exceptions import FileOperationError, DataProcessingError
from core.logger import get_logger

logger = get_logger(__name__)


class DataSaver:
    """
    数据保存器

    支持CSV、Parquet、JSON、GeoJSON等多种格式的数据导出
    """

    def __init__(self, df: pl.LazyFrame, output_dir: str, filename: str):
        """
        初始化数据保存器

        Args:
            df: 待保存的数据
            output_dir: 输出目录
            filename: 文件名（不含扩展名）

        Raises:
            DataProcessingError: 参数无效时抛出
        """
        if df is None:
            raise DataProcessingError("输入数据不能为空")

        if not output_dir:
            raise DataProcessingError("输出目录不能为空")

        if not filename:
            raise DataProcessingError("文件名不能为空")

        self.df = df
        self.output_dir = Path(output_dir).resolve()
        self.filename = filename

        logger.info(f"数据保存器初始化完成 - 目标: {self.output_dir / filename}")

    def _collect(self) -> Optional[pl.DataFrame]:
        """
        将LazyFrame转换为DataFrame

        Returns:
            DataFrame对象，失败返回None

        Raises:
            DataProcessingError: 数据收集失败时抛出
        """
        try:
            if isinstance(self.df, pl.LazyFrame):
                return self.df.collect()
            return self.df

        except Exception as e:
            raise DataProcessingError(
                f"LazyFrame转换DataFrame失败: {str(e)}",
                details={"error": str(e)}
            )

    def save_json(self) -> Path:
        """
        保存为JSON格式

        Returns:
            保存的文件路径

        Raises:
            FileOperationError: 保存失败时抛出
        """
        try:
            df = self._collect()
            path = self.output_dir / f"{self.filename}.json"
            path.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(df, pl.DataFrame):
                df_dict = df.to_dicts()
            elif isinstance(df, pd.DataFrame):
                df_dict = df.to_dict(orient="records")
            else:
                df_dict = df

            with open(path, "w", encoding='utf-8') as f:
                json.dump(df_dict, f, ensure_ascii=False, indent=4, default=str)

            logger.info(f"JSON文件保存成功: {path}")
            return path

        except Exception as e:
            raise FileOperationError(
                f"JSON文件保存失败: {str(e)}",
                details={"path": str(path), "error": str(e)}
            )

    def save_csv(self) -> Path:
        """
        保存为CSV格式

        Returns:
            保存的文件路径

        Raises:
            FileOperationError: 保存失败时抛出
        """
        try:
            path = self.output_dir / f"{self.filename}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(self.df, pl.DataFrame):
                self.df.write_csv(path)
            elif isinstance(self.df, pl.LazyFrame):
                self.df.sink_csv(path)
            elif isinstance(self.df, pd.DataFrame):
                self.df.to_csv(path, index=False)
            else:
                raise FileOperationError(f"不支持的数据类型: {type(self.df)}")

            logger.info(f"CSV文件保存成功: {path}")
            return path

        except Exception as e:
            raise FileOperationError(
                f"CSV文件保存失败: {str(e)}",
                details={"path": str(path), "error": str(e)}
            )

    def save_parquet(self) -> Path:
        """
        保存为Parquet格式

        Returns:
            保存的文件路径

        Raises:
            FileOperationError: 保存失败时抛出
        """
        try:
            path = self.output_dir / f"{self.filename}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(self.df, pl.DataFrame):
                self.df.write_parquet(path)
            elif isinstance(self.df, pl.LazyFrame):
                self.df.sink_parquet(path)
            elif isinstance(self.df, pd.DataFrame):
                self.df.to_parquet(path, index=False)
            else:
                raise FileOperationError(f"不支持的数据类型: {type(self.df)}")

            logger.info(f"Parquet文件保存成功: {path}")
            return path

        except Exception as e:
            raise FileOperationError(
                f"Parquet文件保存失败: {str(e)}",
                details={"path": str(path), "error": str(e)}
            )

    def save_geojson(self) -> Path:
        """
        保存为GeoJSON格式

        Returns:
            保存的文件路径

        Raises:
            FileOperationError: 保存失败时抛出
        """
        try:
            df = self._collect()
            path = self.output_dir / f"{self.filename}.geojson"
            path.parent.mkdir(parents=True, exist_ok=True)

            # 假设df是GeoDataFrame
            df.to_file(path, driver="GeoJSON")

            logger.info(f"GeoJSON文件保存成功: {path}")
            return path

        except Exception as e:
            raise FileOperationError(
                f"GeoJSON文件保存失败: {str(e)}",
                details={"path": str(path), "error": str(e)}
            )

    def save_split_by_mmsi(self) -> Path:
        """
        按MMSI拆分保存数据

        Returns:
            保存的目录路径

        Raises:
            FileOperationError: 保存失败时抛出
        """
        try:
            df_collected = self._collect()

            mmsi_df = df_collected.select(pl.col(MMSI_COL)).unique()
            mmsi_list = mmsi_df[MMSI_COL].to_list()

            output_dir = self.output_dir / "split_by_mmsi"
            output_dir.mkdir(parents=True, exist_ok=True)

            for mmsi in mmsi_list:
                df_mmsi = df_collected.filter(pl.col(MMSI_COL) == mmsi)
                path = output_dir / f"{mmsi}.parquet"

                if isinstance(df_mmsi, pl.LazyFrame):
                    df_mmsi.sink_parquet(path)
                elif isinstance(df_mmsi, pl.DataFrame):
                    df_mmsi.write_parquet(path)

            logger.info(f"按MMSI拆分保存完成 - 目录: {output_dir}, MMSI数: {len(mmsi_list)}")
            return output_dir

        except Exception as e:
            raise FileOperationError(
                f"按MMSI拆分保存失败: {str(e)}",
                details={"output_dir": str(output_dir), "error": str(e)}
            )

    def run(
        self,
        formats: Optional[List[str]] = None,
        split_by_mmsi: bool = False
    ) -> Dict[str, Path]:
        """
        执行数据保存

        Args:
            formats: 保存格式列表，支持 'csv', 'parquet', 'json', 'geojson'
            split_by_mmsi: 是否按MMSI拆分保存

        Returns:
            保存的文件路径字典

        Raises:
            FileOperationError: 保存失败时抛出
        """
        if formats is None:
            formats = ['parquet']

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            saved_paths = {}

            if 'csv' in formats:
                saved_paths['csv'] = self.save_csv()

            if 'parquet' in formats:
                saved_paths['parquet'] = self.save_parquet()

            if 'json' in formats:
                saved_paths['json'] = self.save_json()

            if 'geojson' in formats:
                saved_paths['geojson'] = self.save_geojson()

            if split_by_mmsi:
                saved_paths['split_data'] = self.save_split_by_mmsi()

            logger.info(f"数据保存完成 - 格式: {list(saved_paths.keys())}")
            return saved_paths

        except Exception as e:
            if isinstance(e, FileOperationError):
                raise
            raise FileOperationError(
                f"数据保存执行失败: {str(e)}",
                details={"formats": formats, "error": str(e)}
            )
