"""
企业级数据加载模块
提供CSV和Parquet格式的AIS数据加载功能
"""
import polars as pl
from pathlib import Path
from typing import Optional

from core.exceptions import FileOperationError, DataValidationError
from core.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    AIS数据加载器

    支持CSV和Parquet格式的数据文件加载
    """

    def __init__(self, input_path: str):
        """
        初始化数据加载器

        Args:
            input_path: 数据文件路径

        Raises:
            FileOperationError: 文件不存在或路径无效时抛出
        """
        if not input_path:
            raise FileOperationError("输入路径不能为空")

        self.input_path = Path(input_path)

        if not self.input_path.exists():
            raise FileOperationError(
                f"文件不存在: {input_path}",
                details={"path": str(self.input_path)}
            )

        logger.info(f"数据加载器初始化完成 - 文件: {self.input_path.name}")

    def load_csv(self) -> Optional[pl.LazyFrame]:
        """
        加载CSV格式数据

        Returns:
            LazyFrame对象，加载失败返回None

        Raises:
            FileOperationError: CSV读取失败时抛出
        """
        try:
            df = pl.scan_csv(
                self.input_path,
                ignore_errors=True,
                truncate_ragged_lines=True,
                infer_schema_length=10000
            )

            logger.info(f"CSV文件加载成功: {self.input_path}")
            return df

        except pl.exceptions.PolarsError as e:
            raise FileOperationError(
                f"Polars读取CSV失败: {str(e)}",
                details={"path": str(self.input_path), "error": str(e)}
            )
        except Exception as e:
            raise FileOperationError(
                f"CSV文件读取失败: {str(e)}",
                details={"path": str(self.input_path), "error": str(e)}
            )

    def load_parquet(self) -> Optional[pl.LazyFrame]:
        """
        加载Parquet格式数据

        Returns:
            LazyFrame对象，加载失败返回None

        Raises:
            FileOperationError: Parquet读取失败时抛出
        """
        try:
            df = pl.scan_parquet(self.input_path)

            logger.info(f"Parquet文件加载成功: {self.input_path}")
            return df

        except pl.exceptions.PolarsError as e:
            raise FileOperationError(
                f"Polars读取Parquet失败: {str(e)}",
                details={"path": str(self.input_path), "error": str(e)}
            )
        except Exception as e:
            raise FileOperationError(
                f"Parquet文件读取失败: {str(e)}",
                details={"path": str(self.input_path), "error": str(e)}
            )

    def run(self) -> pl.LazyFrame:
        """
        根据文件扩展名自动选择加载方法

        Returns:
            LazyFrame对象

        Raises:
            FileOperationError: 不支持的文件格式或加载失败时抛出
        """
        suffix = self.input_path.suffix.lower()

        if suffix == '.csv':
            return self.load_csv()
        elif suffix == '.parquet':
            return self.load_parquet()
        else:
            raise FileOperationError(
                f"不支持的文件格式: {suffix} (仅支持 .csv 或 .parquet)",
                details={"path": str(self.input_path), "suffix": suffix}
            )


# 兼容旧代码的别名
ShipDataLoader = DataLoader
