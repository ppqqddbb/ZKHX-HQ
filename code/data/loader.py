import logging
import polars as pl
from pathlib import Path


class DataLoader:
    def __init__(self, input_path: str):
        self.input_path = Path(input_path)

    def load_csv(self) -> pl.LazyFrame | None:
        try:
            df = pl.scan_csv(
                self.input_path,
                ignore_errors=True,            # 关键：跳过格式错误的行
                truncate_ragged_lines=True,     # 关键：截断列数不对的行
                infer_schema_length=10000       # 增加推断行数，防止类型推断错误
                )
            logging.info(f"成功加载 CSV 文件: {self.input_path}")
            return df
        except FileNotFoundError:
            logging.error(f"CSV 文件不存在: {self.input_path}")
        except pl.exceptions.PolarsError as e:
            logging.error(f"Polars 读取 CSV 文件出错 {self.input_path}: {e}")
        except Exception as e:
            logging.error(f"读取 CSV 文件发生未知错误 {self.input_path}: {e}")
        return None

    def load_parquet(self) -> pl.LazyFrame | None:
        try:
            df = pl.scan_parquet(self.input_path)
            logging.info(f"成功加载 Parquet 文件: {self.input_path}")
            return df
        except FileNotFoundError:
            logging.error(f"Parquet 文件不存在: {self.input_path}")
        except pl.exceptions.PolarsError as e:
            logging.error(f"Polars 读取 Parquet 文件出错 {self.input_path}: {e}")
        except Exception as e:
            logging.error(f"读取 Parquet 文件发生未知错误 {self.input_path}: {e}")
        return None

    def run(self) -> pl.LazyFrame | None:
        suffix = self.input_path.suffix.lower()
        
        if suffix == '.csv':
            return self.load_csv()
        elif suffix == '.parquet':
            return self.load_parquet()
        else:
            logging.error(f"不支持的文件格式: {suffix} (仅支持 .csv 或 .parquet)")
            return None