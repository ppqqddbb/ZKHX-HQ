import logging
import polars as pl
from datetime import datetime
from config.settings import *

class DataFilter:
    def __init__(
            self, 
            df: pl.LazyFrame, 
            time_range: tuple[str, str] | None = None, 
            lat_range: tuple[float, float] | None = None, 
            lon_range: tuple[float, float] | None = None,
        ):
        self.df = df
        if time_range is not None:
            self.start_time, self.end_time = time_range
        else:
            self.start_time, self.end_time = None, None

        if lat_range is not None:
            self.lat_min, self.lat_max = lat_range
        else:
            self.lat_min, self.lat_max = None, None
        
        if lon_range is not None:
            self.lon_min, self.lon_max = lon_range
        else:
            self.lon_min, self.lon_max = None, None

    def apply_filters(self) -> pl.LazyFrame:
        try:
            filters = []

            if self.start_time is not None and self.end_time is not None:
                start_dt = datetime.strptime(self.start_time, OutTimeFormat)
                end_dt = datetime.strptime(self.end_time, OutTimeFormat)
                filters.append(
                    pl.col(TIMESTAMP_COL).is_between(start_dt, end_dt, closed="both")
                )

            if self.lat_min is not None and self.lat_max is not None:
                filters.append(
                    pl.col(LATITUDE_COL).is_between(self.lat_min, self.lat_max)
                )

            if self.lon_min is not None and self.lon_max is not None:
                filters.append(
                    pl.col(LONGITUDE_COL).is_between(self.lon_min, self.lon_max)
                )

            if filters:
                self.df = self.df.filter(filters)

            logging.info("数据筛选完成")

            return self.df
        
        except Exception as e:
            logging.error(f"数据筛选出错: {str(e)}")
            return self.df
        
    def run(self):
        logging.info("开始执行数据过滤...")
        
        schema = self.df.schema
        if self.start_time is not None and schema.get(TIMESTAMP_COL) == pl.String:
            logging.warning(f"检测到 {TIMESTAMP_COL} 为字符串，正在尝试临时转换以执行时间过滤")
            self.df = self.df.with_columns(
                pl.col(TIMESTAMP_COL).str.strptime(pl.Datetime, TimeFormat, strict=False)
            )

        # 2. 调用具体的过滤逻辑
        self.df = self.apply_filters()
        
        return self.df