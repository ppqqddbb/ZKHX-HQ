"""
企业级数据过滤模块
提供时间、空间范围的数据过滤功能
"""
import polars as pl
from datetime import datetime
from typing import Optional, Tuple

from core.config import data_config, model_config
from core.exceptions import DataProcessingError, DataValidationError
from core.logger import get_logger

logger = get_logger(__name__)

# 配置变量映射
TIMESTAMP_COL = data_config.TIMESTAMP_COL
LATITUDE_COL = data_config.LATITUDE_COL
LONGITUDE_COL = data_config.LONGITUDE_COL
TimeFormat = data_config.TIME_FORMAT
OutTimeFormat = data_config.OUT_TIME_FORMAT


class DataFilter:
    """
    AIS数据过滤器

    支持按时间范围、纬度范围、经度范围过滤数据
    """

    def __init__(
        self,
        df: pl.LazyFrame,
        time_range: Optional[Tuple[str, str]] = None,
        lat_range: Optional[Tuple[float, float]] = None,
        lon_range: Optional[Tuple[float, float]] = None
    ):
        """
        初始化数据过滤器

        Args:
            df: 输入数据
            time_range: 时间范围 (start, end)，格式: "YYYY-MM-DD HH:MM:SS"
            lat_range: 纬度范围 (min, max)
            lon_range: 经度范围 (min, max)

        Raises:
            DataValidationError: 输入参数无效时抛出
        """
        if df is None:
            raise DataValidationError("输入数据不能为空")

        self.df = df

        # 时间范围
        if time_range is not None:
            if len(time_range) != 2:
                raise DataValidationError("时间范围必须包含起始和结束时间")
            self.start_time, self.end_time = time_range
        else:
            self.start_time, self.end_time = None, None

        # 纬度范围
        if lat_range is not None:
            if len(lat_range) != 2:
                raise DataValidationError("纬度范围必须包含最小值和最大值")
            if lat_range[0] < -90 or lat_range[1] > 90:
                raise DataValidationError(f"纬度必须在[-90, 90]范围内: {lat_range}")
            self.lat_min, self.lat_max = lat_range
        else:
            self.lat_min, self.lat_max = None, None

        # 经度范围
        if lon_range is not None:
            if len(lon_range) != 2:
                raise DataValidationError("经度范围必须包含最小值和最大值")
            if lon_range[0] < -180 or lon_range[1] > 180:
                raise DataValidationError(f"经度必须在[-180, 180]范围内: {lon_range}")
            self.lon_min, self.lon_max = lon_range
        else:
            self.lon_min, self.lon_max = None, None

        logger.info(
            f"数据过滤器初始化完成 - "
            f"时间过滤: {time_range is not None}, "
            f"纬度过滤: {lat_range is not None}, "
            f"经度过滤: {lon_range is not None}"
        )

    def _parse_time_range(self) -> Tuple[datetime, datetime]:
        """
        解析时间范围字符串

        Returns:
            (开始时间, 结束时间)

        Raises:
            DataProcessingError: 时间解析失败时抛出
        """
        try:
            start_dt = datetime.strptime(self.start_time, OutTimeFormat)
            end_dt = datetime.strptime(self.end_time, OutTimeFormat)

            if start_dt > end_dt:
                raise DataValidationError(
                    f"起始时间不能晚于结束时间: {self.start_time} > {self.end_time}"
                )

            return start_dt, end_dt

        except ValueError as e:
            raise DataProcessingError(
                f"时间格式解析失败: {str(e)}",
                details={
                    "start_time": self.start_time,
                    "end_time": self.end_time,
                    "expected_format": OutTimeFormat
                }
            )

    def apply_filters(self) -> pl.LazyFrame:
        """
        应用所有过滤条件

        Returns:
            过滤后的LazyFrame

        Raises:
            DataProcessingError: 过滤失败时抛出
        """
        try:
            filters = []
            filter_count = 0

            # 时间过滤
            if self.start_time is not None and self.end_time is not None:
                start_dt, end_dt = self._parse_time_range()
                filters.append(
                    pl.col(TIMESTAMP_COL).is_between(start_dt, end_dt, closed="both")
                )
                filter_count += 1
                logger.info(f"时间过滤: {self.start_time} 到 {self.end_time}")

            # 纬度过滤
            if self.lat_min is not None and self.lat_max is not None:
                filters.append(
                    pl.col(LATITUDE_COL).is_between(self.lat_min, self.lat_max)
                )
                filter_count += 1
                logger.info(f"纬度过滤: [{self.lat_min}, {self.lat_max}]")

            # 经度过滤
            if self.lon_min is not None and self.lon_max is not None:
                filters.append(
                    pl.col(LONGITUDE_COL).is_between(self.lon_min, self.lon_max)
                )
                filter_count += 1
                logger.info(f"经度过滤: [{self.lon_min}, {self.lon_max}]")

            # 应用过滤
            if filters:
                # 使用all_horizontal组合多个条件
                combined_filter = filters[0]
                for f in filters[1:]:
                    combined_filter = combined_filter & f

                self.df = self.df.filter(combined_filter)
                logger.info(f"数据过滤完成 - 应用了{filter_count}个过滤条件")
            else:
                logger.warning("未指定任何过滤条件")

            return self.df

        except Exception as e:
            if isinstance(e, (DataProcessingError, DataValidationError)):
                raise
            raise DataProcessingError(
                f"数据过滤失败: {str(e)}",
                details={"error": str(e)}
            )

    def run(self) -> pl.LazyFrame:
        """
        执行数据过滤

        Returns:
            过滤后的LazyFrame

        Raises:
            DataProcessingError: 过滤失败时抛出
        """
        try:
            logger.info("开始执行数据过滤")

            # 检查时间列类型，必要时转换
            if self.start_time is not None:
                schema = self.df.collect_schema()
                if schema.get(TIMESTAMP_COL) == pl.String:
                    logger.warning(f"{TIMESTAMP_COL}列为字符串类型，正在转换为Datetime")
                    self.df = self.df.with_columns(
                        pl.col(TIMESTAMP_COL).str.strptime(
                            pl.Datetime, TimeFormat, strict=False
                        )
                    )

            # 应用过滤
            self.df = self.apply_filters()

            logger.info("数据过滤完成")
            return self.df

        except Exception as e:
            if isinstance(e, (DataProcessingError, DataValidationError)):
                raise
            raise DataProcessingError(
                f"数据过滤执行失败: {str(e)}",
                details={"error": str(e)}
            )
