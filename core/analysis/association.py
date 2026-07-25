"""
企业级关联分析模块
基于FP-Growth算法挖掘船舶行为模式的关联规则
"""
import polars as pl
import pandas as pd
from typing import Tuple
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

from core.config import data_config
from core.exceptions import AnalysisError, DataValidationError
from core.logger import get_logger

logger = get_logger(__name__)

# 配置变量映射
MMSI_COL = data_config.MMSI_COL
TIMESTAMP_COL = data_config.TIMESTAMP_COL
LATITUDE_COL = data_config.LATITUDE_COL
LONGITUDE_COL = data_config.LONGITUDE_COL
SOG_COL = data_config.SOG_COL
COG_COL = data_config.COG_COL
SHIPTYPE_COL = data_config.SHIPTYPE_COL


class AssociationAnalyzer:
    """
    关联分析器

    使用FP-Growth算法挖掘船舶行为特征的频繁模式和关联规则
    """

    def __init__(
        self,
        df: pl.LazyFrame,
        min_support: float = 0.01,
        min_confidence: float = 0.5
    ):
        """
        初始化关联分析器

        Args:
            df: 输入AIS数据
            min_support: 最小支持度阈值
            min_confidence: 最小置信度阈值

        Raises:
            DataValidationError: 输入参数无效时抛出
        """
        if df is None:
            raise DataValidationError("输入数据不能为空")

        if not (0 < min_support <= 1):
            raise DataValidationError(f"min_support必须在(0, 1]范围内，当前值: {min_support}")

        if not (0 < min_confidence <= 1):
            raise DataValidationError(f"min_confidence必须在(0, 1]范围内，当前值: {min_confidence}")

        self.df = df
        self.min_support = min_support
        self.min_confidence = min_confidence

        logger.info(
            f"关联分析器初始化完成 - "
            f"min_support={min_support}, "
            f"min_confidence={min_confidence}"
        )

    def _build_transactions(self) -> list:
        """
        构建交易数据集

        Returns:
            交易列表

        Raises:
            AnalysisError: 数据构建失败时抛出
        """
        try:
            df_grouped = (
                self.df
                .group_by(MMSI_COL)
                .agg(
                    pl.col(TIMESTAMP_COL).first().alias("开始时间"),
                    pl.col(TIMESTAMP_COL).last().alias("结束时间"),
                    pl.col(LATITUDE_COL).first().alias("开始纬度"),
                    pl.col(LATITUDE_COL).last().alias("结束纬度"),
                    pl.col(LONGITUDE_COL).first().alias("开始经度"),
                    pl.col(LONGITUDE_COL).last().alias("结束经度"),
                    pl.col(SOG_COL).max().alias("最大速度"),
                    pl.col(SOG_COL).min().alias("最小速度"),
                    pl.col(SOG_COL).mean().alias("平均速度"),
                    pl.col(SOG_COL).std().alias("标准速度"),
                    pl.col(COG_COL).max().alias("最大航向"),
                    pl.col(COG_COL).min().alias("最小航向"),
                    pl.col(COG_COL).mean().alias("平均航向"),
                    pl.col(COG_COL).std().alias("标准航向"),
                    pl.col(SHIPTYPE_COL).first().alias("类型"),
                )
                .collect()
                .to_pandas()
                .astype(str)
                .values
                .tolist()
            )

            if not df_grouped:
                raise AnalysisError("分组数据为空，无法构建交易数据集")

            logger.info(f"交易数据集构建完成 - 交易数: {len(df_grouped)}")
            return df_grouped

        except Exception as e:
            raise AnalysisError(
                f"交易数据集构建失败: {str(e)}",
                details={"error": str(e)}
            )

    def _mine_frequent_patterns(self, transactions: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        挖掘频繁模式和关联规则

        Args:
            transactions: 交易数据列表

        Returns:
            (频繁项集, 关联规则)

        Raises:
            AnalysisError: 挖掘失败时抛出
        """
        try:
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df_trans = pd.DataFrame(te_ary, columns=te.columns_)

            frequent_itemsets = fpgrowth(
                df_trans,
                min_support=self.min_support,
                use_colnames=True
            )

            if len(frequent_itemsets) == 0:
                logger.warning("未发现频繁项集")
                return pd.DataFrame(), pd.DataFrame()

            frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False)

            association_data = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=self.min_confidence
            ).sort_values(by="lift", ascending=False)

            logger.info(
                f"频繁模式挖掘完成 - "
                f"频繁项集数: {len(frequent_itemsets)}, "
                f"关联规则数: {len(association_data)}"
            )

            return frequent_itemsets, association_data

        except Exception as e:
            raise AnalysisError(
                f"频繁模式挖掘失败: {str(e)}",
                details={"error": str(e)}
            )

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        执行关联分析

        Returns:
            (频繁项集, 关联规则)

        Raises:
            AnalysisError: 分析失败时抛出
        """
        try:
            logger.info("开始关联分析")

            transactions = self._build_transactions()
            frequent_itemsets, association_data = self._mine_frequent_patterns(transactions)

            logger.info("关联分析完成")
            return frequent_itemsets, association_data

        except Exception as e:
            if isinstance(e, AnalysisError):
                raise
            raise AnalysisError(
                f"关联分析失败: {str(e)}",
                details={"error": str(e)}
            )
