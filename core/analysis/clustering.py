"""
企业级聚类分析模块
基于DTW距离和层次聚类算法进行船舶轨迹聚类
"""
import numpy as np
import polars as pl
from typing import Tuple
from tslearn.metrics import cdist_dtw
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import linear_sum_assignment

from core.config import data_config
from core.exceptions import AnalysisError, DataValidationError
from core.logger import get_logger

logger = get_logger(__name__)

# 配置变量映射
MMSI_COL = data_config.MMSI_COL
TIMESTAMP_COL = data_config.TIMESTAMP_COL
LATITUDE_COL = data_config.LATITUDE_COL
LONGITUDE_COL = data_config.LONGITUDE_COL
GROUP_COL = data_config.GROUP_COL
PREDICT_GROUP_COL = data_config.PREDICT_GROUP_COL
OutTimeFormat = data_config.OUT_TIME_FORMAT


class ClusteringAnalyzer:
    """
    聚类分析器

    使用动态时间规整(DTW)距离度量和层次聚类算法对船舶轨迹进行聚类
    """

    def __init__(self, df: pl.LazyFrame):
        """
        初始化聚类分析器

        Args:
            df: 输入AIS数据（必须包含group_id列）

        Raises:
            DataValidationError: 输入参数无效时抛出
        """
        if df is None:
            raise DataValidationError("输入数据不能为空")

        self.df = df
        logger.info("聚类分析器初始化完成")

    def _build_trajectories(self) -> pl.DataFrame:
        """
        构建轨迹数据结构

        Returns:
            轨迹数据

        Raises:
            AnalysisError: 轨迹构建失败时抛出
        """
        try:
            schema = self.df.collect_schema()

            # 如果没有group_id列，使用MMSI作为分组
            if GROUP_COL not in schema.names():
                logger.info(f"未找到{GROUP_COL}列，使用{MMSI_COL}作为分组标识")
                self.df = self.df.with_columns(
                    pl.col(MMSI_COL).alias(GROUP_COL)
                )
                schema = self.df.collect_schema()

            select_df = (
                self.df
                .select(
                    pl.col(TIMESTAMP_COL),
                    pl.col(MMSI_COL),
                    pl.col(LATITUDE_COL),
                    pl.col(LONGITUDE_COL),
                    pl.col(GROUP_COL)
                )
            )

            if schema[TIMESTAMP_COL] == pl.String:
                select_df = select_df.with_columns(
                    pl.col(TIMESTAMP_COL)
                    .str.strptime(pl.Datetime, format=OutTimeFormat)
                )

            select_df = select_df.collect()

            if len(select_df) == 0:
                raise AnalysisError("数据为空，无法构建轨迹")

            # 过滤掉时间戳为null的记录
            select_df = select_df.filter(pl.col(TIMESTAMP_COL).is_not_null())

            if len(select_df) == 0:
                raise AnalysisError("过滤null时间戳后数据为空")

            global_start = select_df[TIMESTAMP_COL].min()

            trajectories = (
                select_df
                .sort([MMSI_COL, TIMESTAMP_COL])
                .with_columns(
                    (
                        pl.col(TIMESTAMP_COL) - pl.lit(global_start)
                    )
                    .dt.total_seconds()
                    .alias("time_sec")
                )
                .group_by(MMSI_COL)
                .agg(
                    pl.struct([LATITUDE_COL, LONGITUDE_COL, "time_sec"])
                    .alias("trajectory")
                )
            )

            logger.info(f"轨迹构建完成 - 轨迹数: {len(trajectories)}")
            return trajectories

        except Exception as e:
            raise AnalysisError(
                f"轨迹构建失败: {str(e)}",
                details={"error": str(e)}
            )

    def _compute_dtw_similarity(self, traj_df: pl.DataFrame) -> np.ndarray:
        """
        计算DTW相似度矩阵

        Args:
            traj_df: 轨迹数据

        Returns:
            相似度矩阵

        Raises:
            AnalysisError: DTW计算失败时抛出
        """
        try:
            all_trajs = []
            for row in traj_df.iter_rows(named=True):
                traj = row["trajectory"]
                arr = np.array([
                    [p[LATITUDE_COL], p[LONGITUDE_COL], p["time_sec"]]
                    for p in traj
                ])
                all_trajs.append(arr)

            if len(all_trajs) < 2:
                raise AnalysisError(f"轨迹数量不足，需要至少2条轨迹，当前: {len(all_trajs)}")

            logger.info(f"开始计算DTW距离矩阵 - 轨迹数: {len(all_trajs)}")

            dist_matrix = cdist_dtw(all_trajs)

            max_d = dist_matrix.max() if dist_matrix.max() > 0 else 1
            similarity_matrix = np.exp(-dist_matrix / max_d)
            np.fill_diagonal(similarity_matrix, 1.0)

            logger.info("DTW相似度矩阵计算完成")
            return similarity_matrix

        except Exception as e:
            raise AnalysisError(
                f"DTW相似度计算失败: {str(e)}",
                details={"error": str(e)}
            )

    def _cluster_trajectories(
        self,
        traj_df: pl.DataFrame,
        similarity_matrix: np.ndarray
    ) -> pl.DataFrame:
        """
        执行层次聚类

        Args:
            traj_df: 轨迹数据
            similarity_matrix: 相似度矩阵

        Returns:
            带聚类标签的数据

        Raises:
            AnalysisError: 聚类失败时抛出
        """
        try:
            distance_matrix = 1 - similarity_matrix

            best_score = -1
            best_labels = None
            best_thresh = None

            logger.info("开始寻找最优聚类阈值")

            for thresh in np.arange(0.0, 1.0, 0.01):
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=thresh,
                    metric="precomputed",
                    linkage="average",
                )
                labels = clustering.fit_predict(distance_matrix)
                n_clusters = len(set(labels))

                if n_clusters < 2:
                    continue

                try:
                    score = silhouette_score(distance_matrix, labels, metric="precomputed")
                    if score > best_score:
                        best_score = score
                        best_labels = labels
                        best_thresh = thresh
                except:
                    continue

            if best_labels is None:
                logger.warning("未找到最优聚类，使用默认阈值0.5")
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=0.5,
                    metric="precomputed",
                    linkage="average",
                )
                best_labels = clustering.fit_predict(distance_matrix)
                best_score = -1

            n_clusters = len(set(best_labels))
            logger.info(
                f"聚类完成 - "
                f"聚类数: {n_clusters}, "
                f"轮廓系数: {best_score:.4f}, "
                f"阈值: {best_thresh}"
            )

            label_df = pl.DataFrame({
                MMSI_COL: traj_df[MMSI_COL],
                PREDICT_GROUP_COL: best_labels
            })

            result_df = (
                self.df.collect()
                .join(label_df, on=MMSI_COL, how="left")
            )

            return result_df

        except Exception as e:
            raise AnalysisError(
                f"轨迹聚类失败: {str(e)}",
                details={"error": str(e)}
            )

    def _compute_accuracy(
        self,
        df: pl.DataFrame,
        true_col: str = GROUP_COL,
        pred_col: str = PREDICT_GROUP_COL
    ) -> float:
        """
        计算聚类准确率（使用匈牙利算法对齐标签）

        Args:
            df: 带真实和预测标签的数据
            true_col: 真实标签列名
            pred_col: 预测标签列名

        Returns:
            准确率

        Raises:
            AnalysisError: 准确率计算失败时抛出
        """
        try:
            confusion_df = (
                df
                .group_by([true_col, pred_col])
                .len()
                .rename({"len": "count"})
            )

            true_classes = (
                confusion_df
                .select(true_col)
                .unique()
                .sort(true_col)
                .to_series()
                .to_numpy()
            )

            pred_classes = (
                confusion_df
                .select(pred_col)
                .unique()
                .sort(pred_col)
                .to_series()
                .to_numpy()
            )

            true_index = {v: i for i, v in enumerate(true_classes)}
            pred_index = {v: i for i, v in enumerate(pred_classes)}

            matrix = np.zeros((len(true_classes), len(pred_classes)), dtype=np.int64)

            for row in confusion_df.iter_rows(named=True):
                matrix[
                    true_index[row[true_col]],
                    pred_index[row[pred_col]]
                ] = row["count"]

            row_ind, col_ind = linear_sum_assignment(matrix.max() - matrix)
            acc = matrix[row_ind, col_ind].sum() / df.height

            logger.info(f"聚类准确率: {acc:.4f}")
            return acc

        except Exception as e:
            raise AnalysisError(
                f"准确率计算失败: {str(e)}",
                details={"error": str(e)}
            )

    def run(self) -> Tuple[pl.DataFrame, float]:
        """
        执行聚类分析

        Returns:
            (聚类结果, 准确率)

        Raises:
            AnalysisError: 分析失败时抛出
        """
        try:
            logger.info("开始聚类分析")

            traj_df = self._build_trajectories()
            similarity_matrix = self._compute_dtw_similarity(traj_df)
            result_df = self._cluster_trajectories(traj_df, similarity_matrix)
            accuracy = self._compute_accuracy(result_df)

            logger.info("聚类分析完成")
            return result_df, accuracy

        except Exception as e:
            if isinstance(e, AnalysisError):
                raise
            raise AnalysisError(
                f"聚类分析失败: {str(e)}",
                details={"error": str(e)}
            )
