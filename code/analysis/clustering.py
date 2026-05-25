import logging
import numpy as np
import polars as pl
from tslearn.metrics import dtw
from tslearn.metrics import cdist_dtw
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import linear_sum_assignment
from config.settings import *


class ClusteringAnalyzer:
    def __init__(self, df):
        self.df = df

    def build_data(self):
        schema = self.df.schema
        select_df = (
            self.df
            .select(
                pl.col(TIMESTAMP_COL),
                pl.col(MMSI_COL),
                pl.col(LATITUDE_COL),
                pl.col(LONGITUDE_COL),
                pl.col(GROUP_COL)
            )
            # .with_columns(
            #     pl.col(TIMESTAMP_COL).str.strptime(pl.Datetime, format=OutTimeFormat).alias(TIMESTAMP_COL)
            # )
            # .collect()
        )
        if schema[TIMESTAMP_COL] == pl.String:
            select_df = select_df.with_columns(
                pl.col(TIMESTAMP_COL)
                .str.strptime(pl.Datetime, format=OutTimeFormat)
            )
        select_df = select_df.collect()
        global_start = select_df[TIMESTAMP_COL].min()

        trajectories = (
            select_df
            .sort([MMSI_COL, TIMESTAMP_COL])
            .with_columns(
                (
                    pl.col(TIMESTAMP_COL)-pl.lit(global_start)
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

        return trajectories

    def dtw_distance(self, df: pl.LazyFrame):
        all_trajs = []
        for row in df.iter_rows(named=True):
            traj = row["trajectory"]
            arr = np.array([[p[LATITUDE_COL], p[LONGITUDE_COL], p["time_sec"]] for p in traj])
            all_trajs.append(arr)
        dist_matrix = cdist_dtw(all_trajs)

        max_d = dist_matrix.max() if dist_matrix.max() > 0 else 1
        similarity_matrix = np.exp(-dist_matrix / max_d)
        np.fill_diagonal(similarity_matrix, 1.0)
        return similarity_matrix

    def cluster_ships(self, traj_df: pl.DataFrame, similarity_matrix):
        distance_matrix = 1 - similarity_matrix
        best_score = -1
        best_labels = None
        best_thresh = None
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
            except:
                continue

        label_df = pl.DataFrame({
            MMSI_COL:traj_df[MMSI_COL],
            PREDICT_GROUP_COL: best_labels
        })

        result_df = (
            self.df.collect()
            .join(label_df, on=MMSI_COL, how="left")
        )
        
        return result_df

    def clustering_accuracy_polars(
            self,
            df: pl.DataFrame,
            true_col=GROUP_COL,
            pred_col=PREDICT_GROUP_COL
        ):

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

        return acc

    def run(self):
        traj_df = self.build_data()
        similarity_matrix = self.dtw_distance(traj_df)
        result_df = self.cluster_ships(traj_df, similarity_matrix)
        accuracy = self.clustering_accuracy_polars(result_df)
        print(f"聚类准确率：{accuracy:.5f}")

        return result_df, accuracy