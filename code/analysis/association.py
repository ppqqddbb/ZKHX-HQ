import logging
import numpy as np
import polars as pl
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from config.settings import *

class AssociationAnalyzer:
    def __init__(self, df, min_support: float, min_confidence: float):
        self.df = df
        self.min_support = min_support
        self.min_confidence = min_confidence

    def build_data(self):
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
            .collect().to_pandas().astype(str).values.tolist()
        )
        return df_grouped

    def mine_frequent_patterns(self, df_grouped):
        te = TransactionEncoder()
        te_ary = te.fit(df_grouped).transform(df_grouped)
        df_trans = pd.DataFrame(te_ary, columns=te.columns_)
        
        frequent_itemsets = fpgrowth(
            df_trans,
            min_support=self.min_support,
            use_colnames=True
        )

        frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False)
        if len(frequent_itemsets) > 0:
            association_data = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=self.min_confidence
            ).sort_values(by="lift", ascending=False)
        
        return frequent_itemsets, association_data
    
    def run(self):
        df_grouped = self.build_data()
        frequent_itemsets, association_data = self.mine_frequent_patterns(df_grouped)
        return frequent_itemsets, association_data