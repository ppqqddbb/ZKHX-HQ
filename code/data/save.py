import os
import json
import logging
import polars as pl
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from config.settings import *


class DataSaver:
    def __init__(self, df: pl.LazyFrame, output_dir: str, filename: str):
        self.df = df
        self.output_dir = os.path.abspath(output_dir)
        self.filename = filename

    def _collect(self):
        if isinstance(self.df, pl.LazyFrame):
            try:
                return self.df.collect()
            except Exception as e:
                logging.error(f"LazyFrame转换DataFrame出错: {str(e)}")
                return None
        return self.df

    def save_json(self):
        try:
            df = self._collect()
            if df is None:
                return None

            path = Path(self.output_dir) / f"{self.filename}.json"

            if isinstance(df, pl.DataFrame):
                df_dict = df.to_dicts()
            elif isinstance(df, pd.DataFrame):
                df_dict = df.to_dict(orient="records")
            else:
                df_dict = df

            with open(path, "w", encoding='utf-8') as f:
                json.dump(df_dict, f, ensure_ascii=False, indent=4)

            logging.info(f"JSON保存成功: {path}")
            return path
        except Exception as e:
            logging.error(f"保存JSON文件出错: {str(e)}")
            return None

    def save_csv(self):
        try:
            path = Path(self.output_dir) / f"{self.filename}.csv"

            if isinstance(self.df, pl.DataFrame):
                self.df.write_csv(path)
            elif isinstance(self.df, pl.LazyFrame):
                self.df.sink_csv(path)
            elif isinstance(self.df, pd.DataFrame):
                self.df.to_csv(path, index=False)

            logging.info(f"CSV保存成功: {path}")
            return path
        except Exception as e:
            logging.error(f"保存CSV文件出错: {str(e)}")
            return None

    def save_parquet(self):
        try:
            path = Path(self.output_dir) / f"{self.filename}.parquet"

            if isinstance(self.df, pl.DataFrame):
                self.df.write_parquet(path)
            elif isinstance(self.df, pl.LazyFrame):
                self.df.sink_parquet(path)
            elif isinstance(self.df, pd.DataFrame):
                self.df.to_parquet(path, index=False)

            logging.info(f"Parquet保存成功: {path}")
            return path
        except Exception as e:
            logging.error(f"保存Parquet文件出错: {str(e)}")
            return None
    
    def save_geojson(self):
        try:
            df = self._collect()
            if df is None:
                return None

            path = Path(self.output_dir) / f"{self.filename}.geojson"
            path.parent.mkdir(parents=True, exist_ok=True)

            df.to_file(path, driver="GeoJSON")
            logging.info(f"GeoJSON保存成功: {path}")
            return path
        except Exception as e:
            logging.error(f"保存GeoJSON文件出错: {str(e)}")
            return None
    
    def save_split_data(self):
        try:
            df_collected = self._collect()
            if df_collected is None:
                return

            mmsi_df = df_collected.select(pl.col(MMSI_COL)).unique()
            mmsi_list = mmsi_df[MMSI_COL].to_list()

            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for mmsi in mmsi_list:
                df_mmsi = df_collected.filter(pl.col(MMSI_COL) == mmsi)
                path = output_dir / f"{mmsi}.parquet"

                if isinstance(df_mmsi, pl.LazyFrame):
                    df_mmsi.sink_parquet(path)
                elif isinstance(df_mmsi, pl.DataFrame):
                    df_mmsi.write_parquet(path)
            logging.info(f"按MMSI拆分保存数据成功, 保存到: {output_dir}")
        except Exception as e:
            logging.error(f"按MMSI拆分保存数据出错: {str(e)}")
                
    def run(self, formats=None, split_by_mmsi=False):
        if formats is None:
            formats = ['parquet']  # 默认保存为 parquet，因为性能最好

        try:
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
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
                self.save_split_data()
                saved_paths['split_data'] = str(output_path)

            logging.info(f"所有数据保存任务完成: {saved_paths.keys()}")
            return saved_paths

        except Exception as e:
            logging.error(f"执行 DataSaver.run 时出错: {str(e)}")
            return None