import h3
import numpy as np
import polars as pl
from config.settings import *

def add_h3_index(
        df: pl.LazyFrame, 
        resolution: int
    ):
    return df.with_columns(
        pl.struct(
            [LATITUDE_COL, LONGITUDE_COL]
        )
        .map_elements(
            lambda x:h3.latlng_to_cell(
                x[LATITUDE_COL],
                x[LONGITUDE_COL],
                resolution
            ),
            return_dtype=pl.Utf8
        )
        .alias("h3")
    )


def h3_center(df: pl.LazyFrame):
    return (
        df.group_by("h3")
        .agg(
            pl.mean(LATITUDE_COL).alias(LATITUDE_COL),
            pl.mean(LONGITUDE_COL).alias(LONGITUDE_COL)
        )
    )
