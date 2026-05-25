import polars as pl

df = pl.read_parquet(r"D:\File\VSCodeFile\SXGF\SXGF\upload\AIS_2024_12_31.parquet")
pl.Config.set_tbl_cols(-1)
print(df.describe())
print(df.head(5))