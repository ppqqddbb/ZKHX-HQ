import torch
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent

'''
数据特征
'''
TimeFormat = "%Y-%m-%dT%H:%M:%S"
usTimeFormat = "%Y-%m-%dT%H:%M:%S.%f"
OutTimeFormat = "%Y-%m-%d %H:%M:%S"
MMSI_COL = "MMSI"                               # 用户识别码(int64)
TIMESTAMP_COL = "Timestamp"                     # 时间(datetime[ns])
LATITUDE_COL = "Latitude"                       # 纬度(float64)
LONGITUDE_COL = "Longitude"                     # 经度(float64)
SOG_COL = "SOG"                                 # 航速(float64)
COG_COL = "COG"                                 # 航向(float64)
ROT_COL = "ROT"
HEADING_COL = "Heading"
SHIPTYPE_COL = "ShipType"                       # 船型(int64)
Status_COL = "Status"                           # 状态(int64)
Length_COL = "Length"                           # 船长(int64)
Width_COL = "Width"                             # 船宽(int64)
Draft_COL = "Draft"                             # 深度(float64)
Cargo_COL = "Cargo"                             # 货物类型
GROUP_COL = "group_id"
PREDICT_GROUP_COL = "predict_cluster"

'''
特征重命名
'''
RENAME_NAME = {
    "BaseDateTime": "Timestamp",
    "MMSI": "MMSI",
    "LAT": "Latitude",
    "LON": "Longitude",
    "SOG": "SOG",
    "COG": "COG",
    "Heading": "Heading",
    "VesselType": "ShipType",
    "Status": "Status",
    "Length": "Length",
    "Width": "Width",
    "Draft": "Draft",
    "Cargo": "Cargo"
}
# RENAME_NAME = {
#     "Timestamp": "Timestamp",
#     "MMSI": "MMSI",
#     "Latitude": "Latitude",
#     "Longitude": "Longitude",
#     "SOG": "SOG",
#     "COG": "COG",
#     "Heading": "Heading",
#     "ShipType": "ShipType",
# }

NUM_COLUMNS = [LONGITUDE_COL, LATITUDE_COL, SOG_COL, COG_COL, HEADING_COL]
COLUMNS = [TIMESTAMP_COL, MMSI_COL, LONGITUDE_COL, LATITUDE_COL, SOG_COL, COG_COL, HEADING_COL, SHIPTYPE_COL]
ENV_COLUMNS = ["wind_speed","wind_direction","wave_height","wave_period","current_speed","current_direction","water_depth","sea_temperature","salinity"]
'''
航迹预测
'''
select_columns = [
    TIMESTAMP_COL,
    MMSI_COL,
    LONGITUDE_COL,
    LATITUDE_COL,
    "time_sin",
    "time_cos",
    # "x_meters",
    # "y_meters",
    "cog_sin",
    "cog_cos",
    "heading_sin",
    "heading_cos",
    "v_x",
    "v_y",
    "delta_x",
    "delta_y",
    "delta_sog",
    "delta_cog",
    "rel_x",
    "rel_y",
    "drift_angle",
    "curvature"
]

train_columns = [
    "time_sin",
    "time_cos",
    # "x_meters",
    # "y_meters",
    "cog_sin",
    "cog_cos",
    "heading_sin",
    "heading_cos",
    "v_x",
    "v_y",
    "delta_x",
    "delta_y",
    "delta_sog",
    "delta_cog",
    "rel_x",
    "rel_y",
    "drift_angle",
    "curvature"
]

scaler_columns = [
    "time_sin",
    "time_cos",
    # "x_meters",
    # "y_meters",
    "cog_sin",
    "cog_cos",
    "heading_sin",
    "heading_cos",
    "v_x",
    "v_y",
    "delta_x",
    "delta_y",
    "delta_sog",
    "delta_cog",
    "rel_x",
    "rel_y",
    "drift_angle",
    "curvature"
]

freq = '5m'
is_scaler = False
seq_len = 96
pred_len = 60

R = 6378137.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d_model = 256
nhead = 8
num_encoder_layers = 6
dim_feedforward = 1024
dropout = 0.2
activation = 'gelu'
batch_first = True
norm_first = True

lr = 0.0005
weight_decay = 1e-4
epochs = 100
batch_size = 16

file_paths = [base_dir / f"dataset/AIS_2024_12_{i:02d}.csv" for i in range(1, 6)]

scaler_df = base_dir / "dataset/scaler.parquet"
scaler_path = base_dir / "dataset/MeanStd.joblib"
best_model_path = base_dir / "dataset/best_patchtst_model.pt"

'''
静态参数
'''
StaticFiles_path = base_dir / "static/swagger-ui-5.30.2/dist"

'''
日志路径
'''
log_dir = base_dir / "log"

'''
输入文件
'''
upload_dir = base_dir / "upload"

'''
输出文件
'''
output_dir = base_dir / "output"

'''
数据库路径
'''
db_path = base_dir / "dataset/sxgf.db"