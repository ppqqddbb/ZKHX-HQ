import os
import time
import shutil 
import joblib
import uvicorn
import asyncio
import aiofiles
import logging
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, List, Tuple
from fastapi import FastAPI, HTTPException, File, UploadFile 
from fastapi.responses import FileResponse 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles

from code.data import (
    DataLoader,
    DataProcessor,
    DataFilter,
    DataSaver
)

from code.analysis import (
    HotspotAnalyzer, 
    TargetAnalyzer, 
    AssociationAnalyzer, 
    ClusteringAnalyzer, 
    FrequentAnalyzer, 
)

from code.model import (
    PatchTSTManager,
    GlobalDataScaler,
)

from config.settings import *

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/swagger",  # 前端访问路径
    StaticFiles(directory=StaticFiles_path),  # 本地 Swagger UI 静态文件目录
    name="swagger-ui"
)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",  # 仍指向这个接口
        title=app.title + " - Swagger UI",
        swagger_js_url="/swagger/swagger-ui-bundle.js",
        swagger_css_url="/swagger/swagger-ui.css",
        swagger_favicon_url="/swagger/favicon-32x32.png",
    )

@app.get("/openapi.json", include_in_schema=False)
async def get_custom_openapi_json():
    return get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
        servers=app.servers,
    )

# 创建输入目录，输出目录
upload_dir = Path("./upload") 
output_dir = Path("./output")
log_dir = Path("./log")
dataset_dir = Path("./dataset")

upload_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)
dataset_dir.mkdir(parents=True, exist_ok=True)

# --- 所有请求模型保持不变 ---
class DataProcessRequest(BaseModel):
    input_path: str
    time_range: Optional[Tuple[str, str]] = None
    lat_range: Optional[Tuple[float, float]] = None
    lon_range: Optional[Tuple[float, float]] = None

class HotspotRequest(BaseModel):
    input_path: str
    time_range: Optional[Tuple[str, str]] = None
    lat_range: Optional[Tuple[float, float]] = None
    lon_range: Optional[Tuple[float, float]] = None
    min_cluster_size: int = 5
    min_samples: int = 2
    resolution: int = 6

class TargetRequest(BaseModel):
    input_path: str
    time_range: Optional[Tuple[str, str]] = None
    lat_range: Optional[Tuple[float, float]] = None
    lon_range: Optional[Tuple[float, float]] = None
    eps_km: float = 0.5
    min_samples: int = 3
    zones: str = "upload/zones.json"

class AssociationRequest(BaseModel):
    input_path: str
    time_range: Optional[Tuple[str, str]] = None
    lat_range: Optional[Tuple[float, float]] = None
    lon_range: Optional[Tuple[float, float]] = None
    min_support: float = 0.05
    min_confidence: float = 0.6

class ClusteringRequest(BaseModel):
    input_path: str
    time_range: Optional[Tuple[str, str]] = None
    lat_range: Optional[Tuple[float, float]] = None
    lon_range: Optional[Tuple[float, float]] = None

class FrequentRequest(BaseModel):
    input_path: str
    time_range: Optional[Tuple[str, str]] = None
    lat_range: Optional[Tuple[float, float]] = None
    lon_range: Optional[Tuple[float, float]] = None
    grid_size: float = 0.01
    sigma: float = 1.5
    percentile_thresh: float = 70
    visulize: bool = False

class TrainRequest(BaseModel):
    scaler_path: str = r"D:\InstallFiles\File\Project\dataset\MeanStd.joblib"
    pretrain_model_path: str = None
    best_model_path: str = r"D:\InstallFiles\File\Project\dataset\best_patchtst_model.pt"

class PredictRequest(BaseModel):
    input_path: str = r"D:\InstallFiles\File\Project\upload\AIS_2024_12_31.parquet"
    time_range: Optional[Tuple[str, str]] = None
    lat_range: Optional[Tuple[float, float]] = None
    lon_range: Optional[Tuple[float, float]] = None
    best_model_path: str = r"D:\InstallFiles\File\Project\dataset\best_patchtst_model.pt"
    total_steps: int = 8

# --- 异步包装函数（用于执行阻塞操作）---
async def run_in_thread(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args, **kwargs)

@app.post("/Upload", summary="上传文件")
async def upload_file(files: list[UploadFile] = File(..., description="上传多个文件")):
    try:
        for file in files:
            file_name = Path(file.filename).name
            input_path = upload_dir / file_name
            CHUNK_SIZE = 4 * 1024 * 1024
            total_bytes = 0
            async with aiofiles.open(input_path, "wb") as f:
                while content := await file.read(CHUNK_SIZE):
                    await f.write(content)
                    total_bytes += len(content)
        return {
            "status": "success",
            "file_name": file_name,
            "total_bytes": total_bytes,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")
    
@app.get("/Files/download", summary="下载文档")
async def download_file(filename: str):
    # 构建完整路径并防止目录遍历攻击
    file_path = Path(output_dir) / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return FileResponse(
        path=file_path, 
        filename=filename, 
        media_type='application/octet-stream'
    )

@app.get("/ListUploadFolder", summary="显示上传文档")
async def ListOutputFiles():
    try:
        if not upload_dir.exists():
            raise HTTPException(status_code=404, detail="目录不存在")
        files_info = []
        for root, dirs, files in os.walk(upload_dir):
            for file in files:
                file_path = Path(root) / file
                info = {
                    "file": file_path.name,
                    "size_kb": round(file_path.stat().st_size / 1024, 2),
                    "modified": time.strftime(
                        "%Y-%m-%d %H:%M:%S", 
                        time.localtime(file_path.stat().st_mtime)
                    ),
                    "path": str(file_path)
                }
                files_info.append(info)
        return {
            "status": "success",
            "path": str(upload_dir),
            "total_files": len(files_info),
            "files": files_info
        }
    except Exception as e:
        logging.error(f"读取上传目录失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"无法读取目录: {str(e)}")

@app.get("/ListOutputFolder", summary="显示输出文档")
async def ListOutputFiles():
    try:
        if not upload_dir.exists():
            raise HTTPException(status_code=404, detail="目录不存在")
        files_info = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = Path(root) / file
                info = {
                    "file": file_path.name,
                    "size_kb": round(file_path.stat().st_size / 1024, 2),
                    "modified": time.strftime(
                        "%Y-%m-%d %H:%M:%S", 
                        time.localtime(file_path.stat().st_mtime)
                    ),
                    "path": str(file_path)
                }
                files_info.append(info)
        return {
            "status": "success",
            "path": str(upload_dir),
            "total_files": len(files_info),
            "files": files_info
        }
    except Exception as e:
        logging.error(f"读取上传目录失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"无法读取目录: {str(e)}")
    
# --- 异步路由 ---
@app.post("/Data", summary="数据处理")
async def data(req: DataProcessRequest):
    try:
        # 在线程中执行整个数据处理流程
        path = await run_in_thread(_data_process_sync, req)
        return {
            "status": "success",
            "function": "数据处理分析完成",
            "path": path
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "function": "数据处理分析失败",
        }

def _data_process_sync(req: DataProcessRequest):
    input_path = req.input_path
    time_range = req.time_range
    lat_range = req.lat_range
    lon_range = req.lon_range

    original_df = DataLoader(input_path).run()

    process_df = DataProcessor(original_df).run()

    filter_df = DataFilter(
        df=process_df,
        time_range=time_range,
        lat_range=lat_range,
        lon_range=lon_range,
    ).run()

    return DataSaver(
        df=filter_df,
        output_dir=upload_dir,
        filename="process"
    ).save_parquet()

@app.post("/HotSpot", summary="热点区域分析")
async def hotspot(req: HotspotRequest):
    try:
        path = await run_in_thread(_hotspot_sync, req)
        return {
            "status": "success",
            "function": "热点区域分析完成",
            "path": path
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "function": "热点区域分析失败",
        }

def _hotspot_sync(req: HotspotRequest):
    df = DataLoader(req.input_path).run()
    filter_df = DataFilter(
        df=df,
        time_range=req.time_range,
        lat_range=req.lat_range,
        lon_range=req.lon_range,
    ).run()

    analyzer_df, coverage_df = HotspotAnalyzer(
        df=filter_df,
        min_cluster_size=req.min_cluster_size,
        min_samples=req.min_samples,
        resolution=req.resolution
    ).run()

    return DataSaver(
        df=analyzer_df,
        output_dir=output_dir,
        filename="hotspot"
    ).save_csv(), DataSaver(
        df=coverage_df,
        output_dir=output_dir,
        filename="hotspot_coverage"
    ).save_json()

@app.post("/TargetAnalysis", summary="目标分析")
async def target(req: TargetRequest):
    try:
        activate_areas_path, sog_change_path, dwell_positions_path, total_distance_path, zones_path = await run_in_thread(_target_sync, req)
        return {
            "status": "success",
            "function": "目标分析完成",
            "活动区域": activate_areas_path,
            "航速变化": sog_change_path,
            "驻留位置": dwell_positions_path,
            "总体航程": total_distance_path,
            "自定义区域覆盖度": zones_path
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "function": "目标分析失败",
        }

def _target_sync(req: TargetRequest):
    df = DataLoader(req.input_path).run()
    filter_df = DataFilter(
        df=df,
        time_range=req.time_range,
        lat_range=req.lat_range,
        lon_range=req.lon_range,
    ).run()

    activate_areas, dwell_positions, speed, total_distance, zones = TargetAnalyzer(
        df=filter_df,
        eps_km=req.eps_km,
        min_samples=req.min_samples,
        zones=req.zones
    ).run()

    activate_areas_path = DataSaver(df=activate_areas, output_dir=output_dir, filename="activate_areas").save_json()
    sog_change_path = DataSaver(df=speed, output_dir=output_dir, filename="sog_change").save_json()
    dwell_positions_path = DataSaver(df=dwell_positions, output_dir=output_dir, filename="dwell_positions").save_csv()
    total_distance_path = DataSaver(df=total_distance, output_dir=output_dir, filename="total_distance").save_json()
    zones_path = DataSaver(df=zones, output_dir=output_dir, filename="overall_zone_coverage").save_json()

    return activate_areas_path, sog_change_path, dwell_positions_path, total_distance_path, zones_path

@app.post("/TargetFPGrowthAnalysis", summary="船舶关联规则分析")
async def association(req: AssociationRequest):
    try:
        frequent_itemsets_path, association_data_path = await run_in_thread(_association_sync, req)
        return {
            "status": "success",
            "function": "规则关联分析完成",
            "频繁项集": frequent_itemsets_path,
            "关联规则": association_data_path
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "function": "规则关联分析失败",
        }

def _association_sync(req: AssociationRequest):
    df = DataLoader(req.input_path).run()

    filter_df = DataFilter(
        df=df,
        time_range=req.time_range,
        lat_range=req.lat_range,
        lon_range=req.lon_range,
    ).run()

    frequent_itemsets, association_data = AssociationAnalyzer(
        df=filter_df,
        min_support=req.min_support,
        min_confidence=req.min_confidence,
    ).run()

    frequent_itemsets_path = DataSaver(
        df=frequent_itemsets,
        output_dir=output_dir,
        filename="frequent_itemsets" 
    ).save_csv()

    association_data_path = DataSaver(
        df=association_data,
        output_dir=output_dir,
        filename="association_rules" 
    ).save_csv()

    return frequent_itemsets_path, association_data_path

@app.post("/TrajectorySimilarity", summary="船舶编群分析")
async def clustering(req: ClusteringRequest):
    try:
        path, accuracy = await run_in_thread(_clustering_sync, req)
        return {
            "status": "success",
            "function": "编群分析完成",
            "path": path,
            "准确率": accuracy
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "function": "编群分析失败",
        }

def _clustering_sync(req: ClusteringRequest):
    df = DataLoader(req.input_path).run()
    filter_df = DataFilter(
        df=df,
        time_range=req.time_range,
        lat_range=req.lat_range,
        lon_range=req.lon_range,
    ).run()
    analyzer_df, accuracy = ClusteringAnalyzer(df=filter_df).run()
    return DataSaver(
        df=analyzer_df,
        output_dir=output_dir,
        filename="cluster" 
    ).save_csv(), accuracy

@app.post("/FrequentTrajectory", summary="频繁轨迹挖掘")
async def frequent(req: FrequentRequest):
    try:
        geo_path, csv_path = await run_in_thread(_frequent_sync, req)
        return {
            "status": "success",
            "function": "频繁轨迹分析完成",
            "geo_path": geo_path,
            "csv_path": csv_path
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "function": "频繁轨迹分析失败",
        }

def _frequent_sync(req: FrequentRequest):
    df = DataLoader(req.input_path).run()

    filter_df = DataFilter(
        df=df,
        time_range=req.time_range,
        lat_range=req.lat_range,
        lon_range=req.lon_range,
    ).run()

    gdf, records_df = FrequentAnalyzer(
        df=filter_df,
        grid_size=req.grid_size,
        sigma=req.sigma,
        percentile_thresh=req.percentile_thresh,
        visulize=req.visulize
    ).run()
    return DataSaver(gdf, output_dir=output_dir, filename="frequent_trajectory").save_geojson(), DataSaver(records_df, output_dir=output_dir, filename="frequent_trajectory").save_csv()

@app.post("/WaterSurfaceModelTrajectoryPredict", summary="水面模型航迹预测")
async def predict(req: PredictRequest):
    try:
        path = await run_in_thread(_predict_sync, req)
        return {
            "status": "success",
            "function": "航迹预测完成",
            "path": path
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "function": "航迹预测失败",
        }

def _predict_sync(req: PredictRequest):
    manager = PatchTSTManager(pretrain_model_path=best_model_path)

    data = manager.run_rolling_inference(
        file_path=req.input_path,
        weight_path=req.best_model_path,
        total_steps=req.total_steps
    )

    return DataSaver(
        df=data, 
        output_dir=output_dir, 
        filename="predict"
    ).save_csv()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("./log/log", encoding="utf-8"),
            logging.StreamHandler()
            ]
        )
    uvicorn.run("api.api_server:app", host="0.0.0.0", port=5000, reload=True)