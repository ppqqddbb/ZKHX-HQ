"""
AIS数据分析系统 FastAPI接口
提供RESTful API访问AIS数据分析功能
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pathlib import Path
import polars as pl
import tempfile
import uuid
import json

# 核心模块导入
from core import (
    DataLoader, DataProcessor,
    HotspotAnalyzer, FrequentAnalyzer, TargetAnalyzer,
    AssociationAnalyzer, ClusteringAnalyzer,
    PatchTSTManager,
    get_logger,
    AISBaseException
)

logger = get_logger(__name__)

app = FastAPI(
    title="AIS数据分析系统API",
    description="提供船舶AIS数据的多维度分析功能",
    version="1.0.0"
)

# ============================================================================
# 请求/响应模型
# ============================================================================

class AnalysisResponse(BaseModel):
    """通用分析响应"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    task_id: Optional[str] = None


class HotspotRequest(BaseModel):
    """热点分析请求"""
    min_cluster_size: int = Field(50, description="最小聚类大小")
    min_samples: int = Field(10, description="最小样本数")
    resolution: int = Field(8, description="H3空间索引分辨率")


class FrequentRequest(BaseModel):
    """频繁轨迹分析请求"""
    grid_size: float = Field(0.01, description="网格大小（度）")
    sigma: float = Field(2.0, description="高斯平滑参数")
    percentile_thresh: int = Field(90, description="密度百分位阈值")


class TargetRequest(BaseModel):
    """目标分析请求"""
    eps_km: float = Field(1.0, description="DBSCAN聚类半径（公里）")
    min_samples: int = Field(3, description="最小样本数")
    zones: Optional[List[Dict]] = Field(None, description="自定义区域配置")


class AssociationRequest(BaseModel):
    """关联分析请求"""
    min_support: float = Field(0.01, description="最小支持度")
    min_confidence: float = Field(0.5, description="最小置信度")


class PredictionRequest(BaseModel):
    """航迹预测请求"""
    weight_path: str = Field(..., description="模型权重路径")
    scaler_path: str = Field(..., description="标准化器路径")
    mmsi_list: Optional[List[int]] = Field(None, description="指定MMSI列表")
    max_ships: int = Field(10, description="最大预测船舶数")


# ============================================================================
# 工具函数
# ============================================================================

def save_upload_file(upload_file: UploadFile) -> Path:
    """保存上传的文件"""
    try:
        suffix = Path(upload_file.filename).suffix
        temp_file = Path(tempfile.gettempdir()) / f"{uuid.uuid4()}{suffix}"

        with open(temp_file, "wb") as f:
            content = upload_file.file.read()
            f.write(content)

        logger.info(f"文件已保存: {temp_file}")
        return temp_file

    except Exception as e:
        logger.error(f"文件保存失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")


def cleanup_file(file_path: Path):
    """清理临时文件"""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"临时文件已删除: {file_path}")
    except Exception as e:
        logger.warning(f"清理文件失败: {e}")


# ============================================================================
# API端点
# ============================================================================

@app.get("/", summary="API根路径")
async def root():
    """返回API基本信息"""
    return {
        "name": "AIS数据分析系统API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "process": "/api/v1/data/process",
            "hotspot": "/api/v1/analysis/hotspot",
            "frequent": "/api/v1/analysis/frequent",
            "target": "/api/v1/analysis/target",
            "association": "/api/v1/analysis/association",
            "clustering": "/api/v1/analysis/clustering",
            "prediction": "/api/v1/model/predict"
        }
    }


@app.get("/health", summary="健康检查")
async def health_check():
    """检查服务健康状态"""
    return {"status": "healthy", "service": "ais-analysis-api"}


@app.post("/api/v1/data/process", response_model=AnalysisResponse, summary="数据处理")
async def process_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="AIS数据文件（CSV格式）")
):
    """
    上传并处理AIS数据

    - 数据清洗
    - 异常值处理
    - 缺失值填充
    - 返回处理后的数据统计
    """
    temp_file = None

    try:
        temp_file = save_upload_file(file)
        logger.info(f"开始处理数据: {temp_file}")

        loader = DataLoader(str(temp_file))
        df = loader.run()
        logger.info("数据加载完成")

        processor = DataProcessor(df=df)
        df_processed = processor.run()
        logger.info("数据处理完成")

        df_collected = df_processed.collect()
        logger.info(f"数据收集完成，记录数: {df_collected.height}")

        background_tasks.add_task(cleanup_file, temp_file)

        return AnalysisResponse(
            success=True,
            message="数据处理完成",
            data={
                "records": df_collected.height,
                "columns": df_collected.columns,
                "shape": {"rows": df_collected.height, "cols": len(df_collected.columns)}
            }
        )

    except AISBaseException as e:
        logger.error(f"数据处理失败: {e}")
        if temp_file:
            cleanup_file(temp_file)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"未知错误: {e}", exc_info=True)
        if temp_file:
            cleanup_file(temp_file)
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@app.post("/api/v1/analysis/hotspot", response_model=AnalysisResponse, summary="热点区域分析")
async def analyze_hotspot(
    background_tasks: BackgroundTasks,
    min_cluster_size: int = Query(50, description="最小聚类大小"),
    min_samples: int = Query(10, description="最小样本数"),
    resolution: int = Query(8, description="H3空间索引分辨率"),
    file: UploadFile = File(..., description="AIS数据文件")
):
    """
    识别船舶活动热点区域

    - 基于HDBSCAN聚类算法
    - 计算时间和空间覆盖度
    - 返回热点区域统计

    **在Swagger UI中使用：**
    1. 在URL参数中设置 min_cluster_size, min_samples, resolution
    2. 然后上传文件
    """
    temp_file = None

    try:
        temp_file = save_upload_file(file)

        loader = DataLoader(str(temp_file))
        df = loader.run()

        processor = DataProcessor(df=df)
        df_processed = processor.run()

        analyzer = HotspotAnalyzer(
            df=df_processed,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            resolution=resolution
        )

        df_hotspot, coverage = analyzer.run()

        background_tasks.add_task(cleanup_file, temp_file)

        return AnalysisResponse(
            success=True,
            message="热点分析完成",
            data={
                "hotspot_points": df_hotspot.collect().height,
                "coverage_by_mmsi": coverage,
                "total_mmsi": len(coverage)
            }
        )

    except AISBaseException as e:
        logger.error(f"热点分析失败: {e}")
        if temp_file:
            cleanup_file(temp_file)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"未知错误: {e}", exc_info=True)
        if temp_file:
            cleanup_file(temp_file)
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@app.post("/api/v1/analysis/frequent", response_model=AnalysisResponse, summary="频繁轨迹分析")
async def analyze_frequent(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="AIS数据文件"),
    grid_size: float = Query(0.01, description="网格大小（度）"),
    sigma: float = Query(2.0, description="高斯平滑参数"),
    percentile_thresh: int = Query(90, description="密度百分位阈值")
):
    """
    识别频繁航道

    - 密度网格分析
    - 骨架提取
    - 返回航道数量和坐标
    """
    temp_file = None

    try:
        temp_file = save_upload_file(file)

        loader = DataLoader(str(temp_file))
        df = loader.run()

        processor = DataProcessor(df=df)
        df_processed = processor.run()

        analyzer = FrequentAnalyzer(
            df=df_processed,
            grid_size=grid_size,
            sigma=sigma,
            percentile_thresh=percentile_thresh
        )

        gdf, records_df = analyzer.run()

        background_tasks.add_task(cleanup_file, temp_file)

        return AnalysisResponse(
            success=True,
            message="频繁轨迹分析完成",
            data={
                "route_count": len(gdf),
                "route_points": len(records_df)
            }
        )

    except AISBaseException as e:
        logger.error(f"频繁轨迹分析失败: {e}")
        if temp_file:
            cleanup_file(temp_file)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"未知错误: {e}", exc_info=True)
        if temp_file:
            cleanup_file(temp_file)
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@app.post("/api/v1/analysis/target", response_model=AnalysisResponse, summary="目标分析")
async def analyze_target(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="AIS数据文件"),
    eps_km: float = Query(1.0, description="DBSCAN聚类半径（公里）"),
    min_samples: int = Query(3, description="最小样本数")
):
    """
    船舶目标多维度分析

    - 活动区域
    - 停留位置
    - 速度统计
    - 航程计算
    """
    temp_file = None

    try:
        temp_file = save_upload_file(file)

        loader = DataLoader(str(temp_file))
        df = loader.run()

        processor = DataProcessor(df=df)
        df_processed = processor.run()

        analyzer = TargetAnalyzer(
            df=df_processed,
            eps_km=eps_km,
            min_samples=min_samples,
            zones=None
        )

        activate_areas, dwell_positions, speed, total_distance, zones = analyzer.run()

        background_tasks.add_task(cleanup_file, temp_file)

        # 转换Polars DataFrame/list为Python原生类型（避免序列化错误）
        speed_dict = speed.to_dicts() if len(speed) > 0 else []
        distance_dict = total_distance.to_dicts() if len(total_distance) > 0 else []

        # dwell_positions也是DataFrame，需要转换
        if isinstance(dwell_positions, pl.DataFrame):
            dwell_dict = dwell_positions.to_dicts() if len(dwell_positions) > 0 else []
        else:
            dwell_dict = dwell_positions  # 如果已经是list，直接使用

        return AnalysisResponse(
            success=True,
            message="目标分析完成",
            data={
                "activate_areas_count": len(activate_areas),
                "activate_areas": activate_areas,
                "dwell_positions_count": len(dwell_dict),
                "dwell_positions": dwell_dict,
                "speed_statistics": speed_dict,
                "total_distance": distance_dict,
                "zone_coverage": zones
            }
        )

    except AISBaseException as e:
        logger.error(f"目标分析失败: {e}")
        if temp_file:
            cleanup_file(temp_file)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"未知错误: {e}", exc_info=True)
        if temp_file:
            cleanup_file(temp_file)
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@app.post("/api/v1/analysis/association", response_model=AnalysisResponse, summary="关联分析")
async def analyze_association(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="AIS数据文件"),
    min_support: float = Query(0.01, description="最小支持度"),
    min_confidence: float = Query(0.5, description="最小置信度")
):
    """
    船舶行为关联分析

    - FP-Growth算法
    - 频繁模式挖掘
    - 关联规则生成
    """
    temp_file = None

    try:
        temp_file = save_upload_file(file)

        loader = DataLoader(str(temp_file))
        df = loader.run()

        processor = DataProcessor(df=df)
        df_processed = processor.run()

        analyzer = AssociationAnalyzer(
            df=df_processed,
            min_support=min_support,
            min_confidence=min_confidence
        )

        frequent_itemsets, association_rules = analyzer.run()

        background_tasks.add_task(cleanup_file, temp_file)

        # 转换为Python原生类型
        top_rules = association_rules.head(10).to_dicts() if len(association_rules) > 0 else []

        return AnalysisResponse(
            success=True,
            message="关联分析完成",
            data={
                "frequent_itemsets_count": len(frequent_itemsets),
                "association_rules_count": len(association_rules),
                "top_rules": top_rules
            }
        )

    except AISBaseException as e:
        logger.error(f"关联分析失败: {e}")
        if temp_file:
            cleanup_file(temp_file)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"未知错误: {e}", exc_info=True)
        if temp_file:
            cleanup_file(temp_file)
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@app.post("/api/v1/analysis/clustering", response_model=AnalysisResponse, summary="轨迹聚类分析")
async def analyze_clustering(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    船舶轨迹聚类

    - DTW距离度量
    - 层次聚类
    - 聚类准确率评估
    """
    temp_file = None

    try:
        temp_file = save_upload_file(file)

        loader = DataLoader(str(temp_file))
        df = loader.run()

        processor = DataProcessor(df=df)
        df_processed = processor.run()

        analyzer = ClusteringAnalyzer(df=df_processed)
        result_df, accuracy = analyzer.run()

        background_tasks.add_task(cleanup_file, temp_file)

        return AnalysisResponse(
            success=True,
            message="聚类分析完成",
            data={
                "total_records": len(result_df),
                "accuracy": float(accuracy),
                "cluster_count": result_df["predict_cluster"].n_unique()
            }
        )

    except AISBaseException as e:
        logger.error(f"聚类分析失败: {e}")
        if temp_file:
            cleanup_file(temp_file)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"未知错误: {e}", exc_info=True)
        if temp_file:
            cleanup_file(temp_file)
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@app.post("/api/v1/model/predict", response_model=AnalysisResponse, summary="航迹预测")
async def predict_trajectory(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="AIS数据文件"),
    weight_path: str = Query(..., description="模型权重路径"),
    scaler_path: str = Query(..., description="标准化器路径"),
    max_ships: int = Query(10, description="最大预测船舶数")
):
    """
    船舶航迹预测

    - PatchTST深度学习模型
    - 多步预测
    - 返回预测轨迹坐标
    """
    temp_file = None

    try:
        temp_file = save_upload_file(file)

        manager = PatchTSTManager(
            pretrain_model_path=None,
            scaler_path=scaler_path
        )

        predictions = manager.predict(
            file_path=str(temp_file),
            weight_path=weight_path,
            mmsi_list=None,
            max_ships=max_ships
        )

        background_tasks.add_task(cleanup_file, temp_file)

        return AnalysisResponse(
            success=True,
            message="航迹预测完成",
            data={
                "prediction_points": len(predictions),
                "predicted_mmsi_count": predictions["MMSI"].n_unique() if len(predictions) > 0 else 0
            }
        )

    except AISBaseException as e:
        logger.error(f"航迹预测失败: {e}")
        if temp_file:
            cleanup_file(temp_file)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"未知错误: {e}", exc_info=True)
        if temp_file:
            cleanup_file(temp_file)
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


# ============================================================================
# 启动配置
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
