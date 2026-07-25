"""
AIS数据分析系统 FastAPI接口 - 异步版本
支持并发处理多个分析请求
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pathlib import Path
import polars as pl
import tempfile
import uuid
import json
import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial

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
    title="AIS数据分析系统API（异步版）",
    description="支持并发处理多个分析请求的船舶AIS数据分析系统",
    version="2.0.0-async"
)

# 创建进程池执行器（用于CPU密集型任务）
executor = ProcessPoolExecutor(max_workers=4)

# ============================================================================
# 请求/响应模型
# ============================================================================

class AnalysisResponse(BaseModel):
    """通用分析响应"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    task_id: Optional[str] = None


# ============================================================================
# 工具函数
# ============================================================================

def save_upload_file(upload_file: UploadFile) -> Path:
    """保存上传的文件"""
    try:
        suffix = Path(upload_file.filename).suffix
        temp_file = Path(tempfile.gettempdir()) / f"{uuid.uuid4()}{suffix}"

        with open(temp_file, "wb") as f:
            f.write(upload_file.file.read())

        logger.info(f"文件已保存: {temp_file}")
        return temp_file

    except Exception as e:
        logger.error(f"保存文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"保存文件失败: {str(e)}")


def cleanup_file(file_path: Path):
    """清理临时文件"""
    try:
        if file_path and file_path.exists():
            file_path.unlink()
            logger.info(f"临时文件已删除: {file_path}")
    except Exception as e:
        logger.warning(f"删除临时文件失败: {e}")


async def run_in_executor(func, *args):
    """在进程池中异步执行CPU密集型任务"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)


# ============================================================================
# CPU密集型任务包装函数（在子进程中执行）
# ============================================================================

def process_data_task(file_path: str):
    """数据处理任务"""
    loader = DataLoader(file_path)
    df = loader.run()
    processor = DataProcessor(df=df)
    df_processed = processor.run()

    return {
        "total_records": df_processed.select(pl.len()).collect().item(),
        "columns": df_processed.columns
    }


def hotspot_analysis_task(file_path: str, min_cluster_size: int, min_samples: int, resolution: int):
    """热点分析任务"""
    loader = DataLoader(file_path)
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

    return {
        "hotspot_points": df_hotspot.collect().height,
        "coverage_by_mmsi": coverage,
        "total_mmsi": len(coverage)
    }


def target_analysis_task(file_path: str, eps_km: float, min_samples: int):
    """目标分析任务"""
    loader = DataLoader(file_path)
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

    # 转换Polars类型为Python原生类型
    speed_dict = speed.to_dicts() if len(speed) > 0 else []
    distance_dict = total_distance.to_dicts() if len(total_distance) > 0 else []

    if isinstance(dwell_positions, pl.DataFrame):
        dwell_dict = dwell_positions.to_dicts() if len(dwell_positions) > 0 else []
    else:
        dwell_dict = dwell_positions

    return {
        "activate_areas_count": len(activate_areas),
        "activate_areas": activate_areas,
        "dwell_positions_count": len(dwell_dict),
        "dwell_positions": dwell_dict[:100],  # 只返回前100个，避免数据过大
        "speed_statistics": speed_dict,
        "total_distance": distance_dict,
        "zone_coverage": zones
    }


def frequent_analysis_task(file_path: str, grid_size: float, sigma: float, percentile_thresh: int):
    """频繁轨迹分析任务"""
    loader = DataLoader(file_path)
    df = loader.run()
    processor = DataProcessor(df=df)
    df_processed = processor.run()

    analyzer = FrequentAnalyzer(
        df=df_processed,
        grid_size=grid_size,
        sigma=sigma,
        percentile_thresh=percentile_thresh
    )

    frequent_channels = analyzer.run()

    return {
        "frequent_channels_count": len(frequent_channels),
        "channels": [json.loads(geom) for geom in frequent_channels['geometry'][:10]]
    }


def clustering_analysis_task(file_path: str):
    """聚类分析任务"""
    loader = DataLoader(file_path)
    df = loader.run()
    processor = DataProcessor(df=df)
    df_processed = processor.run()

    analyzer = ClusteringAnalyzer(df=df_processed)
    result_df, accuracy = analyzer.run()  # 返回元组 (pandas DataFrame, float)

    # result_df 是 pandas DataFrame，不需要 collect()
    return {
        "total_records": len(result_df),
        "accuracy": float(accuracy),
        "cluster_count": int(result_df['cluster'].nunique()) if 'cluster' in result_df.columns else 0
    }


def association_analysis_task(file_path: str, min_support: float, min_confidence: float):
    """关联分析任务"""
    loader = DataLoader(file_path)
    df = loader.run()
    processor = DataProcessor(df=df)
    df_processed = processor.run()

    analyzer = AssociationAnalyzer(
        df=df_processed,
        min_support=min_support,
        min_confidence=min_confidence
    )

    frequent_itemsets, association_rules = analyzer.run()

    top_rules = association_rules.head(10).to_dicts() if len(association_rules) > 0 else []

    return {
        "frequent_itemsets_count": len(frequent_itemsets),
        "association_rules_count": len(association_rules),
        "top_rules": top_rules
    }


# ============================================================================
# API端点
# ============================================================================

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "version": "2.0.0-async", "concurrent": True}


@app.post("/api/v1/data/process", response_model=AnalysisResponse, summary="数据处理")
async def process_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="AIS数据文件")
):
    """
    异步数据处理

    支持并发处理多个文件
    """
    temp_file = None

    try:
        temp_file = save_upload_file(file)

        # 在进程池中异步执行
        result = await run_in_executor(process_data_task, str(temp_file))

        background_tasks.add_task(cleanup_file, temp_file)

        return AnalysisResponse(
            success=True,
            message="数据处理完成",
            data=result
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


@app.post("/api/v1/analysis/hotspot", response_model=AnalysisResponse, summary="热点分析")
async def analyze_hotspot(
    background_tasks: BackgroundTasks,
    min_cluster_size: int = Query(50, description="最小聚类大小"),
    min_samples: int = Query(10, description="最小样本数"),
    resolution: int = Query(8, description="H3空间索引分辨率"),
    file: UploadFile = File(..., description="AIS数据文件")
):
    """
    异步热点区域分析

    支持多个热点分析请求并发执行
    """
    temp_file = None

    try:
        temp_file = save_upload_file(file)

        # 在进程池中异步执行
        result = await run_in_executor(
            hotspot_analysis_task,
            str(temp_file),
            min_cluster_size,
            min_samples,
            resolution
        )

        background_tasks.add_task(cleanup_file, temp_file)

        return AnalysisResponse(
            success=True,
            message="热点分析完成",
            data=result
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


@app.post("/api/v1/analysis/target", response_model=AnalysisResponse, summary="目标分析")
async def analyze_target(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="AIS数据文件"),
    eps_km: float = Query(1.0, description="DBSCAN聚类半径（公里）"),
    min_samples: int = Query(3, description="最小样本数")
):
    """
    异步船舶目标多维度分析

    支持并发处理
    """
    temp_file = None

    try:
        temp_file = save_upload_file(file)

        # 在进程池中异步执行
        result = await run_in_executor(
            target_analysis_task,
            str(temp_file),
            eps_km,
            min_samples
        )

        background_tasks.add_task(cleanup_file, temp_file)

        return AnalysisResponse(
            success=True,
            message="目标分析完成",
            data=result
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


@app.post("/api/v1/analysis/frequent", response_model=AnalysisResponse, summary="频繁轨迹分析")
async def analyze_frequent(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="AIS数据文件"),
    grid_size: float = Query(0.01, description="网格大小（度）"),
    sigma: float = Query(2.0, description="高斯平滑参数"),
    percentile_thresh: int = Query(90, description="密度百分位阈值")
):
    """
    异步频繁轨迹分析

    支持并发处理
    """
    temp_file = None

    try:
        temp_file = save_upload_file(file)

        # 在进程池中异步执行
        result = await run_in_executor(
            frequent_analysis_task,
            str(temp_file),
            grid_size,
            sigma,
            percentile_thresh
        )

        background_tasks.add_task(cleanup_file, temp_file)

        return AnalysisResponse(
            success=True,
            message="频繁轨迹分析完成",
            data=result
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


@app.post("/api/v1/analysis/clustering", response_model=AnalysisResponse, summary="聚类分析")
async def analyze_clustering(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="AIS数据文件")
):
    """
    异步聚类分析

    支持并发处理
    """
    temp_file = None

    try:
        temp_file = save_upload_file(file)

        # 在进程池中异步执行
        result = await run_in_executor(clustering_analysis_task, str(temp_file))

        background_tasks.add_task(cleanup_file, temp_file)

        return AnalysisResponse(
            success=True,
            message="聚类分析完成",
            data=result
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


@app.post("/api/v1/analysis/association", response_model=AnalysisResponse, summary="关联分析")
async def analyze_association(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="AIS数据文件"),
    min_support: float = Query(0.01, description="最小支持度"),
    min_confidence: float = Query(0.5, description="最小置信度")
):
    """
    异步船舶行为关联分析

    支持并发处理
    """
    temp_file = None

    try:
        temp_file = save_upload_file(file)

        # 在进程池中异步执行
        result = await run_in_executor(
            association_analysis_task,
            str(temp_file),
            min_support,
            min_confidence
        )

        background_tasks.add_task(cleanup_file, temp_file)

        return AnalysisResponse(
            success=True,
            message="关联分析完成",
            data=result
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


# ============================================================================
# 应用生命周期
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """应用启动"""
    logger.info("=" * 60)
    logger.info("AIS数据分析系统启动 - 异步版本")
    logger.info("支持并发处理多个分析请求")
    logger.info(f"进程池大小: {executor._max_workers}")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭"""
    logger.info("关闭进程池...")
    executor.shutdown(wait=True)
    logger.info("AIS数据分析系统已关闭")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
