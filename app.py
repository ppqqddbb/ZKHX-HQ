"""
AIS数据分析系统 - 企业级异步API服务
提供船舶AIS数据的多维度分析功能
"""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor
import tempfile
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware

from core import get_logger
from core.exceptions import AISBaseException
from api.services import AnalysisService
from api.models import AnalysisResponse

# 初始化日志
logger = get_logger(__name__)

# 全局进程池执行器
executor: Optional[ProcessPoolExecutor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global executor

    # 启动
    logger.info("=" * 80)
    logger.info("AIS数据分析系统启动 - 企业级异步版本")
    logger.info("=" * 80)

    # 创建进程池（用于CPU密集型任务）
    # 减少工作进程数以降低内存占用
    max_workers = 2  # 从4改为2，减少内存占用
    executor = ProcessPoolExecutor(max_workers=max_workers)
    logger.info(f"进程池已创建 - 工作进程数: {max_workers}")

    # 确保必要的目录存在
    Path("logs").mkdir(exist_ok=True)
    Path("data/input").mkdir(parents=True, exist_ok=True)
    Path("data/output").mkdir(parents=True, exist_ok=True)

    logger.info("应用启动完成")

    yield

    # 关闭
    logger.info("正在关闭应用...")
    if executor:
        executor.shutdown(wait=True)
        logger.info("进程池已关闭")
    logger.info("应用已关闭")


# 创建FastAPI应用
app = FastAPI(
    title="AIS数据分析系统",
    description="企业级船舶AIS数据分析系统，支持异步并发处理",
    version="3.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# 工具函数
# ============================================================================

def save_upload_file(upload_file: UploadFile) -> Path:
    """
    保存上传的文件到临时目录

    Args:
        upload_file: 上传的文件对象

    Returns:
        临时文件路径

    Raises:
        HTTPException: 保存失败时
    """
    try:
        suffix = Path(upload_file.filename).suffix
        temp_file = Path(tempfile.gettempdir()) / f"ais_{uuid.uuid4()}{suffix}"

        with open(temp_file, "wb") as f:
            f.write(upload_file.file.read())

        logger.info(f"文件已保存: {temp_file}")
        return temp_file

    except Exception as e:
        logger.error(f"保存文件失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"保存文件失败: {str(e)}")


def cleanup_file(file_path: Path):
    """
    清理临时文件

    Args:
        file_path: 要删除的文件路径
    """
    try:
        if file_path and file_path.exists():
            file_path.unlink()
            logger.debug(f"临时文件已删除: {file_path}")
    except Exception as e:
        logger.warning(f"删除临时文件失败: {e}")


async def run_in_executor(func, *args):
    """
    在进程池中异步执行CPU密集型任务

    Args:
        func: 要执行的函数
        *args: 函数参数

    Returns:
        函数执行结果
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)


# ============================================================================
# API端点
# ============================================================================

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "version": "3.0.0",
        "service": "AIS数据分析系统",
        "async": True
    }


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "AIS数据分析系统API",
        "version": "3.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.post("/api/v1/data/process", response_model=AnalysisResponse, summary="数据处理")
async def process_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="AIS数据文件（CSV格式）")
):
    """
    数据处理端点

    - 加载AIS数据
    - 数据清洗和预处理
    - 返回处理后的数据统计信息
    """
    temp_file = None

    try:
        temp_file = save_upload_file(file)

        # 在进程池中异步执行
        result = await run_in_executor(
            AnalysisService.process_data,
            str(temp_file)
        )

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


@app.post("/api/v1/analysis/hotspot", response_model=AnalysisResponse, summary="热点区域分析")
async def analyze_hotspot(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="AIS数据文件（CSV格式）"),
    min_cluster_size: int = Query(50, ge=1, description="最小聚类大小"),
    min_samples: int = Query(10, ge=1, description="最小样本数"),
    resolution: int = Query(8, ge=0, le=15, description="H3空间索引分辨率")
):
    """
    热点区域分析

    - 识别船舶活动热点区域
    - 基于HDBSCAN聚类算法
    - 使用H3空间索引
    """
    temp_file = None

    try:
        temp_file = save_upload_file(file)

        result = await run_in_executor(
            AnalysisService.analyze_hotspot,
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
    file: UploadFile = File(..., description="AIS数据文件（CSV格式）"),
    eps_km: float = Query(1.0, gt=0, description="DBSCAN聚类半径（公里）"),
    min_samples: int = Query(3, ge=1, description="最小样本数")
):
    """
    船舶目标多维度分析

    - 活动区域识别
    - 停留位置分析
    - 速度统计
    - 总航行距离
    """
    temp_file = None

    try:
        temp_file = save_upload_file(file)

        result = await run_in_executor(
            AnalysisService.analyze_target,
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
    file: UploadFile = File(..., description="AIS数据文件（CSV格式）"),
    grid_size: float = Query(0.01, gt=0, description="网格大小（度）"),
    sigma: float = Query(2.0, gt=0, description="高斯平滑参数"),
    percentile_thresh: int = Query(90, ge=0, le=100, description="密度百分位阈值")
):
    """
    频繁轨迹挖掘

    - 识别常用航道
    - 基于密度热力图
    - 轮廓提取
    """
    temp_file = None

    try:
        temp_file = save_upload_file(file)

        result = await run_in_executor(
            AnalysisService.analyze_frequent,
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


@app.post("/api/v1/analysis/clustering", response_model=AnalysisResponse, summary="编群分析")
async def analyze_clustering(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="AIS数据文件（CSV格式，必须包含group_id列）")
):
    """
    船舶轨迹编群分析

    - 基于DTW距离的轨迹相似度计算
    - 层次聚类算法
    - 自动确定最优聚类数
    """
    temp_file = None

    try:
        temp_file = save_upload_file(file)

        result = await run_in_executor(
            AnalysisService.analyze_clustering,
            str(temp_file)
        )

        background_tasks.add_task(cleanup_file, temp_file)

        return AnalysisResponse(
            success=True,
            message="编群分析完成",
            data=result
        )

    except AISBaseException as e:
        logger.error(f"编群分析失败: {e}")
        if temp_file:
            cleanup_file(temp_file)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"未知错误: {e}", exc_info=True)
        if temp_file:
            cleanup_file(temp_file)
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@app.post("/api/v1/analysis/association", response_model=AnalysisResponse, summary="目标关联分析")
async def analyze_association(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="AIS数据文件（CSV格式）"),
    min_support: float = Query(0.01, gt=0, le=1, description="最小支持度"),
    min_confidence: float = Query(0.5, gt=0, le=1, description="最小置信度")
):
    """
    船舶行为关联分析

    - 基于Apriori算法
    - 发现船舶行为模式
    - 关联规则挖掘
    """
    temp_file = None

    try:
        temp_file = save_upload_file(file)

        result = await run_in_executor(
            AnalysisService.analyze_association,
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


@app.post("/api/v1/prediction/trajectory", response_model=AnalysisResponse, summary="航迹预测")
async def predict_trajectory(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="AIS数据文件（CSV格式）"),
    forecast_horizon: int = Query(10, ge=1, le=100, description="预测时间步长")
):
    """
    船舶航迹预测

    - 基于PatchTST时序预测模型
    - 多步预测
    - 经纬度坐标预测
    """
    temp_file = None

    try:
        temp_file = save_upload_file(file)

        result = await run_in_executor(
            AnalysisService.predict_trajectory,
            str(temp_file),
            forecast_horizon
        )

        background_tasks.add_task(cleanup_file, temp_file)

        return AnalysisResponse(
            success=True,
            message="航迹预测完成",
            data=result
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
