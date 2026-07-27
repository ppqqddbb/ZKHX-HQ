"""
API数据模型
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class AnalysisResponse(BaseModel):
    """
    通用分析响应模型

    Attributes:
        success: 是否成功
        message: 响应消息
        data: 响应数据
        task_id: 任务ID（用于异步任务跟踪）
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "分析完成",
                "data": {
                    "total_records": 1000,
                    "processing_time": 2.5
                },
                "task_id": "abc123"
            }
        }
    )

    success: bool = Field(..., description="操作是否成功")
    message: str = Field(..., description="响应消息")
    data: Optional[Dict[str, Any]] = Field(None, description="响应数据")
    task_id: Optional[str] = Field(None, description="任务ID")
