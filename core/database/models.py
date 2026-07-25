"""
企业级数据库模型模块
定义数据库表结构
"""
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean
from datetime import datetime

from .database import Base


class User(Base):
    """
    用户表模型

    存储系统用户信息
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, comment="用户ID")
    username = Column(String(50), unique=True, index=True, nullable=False, comment="用户名")
    password = Column(String(255), nullable=False, comment="密码哈希")
    email = Column(String(100), unique=True, nullable=True, comment="邮箱")
    is_active = Column(Boolean, default=True, comment="是否激活")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"


class AnalysisTask(Base):
    """
    分析任务表模型

    记录数据分析任务的执行历史
    """
    __tablename__ = "analysis_tasks"

    id = Column(Integer, primary_key=True, index=True, comment="任务ID")
    task_type = Column(String(50), nullable=False, index=True, comment="任务类型")
    status = Column(String(20), nullable=False, default="pending", comment="任务状态")
    input_file = Column(String(500), nullable=True, comment="输入文件路径")
    output_file = Column(String(500), nullable=True, comment="输出文件路径")
    parameters = Column(String(1000), nullable=True, comment="任务参数(JSON)")
    error_message = Column(String(1000), nullable=True, comment="错误信息")
    created_by = Column(Integer, nullable=True, comment="创建人ID")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    started_at = Column(DateTime, nullable=True, comment="开始时间")
    completed_at = Column(DateTime, nullable=True, comment="完成时间")

    def __repr__(self):
        return f"<AnalysisTask(id={self.id}, type='{self.task_type}', status='{self.status}')>"


class ShipInfo(Base):
    """
    船舶信息表模型

    存储船舶的基本信息
    """
    __tablename__ = "ship_info"

    id = Column(Integer, primary_key=True, index=True, comment="记录ID")
    mmsi = Column(Integer, unique=True, index=True, nullable=False, comment="MMSI")
    ship_name = Column(String(100), nullable=True, comment="船舶名称")
    ship_type = Column(String(50), nullable=True, comment="船舶类型")
    length = Column(Float, nullable=True, comment="船长(米)")
    width = Column(Float, nullable=True, comment="船宽(米)")
    draft = Column(Float, nullable=True, comment="吃水深度(米)")
    callsign = Column(String(20), nullable=True, comment="呼号")
    imo = Column(String(20), nullable=True, comment="IMO编号")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")

    def __repr__(self):
        return f"<ShipInfo(mmsi={self.mmsi}, name='{self.ship_name}')>"
