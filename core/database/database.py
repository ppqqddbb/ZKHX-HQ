"""
企业级SQLite数据库配置模块
提供数据库连接和会话管理
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from pathlib import Path

from core.config import paths
from core.logger import get_logger

logger = get_logger(__name__)

# 数据库路径
db_path = paths.db_path

# 数据库URL
DATABASE_URL = f"sqlite:///{db_path}"

# 创建数据库引擎
try:
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False,
        pool_pre_ping=True
    )
    logger.info(f"数据库引擎创建成功: {db_path}")
except Exception as e:
    logger.error(f"数据库引擎创建失败: {e}")
    raise

# 创建会话工厂
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# 创建基础模型类
Base = declarative_base()


def get_db():
    """
    获取数据库会话

    用于依赖注入，自动管理会话生命周期

    Yields:
        数据库会话对象
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    初始化数据库

    创建所有表结构
    """
    try:
        # 确保数据库目录存在
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # 创建所有表
        Base.metadata.create_all(bind=engine)
        logger.info("数据库初始化完成")

    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise
