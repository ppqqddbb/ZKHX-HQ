# AIS数据分析系统

企业级船舶AIS数据分析系统，提供RESTful API接口。

## 🚀 版本说明

### 同步版（api_server.py）
- 标准FastAPI实现
- 串行处理请求
- 适合单用户/低并发场景

### 异步版（api_server_async.py）⭐ 推荐
- **真正的并发处理** - 多个请求同时执行
- **多进程架构** - ProcessPoolExecutor + asyncio
- **2-3倍性能提升** - 充分利用多核CPU
- 适合多用户/高并发场景

## 快速启动

### 异步版（推荐）
```bash
# 方法1：使用启动脚本
start_async.bat

# 方法2：命令行启动
uvicorn api_server_async:app --host 0.0.0.0 --port 8000 --reload
```

### 同步版
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

## 测试并发性能

```bash
# 测试异步版的并发能力
python test_async.py

# 对比串行vs并发性能
python test_async.py compare
```

## 安装依赖
```bash
pip install -r requirements.txt
```

## 访问API文档
```
http://localhost:8000/docs
```

## 项目结构

```
ZKHX-HQ/
├── api_server.py          # FastAPI主服务器
├── core/                  # 核心业务模块
│   ├── config.py         # 配置管理
│   ├── spatial.py        # 空间索引
│   ├── data/             # 数据处理
│   ├── analysis/         # 分析功能
│   ├── model/            # 预测模型
│   └── database/         # 数据库接口
├── requirements.txt       # 依赖列表
├── test.csv              # 测试数据（200行）
├── test_full.csv         # 完整数据（36,649行）
└── test_group.csv        # 编群测试数据
```

## API接口

| 接口 | 功能 |
|------|------|
| GET /health | 健康检查 |
| POST /api/v1/data/process | 数据处理 |
| POST /api/v1/analysis/hotspot | 热点分析 |
| POST /api/v1/analysis/frequent | 频繁轨迹 |
| POST /api/v1/analysis/target | 目标分析 |
| POST /api/v1/analysis/association | 关联分析 |
| POST /api/v1/analysis/clustering | 聚类分析 |

## 测试数据

- **test.csv**: 200行，适合快速测试
- **test_full.csv**: 36,649行，完整数据
- **test_group.csv**: 100行，编群分析专用

## 依赖

- Python 3.8+
- fastapi, uvicorn
- polars, pandas, numpy
- scikit-learn, hdbscan
- shapely, geopandas, h3

详见 `requirements.txt`
