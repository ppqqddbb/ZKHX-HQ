# AIS数据分析系统 - 使用指南

## 目录
1. [系统概述](#系统概述)
2. [快速开始](#快速开始)
3. [API详细说明](#api详细说明)
4. [测试数据说明](#测试数据说明)
5. [常见问题](#常见问题)
6. [性能优化](#性能优化)

## 系统概述

### 功能矩阵

| 功能模块 | 输入数据 | 核心算法 | 输出结果 | 应用场景 |
|---------|---------|---------|---------|---------|
| 数据处理 | test.csv | Polars数据处理 | 清洗后的数据统计 | 数据预处理 |
| 热点分析 | test.csv | HDBSCAN + H3 | 热点区域分布 | 港口、锚地识别 |
| 目标分析 | test.csv | DBSCAN | 活动区域、停留点 | 船舶行为分析 |
| 频繁轨迹 | test.csv | 密度热力图 | 常用航道 | 航线规划 |
| 编群分析 | test_group.csv | DTW + 层次聚类 | 轨迹分类 | 船队管理 |
| 关联分析 | test.csv | Apriori | 行为关联规则 | 行为模式挖掘 |
| 航迹预测 | test.csv | PatchTST | 未来位置预测 | 轨迹预测 |

### 系统架构

```
┌─────────────────────────────────────────────┐
│           FastAPI Application               │
│  (异步请求处理 + CORS + 生命周期管理)        │
└─────────────────┬───────────────────────────┘
                  │
    ┌─────────────┴─────────────┐
    │   ProcessPoolExecutor     │
    │   (4个工作进程)            │
    └─────────────┬─────────────┘
                  │
    ┌─────────────┴─────────────┐
    │   AnalysisService         │
    │   (业务服务层)             │
    └─────────────┬─────────────┘
                  │
    ┌─────────────┴─────────────┐
    │   Core Modules            │
    ├───────────────────────────┤
    │ • DataLoader/Processor    │
    │ • HotspotAnalyzer         │
    │ • TargetAnalyzer          │
    │ • FrequentAnalyzer        │
    │ • ClusteringAnalyzer      │
    │ • AssociationAnalyzer     │
    │ • PatchTSTManager         │
    └───────────────────────────┘
```

## 快速开始

### 1. 环境准备

**系统要求：**
- Python 3.8+
- 8GB+ RAM（推荐16GB用于大数据集）
- 4核+ CPU（多核性能更好）

**安装依赖：**
```bash
pip install -r requirements.txt
```

### 2. 启动服务

**Windows:**
```bash
scripts\start.bat
```

**Linux/Mac:**
```bash
python app.py
```

### 3. 验证安装

访问 http://localhost:8000/health 应该看到：
```json
{
  "status": "healthy",
  "version": "3.0.0",
  "service": "AIS数据分析系统",
  "async": true
}
```

### 4. 运行测试

```bash
scripts\test.bat
```

## API详细说明

### 1. 数据处理 API

**端点:** `POST /api/v1/data/process`

**功能:** 加载、清洗、预处理AIS数据

**请求参数:**
- `file`: CSV文件（multipart/form-data）

**请求示例:**
```bash
curl -X POST "http://localhost:8000/api/v1/data/process" \
  -F "file=@data/input/test.csv"
```

**响应示例:**
```json
{
  "success": true,
  "message": "数据处理完成",
  "data": {
    "total_records": 200,
    "columns": ["Timestamp", "MMSI", "Latitude", "Longitude", ...],
    "sample_data": [...]
  }
}
```

### 2. 热点区域分析 API

**端点:** `POST /api/v1/analysis/hotspot`

**功能:** 识别船舶活动热点区域（港口、锚地等）

**请求参数:**
- `file`: CSV文件
- `min_cluster_size`: 最小聚类大小（默认50）
- `min_samples`: 最小样本数（默认10）
- `resolution`: H3分辨率0-15（默认8）

**分辨率说明:**
- 0-3: 国家级别（粗粒度）
- 4-6: 区域级别
- 7-9: 城市级别（推荐）
- 10-12: 街区级别
- 13-15: 建筑级别（细粒度）

**请求示例:**
```bash
curl -X POST "http://localhost:8000/api/v1/analysis/hotspot?min_cluster_size=50&min_samples=10&resolution=8" \
  -F "file=@data/input/test.csv"
```

**响应示例:**
```json
{
  "success": true,
  "message": "热点分析完成",
  "data": {
    "hotspot_points": 45,
    "coverage_by_mmsi": {
      "273257120": 0.85,
      "219000431": 0.92
    },
    "total_mmsi": 2,
    "parameters": {
      "min_cluster_size": 50,
      "min_samples": 10,
      "resolution": 8
    }
  }
}
```

### 3. 目标分析 API

**端点:** `POST /api/v1/analysis/target`

**功能:** 多维度船舶行为分析（活动区域、停留点、速度统计）

**请求参数:**
- `file`: CSV文件
- `eps_km`: DBSCAN半径，单位公里（默认1.0）
- `min_samples`: 最小样本数（默认3）

**请求示例:**
```bash
curl -X POST "http://localhost:8000/api/v1/analysis/target?eps_km=1.0&min_samples=3" \
  -F "file=@data/input/test.csv"
```

**响应示例:**
```json
{
  "success": true,
  "message": "目标分析完成",
  "data": {
    "activate_areas_count": 8,
    "activate_areas": [...],
    "dwell_positions_count": 12,
    "dwell_positions": [...],
    "speed_statistics": [...],
    "total_distance": [...],
    "zone_coverage": {...}
  }
}
```

### 4. 频繁轨迹分析 API

**端点:** `POST /api/v1/analysis/frequent`

**功能:** 识别船舶常用航道和航线

**请求参数:**
- `file`: CSV文件
- `grid_size`: 网格大小，单位度（默认0.01）
- `sigma`: 高斯平滑参数（默认2.0）
- `percentile_thresh`: 密度百分位阈值0-100（默认90）

**请求示例:**
```bash
curl -X POST "http://localhost:8000/api/v1/analysis/frequent?grid_size=0.01&sigma=2.0&percentile_thresh=90" \
  -F "file=@data/input/test.csv"
```

**响应示例:**
```json
{
  "success": true,
  "message": "频繁轨迹分析完成",
  "data": {
    "frequent_channels_count": 15,
    "channels": [
      {
        "type": "LineString",
        "coordinates": [[lon1, lat1], [lon2, lat2], ...]
      }
    ],
    "parameters": {...}
  }
}
```

### 5. 编群分析 API

**端点:** `POST /api/v1/analysis/clustering`

**功能:** 基于轨迹相似度的船舶分群（需要group_id列）

**请求参数:**
- `file`: CSV文件（必须包含group_id列）

**请求示例:**
```bash
curl -X POST "http://localhost:8000/api/v1/analysis/clustering" \
  -F "file=@data/input/test_group.csv"
```

**响应示例:**
```json
{
  "success": true,
  "message": "编群分析完成",
  "data": {
    "total_records": 200,
    "accuracy": 0.8750,
    "cluster_count": 2,
    "sample_results": [...]
  }
}
```

### 6. 关联分析 API

**端点:** `POST /api/v1/analysis/association`

**功能:** 船舶行为模式关联规则挖掘

**请求参数:**
- `file`: CSV文件
- `min_support`: 最小支持度0-1（默认0.01）
- `min_confidence`: 最小置信度0-1（默认0.5）

**请求示例:**
```bash
curl -X POST "http://localhost:8000/api/v1/analysis/association?min_support=0.01&min_confidence=0.5" \
  -F "file=@data/input/test.csv"
```

### 7. 航迹预测 API

**端点:** `POST /api/v1/prediction/trajectory`

**功能:** 基于历史轨迹预测未来位置

**请求参数:**
- `file`: CSV文件
- `forecast_horizon`: 预测步长1-100（默认10）

**请求示例:**
```bash
curl -X POST "http://localhost:8000/api/v1/prediction/trajectory?forecast_horizon=10" \
  -F "file=@data/input/test.csv"
```

## 测试数据说明

### test.csv (200行)
- **用途:** 数据处理、热点分析、目标分析、频繁轨迹、关联分析、航迹预测
- **特点:** 2艘船（MMSI: 273257120, 219000431）的AIS数据
- **时间范围:** 2025-01-03 00:00:00 开始
- **数据质量:** 包含部分重复数据和Unknown字段，适合测试数据清洗功能

### test_group.csv (200行)
- **用途:** 编群分析（轨迹聚类）
- **特点:** 在test.csv基础上增加了group_id列
- **group_id:** 1和2两个分组，用于验证聚类准确率
- **说明:** 该列是真实分组标签，算法会尝试还原这个分组

## 常见问题

### Q1: 如何处理大数据文件？
**A:** 
1. 增加进程池大小：修改`app.py`中的`max_workers`参数
2. 使用流式处理：Polars的LazyFrame已经实现了延迟计算
3. 分批处理：将大文件拆分成小文件分别处理

### Q2: 编群分析报错"未找到group_id列"？
**A:** 编群分析需要使用`test_group.csv`，该文件包含`group_id`列。如果使用普通数据，系统会自动使用MMSI作为分组标识。

### Q3: 如何调整分析参数获得更好的结果？
**A:**
- **热点分析:** 降低`min_cluster_size`可以发现更多小热点
- **目标分析:** 增大`eps_km`可以合并相近的停留点
- **频繁轨迹:** 提高`percentile_thresh`只保留最频繁的航道
- **关联分析:** 降低`min_support`可以发现更多规则

### Q4: API响应时间过长？
**A:**
- 数据处理和编群分析相对耗时（10-30秒）
- 考虑使用后台任务队列（Celery）
- 增加服务器资源（CPU核心数）

### Q5: 如何部署到生产环境？
**A:**
```bash
# 使用Gunicorn + Uvicorn
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 性能优化

### 1. 进程池配置
```python
# app.py
max_workers = 4  # 根据CPU核心数调整，建议为核心数
```

### 2. 数据预处理优化
```python
# 使用Polars LazyFrame延迟计算
df = pl.scan_csv("data.csv")  # 不立即加载
df = df.filter(...)  # 添加过滤条件
result = df.collect()  # 最后才执行
```

### 3. 内存优化
- 大文件处理时使用流式读取
- 及时清理临时文件
- 限制返回数据量（已在代码中实现）

### 4. 并发优化
- 异步处理多个请求
- CPU密集型任务使用进程池
- IO密集型任务使用asyncio

### 5. 缓存策略
可以添加Redis缓存常用分析结果：
```python
# 伪代码
@cache(expire=3600)
def analyze_hotspot(file_hash, params):
    # 分析逻辑
    pass
```

## 监控和日志

### 日志文件位置
- 日志目录: `logs/`
- 日志格式: 时间戳 - 级别 - 模块 - 消息

### 查看日志
```bash
# Windows
type logs\app.log

# Linux/Mac
tail -f logs/app.log
```

### 监控指标
- 请求响应时间
- 进程池利用率
- 内存使用情况
- 错误率

## 开发指南

### 添加新的分析功能

1. **在core/analysis/创建分析器**
```python
# core/analysis/new_analyzer.py
class NewAnalyzer:
    def __init__(self, df):
        self.df = df
    
    def run(self):
        # 分析逻辑
        return result
```

2. **在api/services.py添加服务方法**
```python
@staticmethod
def analyze_new(file_path: str, params):
    loader = DataLoader(file_path)
    df = loader.run()
    analyzer = NewAnalyzer(df=df)
    return analyzer.run()
```

3. **在app.py添加API端点**
```python
@app.post("/api/v1/analysis/new")
async def analyze_new(...):
    result = await run_in_executor(
        AnalysisService.analyze_new,
        str(temp_file)
    )
    return AnalysisResponse(...)
```

## 技术支持

如有问题，请：
1. 查看日志文件排查错误
2. 运行测试脚本验证功能
3. 提交Issue到项目仓库
4. 参考API文档和代码注释
