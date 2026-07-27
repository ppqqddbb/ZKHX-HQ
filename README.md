# AIS数据分析系统

企业级船舶AIS数据分析系统，提供RESTful API接口，支持异步并发处理。

## 🌟 特性

- **异步并发处理** - 基于FastAPI + ProcessPoolExecutor，充分利用多核CPU
- **企业级架构** - 清晰的代码结构，模块化设计，易于维护和扩展
- **完整的分析功能** - 7大核心分析功能，满足各种AIS数据分析需求
- **健壮的错误处理** - 完善的异常处理和日志记录
- **RESTful API** - 标准的REST接口，易于集成

## 📋 功能列表

| 功能 | 描述 | 算法/技术 |
|------|------|-----------|
| 数据处理 | AIS数据加载、清洗、预处理 | Polars高性能数据处理 |
| 热点区域分析 | 识别船舶活动热点区域 | HDBSCAN聚类 + H3空间索引 |
| 目标分析 | 船舶活动区域、停留点、速度统计 | DBSCAN聚类 |
| 频繁轨迹挖掘 | 识别常用航道和航线 | 密度热力图 + 轮廓提取 |
| 编群分析 | 轨迹聚类和相似度分析 | DTW距离 + 层次聚类 |
| 目标关联分析 | 发现船舶行为模式和关联规则 | Apriori算法 |
| 航迹预测 | 船舶轨迹预测 | PatchTST时序模型 |

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务器

**方法1：使用启动脚本（推荐）**
```bash
scripts/start.bat
```

**方法2：命令行启动**
```bash
python app.py
# 或
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. 访问API文档

启动后访问：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- 健康检查: http://localhost:8000/health

### 4. 运行测试

```bash
scripts/test.bat
# 或
python tests/test_api.py
```

## 📁 项目结构

```
ZKHX-HQ/
├── app.py                    # 主应用入口
├── requirements.txt          # Python依赖
├── README.md                 # 项目文档
│
├── api/                      # API层
│   ├── __init__.py
│   ├── models.py            # 数据模型
│   └── services.py          # 业务服务层
│
├── core/                     # 核心业务模块
│   ├── __init__.py
│   ├── config.py            # 配置管理
│   ├── exceptions.py        # 自定义异常
│   ├── logger.py            # 日志管理
│   ├── validators.py        # 数据验证
│   ├── spatial.py           # 空间索引
│   │
│   ├── data/                # 数据处理模块
│   │   ├── loader.py        # 数据加载
│   │   ├── process.py       # 数据处理
│   │   ├── filter.py        # 数据过滤
│   │   └── save.py          # 数据保存
│   │
│   ├── analysis/            # 分析模块
│   │   ├── hotspot.py       # 热点分析
│   │   ├── target.py        # 目标分析
│   │   ├── frequent.py      # 频繁轨迹
│   │   ├── clustering.py    # 编群分析
│   │   └── association.py   # 关联分析
│   │
│   ├── model/               # 预测模型
│   │   └── prediction.py    # 航迹预测
│   │
│   └── database/            # 数据库模块
│       ├── database.py      # 数据库连接
│       └── models.py        # 数据模型
│
├── data/                     # 数据目录
│   ├── input/               # 输入数据
│   │   ├── test.csv         # 测试数据
│   │   └── test_group.csv   # 编群测试数据
│   └── output/              # 输出数据
│
├── tests/                    # 测试目录
│   └── test_api.py          # API测试
│
├── scripts/                  # 脚本目录
│   ├── start.bat            # 启动脚本
│   └── test.bat             # 测试脚本
│
├── logs/                     # 日志目录（自动生成）
└── docs/                     # 文档目录

```

## 🔌 API接口

### 基础接口

- `GET /` - 根路径
- `GET /health` - 健康检查

### 数据处理

- `POST /api/v1/data/process` - 数据处理

### 分析接口

- `POST /api/v1/analysis/hotspot` - 热点区域分析
- `POST /api/v1/analysis/target` - 目标分析
- `POST /api/v1/analysis/frequent` - 频繁轨迹分析
- `POST /api/v1/analysis/clustering` - 编群分析
- `POST /api/v1/analysis/association` - 目标关联分析

### 预测接口

- `POST /api/v1/prediction/trajectory` - 航迹预测

## 📊 测试数据

项目提供两个测试数据文件：

1. **test.csv** (200行)
   - 用于：数据处理、热点分析、目标分析、频繁轨迹、关联分析、航迹预测
   - 包含字段：时间戳、MMSI、经纬度、速度、航向等

2. **test_group.csv** (200行，含group_id)
   - 用于：编群分析
   - 额外包含：group_id列（用于验证聚类准确率）

## 🛠️ 技术栈

### 核心框架
- **FastAPI** - 现代高性能Web框架
- **Uvicorn** - ASGI服务器
- **Pydantic** - 数据验证

### 数据处理
- **Polars** - 高性能数据处理
- **Pandas** - 数据分析
- **NumPy** - 数值计算

### 机器学习
- **Scikit-learn** - 机器学习算法
- **HDBSCAN** - 层次聚类
- **TSLearn** - 时间序列分析
- **PyTorch** - 深度学习
- **Transformers** - PatchTST模型

### 地理空间
- **GeoPandas** - 地理数据处理
- **Shapely** - 几何对象操作
- **H3** - 六边形空间索引

## 📝 使用示例

### Python客户端示例

```python
import requests

# 数据处理
with open("data/input/test.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/data/process",
        files={"file": f}
    )
    print(response.json())

# 热点分析
with open("data/input/test.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/analysis/hotspot",
        files={"file": f},
        params={
            "min_cluster_size": 50,
            "min_samples": 10,
            "resolution": 8
        }
    )
    print(response.json())

# 编群分析
with open("data/input/test_group.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/analysis/clustering",
        files={"file": f}
    )
    print(response.json())
```

### cURL示例

```bash
# 健康检查
curl http://localhost:8000/health

# 数据处理
curl -X POST "http://localhost:8000/api/v1/data/process" \
  -F "file=@data/input/test.csv"

# 热点分析
curl -X POST "http://localhost:8000/api/v1/analysis/hotspot?min_cluster_size=50&min_samples=10&resolution=8" \
  -F "file=@data/input/test.csv"
```

## 🔧 配置说明

配置文件位于 `core/config.py`，可配置项包括：

- 数据列名映射
- 模型参数
- 分析参数
- 日志配置

## 📈 性能特点

- **异步并发** - 多个请求同时处理，互不阻塞
- **进程池架构** - CPU密集型任务在独立进程中执行
- **内存优化** - 使用Polars LazyFrame延迟计算
- **性能提升** - 相比同步版本提升2-3倍

## ⚠️ 注意事项

1. 编群分析需要数据文件包含 `group_id` 列
2. 大文件处理建议增加进程池大小（修改 `app.py` 中的 `max_workers`）
3. 生产环境建议使用 Gunicorn + Uvicorn 部署

## 📄 许可证

MIT License

## 👥 贡献

欢迎提交Issue和Pull Request

## 📞 联系方式

如有问题，请提交Issue或联系维护者

