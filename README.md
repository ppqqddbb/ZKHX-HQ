# 船舶航迹预测与分析系统 (SXGF)

基于 PatchTST 模型的船舶航迹预测系统，提供 AIS 数据处理、航迹预测、热点分析、聚类分析等功能，并通过 FastAPI 提供 RESTful API 接口。

## 项目结构

```
SXGF/
├── api/                    # API 服务模块
│   ├── api_server.py       # FastAPI 主服务
│   └── api_database.py     # 数据库相关 API
├── code/                   # 核心代码模块
│   ├── analysis/           # 数据分析模块
│   │   ├── association.py  # 关联分析
│   │   ├── clustering.py   # 聚类分析
│   │   ├── frequent.py     # 频繁模式分析
│   │   ├── hotspot.py      # 热点分析
│   │   └── target.py       # 目标分析
│   ├── data/               # 数据处理模块
│   │   ├── loader.py       # 数据加载器
│   │   ├── filter.py       # 数据过滤器
│   │   ├── process.py      # 数据处理器
│   │   └── save.py         # 数据保存器
│   ├── model/              # 模型模块
│   │   ├── prediction.py   # 预测逻辑
│   │   └── pretraining.py  # PatchTST 模型
│   └── sqlite/             # SQLite 数据库模块
│       ├── database.py     # 数据库操作
│       └── models.py       # 数据模型
├── config/                 # 配置模块
│   └── settings.py         # 全局配置参数
├── dataset/                # 数据集目录
├── static/                 # 静态资源 (Swagger UI)
├── main.py                 # 训练/预测主入口
├── requirements.txt        # Python 依赖
├── setup.py                # 安装脚本
└── start_server.bat        # 服务启动脚本
```

## 功能特性

### 航迹预测
- 基于 **PatchTST** 时序模型的船舶航迹预测
- 支持多特征输入：位置、速度、航向、时间特征等
- 模型评估指标：ADE、FDE、最大误差、R²

### 数据分析
- **热点分析**: 识别船舶活动密集区域
- **聚类分析**: 船舶轨迹聚类与模式识别
- **关联分析**: 船舶间关联关系挖掘
- **频繁模式分析**: 常见航路模式提取
- **目标分析**: 特定目标船舶分析

### API 服务
- 基于 **FastAPI** 构建的 RESTful API
- 支持文件上传 (AIS 数据 CSV)
- Swagger UI 文档界面
- CORS 跨域支持

## 技术栈

- **深度学习**: PyTorch, Transformers
- **数据处理**: Polars, Pandas, NumPy
- **机器学习**: scikit-learn, tslearn
- **空间分析**: GeoPandas, Shapely, H3, PyProj
- **API 框架**: FastAPI, Uvicorn
- **数据库**: SQLite, SQLAlchemy
- **可视化**: Matplotlib

## 环境要求

- Python 3.10+
- CUDA (可选，用于 GPU 加速)

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练/预测

编辑 `main.py` 中的 `is_training` 参数：

```python
is_training = True   # 执行训练流程
is_training = False  # 仅执行预测流程
```

运行主程序：

```bash
python main.py
```

### 启动 API 服务

```bash
# Windows
start_server.bat

# 或手动启动
python -m api.api_server
python -m api.api_database
```

API 文档访问：`http://localhost:8000/docs`

## 配置参数

主要配置在 `config/settings.py` 中：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| seq_len | 96 | 输入序列长度 |
| pred_len | 60 | 预测序列长度 |
| batch_size | 16 | 批次大小 |
| epochs | 100 | 训练轮数 |
| lr | 0.0005 | 学习率 |
| d_model | 256 | 模型维度 |
| nhead | 8 | 注意力头数 |
| freq | 5m | 数据采样频率 |

## 数据格式

输入数据为 CSV 格式，包含以下字段：

| 字段 | 说明 |
|------|------|
| MMSI | 船舶识别码 |
| Timestamp | 时间戳 |
| Latitude | 纬度 |
| Longitude | 经度 |
| SOG | 航速 |
| COG | 航向 |
| Heading | 船首向 |
| ShipType | 船型 |
| Status | 状态 |
| Length | 船长 |
| Width | 船宽 |
| Draft | 吃水深度 |
| Cargo | 货物类型 |

## 输出说明

- **模型权重**: `best_patchtst_model.pt`
- **标准化参数**: `dataset/MeanStd.joblib`
- **数据库**: `dataset/sxgf.db`
- **日志**: `log/` 目录
