# 快速参考指南

## 🚀 5分钟快速上手

### 1. 启动服务
```bash
python app.py
```

### 2. 测试健康检查
```bash
curl http://localhost:8000/health
```

### 3. 数据处理示例
```bash
curl -X POST "http://localhost:8000/api/v1/data/process" \
  -F "file=@data/input/test.csv"
```

## 📍 常用命令

| 操作 | 命令 |
|------|------|
| 启动服务 | `python app.py` |
| 运行测试 | `python tests/test_api.py` |
| 查看日志 | `type logs\app.log` (Windows) |
| 验证模块 | `python scripts/verify.py` |

## 🔗 常用URL

| 用途 | URL |
|------|-----|
| API文档 | http://localhost:8000/docs |
| 健康检查 | http://localhost:8000/health |
| ReDoc | http://localhost:8000/redoc |

## 📊 数据文件

| 文件 | 用途 | 位置 |
|------|------|------|
| test.csv | 通用测试 | data/input/test.csv |
| test_group.csv | 编群分析 | data/input/test_group.csv |

## 🎯 7大功能速查

1. **数据处理**: `/api/v1/data/process` - 数据加载和预处理
2. **热点分析**: `/api/v1/analysis/hotspot` - 识别活动热点区域
3. **目标分析**: `/api/v1/analysis/target` - 船舶行为多维分析
4. **频繁轨迹**: `/api/v1/analysis/frequent` - 常用航道识别
5. **编群分析**: `/api/v1/analysis/clustering` - 轨迹相似度聚类
6. **关联分析**: `/api/v1/analysis/association` - 行为模式挖掘
7. **航迹预测**: `/api/v1/prediction/trajectory` - 未来位置预测

## ⚡ 故障排除

| 问题 | 解决方案 |
|------|----------|
| 模块导入错误 | `pip install -r requirements.txt` |
| 端口被占用 | 修改app.py中的端口或关闭占用程序 |
| 测试失败 | 确认服务器已启动 |
| 编群分析报错 | 使用test_group.csv文件 |

## 📚 文档位置

- 快速开始: `README.md`
- 详细指南: `docs/USER_GUIDE.md`
- 重构报告: `docs/REFACTORING_SUMMARY.md`
- 完成总结: `PROJECT_COMPLETE.md`
- 状态报告: `PROJECT_STATUS.txt`

## 💡 提示

- 使用 `/docs` 查看交互式API文档
- 编群分析需要使用 `test_group.csv`
- 查看 `logs/` 目录获取详细日志
- 测试数据仅供功能验证，生产环境请使用实际数据
