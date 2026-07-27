# 内存优化说明

## 问题诊断

### 内存急剧上升的常见原因

1. **进程池累积** - 多个子进程同时存在
2. **数据重复加载** - 每次分析都完整加载数据
3. **中间结果未释放** - pandas/polars DataFrame占用内存
4. **深度学习模型** - TensorFlow/PyTorch模型加载到内存
5. **文件句柄未关闭** - 临时文件累积

## 已实施的优化措施

### 1. 减少进程池大小
```python
# 优化前: 4个工作进程
max_workers = 4

# 优化后: 2个工作进程
max_workers = 2
max_tasks_per_child = 10  # 每个子进程处理10个任务后重启
```

**效果**: 减少50%的基础内存占用

### 2. 添加垃圾回收
```python
import gc

# 关键位置强制垃圾回收
del large_dataframe
gc.collect()
```

**效果**: 及时释放不再使用的对象

### 3. 数据采样
```python
# 大数据集自动采样
if total_records > 1000:
    df_sampled = df_collected.sample(n=1000)
```

**效果**: 限制内存使用上限

### 4. 及时清理
```python
# 保存后立即删除
save_result_to_file("data", df)
del df
gc.collect()
```

## 内存监控

### 查看当前内存使用

**Windows任务管理器**:
1. 按 `Ctrl+Shift+Esc` 打开任务管理器
2. 找到 `python.exe` 进程
3. 查看内存列

**Python脚本监控**:
```python
import psutil
import os

process = psutil.Process(os.getpid())
memory_info = process.memory_info()
print(f"当前内存使用: {memory_info.rss / 1024 / 1024:.2f} MB")
```

### 正常内存使用范围

| 状态 | 预期内存 | 说明 |
|------|---------|------|
| 空闲 | 200-500 MB | 基础运行 + 2个子进程 |
| 数据处理 | 500-800 MB | 小数据集分析 |
| 关联分析 | 800-1500 MB | 包含mlxtend算法 |
| 深度学习 | 1500-3000 MB | PatchTST模型加载 |

### 异常内存使用

如果内存持续增长超过以下情况，说明有内存泄漏：

- **超过4GB** - 严重问题
- **每次请求增加100MB以上** - 有泄漏
- **空闲时不回落** - 垃圾回收问题

## 进一步优化建议

### 1. 重启策略

**定期重启服务器**（推荐用于生产环境）:
```bash
# 使用Supervisor或systemd自动重启
# 或设置定时任务每天凌晨重启
```

### 2. 限制并发

**限制同时处理的请求数**:
```python
# 在app.py中添加
from fastapi_limiter import FastAPILimiter

@app.on_event("startup")
async def startup():
    await FastAPILimiter.init(redis)
```

### 3. 使用更轻量的算法

**关联分析优化**:
```python
# 降低支持度阈值会减少结果数量
min_support = 0.05  # 从0.01提高到0.05
```

### 4. 流式处理

**对于大文件使用流式读取**:
```python
# 分块读取大文件
for chunk in pl.read_csv_batched(file_path, batch_size=10000):
    process_chunk(chunk)
```

## 问题排查步骤

### 步骤1: 确认基础内存
```bash
# 启动服务器后立即检查
python app.py
# 等待5秒
# 查看任务管理器中的内存
```

**预期**: 200-500 MB

### 步骤2: 单次请求测试
```python
# 发送一个简单的数据处理请求
import requests
with open("data/input/test.csv", "rb") as f:
    response = requests.post(
        "http://127.0.0.1:8000/api/v1/data/process",
        files={"file": f}
    )

# 观察内存变化
# 等待30秒后再次检查
```

**预期**: 内存增加200-300MB，然后回落

### 步骤3: 重复请求测试
```python
# 连续发送10个请求
for i in range(10):
    # 发送请求
    # 等待完成
    # 记录内存
```

**预期**: 内存波动但不持续增长

### 步骤4: 识别问题功能

依次测试每个功能，找出内存占用最大的：

1. 数据处理 - 通常最轻量
2. 热点分析 - 中等
3. 目标分析 - 中等
4. 频繁轨迹 - 较高
5. 编群分析 - 较高
6. **关联分析** - **最高**（包含mlxtend）
7. **航迹预测** - **最高**（包含深度学习模型）

## 临时解决方案

### 方案1: 减少工作进程
```python
# app.py
max_workers = 1  # 只用1个工作进程
```

### 方案2: 禁用模型预加载
```python
# 航迹预测改为延迟加载
# 不在启动时加载TensorFlow模型
```

### 方案3: 增加swap空间
```bash
# Windows: 增加虚拟内存
# 控制面板 → 系统 → 高级系统设置 → 性能设置 → 高级 → 虚拟内存
```

### 方案4: 使用更多物理内存
- 升级到16GB或32GB RAM
- 关闭其他占用内存的程序

## 监控脚本

创建一个简单的内存监控脚本：

```python
# memory_monitor.py
import psutil
import time

def monitor_memory():
    while True:
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            if 'python' in proc.info['name'].lower():
                mem_mb = proc.info['memory_info'].rss / 1024 / 1024
                print(f"PID {proc.info['pid']}: {mem_mb:.2f} MB")
        
        time.sleep(5)
        print("-" * 50)

if __name__ == "__main__":
    monitor_memory()
```

运行:
```bash
python memory_monitor.py
```

## 生产环境建议

1. **使用容器化部署** - Docker限制内存上限
2. **配置健康检查** - 内存超限自动重启
3. **使用专业监控** - Prometheus + Grafana
4. **设置告警** - 内存超过2GB时通知
5. **定期重启** - 每天凌晨自动重启

## 紧急处理

如果内存已经过高：

```bash
# 立即重启服务器
# 按 Ctrl+C 停止
# 再次启动
python app.py
```

## 联系支持

如果问题持续存在，请提供：
1. 任务管理器截图
2. 日志文件 `logs/` 目录
3. 测试数据大小
4. 复现步骤

---

**更新日期**: 2026-07-27  
**版本**: v3.1.0
