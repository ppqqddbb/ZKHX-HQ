@echo off
chcp 65001 >nul
echo ============================================================
echo AIS数据分析系统 - 异步版本启动
echo ============================================================
echo.
echo [特性] 支持多个请求并发处理
echo [架构] ProcessPoolExecutor + asyncio
echo [性能] 2-3倍加速比
echo.
echo 正在启动异步服务器...
echo.

uvicorn api_server_async:app --host 0.0.0.0 --port 8000 --reload

pause
