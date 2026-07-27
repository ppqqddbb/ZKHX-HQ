@echo off
REM AIS数据分析系统 - 测试脚本

echo ========================================
echo AIS数据分析系统 - 运行测试
echo ========================================

REM 检查服务器是否运行
echo 检查服务器状态...
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo.
    echo 错误: 服务器未运行
    echo 请先运行 start.bat 启动服务器
    echo.
    pause
    exit /b 1
)

echo 服务器运行正常
echo.
echo 开始运行测试...
echo.

python tests/test_api.py

pause
