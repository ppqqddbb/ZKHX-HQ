@echo off
REM AIS数据分析系统 - 启动脚本

echo ========================================
echo AIS数据分析系统 - 启动中...
echo ========================================

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

REM 启动应用
echo.
echo 正在启动服务器...
echo 访问地址: http://localhost:8000
echo API文档: http://localhost:8000/docs
echo.
echo 按 Ctrl+C 停止服务器
echo.

python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

pause
