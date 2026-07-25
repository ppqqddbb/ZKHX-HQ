@echo off
chcp 65001 >nul
cls
echo ============================================================
echo AIS数据分析系统 - 异步版本测试套件
echo ============================================================
echo.
echo 请选择要执行的操作:
echo.
echo [1] 启动异步服务器
echo [2] 测试并发性能
echo [3] 对比串行vs并发
echo [4] 并发场景演示
echo [5] 查看服务器状态
echo [0] 退出
echo.
echo ============================================================
set /p choice="请输入选项 (0-5): "

if "%choice%"=="1" goto start_server
if "%choice%"=="2" goto test_concurrent
if "%choice%"=="3" goto compare_performance
if "%choice%"=="4" goto demo_scenario
if "%choice%"=="5" goto check_status
if "%choice%"=="0" goto end

echo 无效选项，请重新运行
pause
exit /b

:start_server
echo.
echo [启动异步服务器]
echo 按 Ctrl+C 停止服务器
echo.
uvicorn api_server_async:app --host 0.0.0.0 --port 8000 --reload
goto end

:test_concurrent
echo.
echo [测试并发性能]
echo 将同时发起3个不同的分析请求...
echo.
python test_async.py
pause
goto end

:compare_performance
echo.
echo [串行vs并发对比]
echo 对比相同任务在串行和并发模式下的性能...
echo.
python test_async.py compare
pause
goto end

:demo_scenario
echo.
echo [并发场景演示]
echo 模拟多用户同时访问的真实场景...
echo.
python demo_async.py
pause
goto end

:check_status
echo.
echo [服务器状态检查]
curl -s http://localhost:8000/health | python -m json.tool
echo.
pause
goto end

:end
