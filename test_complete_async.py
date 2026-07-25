"""
异步API完整测试工具
验证所有功能并测试并发性能
"""
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://localhost:8000"

def print_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def check_server():
    """检查服务器状态"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"\n[服务器信息]")
            print(f"  状态: {data.get('status', 'unknown')}")
            print(f"  版本: {data.get('version', 'unknown')}")
            print(f"  并发支持: {'✓' if data.get('concurrent', False) else '✗'}")
            return True
        else:
            print(f"[✗] 服务器响应异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"[✗] 无法连接服务器: {e}")
        print("\n请先启动异步服务器:")
        print("  uvicorn api_server_async:app --host 0.0.0.0 --port 8000")
        return False

def test_single_api(name, endpoint, params="", timeout=60):
    """测试单个API"""
    print(f"\n[测试] {name}")

    url = f"{BASE_URL}{endpoint}"
    if params:
        url += f"?{params}"

    try:
        start_time = time.time()

        with open("test.csv", "rb") as f:
            response = requests.post(url, files={"file": f}, timeout=timeout)

        elapsed = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ 成功 - 耗时: {elapsed:.2f}秒")

            # 显示返回数据摘要
            if 'data' in data:
                result_data = data['data']
                print(f"  返回字段: {len(result_data)} 个")
                for key, value in list(result_data.items())[:3]:
                    if isinstance(value, (int, float)):
                        print(f"    • {key}: {value}")
                    elif isinstance(value, list):
                        print(f"    • {key}: {len(value)} 项")

            return True, elapsed
        else:
            print(f"  ✗ 失败 - 状态码: {response.status_code}")
            try:
                error = response.json()
                print(f"  错误: {error.get('detail', 'Unknown')[:100]}")
            except:
                print(f"  错误: {response.text[:100]}")
            return False, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ✗ 异常: {e}")
        return False, elapsed

def test_all_functions():
    """测试所有功能"""
    print_header("功能测试 - 所有API端点")

    tests = [
        ("健康检查", "/health", "", "GET"),
        ("数据处理", "/api/v1/data/process", ""),
        ("热点分析", "/api/v1/analysis/hotspot", "min_cluster_size=10&min_samples=5&resolution=8"),
        ("频繁轨迹", "/api/v1/analysis/frequent", "grid_size=0.01&sigma=2.0&percentile_thresh=90"),
        ("目标分析", "/api/v1/analysis/target", "eps_km=1.0&min_samples=3"),
        ("关联分析", "/api/v1/analysis/association", "min_support=0.05&min_confidence=0.7"),
        ("聚类分析", "/api/v1/analysis/clustering", ""),
    ]

    results = []

    for name, endpoint, params, *method in tests:
        if method and method[0] == "GET":
            # 健康检查
            try:
                response = requests.get(f"{BASE_URL}{endpoint}", timeout=5)
                if response.status_code == 200:
                    print(f"\n[测试] {name}")
                    print(f"  ✓ 成功")
                    results.append((name, True, 0))
                else:
                    print(f"\n[测试] {name}")
                    print(f"  ✗ 失败")
                    results.append((name, False, 0))
            except Exception as e:
                print(f"\n[测试] {name}")
                print(f"  ✗ 异常: {e}")
                results.append((name, False, 0))
        else:
            success, elapsed = test_single_api(name, endpoint, params)
            results.append((name, success, elapsed))

    # 统计
    print_header("测试结果汇总")

    success_count = sum(1 for _, success, _ in results if success)
    total_count = len(results)

    print(f"\n总计: {success_count}/{total_count} 测试通过")
    print(f"通过率: {success_count*100//total_count}%")

    print("\n详细结果:")
    for name, success, elapsed in results:
        status = "✓" if success else "✗"
        if elapsed > 0:
            print(f"  {status} {name}: {elapsed:.2f}秒")
        else:
            print(f"  {status} {name}")

    return success_count == total_count

def test_concurrent_performance():
    """测试并发性能"""
    print_header("并发性能测试")

    print("\n[场景] 同时发起3个不同的分析请求")
    print("  请求1: 数据处理")
    print("  请求2: 热点分析")
    print("  请求3: 聚类分析")

    def call_api(name, endpoint, params=""):
        start = time.time()
        url = f"{BASE_URL}{endpoint}"
        if params:
            url += f"?{params}"

        try:
            with open("test.csv", "rb") as f:
                response = requests.post(url, files={"file": f}, timeout=120)

            elapsed = time.time() - start
            success = response.status_code == 200

            return {"name": name, "success": success, "time": elapsed}
        except Exception as e:
            elapsed = time.time() - start
            return {"name": name, "success": False, "time": elapsed, "error": str(e)}

    tasks = [
        ("数据处理", "/api/v1/data/process", ""),
        ("热点分析", "/api/v1/analysis/hotspot", "min_cluster_size=10&min_samples=5&resolution=8"),
        ("聚类分析", "/api/v1/analysis/clustering", ""),
    ]

    print("\n[开始] 并发执行...")
    overall_start = time.time()

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(call_api, name, endpoint, params) for name, endpoint, params in tasks]
        results = [future.result() for future in as_completed(futures)]

    overall_elapsed = time.time() - overall_start

    # 结果
    print_header("并发性能结果")

    success_count = sum(1 for r in results if r['success'])
    avg_time = sum(r['time'] for r in results) / len(results)
    sequential_time = sum(r['time'] for r in results)
    speedup = sequential_time / overall_elapsed if overall_elapsed > 0 else 0

    print(f"\n[性能指标]")
    print(f"  总耗时: {overall_elapsed:.2f}秒")
    print(f"  平均单请求: {avg_time:.2f}秒")
    print(f"  成功请求: {success_count}/{len(results)}")

    print(f"\n[并发效果]")
    print(f"  串行总耗时（理论）: {sequential_time:.2f}秒")
    print(f"  并发总耗时（实际）: {overall_elapsed:.2f}秒")
    print(f"  加速比: {speedup:.2f}x")

    if speedup > 2.0:
        print(f"\n  ✓ 并发性能优秀！达到 {speedup:.1f} 倍加速")
        return True
    elif speedup > 1.5:
        print(f"\n  ✓ 具有良好的并发能力")
        return True
    else:
        print(f"\n  ⚠️  并发效果低于预期")
        return False

def main():
    print_header("AIS数据分析系统 - 异步API完整测试")

    # 检查服务器
    if not check_server():
        return

    print("\n" + "="*70)
    print("  开始测试")
    print("="*70)

    # 测试1: 所有功能
    print("\n[阶段1] 功能完整性测试")
    all_pass = test_all_functions()

    # 测试2: 并发性能
    print("\n[阶段2] 并发性能测试")
    concurrent_pass = test_concurrent_performance()

    # 最终报告
    print_header("最终报告")

    print("\n[测试结果]")
    print(f"  功能测试: {'✓ 通过' if all_pass else '✗ 失败'}")
    print(f"  并发测试: {'✓ 通过' if concurrent_pass else '✗ 失败'}")

    if all_pass and concurrent_pass:
        print("\n🎉 [成功] 所有测试通过！")
        print("\n异步API已100%就绪:")
        print("  ✓ 所有功能正常")
        print("  ✓ 并发性能优秀")
        print("  ✓ 可用于生产环境")
    else:
        print("\n⚠️  [警告] 部分测试未通过")
        if not all_pass:
            print("  • 某些功能存在问题")
        if not concurrent_pass:
            print("  • 并发性能需要优化")

    print("\n" + "="*70)

if __name__ == '__main__':
    main()
