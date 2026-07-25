"""
异步API并发测试工具
同时发起多个请求，测试并发处理能力
"""
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://localhost:8000"

def test_concurrent_requests():
    """测试并发请求"""
    print("="*70)
    print("异步API并发测试")
    print("="*70)

    # 检查服务器
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"\n[✓] 服务器运行正常")
            print(f"    版本: {data.get('version', 'unknown')}")
            print(f"    并发支持: {data.get('concurrent', False)}")
        else:
            print("[✗] 服务器响应异常")
            return
    except Exception as e:
        print(f"[✗] 无法连接服务器: {e}")
        print("\n请先启动异步服务器:")
        print("  uvicorn api_server_async:app --host 0.0.0.0 --port 8000")
        return

    print("\n" + "="*70)
    print("测试场景: 同时发起3个不同的分析请求")
    print("="*70)

    def call_api(name, endpoint, params=""):
        """调用API并记录时间"""
        url = f"{BASE_URL}{endpoint}"
        if params:
            url += f"?{params}"

        start_time = time.time()
        thread_id = threading.current_thread().name

        print(f"\n[{thread_id}] 开始请求: {name}")

        try:
            with open("test.csv", "rb") as f:
                response = requests.post(url, files={"file": f}, timeout=120)

            elapsed = time.time() - start_time

            if response.status_code == 200:
                print(f"[{thread_id}] ✓ {name} 完成 - 耗时: {elapsed:.2f}秒")
                return {
                    "name": name,
                    "success": True,
                    "time": elapsed,
                    "thread": thread_id
                }
            else:
                print(f"[{thread_id}] ✗ {name} 失败")
                return {
                    "name": name,
                    "success": False,
                    "time": elapsed,
                    "thread": thread_id
                }

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[{thread_id}] ✗ {name} 异常: {e}")
            return {
                "name": name,
                "success": False,
                "time": elapsed,
                "thread": thread_id,
                "error": str(e)
            }

    # 定义要并发执行的任务
    tasks = [
        ("数据处理", "/api/v1/data/process", ""),
        ("热点分析", "/api/v1/analysis/hotspot", "min_cluster_size=10&min_samples=5&resolution=8"),
        ("目标分析", "/api/v1/analysis/target", "eps_km=1.0&min_samples=3"),
    ]

    print("\n启动并发请求...\n")
    overall_start = time.time()

    # 使用线程池并发执行
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(call_api, name, endpoint, params)
            for name, endpoint, params in tasks
        ]

        results = []
        for future in as_completed(futures):
            results.append(future.result())

    overall_elapsed = time.time() - overall_start

    # 打印结果
    print("\n" + "="*70)
    print("测试结果")
    print("="*70)

    success_count = sum(1 for r in results if r['success'])

    print(f"\n总耗时: {overall_elapsed:.2f}秒")
    print(f"成功: {success_count}/{len(results)}")

    print("\n各任务详情:")
    for result in sorted(results, key=lambda x: x['time']):
        status = "✓" if result['success'] else "✗"
        print(f"  {status} {result['name']}: {result['time']:.2f}秒 (线程: {result['thread']})")

    # 计算并发效率
    total_sequential_time = sum(r['time'] for r in results)
    speedup = total_sequential_time / overall_elapsed if overall_elapsed > 0 else 0

    print(f"\n如果串行执行预计需要: {total_sequential_time:.2f}秒")
    print(f"实际并发执行耗时: {overall_elapsed:.2f}秒")
    print(f"加速比: {speedup:.2f}x")

    if speedup > 1.5:
        print("\n[✓] 并发性能优秀！")
    elif speedup > 1.0:
        print("\n[✓] 具有并发能力")
    else:
        print("\n[!] 并发效果不明显")

    print("\n" + "="*70)


def test_sequential_vs_concurrent():
    """对比串行和并发性能"""
    print("\n" + "="*70)
    print("串行 vs 并发性能对比")
    print("="*70)

    tasks = [
        ("数据处理", "/api/v1/data/process", ""),
        ("聚类分析", "/api/v1/analysis/clustering", ""),
    ]

    # 串行测试
    print("\n[测试1] 串行执行...")
    sequential_start = time.time()

    for name, endpoint, params in tasks:
        url = f"{BASE_URL}{endpoint}"
        if params:
            url += f"?{params}"

        with open("test.csv", "rb") as f:
            response = requests.post(url, files={"file": f}, timeout=120)

        print(f"  完成: {name}")

    sequential_time = time.time() - sequential_start
    print(f"串行总耗时: {sequential_time:.2f}秒")

    # 并发测试
    print("\n[测试2] 并发执行...")
    concurrent_start = time.time()

    def call_api_simple(name, endpoint, params):
        url = f"{BASE_URL}{endpoint}"
        if params:
            url += f"?{params}"
        with open("test.csv", "rb") as f:
            response = requests.post(url, files={"file": f}, timeout=120)
        print(f"  完成: {name}")
        return response.status_code == 200

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(call_api_simple, name, endpoint, params)
            for name, endpoint, params in tasks
        ]
        list(as_completed(futures))

    concurrent_time = time.time() - concurrent_start
    print(f"并发总耗时: {concurrent_time:.2f}秒")

    # 对比
    improvement = ((sequential_time - concurrent_time) / sequential_time * 100)
    print(f"\n性能提升: {improvement:.1f}%")
    print(f"加速比: {sequential_time / concurrent_time:.2f}x")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        test_sequential_vs_concurrent()
    else:
        test_concurrent_requests()
