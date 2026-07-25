"""
同步版 vs 异步版性能对比演示
直观展示并发处理的优势
"""
import requests
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

def print_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def check_server_version():
    """检查服务器版本"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            version = data.get('version', 'unknown')
            concurrent = data.get('concurrent', False)

            print(f"\n[服务器信息]")
            print(f"  版本: {version}")
            print(f"  并发支持: {'✓ 是' if concurrent else '✗ 否'}")

            if concurrent:
                print("\n  ⭐ 正在使用异步版服务器")
            else:
                print("\n  ⚠️  正在使用同步版服务器")
                print("     建议切换到异步版以获得更好性能:")
                print("     uvicorn api_server_async:app --host 0.0.0.0 --port 8000")

            return concurrent
        else:
            print("[✗] 服务器响应异常")
            return None
    except Exception as e:
        print(f"[✗] 无法连接服务器: {e}")
        print("\n请先启动服务器:")
        print("  异步版: uvicorn api_server_async:app --host 0.0.0.0 --port 8000")
        print("  同步版: uvicorn api_server:app --host 0.0.0.0 --port 8000")
        return None

def simulate_concurrent_scenario():
    """模拟并发场景"""
    print_header("场景模拟：3个用户同时发起请求")

    print("\n[场景说明]")
    print("  用户A: 请求数据处理")
    print("  用户B: 请求热点分析")
    print("  用户C: 请求聚类分析")
    print("\n  每个请求预计耗时: 2-3秒")

    is_async = check_server_version()

    if is_async is None:
        return

    if is_async:
        print("\n[异步版预期]")
        print("  ✓ 三个请求同时处理")
        print("  ✓ 总耗时 ≈ 3秒（最慢的单个请求）")
    else:
        print("\n[同步版预期]")
        print("  ✗ 请求排队处理")
        print("  ✗ 总耗时 ≈ 9秒（所有请求累加）")

    input("\n按回车键开始测试...")

    print("\n[开始测试]")
    start_time = time.time()

    import threading

    results = []

    def call_api(user, name, endpoint):
        req_start = time.time()
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {user} 发起请求: {name}")

        try:
            with open("test.csv", "rb") as f:
                response = requests.post(f"{BASE_URL}{endpoint}", files={"file": f}, timeout=120)

            req_time = time.time() - req_start

            if response.status_code == 200:
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] {user} 请求完成: {name} ({req_time:.2f}秒)")
                results.append({"user": user, "success": True, "time": req_time})
            else:
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] {user} 请求失败: {name}")
                results.append({"user": user, "success": False, "time": req_time})
        except Exception as e:
            req_time = time.time() - req_start
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] {user} 请求异常: {e}")
            results.append({"user": user, "success": False, "time": req_time})

    # 创建线程模拟并发
    threads = [
        threading.Thread(target=call_api, args=("用户A", "数据处理", "/api/v1/data/process")),
        threading.Thread(target=call_api, args=("用户B", "热点分析", "/api/v1/analysis/hotspot?min_cluster_size=10&min_samples=5&resolution=8")),
        threading.Thread(target=call_api, args=("用户C", "聚类分析", "/api/v1/analysis/clustering")),
    ]

    # 同时启动所有线程
    for t in threads:
        t.start()

    # 等待所有线程完成
    for t in threads:
        t.join()

    total_time = time.time() - start_time

    # 显示结果
    print_header("测试结果")

    success_count = sum(1 for r in results if r['success'])
    avg_time = sum(r['time'] for r in results) / len(results)

    print(f"\n[性能指标]")
    print(f"  总耗时: {total_time:.2f}秒")
    print(f"  平均单请求: {avg_time:.2f}秒")
    print(f"  成功请求: {success_count}/{len(results)}")

    # 计算理论值
    sequential_time = sum(r['time'] for r in results)
    speedup = sequential_time / total_time if total_time > 0 else 0

    print(f"\n[并发效果]")
    print(f"  串行总耗时（理论）: {sequential_time:.2f}秒")
    print(f"  并发总耗时（实际）: {total_time:.2f}秒")
    print(f"  加速比: {speedup:.2f}x")

    if is_async:
        if speedup > 2.0:
            print(f"\n  ✓ 并发性能优秀！")
            print(f"    异步版实现了 {speedup:.1f} 倍加速")
        elif speedup > 1.5:
            print(f"\n  ✓ 具有良好的并发能力")
        else:
            print(f"\n  ⚠️  并发效果低于预期")
    else:
        print(f"\n  ⚠️  同步版无并发能力")
        print(f"     切换到异步版可获得 ~3倍性能提升")

    print("\n" + "="*70)

def main():
    print_header("AIS数据分析系统 - 并发性能演示")

    print("\n[演示说明]")
    print("  本工具将模拟3个用户同时发起不同的分析请求")
    print("  通过对比总耗时，直观展示异步版的并发优势")

    simulate_concurrent_scenario()

    print("\n[总结]")
    print("  异步版特点:")
    print("    ✓ 多个请求真正并发执行")
    print("    ✓ 充分利用多核CPU")
    print("    ✓ 2-3倍性能提升")
    print("    ✓ 更好的用户体验")

    print("\n  启动异步版:")
    print("    uvicorn api_server_async:app --host 0.0.0.0 --port 8000")

    print("\n" + "="*70)

if __name__ == '__main__':
    main()
