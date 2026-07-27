"""
测试脚本 - 测试所有分析功能
"""
import requests
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"
TEST_DATA_DIR = Path("data/input")


def test_health():
    """测试健康检查"""
    print("\n" + "="*60)
    print("测试健康检查...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    return response.status_code == 200


def test_process_data():
    """测试数据处理"""
    print("\n" + "="*60)
    print("测试数据处理...")

    test_file = TEST_DATA_DIR / "test.csv"
    if not test_file.exists():
        print(f"测试文件不存在: {test_file}")
        return False

    with open(test_file, "rb") as f:
        files = {"file": ("test.csv", f, "text/csv")}
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/v1/data/process", files=files)
        elapsed = time.time() - start_time

    print(f"状态码: {response.status_code}")
    print(f"耗时: {elapsed:.2f}秒")

    if response.status_code == 200:
        data = response.json()
        print(f"成功: {data['success']}")
        print(f"消息: {data['message']}")
        print(f"记录数: {data['data']['total_records']}")
        return True
    else:
        print(f"错误: {response.text}")
        return False


def test_hotspot_analysis():
    """测试热点分析"""
    print("\n" + "="*60)
    print("测试热点分析...")

    test_file = TEST_DATA_DIR / "test.csv"
    if not test_file.exists():
        print(f"测试文件不存在: {test_file}")
        return False

    with open(test_file, "rb") as f:
        files = {"file": ("test.csv", f, "text/csv")}
        params = {
            "min_cluster_size": 50,
            "min_samples": 10,
            "resolution": 8
        }
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/v1/analysis/hotspot",
            files=files,
            params=params
        )
        elapsed = time.time() - start_time

    print(f"状态码: {response.status_code}")
    print(f"耗时: {elapsed:.2f}秒")

    if response.status_code == 200:
        data = response.json()
        print(f"成功: {data['success']}")
        print(f"消息: {data['message']}")
        print(f"热点数: {data['data']['hotspot_points']}")
        return True
    else:
        print(f"错误: {response.text}")
        return False


def test_target_analysis():
    """测试目标分析"""
    print("\n" + "="*60)
    print("测试目标分析...")

    test_file = TEST_DATA_DIR / "test.csv"
    if not test_file.exists():
        print(f"测试文件不存在: {test_file}")
        return False

    with open(test_file, "rb") as f:
        files = {"file": ("test.csv", f, "text/csv")}
        params = {
            "eps_km": 1.0,
            "min_samples": 3
        }
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/v1/analysis/target",
            files=files,
            params=params
        )
        elapsed = time.time() - start_time

    print(f"状态码: {response.status_code}")
    print(f"耗时: {elapsed:.2f}秒")

    if response.status_code == 200:
        data = response.json()
        print(f"成功: {data['success']}")
        print(f"消息: {data['message']}")
        print(f"活动区域数: {data['data']['activate_areas_count']}")
        return True
    else:
        print(f"错误: {response.text}")
        return False


def test_clustering_analysis():
    """测试编群分析"""
    print("\n" + "="*60)
    print("测试编群分析...")

    test_file = TEST_DATA_DIR / "test_group.csv"
    if not test_file.exists():
        print(f"测试文件不存在: {test_file}")
        return False

    with open(test_file, "rb") as f:
        files = {"file": ("test_group.csv", f, "text/csv")}
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/v1/analysis/clustering",
            files=files
        )
        elapsed = time.time() - start_time

    print(f"状态码: {response.status_code}")
    print(f"耗时: {elapsed:.2f}秒")

    if response.status_code == 200:
        data = response.json()
        print(f"成功: {data['success']}")
        print(f"消息: {data['message']}")
        print(f"准确率: {data['data']['accuracy']:.4f}")
        print(f"聚类数: {data['data']['cluster_count']}")
        return True
    else:
        print(f"错误: {response.text}")
        return False


def main():
    """运行所有测试"""
    print("="*60)
    print("AIS数据分析系统 - 功能测试")
    print("="*60)

    tests = [
        ("健康检查", test_health),
        ("数据处理", test_process_data),
        ("热点分析", test_hotspot_analysis),
        ("目标分析", test_target_analysis),
        ("编群分析", test_clustering_analysis),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n测试失败: {e}")
            results.append((name, False))

        time.sleep(1)  # 间隔1秒

    # 打印测试总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{name}: {status}")

    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\n通过率: {passed}/{total} ({100*passed/total:.1f}%)")


if __name__ == "__main__":
    main()
