"""
快速验证异步API所有功能
"""
import requests

BASE_URL = "http://localhost:8000"

def quick_test():
    print("快速验证异步API...")

    # 测试聚类分析
    print("\n[测试] 聚类分析")
    try:
        with open("test.csv", "rb") as f:
            response = requests.post(
                f"{BASE_URL}/api/v1/analysis/clustering",
                files={"file": f},
                timeout=60
            )

        if response.status_code == 200:
            data = response.json()
            print(f"✓ 聚类分析成功")
            print(f"  数据: {data.get('data', {})}")
        else:
            print(f"✗ 失败: {response.status_code}")
            print(f"  错误: {response.text[:200]}")
    except Exception as e:
        print(f"✗ 异常: {e}")

if __name__ == '__main__':
    quick_test()
