"""
快速验证脚本 - 检查所有模块是否可以正常导入
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试所有关键模块的导入"""
    print("=" * 60)
    print("AIS数据分析系统 - 模块导入测试")
    print("=" * 60)

    tests = []

    # 测试API模块
    try:
        from api.models import AnalysisResponse
        from api.services import AnalysisService
        tests.append(("API模块", True, None))
        print("[OK] API模块导入成功")
    except Exception as e:
        tests.append(("API模块", False, str(e)))
        print(f"[FAIL] API模块导入失败: {e}")

    # 测试核心模块
    try:
        from core import (
            get_logger, DataLoader, DataProcessor,
            HotspotAnalyzer, TargetAnalyzer, FrequentAnalyzer,
            ClusteringAnalyzer, AssociationAnalyzer,
            AISBaseException
        )
        tests.append(("核心模块", True, None))
        print("[OK] 核心模块导入成功")
    except Exception as e:
        tests.append(("核心模块", False, str(e)))
        print(f"[FAIL] 核心模块导入失败: {e}")

    # 测试FastAPI
    try:
        from fastapi import FastAPI
        tests.append(("FastAPI", True, None))
        print("[OK] FastAPI导入成功")
    except Exception as e:
        tests.append(("FastAPI", False, str(e)))
        print(f"[FAIL] FastAPI导入失败: {e}")

    # 测试数据处理库
    try:
        import polars as pl
        import pandas as pd
        import numpy as np
        tests.append(("数据处理库", True, None))
        print("[OK] 数据处理库导入成功")
    except Exception as e:
        tests.append(("数据处理库", False, str(e)))
        print(f"[FAIL] 数据处理库导入失败: {e}")

    # 测试机器学习库
    try:
        from sklearn.cluster import DBSCAN
        import hdbscan
        tests.append(("机器学习库", True, None))
        print("[OK] 机器学习库导入成功")
    except Exception as e:
        tests.append(("机器学习库", False, str(e)))
        print(f"[FAIL] 机器学习库导入失败: {e}")

    # 测试地理空间库
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        import h3
        tests.append(("地理空间库", True, None))
        print("[OK] 地理空间库导入成功")
    except Exception as e:
        tests.append(("地理空间库", False, str(e)))
        print(f"[FAIL] 地理空间库导入失败: {e}")

    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for _, success, _ in tests if success)
    total = len(tests)

    for name, success, error in tests:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{name}: {status}")
        if error:
            print(f"  错误: {error}")

    print(f"\n通过率: {passed}/{total} ({100*passed/total:.1f}%)")

    if passed == total:
        print("\n[OK] 所有模块导入成功！系统可以正常运行。")
        return 0
    else:
        print("\n[FAIL] 部分模块导入失败！请检查依赖安装。")
        print("\n建议运行: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(test_imports())
