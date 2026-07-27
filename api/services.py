"""
业务服务层 - 封装所有分析功能的业务逻辑
"""
import json
from typing import Dict, Any
from pathlib import Path
from datetime import datetime
import polars as pl

from core import (
    DataLoader, DataProcessor,
    HotspotAnalyzer, FrequentAnalyzer, TargetAnalyzer,
    AssociationAnalyzer, ClusteringAnalyzer,
    PatchTSTManager,
    get_logger
)

logger = get_logger(__name__)

# 输出目录
OUTPUT_DIR = Path("data/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_result_to_file(result_type: str, data: Any, file_format: str = "json") -> str:
    """
    保存分析结果到文件

    Args:
        result_type: 结果类型（用于文件名）
        data: 要保存的数据
        file_format: 文件格式（json或csv）

    Returns:
        保存的文件路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if file_format == "json":
        filename = f"{result_type}_{timestamp}.json"
        filepath = OUTPUT_DIR / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    elif file_format == "csv":
        filename = f"{result_type}_{timestamp}.csv"
        filepath = OUTPUT_DIR / filename

        if isinstance(data, pl.DataFrame):
            data.write_csv(filepath)
        elif hasattr(data, 'to_csv'):
            data.to_csv(filepath, index=False)
        else:
            raise ValueError(f"不支持的数据类型: {type(data)}")

    logger.info(f"结果已保存到: {filepath}")
    return str(filepath)


class AnalysisService:
    """
    分析服务类

    提供所有分析功能的统一接口，在子进程中执行CPU密集型任务
    """

    @staticmethod
    def process_data(file_path: str) -> Dict[str, Any]:
        """
        数据处理任务

        Args:
            file_path: 数据文件路径

        Returns:
            处理结果字典
        """
        logger.info(f"开始数据处理: {file_path}")

        loader = DataLoader(file_path)
        df = loader.run()

        processor = DataProcessor(df=df)
        df_processed = processor.run()

        # 收集结果
        df_collected = df_processed.collect()

        result = {
            "total_records": len(df_collected),
            "columns": df_processed.columns,
            "sample_data": df_collected.head(5).to_dicts()
        }

        # 保存处理后的数据到CSV
        output_file = save_result_to_file("processed_data", df_collected, "csv")
        result["output_file"] = output_file

        # 保存统计信息到JSON
        stats_file = save_result_to_file("data_stats", result, "json")
        result["stats_file"] = stats_file

        logger.info(f"数据处理完成: {result['total_records']} 条记录")
        return result

    @staticmethod
    def analyze_hotspot(
        file_path: str,
        min_cluster_size: int,
        min_samples: int,
        resolution: int
    ) -> Dict[str, Any]:
        """
        热点区域分析任务

        Args:
            file_path: 数据文件路径
            min_cluster_size: 最小聚类大小
            min_samples: 最小样本数
            resolution: H3分辨率

        Returns:
            分析结果字典
        """
        logger.info(f"开始热点分析: {file_path}")

        loader = DataLoader(file_path)
        df = loader.run()

        processor = DataProcessor(df=df)
        df_processed = processor.run()

        analyzer = HotspotAnalyzer(
            df=df_processed,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            resolution=resolution
        )

        df_hotspot, coverage = analyzer.run()

        # 收集结果
        df_hotspot_collected = df_hotspot.collect()

        result = {
            "hotspot_points": len(df_hotspot_collected),
            "coverage_by_mmsi": coverage,
            "total_mmsi": len(coverage),
            "parameters": {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "resolution": resolution
            }
        }

        # 保存热点数据到CSV
        output_file = save_result_to_file("hotspot_points", df_hotspot_collected, "csv")
        result["output_file"] = output_file

        # 保存统计信息到JSON
        stats_file = save_result_to_file("hotspot_stats", result, "json")
        result["stats_file"] = stats_file

        logger.info(f"热点分析完成: {result['hotspot_points']} 个热点")
        return result

    @staticmethod
    def analyze_target(
        file_path: str,
        eps_km: float,
        min_samples: int
    ) -> Dict[str, Any]:
        """
        目标分析任务

        Args:
            file_path: 数据文件路径
            eps_km: DBSCAN聚类半径（公里）
            min_samples: 最小样本数

        Returns:
            分析结果字典
        """
        logger.info(f"开始目标分析: {file_path}")

        loader = DataLoader(file_path)
        df = loader.run()

        processor = DataProcessor(df=df)
        df_processed = processor.run()

        analyzer = TargetAnalyzer(
            df=df_processed,
            eps_km=eps_km,
            min_samples=min_samples,
            zones=None
        )

        activate_areas, dwell_positions, speed, total_distance, zones = analyzer.run()

        # 转换数据类型
        speed_dict = speed.to_dicts() if len(speed) > 0 else []
        distance_dict = total_distance.to_dicts() if len(total_distance) > 0 else []

        if isinstance(dwell_positions, pl.DataFrame):
            dwell_dict = dwell_positions.to_dicts() if len(dwell_positions) > 0 else []
        else:
            dwell_dict = dwell_positions

        result = {
            "activate_areas_count": len(activate_areas),
            "activate_areas": activate_areas[:50],  # 限制返回数量
            "dwell_positions_count": len(dwell_dict),
            "dwell_positions": dwell_dict[:50],
            "speed_statistics": speed_dict,
            "total_distance": distance_dict,
            "zone_coverage": zones,
            "parameters": {
                "eps_km": eps_km,
                "min_samples": min_samples
            }
        }

        # 保存完整结果到JSON
        complete_result = {
            "activate_areas": activate_areas,
            "dwell_positions": dwell_dict,
            "speed_statistics": speed_dict,
            "total_distance": distance_dict,
            "zone_coverage": zones
        }
        output_file = save_result_to_file("target_analysis", complete_result, "json")
        result["output_file"] = output_file

        logger.info(f"目标分析完成: {result['activate_areas_count']} 个活动区域")
        return result

    @staticmethod
    def analyze_frequent(
        file_path: str,
        grid_size: float,
        sigma: float,
        percentile_thresh: int
    ) -> Dict[str, Any]:
        """
        频繁轨迹分析任务

        Args:
            file_path: 数据文件路径
            grid_size: 网格大小
            sigma: 高斯平滑参数
            percentile_thresh: 密度百分位阈值

        Returns:
            分析结果字典
        """
        logger.info(f"开始频繁轨迹分析: {file_path}")

        loader = DataLoader(file_path)
        df = loader.run()

        processor = DataProcessor(df=df)
        df_processed = processor.run()

        analyzer = FrequentAnalyzer(
            df=df_processed,
            grid_size=grid_size,
            sigma=sigma,
            percentile_thresh=percentile_thresh
        )

        # run() 返回 (GeoDataFrame, DataFrame) 元组
        frequent_gdf, frequent_df = analyzer.run()

        # 将几何对象转换为GeoJSON格式
        channels = []
        for geom in frequent_gdf.geometry[:20]:
            # 使用 shapely 的 mapping 或 __geo_interface__
            geom_dict = {
                "type": geom.geom_type,
                "coordinates": list(geom.coords)
            }
            channels.append(geom_dict)

        result = {
            "frequent_channels_count": len(frequent_gdf),
            "channels": channels,
            "parameters": {
                "grid_size": grid_size,
                "sigma": sigma,
                "percentile_thresh": percentile_thresh
            }
        }

        # 保存完整的航道数据到JSON
        all_channels = []
        for geom in frequent_gdf.geometry:
            geom_dict = {
                "type": geom.geom_type,
                "coordinates": list(geom.coords)
            }
            all_channels.append(geom_dict)

        complete_result = {
            "channels": all_channels,
            "parameters": result["parameters"]
        }
        output_file = save_result_to_file("frequent_channels", complete_result, "json")
        result["output_file"] = output_file

        logger.info(f"频繁轨迹分析完成: {result['frequent_channels_count']} 条航道")
        return result

    @staticmethod
    def analyze_clustering(file_path: str) -> Dict[str, Any]:
        """
        编群分析任务

        Args:
            file_path: 数据文件路径（必须包含group_id列）

        Returns:
            分析结果字典
        """
        logger.info(f"开始编群分析: {file_path}")

        loader = DataLoader(file_path)
        df = loader.run()

        processor = DataProcessor(df=df)
        df_processed = processor.run()

        analyzer = ClusteringAnalyzer(df=df_processed)
        result_df, accuracy = analyzer.run()

        # result_df 是 Polars DataFrame
        # Polars 的 to_dict() 不接受参数，需要先转换为 dicts 列表
        sample_results = []
        if len(result_df) > 0:
            sample_results = result_df.head(10).to_dicts()

        result = {
            "total_records": len(result_df),
            "accuracy": float(accuracy),
            "cluster_count": int(result_df['cluster'].n_unique()) if 'cluster' in result_df.columns else 0,
            "sample_results": sample_results
        }

        # 保存完整的聚类结果到CSV
        output_file = save_result_to_file("clustering_results", result_df, "csv")
        result["output_file"] = output_file

        # 保存统计信息到JSON
        stats_file = save_result_to_file("clustering_stats", result, "json")
        result["stats_file"] = stats_file

        logger.info(f"编群分析完成: 准确率 {accuracy:.4f}")
        return result

    @staticmethod
    def analyze_association(
        file_path: str,
        min_support: float,
        min_confidence: float
    ) -> Dict[str, Any]:
        """
        关联分析任务（优化版 - 内存优化）

        Args:
            file_path: 数据文件路径
            min_support: 最小支持度
            min_confidence: 最小置信度

        Returns:
            分析结果字典
        """
        import gc  # 导入垃圾回收

        logger.info(f"开始关联分析: {file_path}")
        logger.info(f"参数: min_support={min_support}, min_confidence={min_confidence}")

        try:
            loader = DataLoader(file_path)
            df = loader.run()

            processor = DataProcessor(df=df)
            df_processed = processor.run()

            # 性能优化：检查数据量
            df_collected = df_processed.collect()
            total_records = len(df_collected)
            logger.info(f"总记录数: {total_records}")

            # 如果数据量太大，进行采样以提高性能
            if total_records > 1000:
                logger.warning(f"数据量较大({total_records}条)，进行采样以提高性能")
                sample_size = 1000
                df_sampled = df_collected.sample(n=sample_size, seed=42)
                df_processed = df_sampled.lazy()
                logger.info(f"已采样到 {sample_size} 条记录")
                analyzed_records = sample_size

                # 释放原始数据
                del df_collected
                gc.collect()
            else:
                analyzed_records = total_records

            analyzer = AssociationAnalyzer(
                df=df_processed,
                min_support=min_support,
                min_confidence=min_confidence
            )

            frequent_itemsets, association_rules = analyzer.run()

            # 转换规则为字典列表
            top_rules = []
            if len(association_rules) > 0:
                top_rules_df = association_rules.head(20)
                # 处理frozenset类型的列
                for _, row in top_rules_df.iterrows():
                    rule_dict = {}
                    for col in top_rules_df.columns:
                        val = row[col]
                        # 将frozenset转换为list以便JSON序列化
                        if isinstance(val, frozenset):
                            rule_dict[col] = list(val)
                        else:
                            rule_dict[col] = float(val) if isinstance(val, (int, float, np.number)) else str(val)
                    top_rules.append(rule_dict)

            result = {
                "total_records_analyzed": analyzed_records,
                "original_records": total_records,
                "sampled": total_records > 1000,
                "frequent_itemsets_count": len(frequent_itemsets),
                "association_rules_count": len(association_rules),
                "top_rules": top_rules,
                "parameters": {
                    "min_support": min_support,
                    "min_confidence": min_confidence
                }
            }

            # 保存频繁项集到CSV
            if len(frequent_itemsets) > 0:
                itemsets_file = save_result_to_file("association_itemsets", frequent_itemsets, "csv")
                result["itemsets_file"] = itemsets_file
                del frequent_itemsets  # 释放内存

            # 保存关联规则到CSV
            if len(association_rules) > 0:
                rules_file = save_result_to_file("association_rules", association_rules, "csv")
                result["rules_file"] = rules_file
                del association_rules  # 释放内存

            # 保存统计信息到JSON
            stats_file = save_result_to_file("association_stats", result, "json")
            result["stats_file"] = stats_file

            logger.info(f"关联分析完成: {result['association_rules_count']} 条规则")

            # 强制垃圾回收
            gc.collect()

            return result

        except Exception as e:
            # 确保异常时也清理内存
            gc.collect()
            raise

    @staticmethod
    def predict_trajectory(
        file_path: str,
        forecast_horizon: int
    ) -> Dict[str, Any]:
        """
        航迹预测任务

        Args:
            file_path: 数据文件路径
            forecast_horizon: 预测时间步长

        Returns:
            预测结果字典
        """
        logger.info(f"开始航迹预测: {file_path}")

        loader = DataLoader(file_path)
        df = loader.run()

        processor = DataProcessor(df=df)
        df_processed = processor.run()

        manager = PatchTSTManager(df=df_processed)
        predictions = manager.predict(forecast_horizon=forecast_horizon)

        result = {
            "forecast_horizon": forecast_horizon,
            "predictions_count": len(predictions) if predictions else 0,
            "predictions": predictions[:100] if predictions else []  # 限制返回数量
        }

        # 保存完整预测结果到JSON
        if predictions:
            complete_result = {
                "forecast_horizon": forecast_horizon,
                "predictions": predictions
            }
            output_file = save_result_to_file("trajectory_predictions", complete_result, "json")
            result["output_file"] = output_file

        logger.info(f"航迹预测完成: {result['predictions_count']} 条预测")
        return result
