"""
企业级航迹预测模块
基于PatchTST深度学习模型的船舶轨迹预测
"""
import os
import gc
import joblib
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from transformers import PatchTSTConfig, PatchTSTForPrediction

from core.config import data_config, model_config
from core.data import DataLoader, DataProcessor
from core.exceptions import ModelError, DataValidationError, FileOperationError
from core.logger import get_logger

logger = get_logger(__name__)

# 配置变量映射
MMSI_COL = data_config.MMSI_COL
TIMESTAMP_COL = data_config.TIMESTAMP_COL
LATITUDE_COL = data_config.LATITUDE_COL
LONGITUDE_COL = data_config.LONGITUDE_COL
train_columns = model_config.TRAIN_COLUMNS
scaler_columns = model_config.SCALER_COLUMNS
seq_len = model_config.seq_len
pred_len = model_config.pred_len
device = model_config.device
d_model = model_config.d_model
nhead = model_config.nhead
num_encoder_layers = model_config.num_encoder_layers
dim_feedforward = model_config.dim_feedforward
dropout = model_config.dropout
batch_size = model_config.batch_size
R = model_config.EARTH_RADIUS
freq = model_config.freq


class AISTrajectoryDataset(Dataset):
    """AIS轨迹时序数据集"""

    def __init__(
        self,
        data: pl.LazyFrame,
        seq_len: int = 96,
        pred_len: int = 60
    ):
        """
        初始化数据集

        Args:
            data: 处理后的AIS数据
            seq_len: 输入序列长度
            pred_len: 预测序列长度

        Raises:
            DataValidationError: 数据无效时抛出
        """
        super().__init__()

        if data is None:
            raise DataValidationError("数据不能为空")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_windows = seq_len + pred_len

        df = data.collect() if hasattr(data, "collect") else data

        try:
            self.train_data = df.select(train_columns).to_numpy().astype(np.float32)
        except Exception as e:
            raise DataValidationError(
                f"数据列不匹配: {str(e)}",
                details={"required_columns": train_columns}
            )

        mmsi_array = df.select(pl.col(MMSI_COL)).to_numpy().flatten()
        all_starts = np.arange(len(mmsi_array) - self.total_windows + 1)
        is_valid = (mmsi_array[all_starts] == mmsi_array[all_starts + self.total_windows - 1])
        self.valid_indices = all_starts[is_valid]

        logger.info(f"数据集初始化完成 - 有效样本数: {len(self.valid_indices)}")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_idx = self.valid_indices[idx]
        s_end = start_idx + self.seq_len
        r_end = s_end + self.pred_len

        seq_x = self.train_data[start_idx:s_end]
        seq_y = self.train_data[s_end:r_end]

        return torch.from_numpy(seq_x), torch.from_numpy(seq_y)


class PredictionMetrics:
    """预测指标计算器"""

    @staticmethod
    def calculate_trajectory_metrics(
        all_preds: np.ndarray,
        all_trues: np.ndarray,
        scaler,
        scaler_columns: List[str]
    ) -> Tuple[float, float, float]:
        """
        计算轨迹预测误差指标

        Args:
            all_preds: 预测值 (N, T, C)
            all_trues: 真实值 (N, T, C)
            scaler: 标准化器
            scaler_columns: 特征列名

        Returns:
            (ADE, FDE, 最大误差)

        Raises:
            ModelError: 指标计算失败时抛出
        """
        try:
            N, T, C = all_preds.shape

            num_scaled = scaler.n_features_in_
            flat_preds = all_preds[:, :, :num_scaled].reshape(-1, num_scaled)
            flat_trues = all_trues[:, :, :num_scaled].reshape(-1, num_scaled)

            inv_preds = scaler.inverse_transform(flat_preds).reshape(N, T, num_scaled)
            inv_trues = scaler.inverse_transform(flat_trues).reshape(N, T, num_scaled)

            dx_idx = scaler_columns.index("delta_x")
            dy_idx = scaler_columns.index("delta_y")

            p_dx = inv_preds[:, :, dx_idx]
            p_dy = inv_preds[:, :, dy_idx]
            t_dx = inv_trues[:, :, dx_idx]
            t_dy = inv_trues[:, :, dy_idx]

            dist = np.sqrt((p_dx - t_dx)**2 + (p_dy - t_dy)**2)

            ade = np.mean(dist)
            fde = np.mean(dist[:, -1])
            max_error = np.max(dist)

            return ade, fde, max_error

        except Exception as e:
            raise ModelError(
                f"轨迹指标计算失败: {str(e)}",
                details={"error": str(e)}
            )


class PatchTSTManager:
    """
    PatchTST模型管理器

    负责模型训练、评估和推理
    """

    def __init__(
        self,
        pretrain_model_path: Optional[str] = None,
        scaler_path: Optional[str] = None
    ):
        """
        初始化模型管理器

        Args:
            pretrain_model_path: 预训练模型路径
            scaler_path: 标准化器路径

        Raises:
            ModelError: 模型初始化失败时抛出
            FileOperationError: 文件加载失败时抛出
        """
        try:
            self.config = PatchTSTConfig(
                num_input_channels=len(train_columns),
                context_length=seq_len,
                prediction_length=pred_len,
                patch_length=12,
                stride=6,
                d_model=d_model,
                num_attention_heads=nhead,
                num_hidden_layers=num_encoder_layers,
                ffn_dim=dim_feedforward,
                dropout=dropout,
                scaling='std',
                loss='mse'
            )

            if pretrain_model_path and os.path.exists(pretrain_model_path):
                logger.info(f"从{pretrain_model_path}加载预训练模型")
                self.model = PatchTSTForPrediction.from_pretrained(
                    pretrain_model_path,
                    config=self.config
                )
            else:
                logger.warning("未指定预训练路径，使用随机初始化")
                self.model = PatchTSTForPrediction(self.config)

            self.model.to(device)
            self.loss_fn = nn.HuberLoss(delta=1.0)

            if scaler_path and os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info(f"标准化器加载完成: {scaler_path}")
            else:
                raise FileOperationError(
                    f"标准化器文件不存在: {scaler_path}",
                    details={"path": scaler_path}
                )

            logger.info("模型管理器初始化完成")

        except Exception as e:
            if isinstance(e, (ModelError, FileOperationError)):
                raise
            raise ModelError(
                f"模型初始化失败: {str(e)}",
                details={"error": str(e)}
            )

    def get_dataloader(
        self,
        data: pl.LazyFrame,
        shuffle: bool = False
    ) -> TorchDataLoader:
        """
        创建数据加载器

        Args:
            data: 原始数据
            shuffle: 是否打乱数据

        Returns:
            PyTorch数据加载器

        Raises:
            ModelError: 数据加载器创建失败时抛出
        """
        try:
            df = data.collect() if hasattr(data, "collect") else data

            processor = DataProcessor(df=df)
            processed_df = processor.run_prediction(freq=freq)

            feat_to_scale = processed_df.select(scaler_columns).to_numpy()

            try:
                scaled_feat = self.scaler.transform(feat_to_scale)
                scaled_df_part = pl.DataFrame(scaled_feat, schema=scaler_columns)
                final_df = processed_df.with_columns(scaled_df_part)
            except ValueError as e:
                logger.error(f"Scaler期望特征数: {self.scaler.n_features_in_}")
                logger.error(f"当前输入特征数: {feat_to_scale.shape[1]}")
                raise ModelError(
                    f"特征维度不匹配: {str(e)}",
                    details={
                        "expected": self.scaler.n_features_in_,
                        "actual": feat_to_scale.shape[1]
                    }
                )

            dataset = AISTrajectoryDataset(
                data=final_df,
                seq_len=seq_len,
                pred_len=pred_len,
            )

            del df, processed_df, scaled_feat, scaled_df_part, final_df
            gc.collect()

            return TorchDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0,
                pin_memory=False,
                drop_last=False
            )

        except Exception as e:
            if isinstance(e, ModelError):
                raise
            raise ModelError(
                f"数据加载器创建失败: {str(e)}",
                details={"error": str(e)}
            )

    def train_epoch(
        self,
        dataloader: TorchDataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int
    ) -> float:
        """
        训练一个epoch

        Args:
            dataloader: 训练数据加载器
            optimizer: 优化器
            epoch: 当前epoch

        Returns:
            平均训练损失

        Raises:
            ModelError: 训练失败时抛出
        """
        try:
            self.model.train()
            total_loss = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
            for batch_x, batch_y in pbar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()

                outputs = self.model(past_values=batch_x)
                preds = outputs.prediction_outputs

                loss = self.loss_fn(preds, batch_y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

                del batch_x, batch_y, outputs, loss

            torch.cuda.empty_cache()
            avg_loss = total_loss / len(dataloader)

            logger.info(f"Epoch {epoch} 训练完成 - 平均损失: {avg_loss:.6f}")
            return avg_loss

        except Exception as e:
            raise ModelError(
                f"训练失败: {str(e)}",
                details={"epoch": epoch, "error": str(e)}
            )

    def evaluate(
        self,
        dataloader: TorchDataLoader,
        epoch: int,
        mode: str = "Validation"
    ) -> Tuple[float, float, float, float]:
        """
        评估模型

        Args:
            dataloader: 评估数据加载器
            epoch: 当前epoch
            mode: 评估模式

        Returns:
            (平均损失, ADE, FDE, 最大误差)

        Raises:
            ModelError: 评估失败时抛出
        """
        try:
            self.model.eval()
            total_loss = 0
            all_preds = []
            all_trues = []

            with torch.no_grad():
                pbar = tqdm(dataloader, desc=f"Epoch {epoch} [{mode}]")
                for batch_x, batch_y in pbar:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                    outputs = self.model(past_values=batch_x)
                    preds = outputs.prediction_outputs

                    loss = self.loss_fn(preds, batch_y)
                    total_loss += loss.item()

                    all_preds.append(preds.cpu().numpy())
                    all_trues.append(batch_y.cpu().numpy())

                    pbar.set_postfix(loss=loss.item())
                    del batch_x, batch_y, outputs, loss

            avg_loss = total_loss / len(dataloader)

            all_preds_concat = np.concatenate(all_preds)
            all_trues_concat = np.concatenate(all_trues)

            ade, fde, max_e = PredictionMetrics.calculate_trajectory_metrics(
                all_preds_concat,
                all_trues_concat,
                self.scaler,
                scaler_columns
            )

            del all_preds, all_trues, all_preds_concat, all_trues_concat
            torch.cuda.empty_cache()
            gc.collect()

            logger.info(
                f"Epoch {epoch} 评估完成 - "
                f"损失: {avg_loss:.6f}, ADE: {ade:.2f}m, FDE: {fde:.2f}m, Max: {max_e:.2f}m"
            )

            return avg_loss, ade, fde, max_e

        except Exception as e:
            raise ModelError(
                f"评估失败: {str(e)}",
                details={"epoch": epoch, "mode": mode, "error": str(e)}
            )

    def save_model(self, path: str) -> None:
        """
        保存模型权重

        Args:
            path: 保存路径

        Raises:
            FileOperationError: 保存失败时抛出
        """
        try:
            torch.save(self.model.state_dict(), path)
            logger.info(f"模型已保存: {path}")
        except Exception as e:
            raise FileOperationError(
                f"模型保存失败: {str(e)}",
                details={"path": path, "error": str(e)}
            )

    def load_model(self, path: str) -> None:
        """
        加载模型权重

        Args:
            path: 模型路径

        Raises:
            FileOperationError: 加载失败时抛出
        """
        try:
            if not os.path.exists(path):
                raise FileOperationError(f"模型文件不存在: {path}")

            self.model.load_state_dict(torch.load(path, map_location=device))
            logger.info(f"模型已加载: {path}")
        except Exception as e:
            if isinstance(e, FileOperationError):
                raise
            raise FileOperationError(
                f"模型加载失败: {str(e)}",
                details={"path": path, "error": str(e)}
            )

    @staticmethod
    def _mercator_to_latlon(x_meters: np.ndarray, y_meters: np.ndarray, R: float = 6378137.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        墨卡托坐标转经纬度

        Args:
            x_meters: X坐标（米）
            y_meters: Y坐标（米）
            R: 地球半径

        Returns:
            (纬度, 经度)
        """
        lon = np.degrees(x_meters / R)
        lat = np.degrees(2.0 * np.arctan(np.exp(y_meters / R)) - np.pi / 2.0)
        return lat, lon

    def predict(
        self,
        file_path: str,
        weight_path: str,
        mmsi_list: Optional[List[int]] = None,
        max_ships: int = 100
    ) -> pl.DataFrame:
        """
        执行轨迹预测

        Args:
            file_path: 数据文件路径
            weight_path: 模型权重路径
            mmsi_list: 指定的MMSI列表
            max_ships: 最大预测船舶数

        Returns:
            预测结果DataFrame

        Raises:
            ModelError: 预测失败时抛出
        """
        try:
            self.load_model(weight_path)
            self.model.eval()

            df_raw = DataLoader(file_path).run()
            df_processed = DataProcessor(df_raw).run_prediction(freq=freq).collect()

            unique_mmsis = df_processed.get_column(MMSI_COL).unique().to_list()

            if mmsi_list:
                unique_mmsis = [m for m in unique_mmsis if m in mmsi_list]

            unique_mmsis = unique_mmsis[:max_ships]

            logger.info(f"开始预测 - 船舶数: {len(unique_mmsis)}")

            all_results = []
            dx_idx = scaler_columns.index("delta_x")
            dy_idx = scaler_columns.index("delta_y")

            for mmsi in tqdm(unique_mmsis, desc="预测进度"):
                mmsi_df = df_processed.filter(pl.col(MMSI_COL) == mmsi)
                if len(mmsi_df) < (seq_len + pred_len):
                    continue

                input_df = mmsi_df.head(seq_len)

                last_x = input_df.get_column("x_meters").last()
                last_y = input_df.get_column("y_meters").last()

                feat_in = input_df.select(scaler_columns).to_numpy()
                feat_in_scaled = self.scaler.transform(feat_in)
                batch_x = torch.from_numpy(feat_in_scaled).unsqueeze(0).to(device).float()

                with torch.no_grad():
                    outputs = self.model(past_values=batch_x)
                    preds_scaled = outputs.prediction_outputs.cpu().numpy()[0]

                num_scaled = self.scaler.n_features_in_()
                inv_preds = self.scaler.inverse_transform(preds_scaled[:, :num_scaled])

                pred_dx = inv_preds[:, dx_idx]
                pred_dy = inv_preds[:, dy_idx]

                pred_x_meters = last_x + np.cumsum(pred_dx)
                pred_y_meters = last_y + np.cumsum(pred_dy)

                pred_lats, pred_lons = self._mercator_to_latlon(pred_x_meters, pred_y_meters, R)

                res_df = pl.DataFrame({
                    MMSI_COL: [mmsi] * pred_len,
                    LATITUDE_COL: pred_lats,
                    LONGITUDE_COL: pred_lons
                })
                all_results.append(res_df)

            logger.info("预测完成")
            return pl.concat(all_results) if all_results else pl.DataFrame()

        except Exception as e:
            if isinstance(e, ModelError):
                raise
            raise ModelError(
                f"预测失败: {str(e)}",
                details={"file_path": file_path, "weight_path": weight_path, "error": str(e)}
            )
