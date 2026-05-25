import os
import gc
import glob
import joblib
import logging
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from transformers import PatchTSTConfig, PatchTSTForPrediction
from code.data import (
    DataLoader,
    DataFilter,
    DataProcessor,
    DataSaver,
)
from config.settings import *


class GlobalDataScaler:
    def __init__(self, scaler_save_path, feature_cols):
        """
        :param scaler_save_path: 最终保存 .joblib 文件的路径
        :param feature_cols: 需要标准化的特征列名列表 (即 train_columns)
        """
        self.scaler_save_path = scaler_save_path
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()
        self.has_fitted = False

    def stream_fit(self, file_list, freq):
        logging.info(f"🚀 开始流式统计特征分布 (共 {len(file_list)} 个文件)...")
        
        for file_path in tqdm(file_list, desc="Global Scaling"):
            if not os.path.exists(file_path):
                logging.error(f"❌ 物理路径不存在: {file_path}")
                continue

            try:
                df_raw = DataLoader(file_path).run()
                
                processor = DataProcessor(df_raw)
                df_processed = processor.run_prediction(freq=freq).collect()
                
                data = df_processed.select(self.feature_cols).to_numpy()
                
                if data.shape[0] > 0:
                    self.scaler.partial_fit(data)
                    self.has_fitted = True
                
                # 4. 释放内存
                del df_raw, df_processed, data
                gc.collect()
                
            except Exception as e:
                logging.error(f"  ❌ 文件 {file_path} 统计失败: {e}")

        if self.has_fitted:
            joblib.dump(self.scaler, self.scaler_save_path)
            logging.info(f"✅ 全局标准化器已保存至: {self.scaler_save_path}")
            logging.info(f"📊 累计处理样本总数: {self.scaler.n_samples_seen_}")
        else:
            logging.error("❌ 拟合失败：未提取到任何有效数据特征。")
            
        return self.scaler


class AISTrajectoryDataset(Dataset):
    def __init__(
            self,
            data, 
            seq_len: int = 60, 
            pred_len: int = 60
        ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_windows = seq_len + pred_len

        if hasattr(data, "collect"):
            df = data.collect()
        else:
            df = data
        self.train_data = df.select(train_columns).to_numpy().astype(np.float32)
        mmsi_array = df.select(pl.col(MMSI_COL)).to_numpy().flatten()
        all_starts = np.arange(len(mmsi_array) - self.total_windows + 1)
        is_valid = (mmsi_array[all_starts] == mmsi_array[all_starts + self.total_windows - 1])
        self.valid_indices = all_starts[is_valid]

        logging.info(f"有效样本总数: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        s_end = start_idx + self.seq_len
        r_end = s_end + self.pred_len

        seq_x = self.train_data[start_idx:s_end]
        seq_y = self.train_data[s_end:r_end]

        return torch.from_numpy(seq_x), torch.from_numpy(seq_y)


class PatchTSTManager:
    def __init__(self, pretrain_model_path):
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
            logging.info(f"🔄 正在从 {pretrain_model_path} 加载预训练主干权重...")
            self.model = PatchTSTForPrediction.from_pretrained(
                pretrain_model_path, 
                config=self.config
            )
        else:
            logging.warning("⚠️ 未指定预训练路径，使用随机初始化。")
            self.model = PatchTSTForPrediction(self.config)
        
        self.model.to(device)
        self.scaler = joblib.load(scaler_path)
        self.loss_fn = torch.nn.HuberLoss(delta=1.0)

    def get_dataloader(self, data, shuffle):
        df = data.collect() if hasattr(data, "collect") else data
    
        processor = DataProcessor(df=df)
        processed_df = processor.run_prediction(
            freq=freq,
        )

        feat_to_scale = processed_df.select(scaler_columns).to_numpy()

        try:
            scaled_feat = self.scaler.transform(feat_to_scale)
            scaled_df_part = pl.DataFrame(scaled_feat, schema=scaler_columns)
            final_df = processed_df.with_columns(scaled_df_part)
        except ValueError as e:
            logging.info(f"Scaler 期望特征数: {self.scaler.n_features_in_}")
            logging.info(f"当前输入特征数: {feat_to_scale.shape[1]}")
            raise e
        
        dataset = AISTrajectoryDataset(
            data=final_df,
            seq_len=seq_len,
            pred_len=pred_len,
        )

        del df, processed_df, scaled_feat, scaled_df_part, final_df

        return TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
            drop_last=False
        )

    def train(self, dataloader, optimizer, epoch):
        self.model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
        for i, (batch_x, batch_y) in enumerate(pbar):
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
        return total_loss / len(dataloader)

    def evaluate(self, dataloader, epoch, mode="Validation"):
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

        avg_val_loss = total_loss / len(dataloader)

        all_preds_concat = np.concatenate(all_preds)
        all_trues_concat = np.concatenate(all_trues)
        
        ade, fde, max_e = self.calculate_trajectory_metrics(all_preds_concat, all_trues_concat)

        del all_preds, all_trues, all_preds_concat, all_trues_concat
        torch.cuda.empty_cache()
        gc.collect()
        
        return avg_val_loss, ade, fde, max_e
    
    def calculate_trajectory_metrics(self, all_preds, all_trues):
        N, T, C = all_preds.shape
        
        # 1. 严格切片，确保只取被 scaler 拟合过的前 K 列
        num_scaled = self.scaler.n_features_in_
        flat_preds = all_preds[:, :, :num_scaled].reshape(-1, num_scaled)
        flat_trues = all_trues[:, :, :num_scaled].reshape(-1, num_scaled)
        
        # 2. 反标准化
        inv_preds = self.scaler.inverse_transform(flat_preds).reshape(N, T, num_scaled)
        inv_trues = self.scaler.inverse_transform(flat_trues).reshape(N, T, num_scaled)
        
        # 3. 定位增量列的索引 (假设你预测的是 delta_x, delta_y)
        # 请确保这两个字符串在你的 scaler_columns 列表中
        dx_idx = scaler_columns.index("delta_x")
        dy_idx = scaler_columns.index("delta_y")
        
        # 提取预测位移和真实位移 (单位：米)
        p_dx = inv_preds[:, :, dx_idx]
        p_dy = inv_preds[:, :, dy_idx]
        t_dx = inv_trues[:, :, dx_idx]
        t_dy = inv_trues[:, :, dy_idx]
        
        # 4. 计算欧式距离误差 (单位：米)
        # 直接计算每一步预测增量与真实增量的偏差
        dist = np.sqrt((p_dx - t_dx)**2 + (p_dy - t_dy)**2)
        
        ade = np.mean(dist)
        fde = np.mean(dist[:, -1])
        max_error = np.max(dist)
        
        return ade, fde, max_error

    def process_data(self, raw_files):
        for file_idx, file_path in enumerate(raw_files):
            logging.info(f"[{file_idx+1}/{len(raw_files)}] 流式处理: {os.path.basename(file_path)}")
            df = DataLoader(file_path).run()

            mmsi_col = df.select(MMSI_COL).unique().collect().get_column(MMSI_COL)
            unique_mmsis = mmsi_col.to_list()
            
            train_ids, valid_ids = train_test_split(unique_mmsis, test_size=0.2, random_state=42)
            
            train_df = df.filter(pl.col(MMSI_COL).is_in(train_ids))
            valid_df = df.filter(pl.col(MMSI_COL).is_in(valid_ids))
            
            yield train_df, valid_df

            del mmsi_col, unique_mmsis, train_df, valid_df
            gc.collect()

    def train_file(
        self, 
        epochs: int, 
        file_paths: list[str],
        optimizer,
        scheduler,
        best_model_name: str
    ):
        best_ade = float('inf')
        for epoch in range(1, epochs + 1):
            data_stream = self.process_data(file_paths)
            total_train_loss = 0
            total_valid_loss = 0
            total_ade = 0
            file_count = 0
            for train_df, valid_df in data_stream:
                train_loader = self.get_dataloader(data=train_df, shuffle=True)
                valid_loader = self.get_dataloader(data=valid_df, shuffle=False)
                loss_t = self.train(train_loader, optimizer=optimizer, epoch=epoch)
                loss_v, ade, fde, max_e = self.evaluate(valid_loader, epoch=epoch)

                total_train_loss += loss_t
                total_valid_loss += loss_v
                total_ade += ade
                file_count += 1

                del train_loader, valid_loader, train_df, valid_df
                torch.cuda.empty_cache()

            avg_train_loss = total_train_loss / file_count
            avg_valid_loss = total_valid_loss / file_count
            avg_ade = total_ade / file_count

            curr_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_ade)

            logging.info(f"{'='*40}")
            logging.info(f"Epoch {epoch}/{epochs}")
            logging.info(f"Avg Train Loss: {avg_train_loss:.6f} | Avg Val Loss: {avg_valid_loss:.6f} | LR: {curr_lr:.6f}")
            logging.info(f"Trajectory Error: ADE: {ade:.2f}m | FDE: {fde:.2f}m | Max: {max_e:.2f}m")
            logging.info(f"{'='*40}")

            if avg_ade < best_ade:
                best_ade = avg_ade
                torch.save(self.model.state_dict(), best_model_name)
                logging.info(f"✨ 发现更优物理精度 (ADE: {ade:.2f}m)，已更新权重。")

    def _mercator_to_latlon(self, x_meters, y_meters, R=6378137.0):
        # 经度还原：x = R * lon * (pi/180)
        lon = np.degrees(x_meters / R)
        
        # 纬度还原：y = R * ln(tan(pi/4 + lat/2))
        lat = np.degrees(2.0 * np.arctan(np.exp(y_meters / R)) - np.pi / 2.0)
        
        return lat, lon

    def run_inference(self, file_path, weight_path, total_steps=None, is_plot=False):
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"❌ 关键错误：未找到权重文件 {weight_path}，无法进行有效预测！")
            
        self.model.load_state_dict(torch.load(weight_path, map_location=device))
        logging.info(f"✅ 成功加载权重: {weight_path}")
        self.model.eval()

        df_raw = DataLoader(file_path).run()
        df_processed = DataProcessor(df_raw).run_prediction(freq=freq).collect()
        unique_mmsis = df_processed.get_column(MMSI_COL).unique().to_list()
        
        all_results = [] 
        R_val = 6378137.0  # 与转换时一致的地球半径
        
        dx_idx = scaler_columns.index("delta_x")
        dy_idx = scaler_columns.index("delta_y")

        for mmsi in unique_mmsis[100:500]:
            mmsi_df = df_processed.filter(pl.col(MMSI_COL) == mmsi)
            if len(mmsi_df) < (seq_len + pred_len): continue
                
            input_df = mmsi_df.head(seq_len)
            
            # --- 关键：获取历史序列最后一个点的墨卡托坐标 ---
            last_x = input_df.get_column("x_meters").last()
            last_y = input_df.get_column("y_meters").last()

            # 模型预测标准化数值
            feat_in = input_df.select(scaler_columns).to_numpy()
            feat_in_scaled = self.scaler.transform(feat_in)
            batch_x = torch.from_numpy(feat_in_scaled).unsqueeze(0).to(device).float()
            
            with torch.no_grad():
                outputs = self.model(past_values=batch_x)
                preds_scaled = outputs.prediction_outputs.cpu().numpy()[0]

            # --- 关键：反标准化增量并累加 ---
            # 只取预测的前 K 列（由 scaler 决定的维度）进行 inverse_transform
            num_scaled = self.scaler.n_features_in_
            inv_preds = self.scaler.inverse_transform(preds_scaled[:, :num_scaled])
            
            # 提取 delta_x 和 delta_y (单位：米)
            pred_dx = inv_preds[:, dx_idx]
            pred_dy = inv_preds[:, dy_idx]

            # 累加增量得到预测的墨卡托绝对坐标序列
            # x_curr = x_last + cumsum(delta_x)
            pred_x_meters = last_x + np.cumsum(pred_dx)
            pred_y_meters = last_y + np.cumsum(pred_dy)

            # --- 关键：墨卡托坐标转经纬度 ---
            pred_lats, pred_lons = self._mercator_to_latlon(pred_x_meters, pred_y_meters, R_val)

            # 构造结果
            res_df = pl.DataFrame({
                MMSI_COL: [mmsi] * pred_len,
                LATITUDE_COL: pred_lats,
                LONGITUDE_COL: pred_lons
            })
            all_results.append(res_df)

            if is_plot:
                # 可视化对比
                truth_df = mmsi_df.slice(seq_len, pred_len)
                self._plot_trajectory(mmsi, input_df, truth_df, pred_lats, pred_lons)

        return pl.concat(all_results) if all_results else pl.DataFrame()
    
    def run_long_range_inference(self, file_path, weight_path, total_steps=5, is_plot: bool = False):
        """
        长程推断：确保输入窗口始终保持为 seq_len，并返回预测的经纬度 DataFrame
        """

        total_steps = total_steps * 12

        if os.path.exists(weight_path):
            self.model.load_state_dict(torch.load(weight_path, map_location=device))
            logging.info(f"✅ 权重已加载: {weight_path}")
        self.model.eval()

        # 数据加载与预处理
        df_raw = DataLoader(file_path).run()
        df_processed = DataProcessor(df_raw).run_prediction(freq=freq).collect()
        unique_mmsis = df_processed.get_column(MMSI_COL).unique().to_list()
        
        dx_idx = scaler_columns.index("delta_x")
        dy_idx = scaler_columns.index("delta_y")
        num_scaled = self.scaler.n_features_in_

        # 用于存储所有船只预测结果的列表
        all_ship_results = []

        # 为了演示，这里取 unique_mmsis[100:150]，你可以根据需要修改范围
        for mmsi in unique_mmsis[100:150]:
            mmsi_df = df_processed.filter(pl.col(MMSI_COL) == mmsi)
            if len(mmsi_df) < seq_len: 
                logging.info(f"⚠️ 忽略长度不足的轨迹: {mmsi}")
                continue

            # --- 1. 初始化滑动窗口 ---
            current_window_feat = mmsi_df.head(seq_len).select(scaler_columns).to_numpy()
            last_x = mmsi_df.get_column("x_meters").head(seq_len).last()
            last_y = mmsi_df.get_column("y_meters").head(seq_len).last()

            ship_pred_lats = []
            ship_pred_lons = []
            
            iterations = int(np.ceil(total_steps / pred_len))
            
            for i in range(iterations):
                # --- 2. 模型预测 ---
                feat_in_scaled = self.scaler.transform(current_window_feat)
                batch_x = torch.from_numpy(feat_in_scaled).unsqueeze(0).to(device).float()

                with torch.no_grad():
                    outputs = self.model(past_values=batch_x)
                    preds_scaled = outputs.prediction_outputs.cpu().numpy()[0]

                # --- 3. 反标准化与坐标还原 ---
                inv_preds = self.scaler.inverse_transform(preds_scaled[:, :num_scaled])
                p_dx = inv_preds[:, dx_idx]
                p_dy = inv_preds[:, dy_idx]

                # 累加增量
                pred_x_m = last_x + np.cumsum(p_dx)
                pred_y_m = last_y + np.cumsum(p_dy)
                
                # 转换为经纬度
                lats, lons = self._mercator_to_latlon(pred_x_m, pred_y_m)
                ship_pred_lats.extend(lats)
                ship_pred_lons.extend(lons)

                # --- 4. 更新滑动窗口 ---
                current_window_feat = np.concatenate([
                    current_window_feat[pred_len:], 
                    inv_preds
                ], axis=0)

                # 更新基准坐标点
                last_x, last_y = pred_x_m[-1], pred_y_m[-1]

            # 截取到要求的 total_steps
            final_lats = ship_pred_lats[:total_steps]
            final_lons = ship_pred_lons[:total_steps]

            # --- 5. 构造当前船只的预测结果 DataFrame ---
            ship_res_df = pl.DataFrame({
                MMSI_COL: [mmsi] * len(final_lats),
                LATITUDE_COL: final_lats,
                LONGITUDE_COL: final_lons,
                "step": np.arange(1, len(final_lats) + 1) # 可选：记录预测步数
            })
            all_ship_results.append(ship_res_df)

            # 可视化 (保留原有的可视化逻辑)
            if is_plot:
                truth_df = mmsi_df.slice(seq_len, total_steps)
                self._plot_trajectory(
                    f"{mmsi}_long", 
                    mmsi_df.head(seq_len), 
                    truth_df, 
                    final_lats, 
                    final_lons
                )

        logging.info("✨ 长程推断完成。")
        
        # --- 6. 合并所有结果并返回 ---
        if all_ship_results:
            return pl.concat(all_ship_results)
        else:
            return pl.DataFrame(schema={MMSI_COL: pl.Int64, LATITUDE_COL: pl.Float64, LONGITUDE_COL: pl.Float64})
        
    def run_rolling_inference(self, file_path: str, weight_path: str, total_steps: int=120, is_plot: bool=False):
        """
        滑动窗口预测：
        - 使用真实数据修正输入窗口
        - 每次预测一步，窗口右移1
        - 输出预测经纬度
        :param file_path: AIS 原始数据路径
        :param weight_path: 模型权重路径
        :param total_steps: 最大预测步数
        """
        if os.path.exists(weight_path):
            self.model.load_state_dict(torch.load(weight_path, map_location=device))
            logging.info(f"✅ 权重已加载: {weight_path}")
        self.model.eval()

        total_steps = total_steps * 12

        df_raw = DataLoader(file_path).run()
        df_processed = DataProcessor(df_raw).run_prediction(freq=freq).collect()
        unique_mmsis = df_processed.get_column(MMSI_COL).unique().to_list()

        dx_idx = scaler_columns.index("delta_x")
        dy_idx = scaler_columns.index("delta_y")
        num_scaled = self.scaler.n_features_in_

        all_results = []

        for mmsi in unique_mmsis[100:150]:
            mmsi_df = df_processed.filter(pl.col(MMSI_COL) == mmsi)
            n_points = len(mmsi_df)
            if n_points < seq_len + 1:
                logging.info(f"⚠️ MMSI {mmsi} 数据不足，跳过")
                continue

            pred_lats = []
            pred_lons = []

            # 初始化滑动窗口，用前 seq_len 个真实数据
            window_df = mmsi_df.head(seq_len).select(scaler_columns).to_numpy()
            last_x = mmsi_df.get_column("x_meters").head(seq_len).last()
            last_y = mmsi_df.get_column("y_meters").head(seq_len).last()

            for t in range(seq_len, min(n_points, seq_len + total_steps)):
                # --- 1. 预测下一步增量 ---
                feat_scaled = self.scaler.transform(window_df)
                batch_x = torch.from_numpy(feat_scaled).unsqueeze(0).to(device).float()
                with torch.no_grad():
                    outputs = self.model(past_values=batch_x)
                    pred_scaled = outputs.prediction_outputs.cpu().numpy()[0, -1, :num_scaled]  # 只取最后一步

                # --- 2. 反标准化增量 ---
                inv_pred = self.scaler.inverse_transform(pred_scaled.reshape(1, -1)).flatten()
                pred_dx = inv_pred[dx_idx]
                pred_dy = inv_pred[dy_idx]

                # --- 3. 累加得到绝对坐标 ---
                pred_x_m = last_x + pred_dx
                pred_y_m = last_y + pred_dy

                # --- 4. 转为经纬度 ---
                pred_lat, pred_lon = self._mercator_to_latlon(pred_x_m, pred_y_m)
                pred_lats.append(pred_lat)
                pred_lons.append(pred_lon)

                # --- 5. 更新滑动窗口 ---
                # 这里使用真实数据更新窗口（teacher forcing）
                next_real_feat = mmsi_df[t].select(scaler_columns).to_numpy().reshape(1, -1)
                window_df = np.vstack([window_df[1:], next_real_feat])
                last_x, last_y = mmsi_df.get_column("x_meters")[t], mmsi_df.get_column("y_meters")[t]

            # --- 构造 DataFrame ---
            res_df = pl.DataFrame({
                MMSI_COL: [mmsi] * len(pred_lats),
                LATITUDE_COL: pred_lats,
                LONGITUDE_COL: pred_lons,
                "step": np.arange(seq_len+1, seq_len+1+len(pred_lats))
            })
            all_results.append(res_df)

            # 可视化
            if is_plot:
                self._plot_trajectory(
                    f"{mmsi}_rolling",
                    mmsi_df.head(seq_len), 
                    mmsi_df.slice(seq_len, 
                    len(pred_lats)), 
                    pred_lats, 
                    pred_lons
                )

        logging.info("✨ 滑动窗口预测完成。")
        return pl.concat(all_results) if all_results else pl.DataFrame()

    def _plot_trajectory(self, mmsi, hist_df, truth_df, p_lats, p_lons):
        """
        可视化历史轨迹、真实未来、预测未来
        """
        plt.figure(figsize=(12, 8))
        
        # 历史数据 (输入序列)
        plt.plot(hist_df[LONGITUDE_COL], hist_df[LATITUDE_COL], 
                 color='red', linestyle='--', alpha=0.7, label='History (Input)')
        
        # 真实未来数据 (对比序列)
        plt.plot(truth_df[LONGITUDE_COL], truth_df[LATITUDE_COL], 
                 color='blue', linestyle='--', marker='o', markersize=3, label='True Future', alpha=0.6)
        
        # 预测未来轨迹
        plt.plot(p_lons, p_lats, 
                 color='black', linestyle='--', marker='x', markersize=4, label='Predicted Future')
        
        # 标记预测起始点
        plt.scatter(hist_df[LONGITUDE_COL].last(), hist_df[LATITUDE_COL].last(), 
                    color='black', s=80, zorder=5, label='Start Point')

        plt.title(f"MMSI: {mmsi} - Trajectory Prediction (PatchTST)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.5)
        
        os.makedirs("./output/inference_results", exist_ok=True)
        plt.savefig(f"./output/inference_results/traj_{mmsi}.png", dpi=200)
        plt.close()
        logging.info(f"📊 已保存船只 {mmsi} 的可视化结果")
    
def main():
    if not os.path.exists(scaler_path):
        logging.info("📢 未找到标准化器，开始全局流式拟合...")
        Scaler = GlobalDataScaler(scaler_save_path=scaler_path, feature_cols=scaler_columns)
        scaler_obj = Scaler.stream_fit(file_paths, freq=freq)
    else:
        logging.info(f"✅ 检测到已存在的标准化器: {scaler_path}，正在加载...")
        scaler_obj = joblib.load(scaler_path)

    manager = PatchTSTManager(pretrain_model_path=None)
    optimizer=torch.optim.AdamW(manager.model.parameters(), lr=lr)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    manager.train_file(
        epochs=epochs,
        file_paths=file_paths,
        optimizer=optimizer,
        scheduler=scheduler,
        best_model_name=best_model_path,
    )

    # data = manager.run_rolling_inference(
    #     file_path=r"D:\InstallFiles\File\Project\dataset\AIS_2024_12_31.csv",
    #     weight_path=r"D:\InstallFiles\File\Project\dataset\best_patchtst_model.pt",
    #     is_plot=True,
    #     total_steps=8
    # )
    # DataSaver(df=data, output_dir=output_dir, filename="predict").save_csv()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("./log/log", encoding="utf-8"),
            logging.StreamHandler()
            ]
        )
    main()