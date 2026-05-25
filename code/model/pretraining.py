import gc
import os
import torch
import joblib
import logging
import numpy as np
import polars as pl
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import PatchTSTConfig, PatchTSTForPretraining
from code.data import (
    DataLoader,
)
from code.data import DataProcessor
from config.settings import *


class AISPretrainDataset(Dataset):
    def __init__(self, data: pl.LazyFrame, seq_len: int = 96):
        collected_data = data.collect()
        self.np_data = collected_data.select(train_columns).to_numpy().astype(np.float32)
        
        mmsi_array = collected_data.select(pl.col(MMSI_COL)).to_numpy().flatten()
        self.seq_len = seq_len

        all_starts = np.arange(len(mmsi_array) - self.seq_len + 1)
        is_valid = (mmsi_array[all_starts] == mmsi_array[all_starts + self.seq_len - 1])
        self.valid_indices = all_starts[is_valid]

        logging.info(f"有效样本总数: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        seq_x = self.np_data[start : start + self.seq_len]
        return torch.from_numpy(seq_x), 0
    

class PatchTSTPretrainManager(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PatchTSTConfig(
            num_input_channels=len(train_columns),
            context_length=96,
            prediction_length=0,
            mask_ratio=0.4,
            patch_length=12,
            stride=6,
            d_model=d_model,
            num_attention_heads=nhead,
            num_hidden_layers=num_encoder_layers,
            ffn_dim=dim_feedforward,
            dropout=dropout,
            loss='mse'
        )
        self.model = PatchTSTForPretraining(self.config).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.scaler = joblib.load(scaler_path) 

    def train(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Pretrain Epoch {epoch}")
        for batch_x, _ in pbar:
            batch_x = batch_x.to(device)
            self.optimizer.zero_grad()

            outputs = self.model(past_values=batch_x)
            loss = outputs.loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(mae_loss=f"{loss.item():.6f}")

        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader, epoch):
        self.model.eval() 
        total_loss = 0
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Pretrain Epoch {epoch}")
            for batch_x, _ in pbar:
                batch_x = batch_x.to(device)
                outputs = self.model(past_values=batch_x)
                
                loss = outputs.loss 
                total_loss += loss.item()
                
        return total_loss / len(dataloader)

    def process_data(self, raw_files):
        os.makedirs(output_dir, exist_ok=True)
        for idx, file_path in enumerate(raw_files):
            base_name = os.path.basename(file_path).replace('.csv', '')
            logging.info(f"[{idx+1}/{len(raw_files)}] Converting {os.path.basename(file_path)}...")
            
            df = ShipDataLoader(file_path).run()
            unique_mmsis = (
                df.select(MMSI_COL)
                .unique()
                .collect()
                .get_column(MMSI_COL)
                .to_list()
            )
            train_ids, valid_ids = train_test_split(unique_mmsis, test_size=0.2, random_state=42)
            train_df, valid_df = df.filter(pl.col(MMSI_COL).is_in(train_ids)), df.filter(pl.col(MMSI_COL).is_in(valid_ids))
            train_df, valid_df = DataProcessor(train_df).run_prediction(freq=freq), DataProcessor(valid_df).run_prediction(freq=freq)
            # train_df, valid_df = self.scaler.transform(train_df.select(train_columns)), self.scaler.transform(valid_df.select(train_columns))

            yield train_df, valid_df

            del df, train_ids, valid_ids, unique_mmsis, train_df, valid_df

    def train_file(
        self,
        epochs: int, 
        file_paths: list[str],
    ):
        best_val_loss = float('inf')
        for epoch in range(1, epochs + 1):
            data_stream = self.process_data(file_paths)
            for train_path, valid_path in data_stream:
                train_dataset = AISPretrainDataset(train_path)
                valid_dataset = AISPretrainDataset(valid_path)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

                train_loss = self.train(train_loader, epoch)
                val_loss = self.evaluate(valid_loader, epoch)
                logging.info(f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.model.save_pretrained("best_pretrain_model")
                    logging.info(f"🌟 发现更好的模型，已保存至 best_pretrain_model.pth")

                del train_loader, valid_loader, train_dataset, valid_dataset
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logging.info("文件处理完毕，内存已清理。")
def main():
    df_list = [f"D:/InstallFiles/File/Project/dataset/AIS_2024_12_{i:02d}.csv" for i in range(1, 31)]

    manager = PatchTSTPretrainManager()
    manager.train_file(
        epochs=100,
        file_paths=df_list,
    )


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