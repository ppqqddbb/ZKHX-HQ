import os
import sys
import torch
import logging
import joblib
import polars as pl
from tqdm import tqdm
from code.model import (
    PatchTSTManager,
    DataProcessor
)
from code.data import (
    ShipDataLoader,
    DataFilter,
    # DataProcessor,
    DataSaver,
    DataScaler,
)
from sklearn.model_selection import train_test_split
from config.settings import *

def main():
    # 训练集
    # loader = ShipDataLoader(r"D:\InstallFiles\File\Project\upload\AIS_2024_12_31.csv")
    # df = loader.run()
    # saver = DataSaver(df, r"D:\InstallFiles\File\Project\dataset", filename="train")
    # path_dict = saver.run(formats=['parquet'])
    # train_path = str(path_dict['parquet'])

    # # 验证集
    # loader = ShipDataLoader(r"D:\InstallFiles\File\Project\upload\AIS_2024_12_30.csv")
    # df = loader.run()
    # saver = DataSaver(df, r"D:\InstallFiles\File\Project\dataset", filename="valid")
    # path_dict = saver.run(formats=['parquet'])
    # valid_path = str(path_dict['parquet'])

    loader = ShipDataLoader(r"D:\InstallFiles\File\Project\upload\AIS_2024_12_31.csv")
    df = loader.run()
    unique_mmsis = (
        df.select(MMSI_COL)
        .unique()
        .collect()
        .get_column(MMSI_COL)
        .to_list()
    )
    train_ids, valid_ids = train_test_split(unique_mmsis, test_size=0.2, random_state=42)

    train_df = df.filter(pl.col(MMSI_COL).is_in(train_ids))
    valid_df = df.filter(pl.col(MMSI_COL).is_in(valid_ids))

    print(f"原始训练集数量:{len(train_ids)}, 原始验证集数量:{len(valid_ids)}")
    print(f"原始训练集行数: {train_df.collect().height}, 原始验证集行数: {valid_df.collect().height}")

    saver = DataSaver(train_df, r"D:\InstallFiles\File\Project\dataset", filename="train")
    path_dict = saver.run(formats=['parquet'])
    train_path = str(path_dict['parquet'])

    saver = DataSaver(valid_df, r"D:\InstallFiles\File\Project\dataset", filename="valid")
    path_dict = saver.run(formats=['parquet'])
    valid_path = str(path_dict['parquet'])

    is_training = False  

    # 初始化管理器
    manager = PatchTSTManager()

    if is_training:
        train_loader, _ = manager.get_dataloader(file_path=train_path, is_scaler=True)
        valid_loader, _ = manager.get_dataloader(file_path=valid_path, is_scaler=False)

        train_scaler = joblib.load(scaler_path)

        optimizer = torch.optim.AdamW(manager.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
        best_val_loss = float('inf')

        for epoch in range(1, epochs + 1):
            avg_train_loss = manager.train(train_loader, optimizer, epoch)
            avg_val_loss, val_r2, ade, fde, max_e = manager.evaluate(
                valid_loader, epoch, scaler=train_scaler
            )
            curr_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_val_loss)

            logging.info(f"{'='*40}")
            logging.info(f"Epoch {epoch}/{epochs}")
            logging.info(f"Avg Train Loss: {avg_train_loss:.6f} | Avg Val Loss: {avg_val_loss:.6f} | LR: {curr_lr:.6f} | R2: {val_r2:.4f}")
            logging.info(f"Trajectory Error: ADE: {ade:.2f}m | FDE: {fde:.2f}m | Max: {max_e:.2f}m")
            logging.info(f"{'='*40}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(manager.model.state_dict(), "best_patchtst_model.pt")
                logging.info(f"✨ 检测到更优模型，已保存权重。")
    else:
        logging.info("🔍 仅执行预测流程 (跳过训练)...")

    loader = ShipDataLoader(r"D:\InstallFiles\File\Project\upload\processed_top200.csv")
    df = loader.run()
    saver = DataSaver(df, r"D:\InstallFiles\File\Project\upload", filename="processed_top200")
    path_dict = saver.run(formats=['parquet'])
    test_path = str(path_dict['parquet'])
    test_loader, df_test = manager.get_dataloader(test_path, is_scaler=False)

    if os.path.exists("best_patchtst_model.pt"):
        logging.info("正在加载最优权重...")
        manager.model.load_state_dict(torch.load("best_patchtst_model.pt", map_location=device))

        manager.predict_all_mmsi(df_test)
    else:
        logging.error("❌ 未找到 'best_patchtst_model.pt' 文件，请先将 is_training 设为 True 运行训练。")


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
    # data = pl.scan_csv(r"D:\InstallFiles\File\Project\upload\AIS_2024_12_29.csv")
    # pl.Config.set_tbl_cols(-1)
    # print(data.collect().head(5))
    # processor = DataProcessor(data)
    # data = processor.rename_columns()
    
    # df_to_save = data.collect().head(2000) if isinstance(data, pl.LazyFrame) else data.head(2000)
    # save_path = r"D:\InstallFiles\File\Project\upload\processed_top200.csv"
    # df_to_save.write_csv(save_path)
    # print(f"✅ 已成功保存前 200 条处理后的数据至: {save_path}")

