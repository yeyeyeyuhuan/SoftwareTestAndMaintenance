# sock_shop_experiment.py

import os
import time
from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import tfad
from tfad.ts import TimeSeries, TimeSeriesDataset
from tfad.ts import transforms as tr
from tfad.model import TFAD, TFADDataModule

import numpy as np
import pandas as pd


def load_sock_shop_data(
    train_path: Union[str, Path],
    test_path: Union[str, Path],
    timestamp_col: str = "timestamp",
    label_col: str = "Label",
):
    """
    加载 sock-shop 数据集，自动处理15维特征
    """
    print("Loading sock-shop datasets...")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 只保留数值型特征（排除 timestamp, Label, datetime, pod 等）
    feature_cols = [
        col for col in train_df.columns
        if col not in [timestamp_col, label_col, 'datetime', 'pod']
        and not col.startswith("Unnamed:")
        and train_df[col].dtype in ['int64', 'float64', 'int32', 'float32']
    ]
    
    print(f"Found {len(feature_cols)} features: {feature_cols}")
    
    # 处理 cpu_throttled 可能为空字符串的情况
    for col in feature_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)
    
    # 提取特征和标签
    train_values = train_df[feature_cols].to_numpy().astype(np.float32)
    train_labels = train_df[label_col].to_numpy().astype(np.float32)
    test_values = test_df[feature_cols].to_numpy().astype(np.float32)
    test_labels = test_df[label_col].to_numpy().astype(np.float32)
    
    # 处理 NaN
    train_values = np.nan_to_num(train_values)
    test_values = np.nan_to_num(test_values)
    
    # 创建 TimeSeriesDataset
    train_dataset = TimeSeriesDataset([
        TimeSeries(values=train_values, labels=train_labels, item_id="sock_shop_train")
    ])
    test_dataset = TimeSeriesDataset([
        TimeSeries(values=test_values, labels=test_labels, item_id="sock_shop_test")
    ])
    
    print(f"Train dataset: {len(train_dataset[0].values)} timesteps, {train_dataset[0].shape[1]} features")
    print(f"Test dataset: {len(test_dataset[0].values)} timesteps, {test_dataset[0].shape[1]} features")
    
    return train_dataset, test_dataset


def sock_shop_pipeline(
    train_csv: str,
    test_csv: str,
    model_dir: str = "models",
    log_dir: str = "logs",
    exp_name: Optional[str] = None,
    epochs: int = 100,
    gpus: int = 0,
    window_length: int = 200,
    suspect_window_length: int = 50,
    num_series_in_train_batch: int = 4,
    num_crops_per_series: int = 8,
    learning_rate: float = 3e-4,
    rnd_seed: int = 123,
):
    """
    针对 sock-shop 数据集的 TFAD 训练管道
    """
    # 设置随机种子
    pl.trainer.seed_everything(rnd_seed)
    
    # 加载数据
    train_set, test_set = load_sock_shop_data(train_csv, test_csv)
    
    # 数据标准化
    scaler = tr.TimeSeriesScaler(type="robust")
    train_set = TimeSeriesDataset(tfad.utils.take_n_cycle(scaler(train_set), len(train_set)))
    test_set = TimeSeriesDataset(tfad.utils.take_n_cycle(scaler(test_set), len(test_set)))
    
    ts_channels = train_set[0].shape[1]
    print(f"Number of channels (features): {ts_channels}")
    
    # 划分验证集
    train_set, validation_set, _ = tfad.ts.split_train_val_test(
        data=train_set,
        val_portion=0.3,
        test_portion=0.0,
        split_method="past_future_with_warmup",
        split_warmup_length=window_length - suspect_window_length,
        verbose=False,
    )
    
    # 创建 DataModule
    data_module = TFADDataModule(
        train_ts_dataset=train_set,
        validation_ts_dataset=validation_set,
        test_ts_dataset=test_set,
        window_length=window_length,
        suspect_window_length=suspect_window_length,
        num_series_in_train_batch=num_series_in_train_batch,
        num_crops_per_series=num_crops_per_series,
        label_reduction_method="any",
        stride_val_and_test=10,
        num_workers=0,
    )
    
    # 距离函数
    distance = tfad.model.distances.CosineDistance()
    
    # 创建模型
    model = TFAD(
        ts_channels=ts_channels,
        window_length=window_length,
        suspect_window_length=suspect_window_length,
        tcn_kernel_size=7,
        tcn_layers=10,
        tcn_out_channels=16,
        tcn_maxpool_out_channels=29,
        embedding_rep_dim=66,
        normalize_embedding=True,
        distance=distance,
        classification_loss=nn.BCELoss(),
        classifier_threshold=0.45,
        threshold_grid_length_val=0.10,
        threshold_grid_length_test=0.05,
        coe_rate=0.5,
        mixup_rate=2.0,
        stride_rolling_val_test=10,
        val_labels_adj=True,
        test_labels_adj=True,
        max_windows_unfold_batch=5000,
        learning_rate=learning_rate,
    )
    
    # 实验名称
    if exp_name is None:
        exp_name = f"sock_shop-{time.strftime('%Y-%m-%d-%H%M%S')}"
    
    # 训练
    logger = TensorBoardLogger(save_dir=log_dir, name=exp_name)
    checkpoint_cb = ModelCheckpoint(
        monitor="val_f1",
        dirpath=model_dir,
        filename=f"tfad-sockshop-{exp_name}-{{epoch:02d}}-{{val_f1:.4f}}",
        save_top_k=1,
        mode="max",
    )
    
    trainer = Trainer(
        gpus=gpus,
        default_root_dir=model_dir,
        logger=logger,
        min_epochs=epochs,
        max_epochs=epochs,
        limit_val_batches=1.0,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=25,
        callbacks=[checkpoint_cb],
        auto_lr_find=False,
    )
    
    trainer.fit(model=model, datamodule=data_module)
    
    # 加载最佳模型
    ckpt_files = [f for f in os.listdir(model_dir) if f.startswith(f"tfad-sockshop-{exp_name}") and f.endswith(".ckpt")]
    if ckpt_files:
        ckpt_path = Path(model_dir) / ckpt_files[-1]
        model = TFAD.load_from_checkpoint(ckpt_path)
    
    # 测试
    results = trainer.test(model=model, datamodule=data_module)
    
    # 打印结果
    print("\n" + "="*50)
    print("📋 测试结果")
    print("="*50)
    for key, value in results[0].items():
        print(f"{key}: {value}")
    print("="*50)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TFAD for sock-shop dataset")
    parser.add_argument("--train_csv", type=str, required=True, help="训练数据路径")
    parser.add_argument("--test_csv", type=str, required=True, help="测试数据路径")
    parser.add_argument("--model_dir", type=str, default="models", help="模型保存目录")
    parser.add_argument("--log_dir", type=str, default="logs", help="日志目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--gpus", type=int, default=0, help="GPU数量")
    parser.add_argument("--window_length", type=int, default=200, help="窗口长度")
    parser.add_argument("--suspect_window_length", type=int, default=50, help="异常窗口长度")
    parser.add_argument("--num_series_in_train_batch", type=int, default=4, help="批次系列数")
    parser.add_argument("--num_crops_per_series", type=int, default=8, help="每系列裁剪数")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="学习率")
    
    args = parser.parse_args()
    
    sock_shop_pipeline(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        epochs=args.epochs,
        gpus=args.gpus,
        window_length=args.window_length,
        suspect_window_length=args.suspect_window_length,
        num_series_in_train_batch=args.num_series_in_train_batch,
        num_crops_per_series=args.num_crops_per_series,
        learning_rate=args.learning_rate,
    )