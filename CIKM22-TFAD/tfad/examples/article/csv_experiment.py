# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import os
import time
from pathlib import Path

from typing import Optional, Union

import itertools

import json

import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import tfad
from tfad.ts import TimeSeriesDataset
from tfad.ts import transforms as tr
from tfad.model import TFAD, TFADDataModule


def csv_inject_anomalies(
    dataset: TimeSeriesDataset,
    rate_true_anomalies_used: float = 1.0,
    injection_method: str = ["None", "local_outliers"][0],
    ratio_injected_spikes: float = None,
) -> TimeSeriesDataset:
    """Inject anomalies into the dataset for training"""

    # Transform using LabelNoise to ignore some true labels
    ts_transform = tr.LabelNoise(
        p_flip_1_to_0=1.0 - rate_true_anomalies_used
    )

    if injection_method == "None":
        ts_transform_iterator = ts_transform(dataset)
        dataset_transformed = tfad.utils.take_n_cycle(ts_transform_iterator, len(dataset))
        dataset_transformed = TimeSeriesDataset(dataset_transformed)
    elif injection_method == "local_outliers":
        if ratio_injected_spikes is None:
            raise ValueError("ratio_injected_spikes must be specified for local_outliers")
        
        anom_transform = tr.LocalOutlier(
            area_radius=2000,
            num_spikes=ratio_injected_spikes,
            spike_multiplier_range=(1.0, 4.0),
            direction_options=["increase"],
        )
        ts_transform = ts_transform + anom_transform

        multiplier = 5
        ts_transform_iterator = ts_transform(itertools.cycle(dataset))
        dataset_transformed = tfad.utils.take_n_cycle(
            ts_transform_iterator, multiplier * len(dataset)
        )
        dataset_transformed = TimeSeriesDataset(dataset_transformed)
    else:
        raise ValueError(f"injection_method = {injection_method} not supported!")

    return dataset_transformed


def csv_pipeline(
    train_csv_path: Union[str, Path],
    test_csv_path: Union[str, Path],
    model_dir: Union[str, Path],
    log_dir: Union[str, Path],
    ## General
    exp_name: Optional[str] = None,
    ## For trainer
    epochs: int = 500,
    gpus: int = 1 if torch.cuda.is_available() else 0,
    limit_val_batches: float = 1.0,
    num_sanity_val_steps: int = 1,
    ## For injection
    injection_method: str = ["None", "local_outliers"][0],
    ratio_injected_spikes: float = None,
    ## For DataLoader
    window_length: int = 2000,
    suspect_window_length: int = 50,
    validation_portion: float = 0.3,
    train_split_method: str = "past_future_with_warmup",
    num_series_in_train_batch: int = 8,
    num_crops_per_series: int = 16,
    rate_true_anomalies_used: float = 0.0,
    num_workers_loader: int = 0,
    ## For model definition
    # hpars for encoder
    tcn_kernel_size: int = 7,
    tcn_layers: int = 10,
    tcn_out_channels: int = 16,
    tcn_maxpool_out_channels: int = 29,
    embedding_rep_dim: int = 66,
    normalize_embedding: bool = True,
    # hpars for classifier
    distance: str = ["cosine", "L2", "non-contrastive"][0],
    classifier_threshold: float = 0.5,
    threshold_grid_length_val: float = 0.10,
    threshold_grid_length_test: float = 0.05,
    # hpars for anomalizers
    coe_rate: float = 0.5,
    mixup_rate: float = 2.0,
    # hpars for optimizer
    learning_rate: float = 3e-4,
    # hpars for validation and test
    check_val_every_n_epoch: int = 25,
    stride_roll_pred_val_test: int = 10,
    val_labels_adj: bool = True,
    test_labels_adj: bool = True,
    max_windows_unfold_batch: Optional[int] = 5000,
    evaluation_result_path: Optional[Union[str, Path]] = None,
    # For reproducibility
    rnd_seed: int = 123,
    **kwargs,
):
    """
    Run TFAD pipeline on custom CSV time series data.
    
    Args:
        train_csv_path: Path to training CSV file
        test_csv_path: Path to test CSV file
        model_dir: Directory to save model checkpoints
        log_dir: Directory to save logs
        ... other parameters for model configuration
    """

    # Expand user paths
    train_csv_path = Path(train_csv_path).expanduser() if str(train_csv_path).startswith("~") else Path(train_csv_path)
    test_csv_path = Path(test_csv_path).expanduser() if str(test_csv_path).startswith("~") else Path(test_csv_path)
    model_dir = Path(model_dir).expanduser() if str(model_dir).startswith("~") else Path(model_dir)
    log_dir = Path(log_dir).expanduser() if str(log_dir).startswith("~") else Path(log_dir)

    # Create directories if inexistent
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if (not os.path.exists(log_dir)) and (not str(log_dir).startswith("s3://")):
        os.makedirs(log_dir)

    # Set random seed
    pl.trainer.seed_everything(rnd_seed)

    #####     Load Data     #####

    train_set, test_set = tfad.datasets.csv_dataset(
        train_path=train_csv_path,
        test_path=test_csv_path,
    )
    
    # Standardize TimeSeries values (subtract median, divide by interquartile range)
    scaler = tr.TimeSeriesScaler(type="robust")
    train_set = TimeSeriesDataset(tfad.utils.take_n_cycle(scaler(train_set), len(train_set)))
    test_set = TimeSeriesDataset(tfad.utils.take_n_cycle(scaler(test_set), len(test_set)))
    
    # Number of channels in TimeSeries
    ts_channels = train_set[0].shape[1]
    print(f"Number of channels (features): {ts_channels}")
    assert all(shape[1] == ts_channels for shape in train_set.shape)
    assert all(shape[1] == ts_channels for shape in test_set.shape)

    # Split dataset
    train_set, validation_set, _ = tfad.ts.split_train_val_test(
        data=train_set,
        val_portion=validation_portion,
        test_portion=0.0,
        split_method=train_split_method,
        split_warmup_length=window_length - suspect_window_length
        if train_split_method == "past_future_with_warmup"
        else None,
        verbose=False,
    )

    #### inject anomalies on train dataset ###
    train_set_transformed = csv_inject_anomalies(
        dataset=train_set,
        rate_true_anomalies_used=rate_true_anomalies_used,
        injection_method=injection_method,
        ratio_injected_spikes=ratio_injected_spikes,
    )

    # Define DataModule for training with pytorch lighting
    data_module = TFADDataModule(
        train_ts_dataset=train_set_transformed,
        validation_ts_dataset=validation_set,
        test_ts_dataset=test_set,
        window_length=window_length,
        suspect_window_length=suspect_window_length,
        num_series_in_train_batch=num_series_in_train_batch,
        num_crops_per_series=num_crops_per_series,
        label_reduction_method="any",
        stride_val_and_test=stride_roll_pred_val_test,
        num_workers=num_workers_loader,
    )

    if distance == "cosine":
        distance = tfad.model.distances.CosineDistance()
    elif distance == "L2":
        distance = tfad.model.distances.LpDistance(p=2)
    elif distance == "non-contrastive":
        distance = tfad.model.distances.BinaryOnX1(rep_dim=embedding_rep_dim, layers=1)

    # Instantiate model
    model = TFAD(
        ts_channels=ts_channels,
        window_length=window_length,
        suspect_window_length=suspect_window_length,
        # hpars for encoder
        tcn_kernel_size=tcn_kernel_size,
        tcn_layers=tcn_layers,
        tcn_out_channels=tcn_out_channels,
        tcn_maxpool_out_channels=tcn_maxpool_out_channels,
        embedding_rep_dim=embedding_rep_dim,
        normalize_embedding=normalize_embedding,
        # hpars for classifier
        distance=distance,
        classification_loss=nn.BCELoss(),
        classifier_threshold=classifier_threshold,
        threshold_grid_length_val=threshold_grid_length_val,
        threshold_grid_length_test=threshold_grid_length_test,
        # hpars for anomalizers
        coe_rate=coe_rate,
        mixup_rate=mixup_rate,
        # hpars for validation and test
        stride_rolling_val_test=stride_roll_pred_val_test,
        val_labels_adj=val_labels_adj,
        test_labels_adj=test_labels_adj,
        max_windows_unfold_batch=max_windows_unfold_batch,
        # hpars for optimizer
        learning_rate=learning_rate,
    )

    # Experiment name
    if exp_name is None:
        time_now = time.strftime("%Y-%m-%d-%H%M%S", time.localtime())
        exp_name = f"csv-{time_now}"

    ### Training the model ###

    logger = TensorBoardLogger(save_dir=log_dir, name=exp_name)

    # Checkpoint callback, monitoring 'val_f1'
    checkpoint_cb = ModelCheckpoint(
        monitor="val_f1",
        dirpath=model_dir,
        filename="tfad-model-" + exp_name + "-{epoch:02d}-{val_f1:.4f}",
        save_top_k=1,
        mode="max",
    )

    trainer = Trainer(
        gpus=gpus,
        default_root_dir=model_dir,
        logger=logger,
        min_epochs=epochs,
        max_epochs=epochs,
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=num_sanity_val_steps,
        check_val_every_n_epoch=check_val_every_n_epoch,
        callbacks=[checkpoint_cb],
        auto_lr_find=False,
    )

    trainer.fit(
        model=model,
        datamodule=data_module,
    )

    # Load top performing checkpoint
    ckpt_file = [
        file
        for file in os.listdir(model_dir)
        if (file.endswith(".ckpt") and file.startswith("tfad-model-" + exp_name))
    ][-1]
    ckpt_path = model_dir / ckpt_file
    model = TFAD.load_from_checkpoint(ckpt_path)

    # Metrics on validation and test data
    evaluation_result = trainer.test(model=model, datamodule=data_module)
    evaluation_result = evaluation_result[0]

    # Save evaluation results
    if evaluation_result_path is not None:
        path = evaluation_result_path
        path = Path(path).expanduser() if str(path).startswith("~") else Path(path)
        with open(path, "w") as f:
            json.dump(evaluation_result, f, cls=tfad.utils.NpEncoder)

    for key, value in evaluation_result.items():
        print(f"{key}={value}")

    print(f"TFAD on CSV dataset finished successfully!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run TFAD on custom CSV time series data")
    
    # Required arguments
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory to save logs")
    
    # Optional arguments with defaults
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--gpus", type=int, default=1 if torch.cuda.is_available() else 0, help="Number of GPUs")
    parser.add_argument("--window_length", type=int, default=2000, help="Window length for training")
    parser.add_argument("--suspect_window_length", type=int, default=50, help="Suspect window length")
    parser.add_argument("--num_series_in_train_batch", type=int, default=8, help="Number of series per batch")
    parser.add_argument("--num_crops_per_series", type=int, default=16, help="Number of crops per series")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--rnd_seed", type=int, default=123, help="Random seed")
    
    args = parser.parse_args()
    
    # Run the pipeline
    csv_pipeline(
        train_csv_path=args.train_csv,
        test_csv_path=args.test_csv,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        exp_name=args.exp_name,
        epochs=args.epochs,
        gpus=args.gpus,
        window_length=args.window_length,
        suspect_window_length=args.suspect_window_length,
        num_series_in_train_batch=args.num_series_in_train_batch,
        num_crops_per_series=args.num_crops_per_series,
        learning_rate=args.learning_rate,
        rnd_seed=args.rnd_seed,
    )