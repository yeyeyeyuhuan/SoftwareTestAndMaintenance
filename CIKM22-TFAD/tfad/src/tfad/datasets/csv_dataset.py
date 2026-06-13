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

from typing import Union
from pathlib import Path

import numpy as np
import pandas as pd

from tfad.ts import TimeSeries, TimeSeriesDataset


def csv_dataset(
    train_path: Union[Path, str],
    test_path: Union[Path, str],
    timestamp_col: str = "timestamp",
    label_col: str = "label",
    *args, **kwargs
) -> TimeSeriesDataset:
    """
    Load multivariate time series from CSV files.
    
    Args:
        train_path : Path to the training CSV file.
        test_path : Path to the test CSV file.
        timestamp_col : Name of the timestamp column (default: "timestamp")
        label_col : Name of the label column (default: "label")
        
    Returns:
        train_dataset, test_dataset : Tuple of TimeSeriesDataset objects
    """
    print("Loading CSV datasets...")
    
    # Expand paths
    train_path = Path(train_path).expanduser()
    test_path = Path(test_path).expanduser()
    
    assert train_path.is_file(), f"Train file not found: {train_path}"
    assert test_path.is_file(), f"Test file not found: {test_path}"

    # Load CSV files
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Extract feature columns (all columns except timestamp, label, and Unnamed columns)
    feature_cols = [
        col for col in train_df.columns 
        if col not in [timestamp_col, label_col] and not col.startswith("Unnamed:")
    ]
    
    # Ensure feature columns exist in both train and test data
    feature_cols = [col for col in feature_cols if col in test_df.columns]
    
    print(f"Found {len(feature_cols)} features: {feature_cols}")
    
    # Extract values and labels
    train_values = train_df[feature_cols].to_numpy().astype(np.float32)
    train_labels = train_df[label_col].to_numpy().astype(np.float32)
    
    test_values = test_df[feature_cols].to_numpy().astype(np.float32)
    test_labels = test_df[label_col].to_numpy().astype(np.float32)
    
    # Check for NaN values
    if np.isnan(train_values).any():
        print("Warning: NaN values found in training data")
        train_values = np.nan_to_num(train_values)
    
    if np.isnan(test_values).any():
        print("Warning: NaN values found in test data")
        test_values = np.nan_to_num(test_values)
    
    # Create TimeSeriesDataset
    train_dataset = TimeSeriesDataset([
        TimeSeries(
            values=train_values,
            labels=train_labels,
            item_id="csv_train",
        )
    ])
    
    test_dataset = TimeSeriesDataset([
        TimeSeries(
            values=test_values,
            labels=test_labels,
            item_id="csv_test",
        )
    ])
    
    print(f"Train dataset: {len(train_dataset[0].values)} timesteps, {train_dataset[0].shape[1]} features")
    print(f"Test dataset: {len(test_dataset[0].values)} timesteps, {test_dataset[0].shape[1]} features")
    
    return train_dataset, test_dataset