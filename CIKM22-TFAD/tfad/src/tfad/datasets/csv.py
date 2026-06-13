from typing import Union, Iterable, Optional
from pathlib import PosixPath

import pandas as pd

from tfad.ts import TimeSeries, TimeSeriesDataset


def from_csv(
    path: Union[PosixPath, str],
    timestamp_col: Optional[str] = None,
    value_cols: Optional[Iterable[str]] = None,
    label_col: Optional[str] = None,
    sep: str = ",",
) -> TimeSeriesDataset:
    """Load a CSV file into a TimeSeriesDataset.

    Assumptions:
    - If `value_cols` is None, all numeric columns except `timestamp_col` and `label_col` are used.
    - Returns a dataset with a single TimeSeries item (multivariate if multiple value_cols).

    Args:
        path: CSV file path
        timestamp_col: optional timestamp column name (ignored except for dropping)
        value_cols: iterable of column names to use as values
        label_col: optional label column name containing anomaly flags (0/1)
        sep: CSV separator
    """
    p = PosixPath(path)
    df = pd.read_csv(p, sep=sep)

    # Determine value columns
    cols = list(df.columns)
    exclude = set()
    if timestamp_col:
        exclude.add(timestamp_col)
    if label_col:
        exclude.add(label_col)

    if value_cols is None:
        value_cols = [c for c in cols if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    values = df[list(value_cols)].to_numpy()

    labels = None
    if label_col and label_col in df.columns:
        labels = df[label_col].to_numpy()

    dataset = TimeSeriesDataset()
    dataset.append(TimeSeries(values=values, labels=labels, item_id=p.stem))
    return dataset
