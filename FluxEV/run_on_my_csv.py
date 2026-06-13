import argparse
import numpy as np
import pandas as pd

from eval_methods import adjust_predicts
from main import detect


def _parse_timestamp_to_epoch_seconds(ts_series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(ts_series):
        return ts_series.to_numpy(dtype=np.int64)

    ts = pd.to_datetime(ts_series, utc=True, errors="coerce")
    mask = ts.notna().to_numpy()
    if not np.any(mask):
        raise ValueError("timestamp 列无法解析为时间")
    ts = ts[mask]
    return (ts.astype("int64") // 1_000_000_000).to_numpy(dtype=np.int64), mask


def _infer_interval_seconds_from_datetime(ts_series: pd.Series) -> int:
    ts = pd.to_datetime(ts_series, utc=True, errors="coerce").dropna()
    if len(ts) < 2:
        return 1
    diffs = ts.sort_values().diff().dt.total_seconds().dropna()
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1
    return max(int(round(float(diffs.median()))), 1)


def _align_complete_interpolate(
    timestamp_s: np.ndarray,
    values: np.ndarray,
    interval_s: int,
    labels: np.ndarray | None = None,
    phases: np.ndarray | None = None,
    agg: str = "sum",
) -> tuple[np.ndarray, np.ndarray, dict]:
    timestamp_s = np.asarray(timestamp_s, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)

    if len(timestamp_s) != len(values):
        raise ValueError("timestamp/value 长度不一致")
    if len(values) == 0:
        raise ValueError("输入数据为空")

    order = np.argsort(timestamp_s)
    timestamp_s = timestamp_s[order]
    values = values[order]
    if labels is not None:
        labels = labels[order]
    if phases is not None:
        phases = phases[order]

    start = int(timestamp_s[0])
    aligned_ts = start + ((timestamp_s - start + interval_s // 2) // interval_s) * interval_s

    if agg == "sum":
        value_by_ts = pd.Series(values, index=aligned_ts).groupby(level=0).sum().sort_index()
    elif agg == "mean":
        value_by_ts = pd.Series(values, index=aligned_ts).groupby(level=0).mean().sort_index()
    else:
        raise ValueError("agg 仅支持 sum 或 mean")

    label_by_ts = None
    if labels is not None:
        label_by_ts = pd.Series(labels, index=aligned_ts).groupby(level=0).max().sort_index()

    phase_by_ts = None
    if phases is not None:
        phase_by_ts = (
            pd.Series(phases, index=aligned_ts)
            .groupby(level=0)
            .agg(lambda s: s.value_counts().index[0] if len(s) else np.nan)
            .sort_index()
        )

    full_ts = np.arange(
        int(value_by_ts.index.min()),
        int(value_by_ts.index.max()) + interval_s,
        interval_s,
        dtype=np.int64,
    )

    full_values = value_by_ts.reindex(full_ts).to_numpy(dtype=np.float64)
    missing = np.isnan(full_values).astype(np.int32)

    filled_values = (
        pd.Series(full_values)
        .interpolate(method="linear", limit_direction="both")
        .bfill()
        .ffill()
        .to_numpy(dtype=np.float64)
    )

    full_labels = None
    if label_by_ts is not None:
        full_labels = (
            label_by_ts.reindex(full_ts)
            .fillna(0)
            .astype(np.int32)
            .to_numpy(dtype=np.int32)
        )

    full_phases = None
    if phase_by_ts is not None:
        full_phases = (
            phase_by_ts.reindex(full_ts)
            .ffill()
            .bfill()
            .astype(str)
            .to_numpy()
        )

    max_missing_num = 0
    if np.any(missing):
        run = 0
        for x in missing:
            if x:
                run += 1
                max_missing_num = max(max_missing_num, run)
            else:
                run = 0

    meta = {
        "interval_s": int(interval_s),
        "missing_count": int(np.sum(missing)),
        "max_missing_num": float(max_missing_num),
        "missing": missing,
        "label": full_labels,
        "phase": full_phases,
    }
    return full_ts, filled_values, meta


def load_as_fluxev_univariate(
    data_path: str,
    value_col: str | None = None,
    preprocess: bool = True,
    agg: str = "sum",
) -> tuple[np.ndarray, np.ndarray, dict]:
    df = pd.read_csv(data_path)

    timestamp_col = None
    for candidate in ["timestamp_s", "timestamp"]:
        if candidate in df.columns:
            timestamp_col = candidate
            break
    if timestamp_col is None:
        raise ValueError("CSV 必须包含 timestamp_s 或 timestamp 列")

    if value_col is None:
        if "value" in df.columns:
            value_col = "value"
        elif len(df.columns) >= 2:
            value_col = df.columns[1]
        else:
            raise ValueError("CSV 至少需要两列：时间列和数值列")

    if value_col not in df.columns:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        raise ValueError(f"CSV 不包含 {value_col} 列，可选数值列: {numeric_cols}")

    values = df[value_col].to_numpy(dtype=np.float64)

    labels = None
    if "label" in df.columns:
        labels = (df["label"].astype(str).str.lower() != "normal").astype(np.int32).to_numpy(dtype=np.int32)

    phases = None
    if "phase" in df.columns:
        phases = df["phase"].astype(str).to_numpy()

    if pd.api.types.is_numeric_dtype(df[timestamp_col]):
        timestamp_s = df[timestamp_col].to_numpy(dtype=np.int64)
        valid_mask = np.ones(len(df), dtype=bool)
    else:
        timestamp_s, valid_mask = _parse_timestamp_to_epoch_seconds(df[timestamp_col])
        values = values[valid_mask]
        if labels is not None:
            labels = labels[valid_mask]
        if phases is not None:
            phases = phases[valid_mask]

    if pd.api.types.is_numeric_dtype(df[timestamp_col]):
        interval_s = 1
        if len(timestamp_s) >= 2:
            diffs = np.diff(np.sort(timestamp_s))
            diffs = diffs[diffs > 0]
            if len(diffs) > 0:
                interval_s = max(int(np.median(diffs)), 1)
    else:
        interval_s = _infer_interval_seconds_from_datetime(df[timestamp_col])

    if preprocess:
        timestamp_s, values, prep_meta = _align_complete_interpolate(
            timestamp_s=timestamp_s,
            values=values,
            interval_s=interval_s,
            labels=labels,
            phases=phases,
            agg=agg,
        )
        prep_meta["value_col"] = value_col
        return timestamp_s, values, prep_meta

    order = np.argsort(timestamp_s)
    timestamp_s = timestamp_s[order]
    values = values[order]
    if labels is not None:
        labels = labels[order]
    if phases is not None:
        phases = phases[order]

    meta = {
        "interval_s": int(interval_s),
        "max_missing_num": 0.0,
        "missing_count": 0,
        "value_col": value_col,
        "phase": phases,
        "label": labels,
    }
    return timestamp_s, values, meta


def infer_train_len(data_len: int, phases: np.ndarray | None = None) -> int:
    if phases is not None and len(phases) == data_len and data_len > 1:
        if str(phases[0]) == "baseline":
            change_idx = np.where(phases != "baseline")[0]
            if len(change_idx) > 0:
                return int(change_idx[0])
    return data_len // 2


def run_fluxev(
    values: np.ndarray,
    train_len: int,
    period: int,
    s_w: int,
    p_w: int,
    half_d_w: int,
    q: float,
    estimator: str,
    smoothing: int,
) -> tuple[np.ndarray, int, int, int]:
    data_len = len(values)
    fs_idx = s_w * 2
    ss_idx = fs_idx + half_d_w + period * (p_w - 1)

    min_init_idx = ss_idx
    if smoothing == 2:
        if train_len - 1 < ss_idx or data_len - 1 < ss_idx:
            smoothing = 1
            min_init_idx = fs_idx
    else:
        smoothing = 1
        min_init_idx = fs_idx

    if data_len - 1 < min_init_idx:
        raise ValueError(
            f"数据太短无法初始化检测器: data_len={data_len}, 需要至少 {min_init_idx + 2} 个点"
        )

    if train_len - 1 < min_init_idx:
        train_len = min(min_init_idx + 1, data_len - 1)

    alarms_test = detect(
        values,
        train_len=train_len,
        period=period,
        smoothing=smoothing,
        s_w=s_w,
        p_w=p_w,
        half_d_w=half_d_w,
        q=q,
        estimator=estimator,
    )

    alarms_full = np.zeros(data_len, dtype=np.int32)
    alarms_full[train_len:] = alarms_test.astype(np.int32)
    return alarms_full, train_len, smoothing, min_init_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FluxEV on prepared univariate CSV")
    parser.add_argument(
        "--data_path",
        type=str,
        default=r"data\search_long_sequence_20260611_154424.csv",
    )
    parser.add_argument("--value_col", type=str, default=None)
    parser.add_argument("--delay", type=int, default=7)
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--no_preprocess", action="store_true")
    parser.add_argument("--agg", type=str, default="sum", choices=["sum", "mean"])
    parser.add_argument("--period_points", type=int, default=30)
    parser.add_argument("--smoothing", type=int, default=2, choices=[1, 2])
    parser.add_argument("--invert", action="store_true")

    parser.add_argument("--s_w", type=int, default=10)
    parser.add_argument("--p_w", type=int, default=5)
    parser.add_argument("--half_d_w", type=int, default=2)
    parser.add_argument("--q", type=float, default=0.003)
    parser.add_argument("--estimator", type=str, default="MOM", choices=["MOM", "MLE"])
    parser.add_argument("--train_len", type=int, default=None)

    args = parser.parse_args()

    preprocess = True
    if args.no_preprocess:
        preprocess = False
    elif args.preprocess:
        preprocess = True

    timestamp_s, values, meta = load_as_fluxev_univariate(
        args.data_path,
        value_col=args.value_col,
        preprocess=preprocess,
        agg=args.agg,
    )
    if args.invert:
        values = -values

    data_len = len(values)
    interval_s = meta["interval_s"]
    period = max(int(args.period_points), 1)

    inferred_train_len = infer_train_len(data_len, phases=meta["phase"])
    train_len = args.train_len if args.train_len is not None else inferred_train_len
    train_len = max(min(int(train_len), data_len - 1), 1)

    alarms_full, train_len, smoothing, min_init_idx = run_fluxev(
        values=values,
        train_len=train_len,
        period=period,
        s_w=args.s_w,
        p_w=args.p_w,
        half_d_w=args.half_d_w,
        q=args.q,
        estimator=args.estimator,
        smoothing=args.smoothing,
    )

    anomaly_idx = np.where(alarms_full == 1)[0]
    print(f"data_path: {args.data_path}")
    print(f"value_col: {meta['value_col']}")
    print(f"invert: {bool(args.invert)}")
    print(f"data_len: {data_len}")
    print(f"interval_s: {interval_s} / period_points: {period}")
    print(f"missing_count: {meta['missing_count']} / max_missing_num: {meta['max_missing_num']}")
    print(f"smoothing: {smoothing} / min_init_idx: {min_init_idx}")
    print(f"train_len: {train_len}")
    print(f"q: {args.q} / estimator: {args.estimator}")
    print(f"anomalies: {len(anomaly_idx)}")

    if len(anomaly_idx) > 0:
        show_n = min(30, len(anomaly_idx))
        print("\n异常点(最多展示前 30 个):")
        for i in anomaly_idx[:show_n]:
            print(f"  idx={int(i):6d} ts={int(timestamp_s[i])} value={values[i]:.10f}")

    if meta["label"] is not None:
        label_test = meta["label"][train_len:]
        pred_test = alarms_full[train_len:]
        adj_test = adjust_predicts(pred_test, label_test, delay=args.delay)
        print(f"\n测试段真实异常点数: {int(np.sum(label_test))}")
        print(f"测试段原始检测点数: {int(np.sum(pred_test))}")
        print(f"测试段调整后命中点数: {int(np.sum(adj_test))}")
