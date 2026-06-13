import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from run_on_my_csv import load_as_fluxev_univariate, infer_train_len, run_fluxev


def _build_aligned_anomaly_id_and_phase(
    data_path: str,
    full_ts: np.ndarray,
    interval_s: int,
    start_ts: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw = pd.read_csv(data_path)
    ts = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    raw = raw.assign(_ts=ts).dropna(subset=["_ts"]).copy()
    raw = raw.assign(timestamp_s=(raw["_ts"].astype("int64") // 1_000_000_000).astype("int64"))

    aligned_ts = start_ts + ((raw["timestamp_s"] - start_ts + interval_s // 2) // interval_s) * interval_s

    anomaly_id_by_ts = raw.groupby(aligned_ts)["anomaly_id"].max().sort_index()
    anomaly_id = anomaly_id_by_ts.reindex(full_ts).fillna(0).astype(int).to_numpy()

    if "phase" in raw.columns:
        phase_by_ts = (
            raw["phase"]
            .astype(str)
            .groupby(aligned_ts)
            .agg(lambda s: s.value_counts().index[0] if len(s) else np.nan)
            .sort_index()
        )
        phase = phase_by_ts.reindex(full_ts).ffill().bfill().astype(str).to_numpy()
    else:
        phase = np.full(len(full_ts), "", dtype=object)

    return anomaly_id, phase


def _segments_from_anomaly_id(anomaly_id: np.ndarray) -> list[tuple[int, int, int]]:
    segments: list[tuple[int, int, int]] = []
    in_run = False
    st = 0
    cur = 0
    for i, aid in enumerate(anomaly_id):
        aid = int(aid)
        if aid != 0 and not in_run:
            in_run = True
            st = i
            cur = aid
        elif in_run and aid != cur:
            segments.append((cur, st, i - 1))
            in_run = False
            if aid != 0:
                in_run = True
                st = i
                cur = aid
    if in_run:
        segments.append((cur, st, len(anomaly_id) - 1))
    return segments


def _first_index(arr: np.ndarray, value: str, lo: int, hi: int) -> int | None:
    for i in range(lo, hi + 1):
        if str(arr[i]) == value:
            return i
    return None


def main():
    data_path = r"data\search_long_sequence_20260611_154424.csv"
    value_col = "cpu"

    s_w = 10
    p_w = 5
    half_d_w = 2
    q = 0.003
    estimator = "MOM"
    train_len = None
    smoothing = 2
    period_points = 30

    timestamp_s, values, meta = load_as_fluxev_univariate(
        data_path,
        value_col=value_col,
        preprocess=True,
        agg="sum",
    )
    relative_ts = timestamp_s - timestamp_s[0]

    data_len = len(values)
    interval_s = meta["interval_s"]
    period = period_points

    inferred_train_len = infer_train_len(data_len, phases=meta["phase"])
    train_len = train_len if train_len is not None else inferred_train_len
    train_len = max(min(int(train_len), data_len - 1), 1)

    alarms_full, train_len, smoothing, _ = run_fluxev(
        values=values,
        train_len=train_len,
        period=period,
        s_w=s_w,
        p_w=p_w,
        half_d_w=half_d_w,
        q=q,
        estimator=estimator,
        smoothing=smoothing,
    )
    anomalies = alarms_full == 1

    anomaly_id, phase = _build_aligned_anomaly_id_and_phase(
        data_path=data_path,
        full_ts=timestamp_s,
        interval_s=interval_s,
        start_ts=int(timestamp_s[0]),
    )
    hit_idx = np.where(anomalies)[0]
    hit_phase = pd.Series(phase[hit_idx], name="phase")
    hit_anomaly_id = pd.Series(anomaly_id[hit_idx], name="anomaly_id")
    hits_by_phase = pd.crosstab(hit_anomaly_id, hit_phase)
    hits_by_phase.to_csv("hits_by_anomaly_phase.csv", index=True)

    segments = _segments_from_anomaly_id(anomaly_id)
    offset_rows: list[dict] = []
    for aid, st, ed in segments:
        fault_start = _first_index(phase, "fault", st, ed)
        recovery_start = _first_index(phase, "recovery", st, ed)
        seg_hits = hit_idx[(hit_idx >= st) & (hit_idx <= ed)]
        for idx in seg_hits:
            ph = str(phase[idx])
            if ph == "fault" and fault_start is not None:
                offset_s = int((idx - fault_start) * interval_s)
                offset_rows.append(
                    {
                        "anomaly_id": int(aid),
                        "phase": "fault",
                        "hit_idx": int(idx),
                        "offset_s": offset_s,
                        "offset_min": offset_s / 60.0,
                    }
                )
            elif ph == "recovery" and recovery_start is not None:
                offset_s = int((idx - recovery_start) * interval_s)
                offset_rows.append(
                    {
                        "anomaly_id": int(aid),
                        "phase": "recovery",
                        "hit_idx": int(idx),
                        "offset_s": offset_s,
                        "offset_min": offset_s / 60.0,
                    }
                )

    offsets_df = pd.DataFrame(offset_rows)
    offsets_df.to_csv("hit_offsets_by_phase.csv", index=False)

    bins_min = np.arange(0, 61, 5)
    cum_rows: list[dict] = []
    for ph in ["fault", "recovery"]:
        x = offsets_df[offsets_df["phase"] == ph]["offset_min"].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        total = int(len(x))
        for n in bins_min:
            cnt = int(np.sum(x <= n))
            frac = float(cnt / total) if total else 0.0
            cum_rows.append({"phase": ph, "n_min": int(n), "count_leq": cnt, "total": total, "frac": frac})
    cum_df = pd.DataFrame(cum_rows)
    cum_df.to_csv("hit_cumulative_by_minutes.csv", index=False)

    plt.figure(figsize=(14, 5))
    plt.plot(relative_ts, values, label=value_col, color="steelblue", alpha=0.7)
    if np.any(anomalies):
        plt.scatter(
            relative_ts[anomalies],
            values[anomalies],
            color="crimson",
            s=60,
            label="Detected Anomaly",
            zorder=5,
        )
    plt.axvline(relative_ts[train_len], color="black", linewidth=1, alpha=0.4, label="Train/Test split")
    plt.xlabel("Relative Time (s)")
    plt.ylabel(value_col)
    plt.title(f"FluxEV Detection (smoothing={smoothing}, q={q}, s_w={s_w}, p_w={p_w}, d={half_d_w})")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("detection_result.png", dpi=150)
    plt.show()

    print(f"检测到异常点数量: {int(np.sum(anomalies))}")
    print("\n命中点在 phase 上的分布（anomaly_id x phase）:")
    if hits_by_phase.shape[0] == 0:
        print("(empty)")
    else:
        print(hits_by_phase.to_string())

    print("\n命中点集中度（前 N 分钟命中比例）:")
    summary = cum_df.pivot(index="n_min", columns="phase", values="frac").fillna(0.0)
    print(summary.to_string())

    if hits_by_phase.shape[0] > 0:
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        data = hits_by_phase.to_numpy(dtype=float)
        im = ax.imshow(data, aspect="auto", cmap="Blues")
        ax.set_xlabel("phase")
        ax.set_ylabel("anomaly_id")
        ax.set_xticks(range(len(hits_by_phase.columns)))
        ax.set_xticklabels([str(x) for x in hits_by_phase.columns])
        ax.set_yticks(range(len(hits_by_phase.index)))
        ax.set_yticklabels([str(x) for x in hits_by_phase.index])
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, str(int(data[i, j])), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig("hit_phase_heatmap.png", dpi=150)
        plt.close(fig)

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    for ph in ["fault", "recovery"]:
        sub = cum_df[cum_df["phase"] == ph].sort_values("n_min")
        ax.plot(sub["n_min"], sub["frac"], marker="o", label=ph)
    ax.set_xlabel("N (minutes)")
    ax.set_ylabel("Fraction of hits within first N minutes")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("hit_offset_cdf.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    if not offsets_df.empty:
        for ph, color in [("fault", "tab:orange"), ("recovery", "tab:green")]:
            x = offsets_df[offsets_df["phase"] == ph]["offset_min"].to_numpy(dtype=float)
            ax.hist(x, bins=bins_min, alpha=0.6, label=ph, color=color)
    ax.set_xlabel("Minutes since phase start")
    ax.set_ylabel("Hit count")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("hit_offset_hist.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
