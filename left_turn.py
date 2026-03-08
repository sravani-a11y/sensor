"""
Left-turn detection from 40-sample IMU windows

What this script does
- Reads a CSV file containing IMU data
- Uses a 40-sample rolling window
- Detects left-turn events using the rule-based algorithm:
    1) gx_mean < -4 dps
    2) gx_min < -12 dps
    3) gx_integral < -15 deg
  OR relaxed rule:
    1) gx_mean < -3 dps
    2) gx_min < -10 dps
    3) gx_integral < -12 deg
    4) az_mean < -0.03 g
- Merges overlapping/nearby detections into events
- Prints detected events with sample index and time
- Optionally plots gx with detected regions

How to run
1. Save as: detect_left_turn.py
2. Install packages:
       pip install pandas numpy matplotlib
3. Run:
       python detect_left_turn.py --csv Trip2Sample.csv

Optional:
       python detect_left_turn.py --csv Trip2Sample.csv --plot
       python detect_left_turn.py --csv Trip2Sample.csv --window 40 --merge-gap 20

Notes
- Adjust COLUMN_MAP below if your CSV column names differ.
- This script tries to auto-detect timestamp and IMU columns if possible.
"""

import argparse
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# Configurable parameters
# =========================

DEFAULT_WINDOW = 40
DEFAULT_SMOOTH = 5
DEFAULT_DT = 0.102  # fallback if timestamp is unavailable
DEFAULT_MERGE_GAP = 10  # samples; detections closer than this are merged

# Thresholds from the proposed algorithm
STRICT_GX_MEAN_THR = -4.0
STRICT_GX_MIN_THR = -12.0
STRICT_GX_INT_THR = -15.0

RELAX_GX_MEAN_THR = -3.0
RELAX_GX_MIN_THR = -10.0
RELAX_GX_INT_THR = -12.0
RELAX_AZ_MEAN_THR = -0.03


# =========================
# Data classes
# =========================

@dataclass
class DetectionWindow:
    start_idx: int
    end_idx: int
    gx_mean: float
    gx_min: float
    gx_integral_deg: float
    az_mean: float
    rule_used: str


@dataclass
class Event:
    start_idx: int
    end_idx: int
    peak_negative_gx: float
    mean_gx: float
    integrated_rotation_deg: float


# =========================
# Utility functions
# =========================

def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    """Simple trailing moving average with min_periods=1 behavior."""
    if k <= 1:
        return x.copy()
    return pd.Series(x).rolling(window=k, min_periods=1).mean().to_numpy()


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return first matching column by exact or normalized match."""
    cols = list(df.columns)
    norm_map = {normalize_name(c): c for c in cols}

    for cand in candidates:
        if cand in cols:
            return cand
        nc = normalize_name(cand)
        if nc in norm_map:
            return norm_map[nc]

    for c in cols:
        cn = normalize_name(c)
        for cand in candidates:
            if normalize_name(cand) in cn or cn in normalize_name(cand):
                return c
    return None


def normalize_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "")
    )


def parse_timestamp_column(series: pd.Series) -> Optional[pd.Series]:
    """
    Try to parse a timestamp column.
    Supports:
    - full datetime strings
    - time-only strings
    - numeric seconds
    """
    # Numeric seconds
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series.astype(float), unit="s", errors="coerce")

    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().sum() > 0:
        return parsed

    # Try time-only format by attaching dummy date
    try:
        parsed = pd.to_datetime("1970-01-01 " + series.astype(str), errors="coerce")
        if parsed.notna().sum() > 0:
            return parsed
    except Exception:
        pass

    return None


def estimate_dt_seconds(df: pd.DataFrame, time_col: Optional[str], fallback_dt: float) -> float:
    """Estimate dt from timestamp column if available, else use fallback."""
    if time_col is None:
        return fallback_dt

    parsed = parse_timestamp_column(df[time_col])
    if parsed is None or parsed.notna().sum() < 3:
        return fallback_dt

    diffs = parsed.diff().dt.total_seconds().dropna()
    diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
    if len(diffs) == 0:
        return fallback_dt

    return float(np.median(diffs))


def format_time_value(val) -> str:
    """Nicely format timestamp-like values."""
    if pd.isna(val):
        return "N/A"
    try:
        ts = pd.to_datetime(val)
        return ts.strftime("%H:%M:%S.%f")[:-3]
    except Exception:
        return str(val)


def choose_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Auto-detect timestamp, gx, az columns.
    Adjust candidate lists if your file uses different names.
    """
    time_candidates = [
        "timestamp",
        "time",
        "datetime",
        "DateTime",
        "Time",
        "Timestamp",
    ]

    gx_candidates = [
        "gx(dps)",
        "gx",
        "gyro_x",
        "gyro_x_dps",
        "gX",
        "gyroX",
        "GyroX",
    ]

    az_candidates = [
        "az(g)",
        "az",
        "acc_z",
        "accel_z",
        "aZ",
        "accZ",
        "AccelZ",
    ]

    time_col = find_column(df, time_candidates)
    gx_col = find_column(df, gx_candidates)
    az_col = find_column(df, az_candidates)

    return time_col, gx_col, az_col


# =========================
# Detection logic
# =========================

def detect_left_turn_windows(
    gx: np.ndarray,
    az: np.ndarray,
    dt: float,
    window: int,
    smooth_k: int,
) -> List[DetectionWindow]:
    """
    Run rolling 40-sample left-turn detector.
    Returns all windows that satisfy the detector.
    """
    gx_s = moving_average(gx, smooth_k)
    az_s = moving_average(az, smooth_k)

    detections: List[DetectionWindow] = []

    if len(gx_s) < window:
        return detections

    for start in range(0, len(gx_s) - window + 1):
        end = start + window - 1

        gx_w = gx_s[start : start + window]
        az_w = az_s[start : start + window]

        gx_mean = float(np.mean(gx_w))
        gx_min = float(np.min(gx_w))
        gx_integral_deg = float(np.sum(gx_w) * dt)
        az_mean = float(np.mean(az_w))

        strict_ok = (
            gx_mean < STRICT_GX_MEAN_THR
            and gx_min < STRICT_GX_MIN_THR
            and gx_integral_deg < STRICT_GX_INT_THR
        )

        relax_ok = (
            gx_mean < RELAX_GX_MEAN_THR
            and gx_min < RELAX_GX_MIN_THR
            and gx_integral_deg < RELAX_GX_INT_THR
            and az_mean < RELAX_AZ_MEAN_THR
        )

        if strict_ok or relax_ok:
            detections.append(
                DetectionWindow(
                    start_idx=start,
                    end_idx=end,
                    gx_mean=gx_mean,
                    gx_min=gx_min,
                    gx_integral_deg=gx_integral_deg,
                    az_mean=az_mean,
                    rule_used="strict" if strict_ok else "relaxed",
                )
            )

    return detections


def merge_detection_windows(
    detections: List[DetectionWindow],
    gx: np.ndarray,
    dt: float,
    merge_gap: int,
) -> List[Event]:
    """
    Merge overlapping or nearby detection windows into events.
    merge_gap is in samples.
    """
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d.start_idx)
    merged: List[Event] = []

    cur_start = detections[0].start_idx
    cur_end = detections[0].end_idx

    for det in detections[1:]:
        if det.start_idx <= cur_end + merge_gap:
            cur_end = max(cur_end, det.end_idx)
        else:
            merged.append(build_event(cur_start, cur_end, gx, dt))
            cur_start, cur_end = det.start_idx, det.end_idx

    merged.append(build_event(cur_start, cur_end, gx, dt))
    return merged


def build_event(start_idx: int, end_idx: int, gx: np.ndarray, dt: float) -> Event:
    gx_seg = gx[start_idx : end_idx + 1]
    return Event(
        start_idx=start_idx,
        end_idx=end_idx,
        peak_negative_gx=float(np.min(gx_seg)),
        mean_gx=float(np.mean(gx_seg)),
        integrated_rotation_deg=float(np.sum(gx_seg) * dt),
    )


# =========================
# Reporting
# =========================

def print_summary(
    df: pd.DataFrame,
    time_col: Optional[str],
    dt: float,
    detections: List[DetectionWindow],
    events: List[Event],
) -> None:
    print("\n==============================")
    print("Left Turn Detection Summary")
    print("==============================")
    print(f"Total samples           : {len(df)}")
    print(f"Estimated sample period : {dt:.4f} s")
    print(f"Window duration         : {DEFAULT_WINDOW * dt:.3f} s")
    print(f"Triggered windows       : {len(detections)}")
    print(f"Detected events         : {len(events)}")

    if not events:
        print("\nNo left-turn events detected.")
        return

    print("\nDetected events:\n")
    header = (
        f"{'Event':<6} {'StartIdx':<9} {'EndIdx':<8} "
        f"{'StartTime':<15} {'EndTime':<15} {'Duration(s)':<12} "
        f"{'PeakNegGX':<10} {'MeanGX':<10} {'IntRot(deg)':<12}"
    )
    print(header)
    print("-" * len(header))

    for i, ev in enumerate(events, start=1):
        duration = (ev.end_idx - ev.start_idx + 1) * dt

        if time_col is not None:
            start_time = format_time_value(df.iloc[ev.start_idx][time_col])
            end_time = format_time_value(df.iloc[ev.end_idx][time_col])
        else:
            start_time = f"{ev.start_idx * dt:.2f}s"
            end_time = f"{ev.end_idx * dt:.2f}s"

        print(
            f"{i:<6} {ev.start_idx:<9} {ev.end_idx:<8} "
            f"{start_time:<15} {end_time:<15} {duration:<12.2f} "
            f"{ev.peak_negative_gx:<10.2f} {ev.mean_gx:<10.2f} {ev.integrated_rotation_deg:<12.2f}"
        )


def save_event_csv(
    out_csv: str,
    df: pd.DataFrame,
    time_col: Optional[str],
    dt: float,
    events: List[Event],
) -> None:
    rows = []
    for i, ev in enumerate(events, start=1):
        duration = (ev.end_idx - ev.start_idx + 1) * dt
        rows.append(
            {
                "event_id": i,
                "start_idx": ev.start_idx,
                "end_idx": ev.end_idx,
                "start_time": df.iloc[ev.start_idx][time_col] if time_col else ev.start_idx * dt,
                "end_time": df.iloc[ev.end_idx][time_col] if time_col else ev.end_idx * dt,
                "duration_sec": duration,
                "peak_negative_gx_dps": ev.peak_negative_gx,
                "mean_gx_dps": ev.mean_gx,
                "integrated_rotation_deg": ev.integrated_rotation_deg,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved event summary to: {out_csv}")


def plot_results(
    df: pd.DataFrame,
    time_col: Optional[str],
    gx_col: str,
    events: List[Event],
    dt: float,
) -> None:
    import matplotlib.pyplot as plt

    if time_col is not None:
        x = np.arange(len(df))
        x_label = "Sample Index"
    else:
        x = np.arange(len(df))
        x_label = "Sample Index"

    gx = df[gx_col].astype(float).to_numpy()

    plt.figure(figsize=(14, 5))
    plt.plot(x, gx, label=gx_col)

    for i, ev in enumerate(events, start=1):
        plt.axvspan(ev.start_idx, ev.end_idx, alpha=0.25)
        plt.text(ev.start_idx, np.min(gx), f"E{i}", fontsize=9, verticalalignment="bottom")

    plt.xlabel(x_label)
    plt.ylabel("gx (dps)")
    plt.title("Left-turn detection")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="Detect left turns from IMU CSV data.")
    parser.add_argument("--csv", required=True, help=r"C:\Users\Sravani\Desktop\sensorr\imu_6axis_data_20260308_084155_Trip2_Left.csv")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Rolling window size in samples")
    parser.add_argument("--smooth", type=int, default=DEFAULT_SMOOTH, help="Moving average window")
    parser.add_argument("--dt", type=float, default=DEFAULT_DT, help="Fallback sample period in seconds")
    parser.add_argument("--merge-gap", type=int, default=DEFAULT_MERGE_GAP, help="Merge detections within this many samples")
    parser.add_argument("--plot", action="store_true", help="Plot gx and detected events")
    parser.add_argument("--out-csv", default="detected_left_turn_events.csv", help="Output CSV summary")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    if df.empty:
        print("CSV is empty.")
        sys.exit(1)

    time_col, gx_col, az_col = choose_columns(df)

    print("\nDetected columns:")
    print(f"  time_col: {time_col}")
    print(f"  gx_col  : {gx_col}")
    print(f"  az_col  : {az_col}")

    if gx_col is None or az_col is None:
        print("\nCould not find required IMU columns.")
        print("Please edit choose_columns() candidate names to match your CSV.")
        print("Required columns:")
        print("  - gx(dps) or equivalent")
        print("  - az(g) or equivalent")
        print("\nAvailable columns:")
        for c in df.columns:
            print(f"  - {c}")
        sys.exit(1)

    # Convert to numeric and drop rows with missing gx/az
    work_df = df.copy()
    work_df[gx_col] = pd.to_numeric(work_df[gx_col], errors="coerce")
    work_df[az_col] = pd.to_numeric(work_df[az_col], errors="coerce")
    work_df = work_df.dropna(subset=[gx_col, az_col]).reset_index(drop=True)

    if len(work_df) < args.window:
        print(f"Not enough samples. Need at least {args.window}, found {len(work_df)}")
        sys.exit(1)

    dt = estimate_dt_seconds(work_df, time_col, args.dt)

    gx = work_df[gx_col].to_numpy(dtype=float)
    az = work_df[az_col].to_numpy(dtype=float)

    detections = detect_left_turn_windows(
        gx=gx,
        az=az,
        dt=dt,
        window=args.window,
        smooth_k=args.smooth,
    )

    events = merge_detection_windows(
        detections=detections,
        gx=gx,
        dt=dt,
        merge_gap=args.merge_gap,
    )

    print_summary(
        df=work_df,
        time_col=time_col,
        dt=dt,
        detections=detections,
        events=events,
    )

    save_event_csv(
        out_csv=args.out_csv,
        df=work_df,
        time_col=time_col,
        dt=dt,
        events=events,
    )

    if args.plot:
        plot_results(
            df=work_df,
            time_col=time_col,
            gx_col=gx_col,
            events=events,
            dt=dt,
        )


if __name__ == "__main__":
    main()