from pathlib import Path
import math

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
SEG_ROOT = BASE_DIR / "Storage" / "SegmentedData" / "Dictionary"
OUT_ROOT = BASE_DIR / "Storage" / "PreprocessedData" / "Dictionary"

SAMPLING_RATE = 25.0


def moving_average(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    Kumar-style moving average.

    For positions where a full window is available:
        x'(i) = mean(x[i : i + window_size])

    For the tail where a full window is not available:
        keep the original signal value.
    """
    signal = signal.astype(float)

    if window_size <= 1 or len(signal) == 0:
        return signal.copy()

    smoothed = np.zeros(len(signal), dtype=float)

    i = 0
    while i <= len(signal) - window_size:
        smoothed[i] = np.mean(signal[i:i + window_size])
        i += 1

    j = i
    while j < len(signal):
        smoothed[j] = signal[j]
        j += 1

    return smoothed


def preprocess_frame(frame: pd.DataFrame, sampling_rate: float) -> pd.DataFrame:
    """
    Apply smoothing only.
    No mean removal.
    No normalization.
    No magnitude.
    """
    s = int(math.ceil(0.05 * sampling_rate + 1))#compute the smoothing window size based on the sampling rate
    s = max(1, s)#ensure at least a window of 1 to avoid empty slices

    #create a copy of the frame to hold the smoothed values
    out = frame.copy()
    
    #apply the moving average to each of the x, y, z columns
    out["x"] = moving_average(frame["x"].to_numpy(), s)
    out["y"] = moving_average(frame["y"].to_numpy(), s)
    out["z"] = moving_average(frame["z"].to_numpy(), s)
    return out


def process_file(in_path: Path, out_path: Path, sampling_rate: float) -> dict:
    df = pd.read_csv(in_path)

    required_cols = {"sample_id", "frame_id", "x", "y", "z", "user"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {in_path}: {sorted(missing)}")

    processed_frames = []
    frame_count = 0

    for _, frame in df.groupby("frame_id", sort=False):
        processed_frame = preprocess_frame(frame, sampling_rate)
        processed_frames.append(processed_frame)
        frame_count += 1

    if processed_frames:
        out_df = pd.concat(processed_frames, ignore_index=True)
    else:
        out_df = df.copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    return {
        "rows_in": len(df),
        "rows_out": len(out_df),
        "frames": frame_count,
    }


def run() -> None:
    if not SEG_ROOT.exists():
        raise FileNotFoundError(f"Segmentation root not found: {SEG_ROOT}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    smooth_window = int(math.ceil(0.05 * SAMPLING_RATE+1))

    user_dirs = sorted(
        [p for p in SEG_ROOT.glob("User*") if p.is_dir()],
        key=lambda p: int(p.name.replace("User", "")),
    )

    for user_dir in user_dirs:
        trial_dirs = sorted([p for p in user_dir.iterdir() if p.is_dir()])

        for trial_dir in trial_dirs:
            for sensor_name in ["acc.csv", "gyr.csv"]:
                in_file = trial_dir / sensor_name

                if not in_file.exists():
                    continue

                out_file = OUT_ROOT / user_dir.name / trial_dir.name / sensor_name

                print(
                    f"Processing {in_file} | "
                    f"user={user_dir.name} | trial={trial_dir.name} | "
                    f"sr={SAMPLING_RATE} | smooth_window={smooth_window}"
                )

                info = process_file(in_file, out_file, SAMPLING_RATE)
                info.update({
                    "user": user_dir.name,
                    "trial": trial_dir.name,
                    "sensor": sensor_name.replace(".csv", ""),
                    "sampling_rate": SAMPLING_RATE,
                    "smooth_window": smooth_window,
                })
                summary_rows.append(info)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_ROOT / "preprocessing_summary_dictionary.csv", index=False)
    print("\nDone preprocessing.")


if __name__ == "__main__":
    run()
