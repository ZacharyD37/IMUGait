import os
import math
import argparse
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEG_ROOT = os.path.join(BASE_DIR, "Storage", "SegmentedData", "Genuine")
OUT_ROOT = os.path.join(BASE_DIR, "Storage", "PreprocessedData", "Genuine")
WINDOW_SECONDS = 8


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
    s = int(math.ceil(0.05 * sampling_rate + 1))
    s = max(1, s)

    out = frame.copy()

    out["x"] = moving_average(frame["x"].to_numpy(), s)
    out["y"] = moving_average(frame["y"].to_numpy(), s)
    out["z"] = moving_average(frame["z"].to_numpy(), s)

    return out


def read_segmented_file(in_path: str) -> pd.DataFrame:
    df = pd.read_csv(in_path)
    required_cols = {"sample_id", "frame_id", "x", "y", "z", "user"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {in_path}: {sorted(missing)}")
    return df


def detect_sampling_rate_from_segmented(df: pd.DataFrame) -> int:
    """
    Detect the exact integer sampling rate from the first segmented frame.

    Each frame spans 8 seconds. If frame 1 contains 800 samples
    (for example sample_id 0..799), then the sampling rate is 800 / 8 = 100 Hz.
    """
    if df.empty:
        raise ValueError("Cannot detect sampling rate from an empty dataframe.")

    first_frame_id = df["frame_id"].iloc[0]
    first_frame = df[df["frame_id"] == first_frame_id]

    if first_frame.empty:
        raise ValueError("Could not read the first frame for sampling-rate detection.")

    sample_ids = pd.to_numeric(first_frame["sample_id"], errors="raise").astype(int).to_numpy()
    frame_samples = int(sample_ids.max() - sample_ids.min() + 1)

    if frame_samples != len(first_frame):
        raise ValueError(
            "First frame sample_id values are not contiguous, "
            f"so sampling rate cannot be inferred safely. frame_id={first_frame_id}"
        )

    if frame_samples % WINDOW_SECONDS != 0:
        raise ValueError(
            f"First frame has {frame_samples} samples, which is not divisible by {WINDOW_SECONDS} seconds."
        )

    return int(frame_samples // WINDOW_SECONDS)


def user_sort_key(user_name: str) -> int:
    return int("".join(ch for ch in user_name if ch.isdigit()) or "0")


def discover_datasets(root: str, selected_dataset: str | None) -> list[tuple[str | None, str]]:
    direct_users = [
        name for name in os.listdir(root)
        if name.startswith("User") and os.path.isdir(os.path.join(root, name))
    ]
    datasets = []
    for name in sorted(os.listdir(root), key=str.lower):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if selected_dataset is not None and name != selected_dataset:
            continue
        has_users = any(
            child.startswith("User") and os.path.isdir(os.path.join(path, child))
            for child in os.listdir(path)
        )
        if has_users:
            datasets.append((name, path))

    if selected_dataset is not None and not datasets:
        raise ValueError(f"Dataset not found or contains no User* folders: {selected_dataset}")

    if selected_dataset is None and direct_users and not datasets:
        datasets.append((None, root))

    return datasets


def output_dataset_root(dataset_name: str | None) -> str:
    if dataset_name is None:
        return OUT_ROOT
    return os.path.join(OUT_ROOT, dataset_name)


def process_file(df: pd.DataFrame, out_path: str, sampling_rate: int) -> None:

    processed_frames = []

    for frame_id, frame in df.groupby("frame_id", sort=False):
        processed_frame = preprocess_frame(frame, sampling_rate)
        processed_frames.append(processed_frame)

    out_df = pd.concat(processed_frames, ignore_index=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess every segmented genuine dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset name to process, e.g. Kumar2015. By default, all datasets are processed.",
    )
    return parser.parse_args()


def run(dataset: str | None = None) -> None:
    if not os.path.exists(SEG_ROOT):
        raise FileNotFoundError(f"Segmentation root not found: {SEG_ROOT}")

    datasets = discover_datasets(SEG_ROOT, dataset)
    if not datasets:
        raise FileNotFoundError(f"No segmented genuine datasets found in {SEG_ROOT}")

    for dataset_name, dataset_root in datasets:
        dataset_out_root = output_dataset_root(dataset_name)
        label = dataset_name or os.path.basename(dataset_root)
        print(f"\nProcessing genuine dataset: {label}")

        users = sorted(
            [
                user for user in os.listdir(dataset_root)
                if user.startswith("User") and os.path.isdir(os.path.join(dataset_root, user))
            ],
            key=user_sort_key,
        )

        for user in users:
            user_path = os.path.join(dataset_root, user)

            for split in ["Training", "Testing"]:
                split_path = os.path.join(user_path, split)

                if not os.path.exists(split_path):
                    continue

                for sensor in ["acc.csv", "gyr.csv"]:
                    in_file = os.path.join(split_path, sensor)

                    if not os.path.exists(in_file):
                        continue

                    df = read_segmented_file(in_file)
                    sampling_rate = detect_sampling_rate_from_segmented(df)
                    window_size = int(math.ceil(0.05 * sampling_rate+1))
                    out_file = os.path.join(dataset_out_root, user, split, sensor)

                    print(
                        f"Processing {in_file} | "
                        f"user={user} | sr={sampling_rate} | smooth_window={window_size}"
                    )

                    process_file(df, out_file, sampling_rate)

    print("\nDone preprocessing.")


if __name__ == "__main__":
    run(parse_args().dataset)
