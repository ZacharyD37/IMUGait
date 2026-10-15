"""
ExtractFeaturesGenuine.py
Feature extraction for genuine data, reading from preprocessed CSVs.

Reads from:   Storage/PreprocessedData/Genuine/{Dataset}/{User}/{Training|Testing}/{acc.csv|gyr.csv}
Writes to:    Storage/FeatureData/Genuine/{Dataset}/{User}/{Training|Testing}/
                  FrequencyFeatures/feat_clean_acc.csv
                  TimeFeatures/feat_clean_acc.csv

Output format matches Kumar's original exactly:
  - One CSV per domain per sensor
  - Column names repeat for each axis (x, y, z, m) with no axis suffix
  - Pandas default index as first column
  - No user or frame_id columns, just feature values
"""

import os
import argparse
import sys
import warnings

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from GlobalParameters import DefaultParam
from FeatureExtraction_Kumar import FrequencyDomain, TimeDomain

warnings.filterwarnings("ignore")


PREPROC_ROOT = os.path.join(BASE_DIR, "Storage", "PreprocessedData", "Genuine")
FEATURE_ROOT = os.path.join(BASE_DIR, "Storage", "FeatureData", "Genuine")

SENSORS = ["acc.csv", "gyr.csv"]
WINDOW_SECONDS = 8


def detect_sampling_rate(df: pd.DataFrame) -> int:
    """
    Detect the exact integer sampling rate from the segmented frame structure.

    Examples:
        800 samples per frame -> 100 Hz
        400 samples per frame ->  50 Hz
        200 samples per frame ->  25 Hz
        160 samples per frame ->  20 Hz
    """
    if df.empty:
        raise ValueError("Cannot detect sampling rate from an empty dataframe.")

    first_fid = df["frame_id"].iloc[0]
    first_frame = df[df["frame_id"] == first_fid]

    if first_frame.empty:
        raise ValueError("Could not read the first frame for sampling-rate detection.")

    sample_ids = pd.to_numeric(first_frame["sample_id"], errors="raise").astype(int).to_numpy()
    frame_samples = int(sample_ids.max() - sample_ids.min() + 1)

    if frame_samples != len(first_frame):
        raise ValueError(
            "First frame sample_id values are not contiguous, "
            f"so sampling rate cannot be inferred safely. frame_id={first_fid}"
        )

    if frame_samples % WINDOW_SECONDS != 0:
        raise ValueError(
            f"First frame has {frame_samples} samples, which is not divisible by {WINDOW_SECONDS} seconds."
        )

    return int(frame_samples // WINDOW_SECONDS)


def get_fft_length(frame_length: int) -> int:
    if 64 < frame_length < 128:
        return 128
    if 128 < frame_length < 256:
        return 256
    return 512


def sensor_to_feat_filename(sensor: str) -> str:
    base = sensor.replace(".csv", "")
    return f"feat_clean_{base}.csv"


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
        return FEATURE_ROOT
    return os.path.join(FEATURE_ROOT, dataset_name)


def process_sensor_file(
    in_path: str,
    out_dir: str,
    feat_filename: str,
) -> None:
    df = pd.read_csv(in_path)

    required = {"sample_id", "frame_id", "x", "y", "z", "user"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {in_path}: {sorted(missing)}")

    frame_ids = sorted(df["frame_id"].unique())
    if not frame_ids:
        print(f"  No frames found in {in_path}")
        return

    first_frame = df[df["frame_id"] == frame_ids[0]]
    samplerate = detect_sampling_rate(df)
    fftlength = get_fft_length(len(first_frame))

    fmatrix_freq = []
    fmatrix_time = []

    fdnames = None
    tdnames = None

    for fid in frame_ids:
        frame = df[df["frame_id"] == fid]
        frame_x = frame["x"].to_numpy(dtype=float)
        frame_y = frame["y"].to_numpy(dtype=float)
        frame_z = frame["z"].to_numpy(dtype=float)

        if len(frame_x) < 10:
            print(f"  Skipping frame {fid} (only {len(frame_x)} samples)")
            continue

        frame_m = np.sqrt(frame_x**2 + frame_y**2 + frame_z**2)

        fdnames_x, fdfvector_x = FrequencyDomain.getall_fd_features(
            frame_x,
            DefaultParam.NUM_BINS_FOR_FFT,
            samplerate,
            DefaultParam.WINLENGTH,
            DefaultParam.WINSTEP,
            fftlength,
        )
        fdnames_y, fdfvector_y = FrequencyDomain.getall_fd_features(
            frame_y,
            DefaultParam.NUM_BINS_FOR_FFT,
            samplerate,
            DefaultParam.WINLENGTH,
            DefaultParam.WINSTEP,
            fftlength,
        )
        fdnames_z, fdfvector_z = FrequencyDomain.getall_fd_features(
            frame_z,
            DefaultParam.NUM_BINS_FOR_FFT,
            samplerate,
            DefaultParam.WINLENGTH,
            DefaultParam.WINSTEP,
            fftlength,
        )
        fdnames_m, fdfvector_m = FrequencyDomain.getall_fd_features(
            frame_m,
            DefaultParam.NUM_BINS_FOR_FFT,
            samplerate,
            DefaultParam.WINLENGTH,
            DefaultParam.WINSTEP,
            fftlength,
        )

        tdnames_x, tdfvector_x = TimeDomain.getall_td_feature(frame_x)
        tdnames_y, tdfvector_y = TimeDomain.getall_td_feature(frame_y)
        tdnames_z, tdfvector_z = TimeDomain.getall_td_feature(frame_z)
        tdnames_m, tdfvector_m = TimeDomain.getall_td_feature(frame_m)

        fmatrix_freq.append(fdfvector_x + fdfvector_y + fdfvector_z + fdfvector_m)
        fmatrix_time.append(tdfvector_x + tdfvector_y + tdfvector_z + tdfvector_m)

        if fdnames is None:
            fdnames = fdnames_x + fdnames_y + fdnames_z + fdnames_m
            tdnames = tdnames_x + tdnames_y + tdnames_z + tdnames_m

    if not fmatrix_freq:
        print(f"  No valid frames in {in_path}")
        return

    freq_path = os.path.join(out_dir, "FrequencyFeatures")
    os.makedirs(freq_path, exist_ok=True)
    pd.DataFrame(fmatrix_freq, columns=fdnames).to_csv(os.path.join(freq_path, feat_filename))

    time_path = os.path.join(out_dir, "TimeFeatures")
    os.makedirs(time_path, exist_ok=True)
    pd.DataFrame(fmatrix_time, columns=tdnames).to_csv(os.path.join(time_path, feat_filename))

    n_frames = len(fmatrix_freq)
    print(f"  {n_frames} frames | sr={samplerate} Hz | Freq, Time | {feat_filename}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract features for every preprocessed genuine dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset name to process, e.g. Kumar2015. By default, all datasets are processed.",
    )
    return parser.parse_args()


def run(dataset: str | None = None) -> None:
    if not os.path.exists(PREPROC_ROOT):
        raise FileNotFoundError(f"Preprocessed root not found: {PREPROC_ROOT}")

    os.makedirs(FEATURE_ROOT, exist_ok=True)

    datasets = discover_datasets(PREPROC_ROOT, dataset)
    if not datasets:
        raise FileNotFoundError(f"No preprocessed genuine datasets found in {PREPROC_ROOT}")

    print("Domains: Time + Frequency\n")

    for dataset_name, dataset_root in datasets:
        feature_root = output_dataset_root(dataset_name)
        os.makedirs(feature_root, exist_ok=True)
        label = dataset_name or os.path.basename(dataset_root)

        user_dirs = sorted(
            [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))],
            key=user_sort_key,
        )

        total = len(user_dirs)
        print(f"Processing genuine dataset: {label}")
        print(f"Found {total} users in {dataset_root}")

        for idx, user in enumerate(user_dirs, 1):
            print(f"[{idx}/{total}] {user}")

            for split in ["Training", "Testing"]:
                split_in = os.path.join(dataset_root, user, split)
                if not os.path.exists(split_in):
                    continue

                for sensor in SENSORS:
                    in_file = os.path.join(split_in, sensor)
                    if not os.path.exists(in_file):
                        continue

                    out_dir = os.path.join(feature_root, user, split)
                    feat_filename = sensor_to_feat_filename(sensor)

                    print(f"  -> {split}/{sensor}")
                    process_sensor_file(in_file, out_dir, feat_filename)

    print("\nDone. Feature files saved to:", FEATURE_ROOT)


if __name__ == "__main__":
    args = parse_args()
    run(dataset=args.dataset)
