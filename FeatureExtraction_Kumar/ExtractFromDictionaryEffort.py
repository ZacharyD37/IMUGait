"""
ExtractFromDictionaryEffort.py
Feature extraction for dictionary data, reading from preprocessed segmented CSVs.

Reads from:   Storage/PreprocessedData/Dictionary/{UserX}/{Ux_Ty}/{acc.csv|gyr.csv}
Writes to:    Storage/FeatureData/Dictionary/{Ux_Ty}/
                  FrequencyFeatures/feat_clean_acc.csv
                  TimeFeatures/feat_clean_acc.csv

Notes:
  - Dictionary data has no Training/Testing split in output.
  - Output folder is trial-based (Ux_Ty), not user-based.
  - Output CSV shape and naming are Kumar-compatible.
"""

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from GlobalParameters import DefaultParam
from FeatureExtraction_Kumar import FrequencyDomain, TimeDomain

warnings.filterwarnings("ignore")


PREPROC_ROOT = os.path.join(BASE_DIR, "Storage", "PreprocessedData", "Dictionary")
FEATURE_ROOT = os.path.join(BASE_DIR, "Storage", "FeatureData", "Dictionary")

SENSORS = ["acc.csv", "gyr.csv"]
WINDOW_SECONDS = 8
TRIAL_NAME_PATTERN = re.compile(r"^U(\d+)_T(\d+)$")


def detect_sampling_rate(df: pd.DataFrame) -> int:
    """
    Detect exact integer sampling rate from segmented frame structure.
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
    return f"feat_clean_{sensor.replace('.csv', '')}.csv"


def sort_trial_key(name: str) -> tuple[int, int, str]:
    match = TRIAL_NAME_PATTERN.match(name)
    if not match:
        return (10**9, 10**9, name)
    return (int(match.group(1)), int(match.group(2)), name)


def process_sensor_file(
    in_path: str,
    out_dir: str,
    feat_filename: str,
) -> None:
    df = pd.read_csv(in_path)

    required = {"sample_id", "frame_id", "x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {in_path}: {sorted(missing)}")

    frame_ids = sorted(df["frame_id"].unique())
    if not frame_ids:
        print(f"    No frames found in {in_path}")
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
            print(f"    Skipping frame {fid} (only {len(frame_x)} samples)")
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
        print(f"    No valid frames in {in_path}")
        return

    freq_path = os.path.join(out_dir, "FrequencyFeatures")
    os.makedirs(freq_path, exist_ok=True)
    pd.DataFrame(fmatrix_freq, columns=fdnames).to_csv(os.path.join(freq_path, feat_filename))

    time_path = os.path.join(out_dir, "TimeFeatures")
    os.makedirs(time_path, exist_ok=True)
    pd.DataFrame(fmatrix_time, columns=tdnames).to_csv(os.path.join(time_path, feat_filename))

    print(
        f"    {len(fmatrix_freq)} frames | sr={samplerate} Hz | Freq, Time | {feat_filename}"
    )


def run() -> None:
    #see if the preprocessed dictionary root folder exists
    if not os.path.exists(PREPROC_ROOT):
        raise FileNotFoundError(f"Preprocessed dictionary root not found: {PREPROC_ROOT}")

    #create the output root folder if it doesn't exist
    os.makedirs(FEATURE_ROOT, exist_ok=True)

    #get all user directories, sorted by numeric user ID (e.g. U1, U2, ..., U10, U11, ...)
    user_dirs = sorted(
        [d for d in os.listdir(PREPROC_ROOT) if os.path.isdir(os.path.join(PREPROC_ROOT, d))],
        key=lambda u: int("".join(ch for ch in u if ch.isdigit()) or "0"),
    )

    #build a list of (user, trial) pairs for all trials across all users, sorted by user ID and trial ID
    trial_pairs: list[tuple[str, str]] = []
    # The trial sorting relies on the trial folder names following the pattern "Ux_Ty" where x and y are integers.
    for user in user_dirs:
        user_in = os.path.join(PREPROC_ROOT, user)#get trail directories
        trials = sorted(#sort trail directory
            [d for d in os.listdir(user_in) if os.path.isdir(os.path.join(user_in, d))],
            key=sort_trial_key,
        )
        for trial in trials:#put them in a list
            trial_pairs.append((user, trial))

    total = len(trial_pairs)#find total number of trails
    print(f"Found {len(user_dirs)} users and {total} dictionary trials in {PREPROC_ROOT}")
    print("Domains: Time + Frequency\n")

    seen_trial_names = set()#check for duplicate trail folders across users, which would cause output overwriting

    for idx, (user, trial) in enumerate(trial_pairs, 1):
        if trial in seen_trial_names:
            raise ValueError(f"Duplicate trial folder name detected across users: {trial}")
        seen_trial_names.add(trial)

        print(f"[{idx}/{total}] {trial} (from {user})")
        trial_in = os.path.join(PREPROC_ROOT, user, trial)
        out_dir = os.path.join(FEATURE_ROOT, trial)

        for sensor in SENSORS:
            in_file = os.path.join(trial_in, sensor)
            if not os.path.exists(in_file):
                print(f"    Missing sensor file: {sensor}")
                continue

            print(f"  -> {sensor}")
            process_sensor_file(in_file, out_dir, sensor_to_feat_filename(sensor))

    print("\nDone. Dictionary feature files saved to:", FEATURE_ROOT)


if __name__ == "__main__":
    run()
