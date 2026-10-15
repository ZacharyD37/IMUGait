from pathlib import Path

import numpy as np
import pandas as pd


USER_START = 1
WALKING_CODE = "A"
SYNTHETIC_UNIX_START_MS = 1700000000 * 1000  # 2023-11-14 UTC, used as fallback origin
MIN_UNIX_NS = 946684800 * 1_000_000_000      # 2000-01-01 UTC in nanoseconds
MAX_UNIX_NS = 4102444800 * 1_000_000_000     # 2100-01-01 UTC in nanoseconds

SOURCE_SAMPLING_RATE_HZ = 20.0
TARGET_SAMPLING_RATE_HZ = 25.0
TARGET_INTERVAL_MS = 1000.0 / TARGET_SAMPLING_RATE_HZ

TRAIN_RATIO = 0.70
TEST_RATIO = 0.30


def find_imugait_root(start_path):
    current = Path(start_path).resolve()
    for path in [current] + list(current.parents):
        if path.name == "IMUGait":
            return path
    raise FileNotFoundError("Could not find IMUGait root folder.")


def extract_raw_user_id(filename):
    """Expected filenames: data_1600_accel_watch.txt"""
    parts = filename.split("_")
    if len(parts) < 4 or parts[0] != "data":
        return None
    return parts[1]


def is_plausible_unix_ns(ts_ns):
    return MIN_UNIX_NS <= ts_ns <= MAX_UNIX_NS


def clean_sensor_data_filtered_to_rows(input_file, target_activity):
    """
    Read raw WISDM file and return monotonically non-decreasing rows:
        (raw_timestamp_ns, x, y, z)
    WISDM watch timestamps are raw nanosecond values.
    """
    rows = []
    last_ts = None

    with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            clean_line = line.strip().rstrip(";")
            if not clean_line:
                continue

            parts = clean_line.split(",")
            if len(parts) < 6:
                continue

            if parts[1].strip() != target_activity:
                continue

            raw_ts = parts[2].strip()
            x, y, z = parts[3].strip(), parts[4].strip(), parts[5].strip()

            if not raw_ts or not x or not y or not z:
                continue

            try:
                ts_ns = int(raw_ts)
            except ValueError:
                continue

            if last_ts is not None and ts_ns < last_ts:
                continue

            rows.append((ts_ns, x, y, z))
            last_ts = ts_ns

    return rows


def rows_to_timestamp_ms_dataframe(rows):
    if not rows:
        return pd.DataFrame(columns=["timestamp_ms", "x", "y", "z"])

    df = pd.DataFrame(rows, columns=["raw_timestamp_ns", "x", "y", "z"])

    for col in ["raw_timestamp_ns", "x", "y", "z"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["raw_timestamp_ns", "x", "y", "z"])
    if df.empty:
        return pd.DataFrame(columns=["timestamp_ms", "x", "y", "z"])

    df["raw_timestamp_ns"] = df["raw_timestamp_ns"].astype("int64")
    df = df.sort_values("raw_timestamp_ns", kind="stable")

    first_ts_ns = int(df["raw_timestamp_ns"].iloc[0])

    if is_plausible_unix_ns(first_ts_ns):
        df["timestamp_ms"] = df["raw_timestamp_ns"] / 1_000_000.0
    else:
        elapsed_ms = (df["raw_timestamp_ns"] - first_ts_ns) / 1_000_000.0
        df["timestamp_ms"] = SYNTHETIC_UNIX_START_MS + elapsed_ms

    return df[["timestamp_ms", "x", "y", "z"]]


def resample_rows_to_25hz(rows):
    source_df = rows_to_timestamp_ms_dataframe(rows)

    if source_df.empty:
        return []

    source_df = source_df.groupby("timestamp_ms", as_index=False)[["x", "y", "z"]].mean()
    source_df = source_df.sort_values("timestamp_ms", kind="stable")

    source_ts = source_df["timestamp_ms"].to_numpy(dtype=float)

    if len(source_ts) == 1:
        row = source_df.iloc[0]
        return [(
            float(row["timestamp_ms"]) / 1000.0,
            float(row["x"]),
            float(row["y"]),
            float(row["z"]),
        )]

    start_ms = float(source_ts[0])
    end_ms = float(source_ts[-1])
    target_ts = np.arange(start_ms, end_ms + 0.001, TARGET_INTERVAL_MS)

    if len(target_ts) == 0:
        return []

    resampled = pd.DataFrame({
        "timestamp_s": target_ts / 1000.0,
        "x": np.interp(target_ts, source_ts, source_df["x"].to_numpy(dtype=float)),
        "y": np.interp(target_ts, source_ts, source_df["y"].to_numpy(dtype=float)),
        "z": np.interp(target_ts, source_ts, source_df["z"].to_numpy(dtype=float)),
    })

    return list(resampled.itertuples(index=False, name=None))


def split_rows_train_test(rows, train_ratio=0.70):
    n = len(rows)
    if n < 10:
        return rows, []
    split_idx = int(n * train_ratio)
    return rows[:split_idx], rows[split_idx:]


def write_rows(output_file, rows):
    with open(output_file, "w", encoding="utf-8") as outfile:
        for ts, x, y, z in rows:
            outfile.write(f"{float(ts):.6f} {x} {y} {z}\n")


def extract_watch_walking_data(raw_data_path, output_dir):
    """
    Read ONLY watch data from:
        raw/watch/accel
        raw/watch/gyro

    Keep ONLY walking rows (activity code A), then upsample from 20Hz to 25Hz.

    Save to:
        Storage/CleanData/Genuine/wisdm_25Hz/User{N}/Training/clean_acc.txt
        Storage/CleanData/Genuine/wisdm_25Hz/User{N}/Testing/clean_acc.txt
        Storage/CleanData/Genuine/wisdm_25Hz/User{N}/Training/clean_gyr.txt
        Storage/CleanData/Genuine/wisdm_25Hz/User{N}/Testing/clean_gyr.txt
    """
    raw_data_path = Path(raw_data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    watch_accel_path = raw_data_path / "watch" / "accel"
    watch_gyro_path = raw_data_path / "watch" / "gyro"

    print(f"Using WATCH accel folder: {watch_accel_path}")
    print(f"Using WATCH gyro folder:  {watch_gyro_path}")

    if not watch_accel_path.exists():
        raise FileNotFoundError(f"Watch accel folder not found: {watch_accel_path}")
    if not watch_gyro_path.exists():
        raise FileNotFoundError(f"Watch gyro folder not found: {watch_gyro_path}")

    accel_files = sorted(f for f in watch_accel_path.iterdir() if f.is_file() and f.suffix == ".txt")
    gyro_files = sorted(f for f in watch_gyro_path.iterdir() if f.is_file() and f.suffix == ".txt")

    if not accel_files:
        raise FileNotFoundError(f"No accel txt files found in: {watch_accel_path}")
    if not gyro_files:
        raise FileNotFoundError(f"No gyro txt files found in: {watch_gyro_path}")

    accel_map = {extract_raw_user_id(f.name): f for f in accel_files if extract_raw_user_id(f.name)}
    gyro_map = {extract_raw_user_id(f.name): f for f in gyro_files if extract_raw_user_id(f.name)}

    common_user_ids = sorted(set(accel_map) & set(gyro_map), key=int)

    if not common_user_ids:
        raise RuntimeError("No matching watch accel/gyro user files found.")

    print(f"Found {len(common_user_ids)} matched WATCH users.")

    for idx, raw_user_id in enumerate(common_user_ids, start=USER_START):
        user_name = f"User{idx}"
        print(f"\nProcessing WATCH raw user {raw_user_id} -> {user_name}")

        acc_raw = clean_sensor_data_filtered_to_rows(accel_map[raw_user_id], WALKING_CODE)
        gyr_raw = clean_sensor_data_filtered_to_rows(gyro_map[raw_user_id], WALKING_CODE)

        acc_rows = resample_rows_to_25hz(acc_raw)
        gyr_rows = resample_rows_to_25hz(gyr_raw)

        acc_train, acc_test = split_rows_train_test(acc_rows, train_ratio=TRAIN_RATIO)
        gyr_train, gyr_test = split_rows_train_test(gyr_rows, train_ratio=TRAIN_RATIO)

        user_dir = output_dir / user_name
        train_dir = user_dir / "Training"
        test_dir = user_dir / "Testing"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        write_rows(train_dir / "clean_acc.txt", acc_train)
        write_rows(test_dir / "clean_acc.txt", acc_test)
        write_rows(train_dir / "clean_gyr.txt", gyr_train)
        write_rows(test_dir / "clean_gyr.txt", gyr_test)

        print(f"  Accel rows: {len(acc_raw)} raw -> {len(acc_rows)} at 25Hz")
        print(f"  Gyro rows:  {len(gyr_raw)} raw -> {len(gyr_rows)} at 25Hz")
        print(f"  Saved accel training rows: {len(acc_train)}")
        print(f"  Saved accel testing rows:  {len(acc_test)}")
        print(f"  Saved gyro training rows:  {len(gyr_train)}")
        print(f"  Saved gyro testing rows:   {len(gyr_test)}")


if __name__ == "__main__":
    imugait_root = find_imugait_root(__file__)
    raw_data_path = imugait_root / "Storage" / "RawData" / "Genuine" / "wisdm-dataset" / "raw"
    output_directory = imugait_root / "Storage" / "CleanData" / "Genuine" / "wisdm_25Hz"

    print("Processing WISDM WATCH normal walking data at 25Hz only...")
    print(f"Input root:  {raw_data_path}")
    print(f"Output root: {output_directory}")
    print(f"Activity filter: {WALKING_CODE} (walking)")
    print(f"Source sampling rate: {SOURCE_SAMPLING_RATE_HZ:g}Hz")
    print(f"Target sampling rate: {TARGET_SAMPLING_RATE_HZ:g}Hz")
    print(f"Split: {int(TRAIN_RATIO * 100)}% training / {int(TEST_RATIO * 100)}% testing")

    extract_watch_walking_data(raw_data_path, output_directory)

    print("\nFinished.")
