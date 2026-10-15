from pathlib import Path
import argparse
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[0]
PROJECT_ROOT = BASE_DIR.parent if BASE_DIR.name.lower() == "segmentation" else BASE_DIR
RAW_ROOT = PROJECT_ROOT / "Storage" / "CleanData" / "Genuine"
OUT_ROOT = PROJECT_ROOT / "Storage" / "SegmentedData" / "Genuine"

WINDOW_SECONDS = 8
STEP_SECONDS = 4
VALID_RATES = (20, 25, 50, 100)
# Cleaned files in this project may store timestamps as seconds or milliseconds.
# Older/raw-cleaned files can also carry larger timestamp units, so estimate rates
# against common time scales and keep the candidate closest to a valid IMU rate.
TIME_SCALES_PER_SECOND = (1.0, 1_000.0, 1_000_000.0, 1_000_000_000.0)


def read_sensor_file(path: Path) -> pd.DataFrame:
    """
    Expected format in each txt file:
    unix_timestamp x y z
    separated by whitespace.
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        header=None,
        names=["ts", "x", "y", "z"]
    )

    df["ts"] = pd.to_numeric(df["ts"], errors="raise").astype(float)
    df["x"] = pd.to_numeric(df["x"], errors="raise").astype(float)
    df["y"] = pd.to_numeric(df["y"], errors="raise").astype(float)
    df["z"] = pd.to_numeric(df["z"], errors="raise").astype(float)

    return df


def snap_rate(raw_rate: float) -> int:
    """
    Snap the estimated sampling rate to one of the allowed rates.
    """
    return min(VALID_RATES, key=lambda r: abs(r - raw_rate))


def estimate_rate_from_file(path: Path) -> tuple[float, int]:
    """
    Estimate the actual sampling rate from timestamps, then snap it to
    the nearest valid segmentation rate.

    Returns:
        actual_sampling_rate: estimated from timestamp spacing
        sampling_rate: snapped rate used by segmentation
    """
    df = read_sensor_file(path)
    ts = df["ts"].to_numpy(dtype=float)

    if len(ts) < 2:
        raise ValueError(f"Not enough rows to estimate sampling rate: {path}")

    diffs = np.diff(ts)
    positive_diffs = diffs[diffs > 0]

    if len(positive_diffs) == 0:
        raise ValueError(
            f"Could not estimate sampling rate from non-increasing timestamps: {path}"
        )

    duration = ts[-1] - ts[0]
    median_diff = float(np.median(positive_diffs))

    candidates: list[float] = []

    for scale in TIME_SCALES_PER_SECOND:
        candidates.append(scale / median_diff)

        if duration > 0:
            candidates.append((len(ts) - 1) * scale / duration)

    best_actual_rate = None
    best_sampling_rate = None
    best_score = None

    for candidate in candidates:
        snapped = snap_rate(candidate)
        score = abs(candidate - snapped) / snapped

        if best_score is None or score < best_score:
            best_score = score
            best_actual_rate = candidate
            best_sampling_rate = snapped

    if best_actual_rate is None or best_sampling_rate is None:
        raise ValueError(f"Could not estimate sampling rate for {path}")

    if best_sampling_rate not in VALID_RATES:
        raise ValueError(f"Invalid sampling rate {best_sampling_rate} for {path}")

    return best_actual_rate, best_sampling_rate


def count_possible_frames(n_rows: int, sampling_rate: int) -> int:
    """
    Count how many 8-second windows can be produced with a 4-second slide.
    """
    window_size = WINDOW_SECONDS * sampling_rate
    step_size = STEP_SECONDS * sampling_rate

    if n_rows < window_size:
        return 0

    return 1 + (n_rows - window_size) // step_size


def segment_sensor(
    df: pd.DataFrame,
    user_id: int,
    sampling_rate: int,
    n_frames: int
) -> pd.DataFrame:
    """
    Segment one sensor file into overlapping sliding windows.

    Output CSV format:
        sample_id,x,y,z,user,frame_id
    """
    window_size = WINDOW_SECONDS * sampling_rate
    step_size = STEP_SECONDS * sampling_rate

    columns = ["sample_id", "x", "y", "z", "user", "frame_id"]

    if n_frames == 0:
        return pd.DataFrame(columns=columns)

    base_df = df[["x", "y", "z"]].reset_index(drop=True)

    rows = []
    sample_id = 0

    for frame_index, start in enumerate(
        range(0, len(base_df) - window_size + 1, step_size)
    ):
        if frame_index >= n_frames:
            break

        end = start + window_size
        window = base_df.iloc[start:end]

        for _, row in window.iterrows():
            rows.append([
                sample_id,
                row["x"],
                row["y"],
                row["z"],
                user_id,
                frame_index + 1
            ])
            sample_id += 1

    return pd.DataFrame(rows, columns=columns)


def user_sort_key(path: Path) -> int:
    return int(path.name.replace("User", ""))


def discover_datasets(
    raw_root: Path,
    selected_dataset: str | None
) -> list[tuple[str | None, Path]]:
    """
    Supports either:

    Storage/CleanData/Genuine/User1/...
    or
    Storage/CleanData/Genuine/DatasetName/User1/...
    """
    direct_users = [p for p in raw_root.glob("User*") if p.is_dir()]

    datasets = []

    for dataset_dir in sorted(
        [p for p in raw_root.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower()
    ):
        if selected_dataset is not None and dataset_dir.name != selected_dataset:
            continue

        if any(p.is_dir() for p in dataset_dir.glob("User*")):
            datasets.append((dataset_dir.name, dataset_dir))

    if selected_dataset is not None and not datasets:
        raise ValueError(
            f"Dataset not found or contains no User* folders: {selected_dataset}"
        )

    if selected_dataset is None and direct_users and not datasets:
        datasets.append((None, raw_root))

    return datasets


def output_dataset_root(dataset_name: str | None) -> Path:
    if dataset_name is None:
        return OUT_ROOT

    return OUT_ROOT / dataset_name


def process_one_split(
    user_dir: Path,
    split_name: str,
    dataset_out_root: Path,
    dataset_name: str | None
) -> dict:
    user_id = int(user_dir.name.replace("User", ""))
    split_dir = user_dir / split_name

    acc_path = split_dir / "clean_acc.txt"
    gyr_path = split_dir / "clean_gyr.txt"

    if not acc_path.exists():
        raise FileNotFoundError(f"Missing accelerometer file: {acc_path}")

    if not gyr_path.exists():
        raise FileNotFoundError(f"Missing gyroscope file: {gyr_path}")

    acc_actual_sr, acc_sr = estimate_rate_from_file(acc_path)
    gyr_actual_sr, gyr_sr = estimate_rate_from_file(gyr_path)

    acc_df = read_sensor_file(acc_path)
    gyr_df = read_sensor_file(gyr_path)

    acc_possible_frames = count_possible_frames(len(acc_df), acc_sr)
    gyr_possible_frames = count_possible_frames(len(gyr_df), gyr_sr)

    # Keep ACC and GYR aligned by using only the shared number of frames.
    shared_frames = min(acc_possible_frames, gyr_possible_frames)

    acc_segmented = segment_sensor(acc_df, user_id, acc_sr, shared_frames)
    gyr_segmented = segment_sensor(gyr_df, user_id, gyr_sr, shared_frames)

    out_dir = dataset_out_root / user_dir.name / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    acc_segmented.to_csv(out_dir / "acc.csv", index=False)
    gyr_segmented.to_csv(out_dir / "gyr.csv", index=False)

    return {
        "dataset": dataset_name,
        "user": user_id,
        "split": split_name,

        # Estimated true sampling rates from timestamps.
        "acc_actual_sampling_rate": round(acc_actual_sr, 3),
        "gyr_actual_sampling_rate": round(gyr_actual_sr, 3),

        # Final snapped sampling rates used for segmentation.
        "acc_sampling_rate": acc_sr,
        "gyr_sampling_rate": gyr_sr,

        "acc_frames": acc_possible_frames,
        "gyr_frames": gyr_possible_frames,
        "shared_frames": shared_frames,
        "acc_rows_out": len(acc_segmented),
        "gyr_rows_out": len(gyr_segmented),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment every genuine dataset under Storage/CleanData/Genuine."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Optional dataset name to process, e.g. Kumar2015. "
            "By default, all datasets are processed."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"RAW_ROOT does not exist: {RAW_ROOT}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    datasets = discover_datasets(RAW_ROOT, args.dataset)

    if not datasets:
        raise FileNotFoundError(f"No genuine datasets found in {RAW_ROOT}")

    for dataset_name, dataset_root in datasets:
        dataset_out_root = output_dataset_root(dataset_name)
        dataset_out_root.mkdir(parents=True, exist_ok=True)

        label = dataset_name or dataset_root.name
        print(f"\nProcessing genuine dataset: {label}")

        user_dirs = sorted(
            [p for p in dataset_root.glob("User*") if p.is_dir()],
            key=user_sort_key
        )

        for user_dir in user_dirs:
            for split_name in ["Training", "Testing"]:
                split_dir = user_dir / split_name

                if not split_dir.exists():
                    continue

                info = process_one_split(
                    user_dir=user_dir,
                    split_name=split_name,
                    dataset_out_root=dataset_out_root,
                    dataset_name=dataset_name
                )

                summary_rows.append(info)

                print(
                    f"{user_dir.name} {split_name}: "
                    f"acc_sr={info['acc_sampling_rate']} "
                    f"(actual={info['acc_actual_sampling_rate']}), "
                    f"gyr_sr={info['gyr_sampling_rate']} "
                    f"(actual={info['gyr_actual_sampling_rate']}), "
                    f"shared_frames={info['shared_frames']}"
                )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_ROOT / "segmentation_summary_genuine.csv", index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
