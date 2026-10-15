from pathlib import Path
import re

import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent if BASE_DIR.name.lower() == "segmentation" else BASE_DIR
RAW_ROOT = PROJECT_ROOT / "Storage" / "CleanData" / "Dictionary"
OUT_ROOT = PROJECT_ROOT / "Storage" / "SegmentedData" / "Dictionary"

WINDOW_SECONDS = 8
STEP_SECONDS = 4

# Your dictionary dataset is collected at 25 Hz.
ACTUAL_SAMPLING_RATE = 25

# Sampling rate used for segmentation.
SAMPLING_RATE = 25


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
        names=["ts", "x", "y", "z"],
    )

    #make sure all columns are numeric and have the correct types
    df["ts"] = pd.to_numeric(df["ts"], errors="raise").astype(np.int64)
    df["x"] = pd.to_numeric(df["x"], errors="raise").astype(float)
    df["y"] = pd.to_numeric(df["y"], errors="raise").astype(float)
    df["z"] = pd.to_numeric(df["z"], errors="raise").astype(float)

    return df


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
                frame_index + 1,
            ])
            sample_id += 1

    return pd.DataFrame(rows, columns=columns)


def extract_user_id(user_dir: Path) -> int:
    return int(user_dir.name.replace("User", ""))


def extract_trial_id(trial_dir: Path) -> int:
    """
    Expected trial folder format:
        something_T1
        something_T2
        ...
        something_T12
    """
    match = re.search(r"_T(\d+)$", trial_dir.name)

    if match is None:
        raise ValueError(f"Could not parse trial id from {trial_dir.name}")

    return int(match.group(1))


def process_one_trial(user_dir: Path, trial_dir: Path) -> dict:
    user_id = extract_user_id(user_dir)

    acc_path = trial_dir / "clean_acc.txt"
    gyr_path = trial_dir / "clean_gyr.txt"

    if not acc_path.exists():
        raise FileNotFoundError(f"Missing accelerometer file: {acc_path}")

    if not gyr_path.exists():
        raise FileNotFoundError(f"Missing gyroscope file: {gyr_path}")

    acc_df = read_sensor_file(acc_path)
    gyr_df = read_sensor_file(gyr_path)

    acc_possible_frames = count_possible_frames(len(acc_df), SAMPLING_RATE)
    gyr_possible_frames = count_possible_frames(len(gyr_df), SAMPLING_RATE)

    # Keep ACC and GYR aligned by using only the shared number of frames.
    shared_frames = min(acc_possible_frames, gyr_possible_frames)

    acc_segmented = segment_sensor(
        df=acc_df,
        user_id=user_id,
        sampling_rate=SAMPLING_RATE,
        n_frames=shared_frames
    )

    gyr_segmented = segment_sensor(
        df=gyr_df,
        user_id=user_id,
        sampling_rate=SAMPLING_RATE,
        n_frames=shared_frames
    )

    out_dir = OUT_ROOT / user_dir.name / trial_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    acc_segmented.to_csv(out_dir / "acc.csv", index=False)
    gyr_segmented.to_csv(out_dir / "gyr.csv", index=False)

    return {
        "user": user_id,
        "trial": trial_dir.name,

        # Actual collection rate of your dictionary dataset.
        "actual_sampling_rate": ACTUAL_SAMPLING_RATE,

        # Sampling rate used to calculate 8s window and 4s slide.
        "sampling_rate": SAMPLING_RATE,

        "acc_rows_in": len(acc_df),
        "gyr_rows_in": len(gyr_df),
        "acc_frames": acc_possible_frames,
        "gyr_frames": gyr_possible_frames,
        "shared_frames": shared_frames,
        "acc_rows_out": len(acc_segmented),
        "gyr_rows_out": len(gyr_segmented),
    }


def main() -> None:
    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"RAW_ROOT does not exist: {RAW_ROOT}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    user_dirs = sorted(
        [p for p in RAW_ROOT.glob("User*") if p.is_dir()],
        key=extract_user_id,
    )

    if not user_dirs:
        raise FileNotFoundError(f"No User* folders found in {RAW_ROOT}")

    for user_dir in user_dirs:
        trial_dirs = sorted(
            [p for p in user_dir.iterdir() if p.is_dir()],
            key=extract_trial_id,
        )

        for trial_dir in trial_dirs:
            info = process_one_trial(user_dir, trial_dir)
            summary_rows.append(info)

            print(
                f"{user_dir.name} {trial_dir.name}: "
                f"actual_sr={info['actual_sampling_rate']}, "
                f"sr={info['sampling_rate']}, "
                f"shared_frames={info['shared_frames']}"
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_ROOT / "segmentation_summary_dictionary.csv", index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
