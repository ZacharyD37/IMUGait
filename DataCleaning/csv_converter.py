from pathlib import Path
import re

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_INPUT_DIR = PROJECT_ROOT / "Storage" / "RawData" / "Dictionary"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Storage" / "CleanData" / "Dictionary"
DEFAULT_SAMPLING_RATE_HZ = 25.0

TRIAL_FILE_PATTERN = re.compile(r"^U(?P<user>\d+)_T(?P<trial>\d+)\.csv$", re.IGNORECASE)


def read_csv_with_metadata(file_path: Path) -> pd.DataFrame | None:
    with file_path.open("r", encoding="utf-8") as infile:
        lines = infile.readlines()

    header_line = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            header_line = index
            break

    if header_line is None:
        return None

    df = pd.read_csv(file_path, skiprows=header_line)
    print(f"      Read {len(df)} rows, skipped {header_line} metadata lines")
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}

    for col in df.columns:
        normalized = col.strip().lower()

        if normalized in {"sensor_type", "sensor_ty", "sensor", "type"}:
            rename_map[col] = "sensor_ty"
        elif normalized in {"timestamp", "timestamp_ms", "time", "time_ms", "ts"}:
            rename_map[col] = "timestamp"
        elif normalized == "x":
            rename_map[col] = "x"
        elif normalized == "y":
            rename_map[col] = "y"
        elif normalized == "z":
            rename_map[col] = "z"

    return df.rename(columns=rename_map)


def normalize_sensor_name(sensor_value: object) -> str:
    sensor_name = str(sensor_value).strip().lower()

    if sensor_name in {"accel", "accelerometer", "acc", "linear_acceleration"}:
        return "acc"
    if sensor_name in {"gyro", "gyroscope", "gyr"}:
        return "gyr"

    return sensor_name


def trial_sort_key(path: Path) -> tuple[int, int]:
    match = TRIAL_FILE_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Expected U#_T#.csv filename, got: {path.name}")

    return int(match.group("user")), int(match.group("trial"))


def user_folder_for_trial(csv_path: Path) -> str:
    user_num, _ = trial_sort_key(csv_path)
    return f"User{user_num}"


def ensure_required_columns(df: pd.DataFrame, csv_name: str) -> pd.DataFrame:
    df = normalize_columns(df)

    if "sensor_ty" not in df.columns:
        df["sensor_ty"] = "acc"

    missing = {"x", "y", "z"} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_name} is missing required columns: {sorted(missing)}")

    return df


def write_sensor_file(
    sensor_df: pd.DataFrame,
    output_file: Path,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
) -> int:
    sensor_df = sensor_df.copy().reset_index(drop=True)

    if "timestamp" not in sensor_df.columns:
        step_ms = 1000.0 / sampling_rate_hz
        sensor_df["timestamp"] = [round(index * step_ms, 3) for index in range(len(sensor_df))]

    for col in ["timestamp", "x", "y", "z"]:
        sensor_df[col] = pd.to_numeric(sensor_df[col], errors="coerce")

    sensor_df = sensor_df.dropna(subset=["timestamp", "x", "y", "z"])
    sensor_df = sensor_df[["timestamp", "x", "y", "z"]].sort_values("timestamp", kind="stable")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    sensor_df.to_csv(output_file, sep=" ", index=False, header=False)

    return len(sensor_df)


def create_trial_files(df: pd.DataFrame, output_dir: Path, user_name: str, trial_name: str) -> None:
    df["sensor_ty"] = df["sensor_ty"].apply(normalize_sensor_name)

    output_trial_dir = output_dir / user_name / trial_name

    sensor_outputs = {
        "acc": "clean_acc.txt",
        "gyr": "clean_gyr.txt",
    }

    for sensor_name, filename in sensor_outputs.items():
        sensor_df = df[df["sensor_ty"] == sensor_name]

        if sensor_df.empty:
            print(f"      No {sensor_name} rows found")
            continue

        rows_written = write_sensor_file(sensor_df, output_trial_dir / filename)
        print(f"      Created {user_name}/{trial_name}/{filename}: {rows_written} rows")


def process_trial_csv(csv_path: Path, output_dir: Path) -> None:
    trial_name = csv_path.stem
    user_name = user_folder_for_trial(csv_path)

    print(f"\nProcessing {csv_path.name} -> {user_name}/{trial_name}")

    df = read_csv_with_metadata(csv_path)
    if df is None or df.empty:
        print(f"      Could not read {csv_path.name}")
        return

    df = ensure_required_columns(df, csv_path.name)
    create_trial_files(df, output_dir, user_name, trial_name)


def process_current_file_format(
    base_directory: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    csv_files = [
        file_path for file_path in base_directory.iterdir()
        if file_path.is_file() and TRIAL_FILE_PATTERN.match(file_path.name)
    ]

    if not csv_files:
        raise FileNotFoundError(f"No U#_T#.csv files found in: {base_directory}")

    csv_files.sort(key=trial_sort_key)

    print(f"Found {len(csv_files)} dictionary CSV files")

    for csv_file in csv_files:
        process_trial_csv(csv_file, output_dir)


def process_user_directories(
    base_directory: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    user_dirs = [
        dir_path for dir_path in base_directory.iterdir()
        if dir_path.is_dir() and dir_path.name.startswith("User")
    ]

    if not user_dirs:
        raise FileNotFoundError(f"No User* directories found in: {base_directory}")

    def user_sort_key(path: Path) -> int:
        match = re.match(r"^User(\d+)", path.name)
        if match is None:
            return 0
        return int(match.group(1))

    user_dirs.sort(key=user_sort_key)
    print(f"Found {len(user_dirs)} user directories")

    for user_dir in user_dirs:
        user_num = user_sort_key(user_dir)
        csv_files = sorted(
            [path for path in user_dir.iterdir() if path.is_file() and path.suffix.lower() == ".csv"],
            key=lambda path: path.name,
        )

        print(f"\nProcessing {user_dir.name}: {len(csv_files)} CSV files")

        for trial_index, csv_file in enumerate(csv_files, start=1):
            trial_name = f"U{user_num}_T{trial_index}"
            output_user_name = f"User{user_num}"

            df = read_csv_with_metadata(csv_file)
            if df is None or df.empty:
                print(f"      Could not read {csv_file.name}")
                continue

            df = ensure_required_columns(df, csv_file.name)
            create_trial_files(df, output_dir, output_user_name, trial_name)


def convert_csv_to_directory(
    csv_file_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    csv_file_path = Path(csv_file_path)
    output_dir = Path(output_dir)

    if TRIAL_FILE_PATTERN.match(csv_file_path.name):
        process_trial_csv(csv_file_path, output_dir)
    else:
        df = read_csv_with_metadata(csv_file_path)
        if df is None or df.empty:
            raise ValueError(f"Could not read CSV file: {csv_file_path}")

        df = ensure_required_columns(df, csv_file_path.name)
        create_trial_files(df, output_dir, "User1", csv_file_path.stem)

    print(f"Conversion completed. Data saved in: {output_dir}")


def main() -> None:
    input_dir = DEFAULT_INPUT_DIR
    output_dir = DEFAULT_OUTPUT_DIR

    print("Processing dictionary CSV data")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    has_trial_files = any(
        path.is_file() and TRIAL_FILE_PATTERN.match(path.name)
        for path in input_dir.iterdir()
    )

    if has_trial_files:
        process_current_file_format(input_dir, output_dir)
    else:
        process_user_directories(input_dir, output_dir)

    print("\nConversion completed.")


if __name__ == "__main__":
    main()
