from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = SCRIPT_DIR.parent / "Storage" / "RawData" / "Genuine" / "Kumar2015"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR.parent / "Storage" / "CleanData" / "Genuine" / "Kumar2015"


@dataclass
class FileStats:
    rows_written: int = 0
    rows_skipped: int = 0


def user_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.name)
    if not match:
        return (10**9, path.name.lower())
    return (int(match.group(1)), path.name.lower())


def clean_sensor_file(input_path: Path, output_path: Path) -> FileStats:
    stats = FileStats()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", errors="ignore") as src, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for raw_line in src:
            line = raw_line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                stats.rows_skipped += 1
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
                timestamp_seconds = int(float(parts[3]))
            except ValueError:
                stats.rows_skipped += 1
                continue

            dst.write(f"{timestamp_seconds} {x:.15g} {y:.15g} {z:.15g}\n")
            stats.rows_written += 1

    return stats


def process_dataset(input_root: Path, output_root: Path, delete_profile: bool) -> None:
    if not input_root.exists():
        raise FileNotFoundError(f"Input path not found: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    print("Gyroscope values preserved in original units (degrees/s).")

    total_acc_rows = 0
    total_gyr_rows = 0
    total_skipped = 0
    deleted_profiles = 0

    user_dirs = sorted(
        [p for p in input_root.iterdir() if p.is_dir() and p.name.lower().startswith("user")],
        key=user_sort_key,
    )

    for user_dir in user_dirs:
        for split_name in ("Training", "Testing"):
            input_split = user_dir / split_name
            if not input_split.exists():
                continue

            output_split = output_root / user_dir.name / split_name
            output_split.mkdir(parents=True, exist_ok=True)

            acc_input = input_split / "Accelerometer_Data.txt"
            gyr_input = input_split / "Gyroscope_Data.txt"

            if acc_input.exists():
                acc_stats = clean_sensor_file(acc_input, output_split / "clean_acc.txt")
                total_acc_rows += acc_stats.rows_written
                total_skipped += acc_stats.rows_skipped
                print(
                    f"[{user_dir.name}/{split_name}] clean_acc.txt -> "
                    f"{acc_stats.rows_written} rows ({acc_stats.rows_skipped} skipped)"
                )
            else:
                print(f"[{user_dir.name}/{split_name}] Missing Accelerometer_Data.txt")

            if gyr_input.exists():
                gyr_stats = clean_sensor_file(
                    gyr_input, output_split / "clean_gyr.txt"
                )
                total_gyr_rows += gyr_stats.rows_written
                total_skipped += gyr_stats.rows_skipped
                print(
                    f"[{user_dir.name}/{split_name}] clean_gyr.txt -> "
                    f"{gyr_stats.rows_written} rows ({gyr_stats.rows_skipped} skipped)"
                )
            else:
                print(f"[{user_dir.name}/{split_name}] Missing Gyroscope_Data.txt")

            if delete_profile:
                profile_path = input_split / "Profile.txt"
                if profile_path.exists():
                    profile_path.unlink()
                    deleted_profiles += 1
                    print(f"[{user_dir.name}/{split_name}] Deleted Profile.txt")

    print("\nDone.")
    print(f"Output root: {output_root}")
    print(f"Total accelerometer rows written: {total_acc_rows}")
    print(f"Total gyroscope rows written: {total_gyr_rows}")
    print(f"Total skipped rows: {total_skipped}")
    print(f"Profile.txt files deleted: {deleted_profiles}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean Kumar watch IMU files into WatchHAR-style outputs "
            "(clean_acc.txt, clean_gyr.txt) with format: timestamp x y z. "
            "Preserves raw gyroscope values in their original units (degrees/s)."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Path to Kumar2015 root (default: {DEFAULT_INPUT_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Path to WatchHAR output root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--keep-profile",
        action="store_true",
        help="Keep Profile.txt files instead of deleting them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        delete_profile=not args.keep_profile,
    )


if __name__ == "__main__":
    main()
