from pathlib import Path
import os


class DefaultParam:
    """Shared configuration for the full pipeline."""

    # ── Project paths ──
    PROJECT_ROOT = Path(__file__).resolve().parent

    STORAGE_ROOT = PROJECT_ROOT / "Storage"

    # Genuine dataset paths
    DATA_PATH = STORAGE_ROOT / "CleanData" / "Genuine"
    FEATURE_PATH = STORAGE_ROOT / "FeatureData" / "Genuine"

    # ── Aliases that Prep3 / kNNZDH expect ──
    FEATUREFILE_PATH = str(FEATURE_PATH)

    # Dictionary / High-effort paths (not used for zero-effort, but Prep3 references them)
    DICT_FEATUREFILE_PATH = str(STORAGE_ROOT / "FeatureData" / "Dictionary")
    HIGH_FEATUREFILE_PATH = str(STORAGE_ROOT / "FeatureData" / "High")

    # ── Sensor configuration ──
    # Prep3 builds filenames as:  feat_clean_{sensor}.csv
    # so these must match your feature file names: feat_clean_acc.csv, feat_clean_gyr.csv
    SENSOR_LIST = ["acc", "gyr"]
    SENSOR_FILE_LIST = ["clean_acc.txt", "clean_gyr.txt"]

    # ── Feature domain configuration ──
    FEATURE_DOMAIN_LIST = ["TimeFeatures", "FrequencyFeatures"]
    FEATURE_DOMAIN_LIST_SHORT = ["time", "freq"]

    # ── Column names for raw data ──
    NEW_COLUMN_NAMES = ["timestamps", "x", "y", "z"]

    # ── Sliding window settings ──
    WINLENGTH = 8   # seconds
    WINSTEP = 4     # seconds

    # ── Feature extractor parameters ──
    NUM_BINS_FOR_TD = 16
    NUM_BINS_FOR_FFT = 16
    FFTLENGTH = 512

    # ── Smoothing ──
    SMOOTH_PERCENT = 0.05

    # ── Classification / evaluation parameters (used by Prep3 + kNNZDH) ──
    BAD_USERS = []
    NUM_SAMPLE_IMP = 5              # impostor samples per user
    CLASS_BALANCING = True
    NNEIGHBORS_FOR_OVERSAMPLING = 7
    NORMALIZE_FEATURES = True
    RANDOM_SEED = 21

    # ── Feature selection ──
    APPLY_SUPERVISED_FSELECTION = True

    # ── Attack parameters (not used for zero-effort, kept for compatibility) ──
    HIGHEFFORT_USER_LIST = []
    MODES = ["Training", "Testing"]

    # ── Split policy ──
    TRAIN_TARGET_RATIO = 0.67
    MIN_WINDOWS_PER_SPLIT = 1
    ALLOW_SINGLE_TRIAL_FALLBACK = True
    SINGLE_TRIAL_TRAIN_RATIO = 0.67

    # ── Sampling-rate estimation ──
    DEFAULT_SAMPLING_RATE = 25
    MIN_VALID_SAMPLING_RATE = 5
    MAX_VALID_SAMPLING_RATE = 500

    # ── Parsing ──
    INPUT_SEPARATOR_REGEX = r"[\s,]+"
    RECURSE_USER_FOLDERS = True
