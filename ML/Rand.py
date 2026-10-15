########################################################################################################################
# Zero-effort + dictionary-effort attack evaluation using Random Forest.
#
# Revised with validation-based threshold calibration:
# - Cleaning / segmentation / feature extraction are unchanged.
# - Uses the feature matrices returned by Prep3.get_feature_matrices().
# - Splits the original zero-effort test matrices into validation + held-out final test.
# - Chooses the probability threshold from validation data only.
# - Reports FAR, FRR, HTER on the held-out final test data only.
# - Supports either one threshold per dataset+sensor or one threshold per user+sensor.
########################################################################################################################

def warn(*args, **kwargs):
    pass


import argparse
import os
import re
import sys
import warnings
import zlib

for thread_env_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_env_var, "1")

slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
if slurm_cpus:
    try:
        if int(slurm_cpus) > 0:
            os.environ.setdefault("LOKY_MAX_CPU_COUNT", slurm_cpus)
    except ValueError:
        pass

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)

MODEL_NAME = "Rand"
RESULT_ROOT = os.path.join(PROJECT_ROOT, "Storage", "PerformanceResult")

sys.path.insert(0, PROJECT_ROOT)

from FeatureSelection import Supervised
from GlobalParameters import DefaultParam
from Preprocess import Prep3
from Preprocess import Utilities

warnings.warn = warn
warnings.filterwarnings("ignore")
np.set_printoptions(precision=3, suppress=True)

# Map the simple --dataset selector to the genuine feature folder name under FEATURE_PATH.
# 1 = Kumar2015, 2 = WISDM. (watchHar intentionally not offered for now.)
DATASET_CHOICES = {
    1: "Kumar2015",
    2: "wisdm_25Hz",
}

ATTACKER_PATTERN = re.compile(r"^U(?P<attacker_user>\d+)_T(?P<trial_id>\d+)$")
NORMAL_TRIAL_IDS = {1, 4, 7, 10}

TRIAL_NAME_MAP = {
    1: "Normal",
    2: "Slower_speed",
    3: "Faster_speed",
    4: "Normal",
    5: "Shorter_steps",
    6: "Longer_steps",
    7: "Normal",
    8: "Arm_low",
    9: "Arm_high",
    10: "Normal",
    11: "Arm_in",
    12: "Arm_out",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Random Forest with zero-effort and dictionary-effort attacks."
    )

    parser.add_argument(
        "--dataset",
        type=int,
        choices=sorted(DATASET_CHOICES.keys()),
        default=1,
        help=(
            "Which genuine dataset to train on: "
            + ", ".join(f"{key}={name}" for key, name in sorted(DATASET_CHOICES.items()))
            + ". Default is 1 (Kumar2015)."
        ),
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for GridSearchCV and final RandomForest.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.40,
        help=(
            "Fixed probability threshold for accepting genuine class +1. "
            "Also used inside GridSearchCV scoring as the training-CV reference threshold."
        ),
    )

    parser.add_argument(
        "--threshold-mode",
        choices=["fixed", "validation"],
        default="validation",
        help=(
            "fixed = use --threshold directly. "
            "validation = choose threshold from validation split and evaluate final test separately."
        ),
    )

    parser.add_argument(
        "--threshold-scope",
        choices=["dataset-sensor", "user-sensor"],
        default="dataset-sensor",
        help=(
            "dataset-sensor = one threshold per dataset/sensor combination. "
            "user-sensor = one threshold per genuine user/sensor combination."
        ),
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.40,
        help="Fraction of the original zero-effort test samples to reserve for threshold validation.",
    )

    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.01,
        help="Smallest threshold to try when threshold-mode=validation.",
    )

    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.99,
        help="Largest threshold to try when threshold-mode=validation.",
    )

    parser.add_argument(
        "--threshold-steps",
        type=int,
        default=99,
        help="Number of thresholds to try between threshold-min and threshold-max.",
    )

    parser.add_argument(
        "--frr-weight",
        type=float,
        default=2.0,
        help=(
            "Weight placed on FRR in GridSearchCV scoring and validation threshold selection. "
            "Larger values prefer lower FRR, usually at the cost of higher FAR."
        ),
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.threshold_mode == "validation":
        if not (0.0 < args.validation_ratio < 1.0):
            raise ValueError("--validation-ratio must be between 0 and 1 for validation mode.")
        if args.threshold_steps < 2:
            raise ValueError("--threshold-steps must be at least 2 for validation mode.")
        if args.threshold_min >= args.threshold_max:
            raise ValueError("--threshold-min must be smaller than --threshold-max.")

    if args.frr_weight < 0:
        raise ValueError("--frr-weight must be non-negative.")


def sort_user_name(name: str) -> int:
    return int("".join(ch for ch in name if ch.isdigit()) or "0")


def parse_attacker_name(attacker: str) -> dict:
    match = ATTACKER_PATTERN.match(attacker)

    if not match:
        return {
            "attacker": attacker,
            "attacker_user": None,
            "trial_id": None,
            "trial_code": None,
            "trial_label": "Unknown",
            "trial_group": "unknown",
        }

    attacker_user = int(match.group("attacker_user"))
    trial_id = int(match.group("trial_id"))

    return {
        "attacker": attacker,
        "attacker_user": attacker_user,
        "trial_id": trial_id,
        "trial_code": f"T{trial_id}",
        "trial_label": TRIAL_NAME_MAP.get(trial_id, "Unknown"),
        "trial_group": "normal" if trial_id in NORMAL_TRIAL_IDS else "changed",
    }


def attacker_sort_key(meta: dict) -> tuple[int, int, str]:
    user = meta["attacker_user"] if meta["attacker_user"] is not None else 10**9
    trial = meta["trial_id"] if meta["trial_id"] is not None else 10**9
    return user, trial, meta["attacker"]


def compute_far_frr_hter(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[-1, 1]).ravel()

    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    hter = (far + frr) / 2.0

    return far, frr, hter


def weighted_far_frr_loss_values(far: float, frr: float, frr_weight: float) -> float:
    return (far + frr_weight * frr) / (1.0 + frr_weight)


def weighted_far_frr_loss(y_true: np.ndarray, y_pred: np.ndarray, frr_weight: float) -> float:
    far, frr, _ = compute_far_frr_hter(y_true, y_pred)
    return weighted_far_frr_loss_values(far, frr, frr_weight)


def get_positive_class_probability(model: RandomForestClassifier, x_data: np.ndarray) -> np.ndarray:
    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(f"Positive class +1 not found in model classes: {classes}")

    positive_index = classes.index(1)
    return model.predict_proba(x_data)[:, positive_index]


def predict_scores_with_threshold(positive_scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(positive_scores >= threshold, 1, -1)


def evaluate_scores_with_threshold(
    positive_scores: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    y_pred = predict_scores_with_threshold(positive_scores, threshold)
    return compute_far_frr_hter(y_true, y_pred)


def predict_with_threshold(
    model: RandomForestClassifier,
    x_data: np.ndarray,
    threshold: float,
) -> np.ndarray:
    positive_probs = get_positive_class_probability(model, x_data)
    return predict_scores_with_threshold(positive_probs, threshold)


def evaluate_with_threshold(
    model: RandomForestClassifier,
    x_data: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    y_pred = predict_with_threshold(model, x_data, threshold)
    return compute_far_frr_hter(y_true, y_pred)


def make_weighted_threshold_scorer(threshold: float, frr_weight: float):
    def scorer(estimator, x_data, y_true):
        y_pred = predict_with_threshold(estimator, x_data, threshold)
        loss = weighted_far_frr_loss(y_true, y_pred, frr_weight)
        return -loss

    return scorer


def build_threshold_grid(threshold_min: float, threshold_max: float, threshold_steps: int) -> np.ndarray:
    return np.linspace(threshold_min, threshold_max, threshold_steps)


def select_threshold_from_scores(
    positive_scores: np.ndarray,
    y_true: np.ndarray,
    threshold_grid: np.ndarray,
    frr_weight: float,
) -> dict:
    """
    Select threshold from validation scores only.

    Primary objective: minimize weighted FAR/FRR loss.
    Tie-breaker: prefer lower FRR, then lower HTER, then lower FAR.
    The lower-FRR tie-breaker is intentional because your current issue is high FRR.
    """
    best = None

    for threshold in threshold_grid:
        far, frr, hter = evaluate_scores_with_threshold(positive_scores, y_true, float(threshold))
        loss = weighted_far_frr_loss_values(far, frr, frr_weight)

        candidate = {
            "threshold": float(threshold),
            "far": float(far),
            "frr": float(frr),
            "hter": float(hter),
            "loss": float(loss),
        }

        candidate_key = (candidate["loss"], candidate["frr"], candidate["hter"], candidate["far"])

        if best is None or candidate_key < best["key"]:
            candidate["key"] = candidate_key
            best = candidate

    if best is None:
        raise ValueError("Could not select a threshold; threshold grid was empty.")

    best.pop("key", None)
    return best


def make_stable_seed(*parts: object) -> int:
    base_seed = int(getattr(DefaultParam, "RANDOM_SEED", 21))
    text = "|".join(str(part) for part in parts)
    checksum = zlib.crc32(text.encode("utf-8"))
    return (base_seed + checksum) % (2**32)


def split_matrix_rows(
    data: np.ndarray,
    validation_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split rows/samples/windows into validation and final test.
    This does NOT split feature columns.
    """
    data = np.asarray(data)
    n_samples = data.shape[0]

    if n_samples < 2:
        raise ValueError(
            "Need at least 2 samples/windows to split validation and final test. "
            f"Got n_samples={n_samples}."
        )

    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_val = int(round(n_samples * validation_ratio))
    n_val = max(1, min(n_val, n_samples - 1))

    val_idx = indices[:n_val]
    test_idx = indices[n_val:]

    return data[val_idx], data[test_idx]


def list_genuine_users() -> list[str]:
    all_dirs = os.listdir(DefaultParam.FEATUREFILE_PATH)

    user_names = sorted(
        [
            d
            for d in all_dirs
            if d.startswith("User")
            and os.path.isdir(os.path.join(DefaultParam.FEATUREFILE_PATH, d))
        ],
        key=sort_user_name,
    )

    return [u for u in user_names if u not in DefaultParam.BAD_USERS]


def list_available_genuine_datasets() -> list[str]:
    feature_root = str(DefaultParam.FEATURE_PATH)

    if not os.path.isdir(feature_root):
        return []

    datasets = []

    for dataset_name in os.listdir(feature_root):
        dataset_path = os.path.join(feature_root, dataset_name)

        if not os.path.isdir(dataset_path):
            continue

        has_user_dirs = any(
            d.startswith("User")
            and os.path.isdir(os.path.join(dataset_path, d))
            for d in os.listdir(dataset_path)
        )

        if has_user_dirs:
            datasets.append(dataset_name)

    return sorted(datasets)


def configure_genuine_dataset(dataset_name: str) -> None:
    dataset_path = os.path.join(str(DefaultParam.FEATURE_PATH), dataset_name)

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Genuine feature dataset not found: {dataset_path}")

    DefaultParam.FEATUREFILE_PATH = dataset_path


def list_dictionary_attackers() -> list[dict]:
    all_dirs = os.listdir(DefaultParam.DICT_FEATUREFILE_PATH)

    attacker_names = [
        d
        for d in all_dirs
        if os.path.isdir(os.path.join(DefaultParam.DICT_FEATUREFILE_PATH, d))
    ]

    metas = [parse_attacker_name(name) for name in attacker_names]
    return sorted(metas, key=attacker_sort_key)


def build_results_dir(genuine_dataset: str) -> str:
    result_location = os.path.join(RESULT_ROOT, genuine_dataset, MODEL_NAME)

    if os.path.exists(result_location):
        print("Result directory already exists, will overwrite CSV files")
    else:
        os.makedirs(result_location, exist_ok=True)
        print(f"#__Created a new directory to save results at {result_location}")

    return result_location


def make_rf_grid(n_jobs: int) -> tuple[RandomForestClassifier, dict]:
    estimator = RandomForestClassifier(
        n_jobs=n_jobs,
        random_state=getattr(DefaultParam, "RANDOM_SEED", 21),
    )

    param_grid = {
        "n_estimators": [1000],
        "max_features": [0.6],
        "max_depth": [5],
        "min_samples_leaf": [6],
        "class_weight": [
            None,
            {-1: 1, 1: 2},
            {-1: 1, 1: 3},
        ],
    }

    return estimator, param_grid


def save_summaries(result_df: pd.DataFrame, result_location: str) -> None:
    dict_df = result_df[result_df["attacktype"] == "dict-effort"].copy()

    if dict_df.empty:
        print("No dictionary rows available; skipped summary exports.")
        return

    idx = dict_df.groupby(["user", "sensors", "trial_group"])["far"].idxmax()
    worst_by_group = dict_df.loc[idx].sort_values(["user", "sensors", "trial_group"])
    worst_by_group.to_csv(
        os.path.join(result_location, "RF_dictionary_worst_by_group.csv"),
        index=False,
    )

    idx_overall = dict_df.groupby(["user", "sensors"])["far"].idxmax()
    worst_overall = dict_df.loc[idx_overall].sort_values(["user", "sensors"])
    worst_overall.to_csv(
        os.path.join(result_location, "RF_dictionary_worst_overall.csv"),
        index=False,
    )

    group_summary = (
        dict_df.groupby(["trial_group", "sensors"])
        .agg(
            n_attacks=("far", "size"),
            mean_far=("far", "mean"),
            median_far=("far", "median"),
            max_far=("far", "max"),
            mean_frr=("frr", "mean"),
            median_frr=("frr", "median"),
            max_frr=("frr", "max"),
            mean_hter=("hter", "mean"),
            median_hter=("hter", "median"),
            max_hter=("hter", "max"),
        )
        .reset_index()
        .sort_values(["sensors", "trial_group"])
    )

    group_summary.to_csv(
        os.path.join(result_location, "RF_dictionary_group_summary.csv"),
        index=False,
    )

    trial_summary = (
        dict_df.groupby(["trial_id", "trial_label", "sensors"])
        .agg(
            n_attacks=("far", "size"),
            mean_far=("far", "mean"),
            median_far=("far", "median"),
            max_far=("far", "max"),
            mean_frr=("frr", "mean"),
            median_frr=("frr", "median"),
            max_frr=("frr", "max"),
            mean_hter=("hter", "mean"),
            median_hter=("hter", "median"),
            max_hter=("hter", "max"),
        )
        .reset_index()
        .sort_values(["sensors", "trial_id"])
    )

    trial_summary["trial_id"] = pd.to_numeric(
        trial_summary["trial_id"],
        errors="coerce",
    ).astype("Int64")

    trial_summary.to_csv(
        os.path.join(result_location, "RF_dictionary_trial_summary.csv"),
        index=False,
    )

    print("Saved summary CSV files")


def train_selected_feature_model(
    dataset_name: str,
    user: str,
    curr_comb: tuple,
    ze_imp_user_list: list[str],
    list_num_features: list[int],
    fsselect_method,
    validation_ratio: float,
    scorer,
    n_jobs: int,
) -> dict:
    best_feature_result = None

    for num_features in list_num_features:
        (
            comb_selected_sens_gtr,
            comb_selected_sens_gts,
            comb_selected_sens_itr,
            comb_selected_sens_its,
            normalizers,
            feature_selectors,
        ) = Prep3.get_feature_matrices(
            curr_comb,
            ze_imp_user_list,
            user,
            fsselect_method,
            num_features,
        )

        gts_val, gts_test = split_matrix_rows(
            comb_selected_sens_gts,
            validation_ratio=validation_ratio,
            seed=make_stable_seed(dataset_name, user, curr_comb, num_features, "genuine-test"),
        )
        its_val, its_test = split_matrix_rows(
            comb_selected_sens_its,
            validation_ratio=validation_ratio,
            seed=make_stable_seed(dataset_name, user, curr_comb, num_features, "impostor-test"),
        )

        training_data = np.vstack(
            (
                comb_selected_sens_gtr,
                comb_selected_sens_itr,
            )
        )

        training_labels = np.concatenate(
            (
                Prep3.get_labels(comb_selected_sens_gtr, 1),
                Prep3.get_labels(comb_selected_sens_itr, -1),
            )
        )

        rf_estimator, param_grid = make_rf_grid(n_jobs=1)

        search = GridSearchCV(
            estimator=rf_estimator,
            param_grid=param_grid,
            cv=3,
            scoring=scorer,
            n_jobs=n_jobs,
            verbose=0,
        )

        search.fit(training_data, training_labels)
        best_params = search.best_params_

        model = RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            max_features=best_params["max_features"],
            min_samples_leaf=best_params["min_samples_leaf"],
            class_weight=best_params["class_weight"],
            n_jobs=n_jobs,
            random_state=getattr(DefaultParam, "RANDOM_SEED", 21),
        )

        model.fit(training_data, training_labels)

        zero_val_data = np.vstack((gts_val, its_val))
        zero_val_labels = np.concatenate(
            (
                Prep3.get_labels(gts_val, 1),
                Prep3.get_labels(its_val, -1),
            )
        )

        zero_test_data = np.vstack((gts_test, its_test))
        zero_test_labels = np.concatenate(
            (
                Prep3.get_labels(gts_test, 1),
                Prep3.get_labels(its_test, -1),
            )
        )

        current_feature_result = {
            "user": user,
            "sensors": curr_comb,
            "num_features": num_features,
            "model": model,
            "best_params": best_params,
            "grid_score": float(search.best_score_),
            "train_scores": get_positive_class_probability(model, training_data),
            "train_labels": training_labels,
            "zero_val_scores": get_positive_class_probability(model, zero_val_data),
            "zero_val_labels": zero_val_labels,
            "zero_test_scores": get_positive_class_probability(model, zero_test_data),
            "zero_test_labels": zero_test_labels,
            "gts_test": gts_test,
            "gts_test_scores": get_positive_class_probability(model, gts_test),
            "normalizers": normalizers,
            "feature_selectors": feature_selectors,
            "n_train_genuine": int(comb_selected_sens_gtr.shape[0]),
            "n_train_impostor": int(comb_selected_sens_itr.shape[0]),
            "n_val_genuine": int(gts_val.shape[0]),
            "n_val_impostor": int(its_val.shape[0]),
            "n_test_genuine": int(gts_test.shape[0]),
            "n_test_impostor": int(its_test.shape[0]),
        }

        # Important:
        # Select feature count by CV score only.
        # Do not select it using validation or final test FAR/FRR/HTER.
        if best_feature_result is None:
            best_feature_result = current_feature_result
        elif current_feature_result["grid_score"] > best_feature_result["grid_score"]:
            best_feature_result = current_feature_result

    if best_feature_result is None:
        raise RuntimeError(f"Could not train a selected Random Forest model for {user}, {curr_comb}")

    return best_feature_result


def threshold_info_to_row(
    dataset_name: str,
    sensors: tuple,
    user: str,
    threshold_scope: str,
    threshold_info: dict,
    n_val_genuine: int,
    n_val_impostor: int,
    frr_weight: float,
) -> dict:
    return {
        "dataset": dataset_name,
        "model": MODEL_NAME,
        "threshold_scope": threshold_scope,
        "user": user,
        "sensors": sensors,
        "threshold": threshold_info["threshold"],
        "validation_far": threshold_info["far"],
        "validation_frr": threshold_info["frr"],
        "validation_hter": threshold_info["hter"],
        "validation_loss": threshold_info["loss"],
        "frr_weight": frr_weight,
        "n_val_genuine": n_val_genuine,
        "n_val_impostor": n_val_impostor,
    }


def add_common_result_fields(
    row: dict,
    task: dict,
    threshold: float,
    threshold_mode: str,
    threshold_scope: str,
    validation_ratio: float,
    frr_weight: float,
    train_metrics: tuple[float, float, float],
    val_metrics: tuple[float, float, float],
    zero_test_metrics: tuple[float, float, float],
) -> dict:
    best_params = task["best_params"]
    tr_far, tr_frr, tr_hter = train_metrics
    val_far, val_frr, val_hter = val_metrics
    tsz_far, tsz_frr, tsz_hter = zero_test_metrics

    row.update(
        {
            "sensors": task["sensors"],
            "nfeatures": task["num_features"],
            "threshold": threshold,
            "threshold_mode": threshold_mode,
            "threshold_scope": threshold_scope,
            "validation_ratio": validation_ratio,
            "frr_weight": frr_weight,
            "grid_score": task["grid_score"],
            "n_estimators": best_params["n_estimators"],
            "max_depth": best_params["max_depth"],
            "max_features": best_params["max_features"],
            "min_samples_leaf": best_params["min_samples_leaf"],
            "class_weight": best_params["class_weight"],
            "val_far": val_far,
            "val_frr": val_frr,
            "val_hter": val_hter,
            "train_far": tr_far,
            "train_frr": tr_frr,
            "train_hter": tr_hter,
            "zero_test_far": tsz_far,
            "zero_test_frr": tsz_frr,
            "zero_test_hter": tsz_hter,
            "n_train_genuine": task["n_train_genuine"],
            "n_train_impostor": task["n_train_impostor"],
            "n_val_genuine": task["n_val_genuine"],
            "n_val_impostor": task["n_val_impostor"],
            "n_test_genuine": task["n_test_genuine"],
            "n_test_impostor": task["n_test_impostor"],
            # Backward-compatible reference columns used by your old plotting/summary code.
            "ref_far": tr_far,
            "ref_frr": tr_frr,
            "refhter": tr_hter,
        }
    )
    return row


def evaluate_sensor_tasks(
    dataset_name: str,
    curr_comb: tuple,
    sensor_tasks: list[dict],
    dict_attacker_metas: list[dict],
    threshold_mode: str,
    threshold_scope: str,
    fixed_threshold: float,
    validation_ratio: float,
    threshold_grid: np.ndarray,
    frr_weight: float,
    user_rows_by_user: dict[str, list[dict]],
    result_rows: list[dict],
    threshold_summary_rows: list[dict],
) -> None:
    if threshold_mode == "validation" and threshold_scope == "dataset-sensor":
        pooled_val_scores = np.concatenate([task["zero_val_scores"] for task in sensor_tasks])
        pooled_val_labels = np.concatenate([task["zero_val_labels"] for task in sensor_tasks])
        pooled_info = select_threshold_from_scores(
            pooled_val_scores,
            pooled_val_labels,
            threshold_grid,
            frr_weight,
        )
        selected_threshold_by_user = {task["user"]: pooled_info["threshold"] for task in sensor_tasks}

        threshold_summary_rows.append(
            threshold_info_to_row(
                dataset_name=dataset_name,
                sensors=curr_comb,
                user="ALL",
                threshold_scope=threshold_scope,
                threshold_info=pooled_info,
                n_val_genuine=int(np.sum(pooled_val_labels == 1)),
                n_val_impostor=int(np.sum(pooled_val_labels == -1)),
                frr_weight=frr_weight,
            )
        )

        print(
            f"#__Selected dataset/sensor threshold | dataset={dataset_name} | sensors={curr_comb} | "
            f"threshold={pooled_info['threshold']:.4f} | "
            f"VAL FAR={pooled_info['far']:.4f} | VAL FRR={pooled_info['frr']:.4f} | "
            f"VAL HTER={pooled_info['hter']:.4f}",
            flush=True,
        )
    else:
        selected_threshold_by_user = {}

    for task in sensor_tasks:
        user = task["user"]
        model = task["model"]

        if threshold_mode == "validation" and threshold_scope == "user-sensor":
            user_info = select_threshold_from_scores(
                task["zero_val_scores"],
                task["zero_val_labels"],
                threshold_grid,
                frr_weight,
            )
            threshold = user_info["threshold"]
            threshold_summary_rows.append(
                threshold_info_to_row(
                    dataset_name=dataset_name,
                    sensors=curr_comb,
                    user=user,
                    threshold_scope=threshold_scope,
                    threshold_info=user_info,
                    n_val_genuine=task["n_val_genuine"],
                    n_val_impostor=task["n_val_impostor"],
                    frr_weight=frr_weight,
                )
            )
        elif threshold_mode == "validation" and threshold_scope == "dataset-sensor":
            threshold = selected_threshold_by_user[user]
        else:
            threshold = fixed_threshold

        train_metrics = evaluate_scores_with_threshold(
            task["train_scores"],
            task["train_labels"],
            threshold,
        )
        val_metrics = evaluate_scores_with_threshold(
            task["zero_val_scores"],
            task["zero_val_labels"],
            threshold,
        )
        zero_test_metrics = evaluate_scores_with_threshold(
            task["zero_test_scores"],
            task["zero_test_labels"],
            threshold,
        )

        tsz_far, tsz_frr, tsz_hter = zero_test_metrics

        zero_row = {
            "user": user,
            "attacktype": "zero-effort",
            "attacker": "rest",
            "attacker_user": None,
            "trial_id": None,
            "trial_code": None,
            "trial_label": None,
            "trial_group": None,
            "far": tsz_far,
            "frr": tsz_frr,
            "hter": tsz_hter,
        }

        zero_row = add_common_result_fields(
            zero_row,
            task=task,
            threshold=threshold,
            threshold_mode=threshold_mode,
            threshold_scope=threshold_scope,
            validation_ratio=validation_ratio,
            frr_weight=frr_weight,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            zero_test_metrics=zero_test_metrics,
        )

        user_rows_by_user[user].append(zero_row)
        result_rows.append(zero_row)

        max_far_all = -1.0
        max_far_normal = -1.0
        max_far_changed = -1.0

        best_all = "---"
        best_normal = "---"
        best_changed = "---"

        worst_dict_hter_all = -1.0
        worst_dict_hter_attacker = "---"
        worst_dict_frr = None

        for meta in dict_attacker_metas:
            dict_attack_data = Prep3.get_feature_matrix_for_dict_attack(
                curr_comb,
                meta["attacker"],
                task["feature_selectors"],
                task["normalizers"],
            )

            dict_attack_scores = get_positive_class_probability(model, dict_attack_data)

            dict_eval_scores = np.concatenate(
                (
                    task["gts_test_scores"],
                    dict_attack_scores,
                )
            )

            dict_eval_labels = np.concatenate(
                (
                    Prep3.get_labels(task["gts_test"], 1),
                    Prep3.get_labels(dict_attack_data, -1),
                )
            )

            tsd_far, tsd_frr, tsd_hter = evaluate_scores_with_threshold(
                dict_eval_scores,
                dict_eval_labels,
                threshold,
            )

            if tsd_far > max_far_all:
                max_far_all = tsd_far
                best_all = meta["attacker"]

            if meta["trial_group"] == "normal" and tsd_far > max_far_normal:
                max_far_normal = tsd_far
                best_normal = meta["attacker"]

            if meta["trial_group"] == "changed" and tsd_far > max_far_changed:
                max_far_changed = tsd_far
                best_changed = meta["attacker"]

            if tsd_hter > worst_dict_hter_all:
                worst_dict_hter_all = tsd_hter
                worst_dict_hter_attacker = meta["attacker"]
                worst_dict_frr = tsd_frr

            dict_row = {
                "user": user,
                "attacktype": "dict-effort",
                "attacker": meta["attacker"],
                "attacker_user": meta["attacker_user"],
                "trial_id": meta["trial_id"],
                "trial_code": meta["trial_code"],
                "trial_label": meta["trial_label"],
                "trial_group": meta["trial_group"],
                "far": tsd_far,
                "frr": tsd_frr,
                "hter": tsd_hter,
            }

            dict_row = add_common_result_fields(
                dict_row,
                task=task,
                threshold=threshold,
                threshold_mode=threshold_mode,
                threshold_scope=threshold_scope,
                validation_ratio=validation_ratio,
                frr_weight=frr_weight,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                zero_test_metrics=zero_test_metrics,
            )

            # For dictionary rows, the reference zero-effort metrics should be the held-out zero-effort test.
            dict_row["ref_far"] = tsz_far
            dict_row["ref_frr"] = tsz_frr
            dict_row["refhter"] = tsz_hter

            user_rows_by_user[user].append(dict_row)
            result_rows.append(dict_row)

        tr_far, tr_frr, tr_hter = train_metrics
        val_far, val_frr, val_hter = val_metrics

        print("\n" + "=" * 100)
        print(f"USER: {user} | SENSORS: {curr_comb} | SELECTED FEATURES: {task['num_features']}")
        print(
            f"MODEL: threshold={threshold:.4f} | threshold_mode={threshold_mode} | "
            f"threshold_scope={threshold_scope} | frr_weight={frr_weight:.2f} | "
            f"grid_score={task['grid_score']:.4f}"
        )
        print(
            f"SPLIT COUNTS           -> "
            f"VAL genuine/impostor: {task['n_val_genuine']}/{task['n_val_impostor']} | "
            f"TEST genuine/impostor: {task['n_test_genuine']}/{task['n_test_impostor']}"
        )
        print(
            f"TRAIN CHECK           -> "
            f"FAR: {tr_far:.4f} | FRR: {tr_frr:.4f} | HTER: {tr_hter:.4f}"
        )
        print(
            f"VALIDATION CALIBRATION -> "
            f"FAR: {val_far:.4f} | FRR: {val_frr:.4f} | HTER: {val_hter:.4f}"
        )
        print(
            f"ZERO-EFFORT FINAL TEST -> "
            f"FAR: {tsz_far:.4f} | FRR: {tsz_frr:.4f} | HTER: {tsz_hter:.4f}"
        )
        print(
            f"DICT-EFFORT FINAL TEST -> "
            f"MAX FAR: {max_far_all:.4f} [{best_all}] | "
            f"WORST HTER: {worst_dict_hter_all:.4f} [{worst_dict_hter_attacker}] | "
            f"DICT FRR: {worst_dict_frr:.4f}"
        )
        print(
            f"DICT NORMAL/CHANGED    -> "
            f"NORMAL MAX FAR: {max_far_normal:.4f} [{best_normal}] | "
            f"CHANGED MAX FAR: {max_far_changed:.4f} [{best_changed}]"
        )
        print("=" * 100, flush=True)


def evaluate_dataset(
    dataset_name: str,
    dict_attacker_metas: list[dict],
    n_jobs: int,
    threshold: float,
    threshold_mode: str,
    threshold_scope: str,
    validation_ratio: float,
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
    frr_weight: float,
) -> None:
    configure_genuine_dataset(dataset_name)

    gen_user_list = list_genuine_users()
    ze_imp_user_list = list_genuine_users()

    if not gen_user_list:
        raise ValueError(f"No genuine user feature folders found in dataset: {dataset_name}")

    sensor_configurations = Utilities.getallcombinations(DefaultParam.SENSOR_LIST)
    list_num_features = [20, 30, 50]
    fsselect_method = Supervised.fs_using_MI
    result_location = build_results_dir(dataset_name)
    threshold_grid = build_threshold_grid(threshold_min, threshold_max, threshold_steps)

    print(f"#__Running Random Forest for sensor configurations: {sensor_configurations}")
    print(f"#__Feature counts: {list_num_features}")
    print(f"#__Threshold mode: {threshold_mode}")
    print(f"#__Threshold scope: {threshold_scope}")
    print(f"#__Fixed/GridSearchCV reference threshold: {threshold}")
    print(f"#__Validation ratio: {validation_ratio}")
    print(f"#__Threshold grid: min={threshold_min}, max={threshold_max}, steps={threshold_steps}")
    print(f"#__FRR weight in scoring/threshold selection: {frr_weight}")

    if DefaultParam.CLASS_BALANCING:
        print("#__Applying SMOTE for CLASS BALANCING")

    if DefaultParam.FEATURE_DOMAIN_LIST:
        print(f"Using features from: {DefaultParam.FEATURE_DOMAIN_LIST}")

    print(f"Using genuine features from: {DefaultParam.FEATUREFILE_PATH}")
    print(f"#__Total users to evaluate: {len(gen_user_list)}")
    print(f"#__Total dictionary attackers: {len(dict_attacker_metas)}")
    print(f"Genuine user list: {gen_user_list}")

    scorer = make_weighted_threshold_scorer(
        threshold=threshold,
        frr_weight=frr_weight,
    )

    result_rows: list[dict] = []
    threshold_summary_rows: list[dict] = []
    user_rows_by_user: dict[str, list[dict]] = {user: [] for user in gen_user_list}

    # Process one sensor configuration at a time. This allows dataset/sensor threshold calibration
    # without changing cleaning/segmentation code.
    for curr_comb in sensor_configurations:
        print(f"\n#__Training selected models for dataset={dataset_name}, sensors={curr_comb}", flush=True)
        sensor_tasks: list[dict] = []

        for user in gen_user_list:
            task = train_selected_feature_model(
                dataset_name=dataset_name,
                user=user,
                curr_comb=curr_comb,
                ze_imp_user_list=ze_imp_user_list,
                list_num_features=list_num_features,
                fsselect_method=fsselect_method,
                validation_ratio=validation_ratio,
                scorer=scorer,
                n_jobs=n_jobs,
            )
            sensor_tasks.append(task)

        evaluate_sensor_tasks(
            dataset_name=dataset_name,
            curr_comb=curr_comb,
            sensor_tasks=sensor_tasks,
            dict_attacker_metas=dict_attacker_metas,
            threshold_mode=threshold_mode,
            threshold_scope=threshold_scope,
            fixed_threshold=threshold,
            validation_ratio=validation_ratio,
            threshold_grid=threshold_grid,
            frr_weight=frr_weight,
            user_rows_by_user=user_rows_by_user,
            result_rows=result_rows,
            threshold_summary_rows=threshold_summary_rows,
        )

        # Release models for this sensor before moving to the next sensor.
        del sensor_tasks

    for user in gen_user_list:
        cuser_df = pd.DataFrame(user_rows_by_user[user])

        for int_col in ("attacker_user", "trial_id"):
            if int_col in cuser_df.columns:
                cuser_df[int_col] = pd.to_numeric(
                    cuser_df[int_col],
                    errors="coerce",
                ).astype("Int64")

        cuser_df.to_csv(
            os.path.join(result_location, user + ".csv"),
            index=False,
        )

    result_df = pd.DataFrame(result_rows)

    for int_col in ("attacker_user", "trial_id"):
        if int_col in result_df.columns:
            result_df[int_col] = pd.to_numeric(
                result_df[int_col],
                errors="coerce",
            ).astype("Int64")

    combined_path = os.path.join(
        result_location,
        os.path.basename(__file__)[:-3] + "_combined.csv",
    )

    result_df.to_csv(combined_path, index=False)
    print(f"Saved combined results to: {combined_path}")

    threshold_df = pd.DataFrame(threshold_summary_rows)
    threshold_path = os.path.join(result_location, "RF_threshold_calibration.csv")
    threshold_df.to_csv(threshold_path, index=False)
    print(f"Saved threshold calibration summary to: {threshold_path}")

    save_summaries(result_df, result_location)


def main() -> None:
    args = parse_args()
    validate_args(args)

    print(f"#__Detected os.cpu_count(): {os.cpu_count()}", flush=True)
    print(f"#__SLURM_CPUS_PER_TASK: {os.environ.get('SLURM_CPUS_PER_TASK', 'not set')}", flush=True)
    print(f"#__LOKY_MAX_CPU_COUNT: {os.environ.get('LOKY_MAX_CPU_COUNT', 'not set')}", flush=True)
    print(f"#__GridSearchCV / final RF n_jobs: {args.n_jobs}", flush=True)
    print(f"#__Threshold mode: {args.threshold_mode}", flush=True)
    print(f"#__Threshold scope: {args.threshold_scope}", flush=True)
    print(f"#__Fixed/GridSearchCV reference threshold: {args.threshold}", flush=True)
    print(f"#__Validation ratio: {args.validation_ratio}", flush=True)
    print(f"#__Threshold grid: {args.threshold_min} to {args.threshold_max} in {args.threshold_steps} steps", flush=True)
    print(f"#__FRR weight: {args.frr_weight}", flush=True)

    dataset_name = DATASET_CHOICES[args.dataset]
    dataset_path = os.path.join(str(DefaultParam.FEATURE_PATH), dataset_name)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"Selected genuine feature dataset not found: {dataset_path}"
        )

    dict_attacker_metas = list_dictionary_attackers()

    if not dict_attacker_metas:
        raise ValueError(
            f"No dictionary attacker folders found under {DefaultParam.DICT_FEATUREFILE_PATH}"
        )

    print(f"#__Selected genuine dataset: {args.dataset} ({dataset_name})", flush=True)
    print(f"#__Dictionary attackers: {len(dict_attacker_metas)}", flush=True)

    print(f"\n#__======== Dataset: {dataset_name} ========", flush=True)
    evaluate_dataset(
        dataset_name=dataset_name,
        dict_attacker_metas=dict_attacker_metas,
        n_jobs=args.n_jobs,
        threshold=args.threshold,
        threshold_mode=args.threshold_mode,
        threshold_scope=args.threshold_scope,
        validation_ratio=args.validation_ratio,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_steps=args.threshold_steps,
        frr_weight=args.frr_weight,
    )


if __name__ == "__main__":
    main()
