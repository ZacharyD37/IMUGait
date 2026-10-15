########################################################################################################################
# SVM zero-effort + dictionary-effort attack evaluation for the current 12-trial protocol.
#
# Revised to use the same general threshold strategy as Rand.py:
# - Uses SVM decision_function + manual threshold.
# - Uses a thresholded FAR/FRR scorer inside GridSearchCV.
# - Adds FRR weighting during CV model selection.
# - Adds class_weight options so the grid can prefer genuine recall when helpful.
# - Selects feature count by CV score only, not by zero-effort or dictionary test performance.
# - Prints TEST FAR, FRR, HTER clearly for each user/sensor.
#
# Label convention:
# - genuine user = +1
# - impostor / attacker = -1
#
# Trial grouping:
# - Normal trials:  T1, T4, T7, T10
# - Changed trials: T2, T3, T5, T6, T8, T9, T11, T12
########################################################################################################################

# !/usr/bin/env python -W ignore::DeprecationWarning

def warn(*args, **kwargs):
    pass


import argparse
import os
import re
import sys
import warnings


# Pin BLAS/OpenMP to one thread per GridSearch worker. This prevents oversubscription
# when GridSearchCV launches multiple parallel processes.
for thread_env_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_env_var, "1")


def get_default_parallel_worker_count() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass

    manual_jobs = os.environ.get("IMUGAIT_N_JOBS")
    if manual_jobs:
        try:
            return max(1, int(manual_jobs))
        except ValueError:
            pass

    return max(1, (os.cpu_count() or 2) - 1)


DEFAULT_N_JOBS = get_default_parallel_worker_count()
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(DEFAULT_N_JOBS))

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)

MODEL_NAME = "svm"
RESULT_ROOT = os.path.join(PROJECT_ROOT, "Storage", "PerformanceResult")

# Add the project root so imports work regardless of current working directory.
sys.path.insert(0, PROJECT_ROOT)

from FeatureSelection import Supervised
from GlobalParameters import DefaultParam
from Preprocess import Prep3
from Preprocess import Utilities

warnings.warn = warn
warnings.filterwarnings("ignore")
np.set_printoptions(precision=3, suppress=True)

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

FEATURE_COUNTS_TO_TEST = [20, 30, 50]

# Selectable genuine datasets. The numeric key is what the user passes via --dataset.
# Each value is a keyword matched (case-insensitively) against the dataset folder names
# under DefaultParam.FEATURE_PATH. watchHAR is intentionally not offered here.
DATASET_CHOICES = {
    1: "Kumar2015",
    2: "wisdm",
}

# Useful for a small debug run. Do not turn this on for a full run unless you want a lot of output.
THRESHOLD_SWEEP_VALUES = [
    0.00,
    -0.05,
    -0.10,
    -0.15,
    -0.20,
    -0.25,
    -0.30,
    -0.35,
    -0.40,
    -0.45,
    -0.50,
    -0.55,
    -0.60,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SVM with zero-effort and dictionary-effort attacks."
    )

    parser.add_argument(
        "--dataset",
        type=int,
        default=1,
        choices=sorted(DATASET_CHOICES.keys()),
        help=(
            "Which genuine dataset to evaluate: 1 = Kumar2015, 2 = WISDM. "
            "watchHAR is not selectable. Defaults to 1 (Kumar2015)."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=-0.45,
        help=(
            "Decision threshold on SVM decision_function. Lower values accept more samples "
            "as genuine, usually lowering FRR but raising FAR. Suggested first try: -0.45."
        ),
    )

    parser.add_argument(
        "--frr-weight",
        type=float,
        default=3.0,
        help=(
            "Weight placed on FRR in GridSearchCV scoring. Larger values push model selection "
            "toward lower FRR. Suggested first try: 3.0."
        ),
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help="Parallel jobs for GridSearchCV. Defaults to SLURM_CPUS_PER_TASK, IMUGAIT_N_JOBS, or local CPU count - 1.",
    )

    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of CV folds for GridSearchCV. Use 3 for faster testing; default keeps your original SVM setting of 5.",
    )

    parser.add_argument(
        "--debug-threshold-sweep",
        action="store_true",
        help="Print FAR/FRR/HTER for several thresholds on the zero-effort test split. Use only for small/debug runs.",
    )

    return parser.parse_args()


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


def predict_with_threshold(model, x_data: np.ndarray, threshold: float) -> np.ndarray:
    """Predict using SVM decision_function instead of model.predict().

    sklearn SVC.predict() uses an implicit threshold around 0.0. Using
    decision_function manually lets us move the threshold to control the
    FRR/FAR tradeoff:
        lower threshold  -> lower FRR, higher FAR
        higher threshold -> lower FAR, higher FRR
    """
    scores = model.decision_function(x_data)
    return np.where(scores >= threshold, 1, -1)


def evaluate_with_threshold(
    model,
    x_data: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    y_pred = predict_with_threshold(model, x_data, threshold)
    return compute_far_frr_hter(y_true, y_pred)


def weighted_far_frr_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    frr_weight: float,
) -> float:
    far, frr, _ = compute_far_frr_hter(y_true, y_pred)
    return (far + frr_weight * frr) / (1.0 + frr_weight)


def make_weighted_threshold_scorer(threshold: float, frr_weight: float):
    """Create a scorer for GridSearchCV.

    GridSearchCV maximizes the returned score. Because we want to minimize
    weighted FAR/FRR loss, this scorer returns negative loss.
    """

    def scorer(estimator, x_data, y_true):
        y_pred = predict_with_threshold(estimator, x_data, threshold)
        loss = weighted_far_frr_loss(y_true, y_pred, frr_weight)
        return -loss

    return scorer


def print_threshold_sweep(
    model,
    user: str,
    curr_comb,
    genuine_test: np.ndarray,
    impostor_test: np.ndarray,
) -> None:
    actual_labels = np.concatenate(
        (
            np.ones(len(genuine_test), dtype=int),
            -np.ones(len(impostor_test), dtype=int),
        )
    )

    print(f"#__Threshold sweep for {user} | {curr_comb}", flush=True)

    for threshold in THRESHOLD_SWEEP_VALUES:
        pred_labels = np.concatenate(
            (
                predict_with_threshold(model, genuine_test, threshold),
                predict_with_threshold(model, impostor_test, threshold),
            )
        )
        far, frr, hter = compute_far_frr_hter(actual_labels, pred_labels)
        print(
            f"   threshold:{threshold:>5.2f} | "
            f"FAR:{far:.3f} | FRR:{frr:.3f} | HTER:{hter:.3f}",
            flush=True,
        )


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


def resolve_dataset_choice(choice: int, available_datasets: list[str]) -> str:
    """Map a --dataset number to an actual dataset folder name.

    The numeric choice is mapped to a keyword via DATASET_CHOICES, which is then
    matched case-insensitively against the folder names found on disk (so that
    e.g. choice 2 -> "wisdm" resolves to the real folder "wisdm_25Hz").
    """
    keyword = DATASET_CHOICES.get(choice)

    if keyword is None:
        raise ValueError(
            f"Unknown dataset choice: {choice}. Valid choices: {sorted(DATASET_CHOICES.keys())}"
        )

    for name in available_datasets:
        if keyword.lower() in name.lower():
            return name

    raise FileNotFoundError(
        f"No genuine dataset folder matching '{keyword}' (choice {choice}) found. "
        f"Available datasets: {available_datasets}"
    )


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


def make_svm_grid() -> tuple[svm.SVC, dict]:
    estimator = svm.SVC(
        random_state=getattr(DefaultParam, "RANDOM_SEED", 21),
        tol=1e-5,
    )

    param_grid = {
        "C": [0.2, 0.4, 0.8, 1.6, 3.2],
        "gamma": ["scale"],
        "kernel": ["linear", "rbf"],
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
        os.path.join(result_location, "svm_dictionary_worst_by_group.csv"),
        index=False,
    )

    idx_overall = dict_df.groupby(["user", "sensors"])["far"].idxmax()
    worst_overall = dict_df.loc[idx_overall].sort_values(["user", "sensors"])
    worst_overall.to_csv(
        os.path.join(result_location, "svm_dictionary_worst_overall.csv"),
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
        os.path.join(result_location, "svm_dictionary_group_summary.csv"),
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
        os.path.join(result_location, "svm_dictionary_trial_summary.csv"),
        index=False,
    )

    print("Saved summary CSV files:")
    print(" - svm_dictionary_worst_by_group.csv")
    print(" - svm_dictionary_worst_overall.csv")
    print(" - svm_dictionary_group_summary.csv")
    print(" - svm_dictionary_trial_summary.csv")


def evaluate_dataset(
    dataset_name: str,
    dict_attacker_metas: list[dict],
    threshold: float,
    frr_weight: float,
    n_jobs: int,
    cv: int,
    debug_threshold_sweep: bool,
) -> None:
    configure_genuine_dataset(dataset_name)

    gen_user_list = list_genuine_users()
    ze_imp_user_list = list_genuine_users()

    if not gen_user_list:
        raise ValueError(f"No genuine user feature folders found in dataset: {dataset_name}")

    sensor_configurations = Utilities.getallcombinations(DefaultParam.SENSOR_LIST)
    list_num_features = FEATURE_COUNTS_TO_TEST
    fsselect_method = Supervised.fs_using_MI
    result_location = build_results_dir(dataset_name)

    print(f"#__Running svm for sensor configurations: {sensor_configurations}")
    print(f"#__Feature counts: {list_num_features}")
    print(f"#__Decision threshold: {threshold}")
    print(f"#__FRR weight in scoring: {frr_weight}")
    print(f"#__CV folds: {cv}")

    if DefaultParam.CLASS_BALANCING:
        print("#__Applying SMOTE for CLASS BALANCING")

    if DefaultParam.FEATURE_DOMAIN_LIST:
        print(f"Using features from: {DefaultParam.FEATURE_DOMAIN_LIST}")

    print(f"Using genuine features from: {DefaultParam.FEATUREFILE_PATH}")
    print(f"#__Total users to evaluate: {len(gen_user_list)}")
    print(f"#__Total dictionary attackers: {len(dict_attacker_metas)}")
    print(f"Genuine user list: {gen_user_list}")

    result_rows: list[dict] = []

    scorer = make_weighted_threshold_scorer(
        threshold=threshold,
        frr_weight=frr_weight,
    )

    for user in gen_user_list:
        user_rows: list[dict] = []

        for curr_comb in sensor_configurations:
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

                svm_estimator, param_grid = make_svm_grid()

                search = GridSearchCV(
                    estimator=svm_estimator,
                    param_grid=param_grid,
                    cv=cv,
                    scoring=scorer,
                    n_jobs=n_jobs,
                    verbose=0,
                )

                search.fit(training_data, training_labels)
                best_params = search.best_params_

                model = svm.SVC(
                    C=best_params["C"],
                    gamma=best_params["gamma"],
                    kernel=best_params["kernel"],
                    class_weight=best_params["class_weight"],
                    random_state=getattr(DefaultParam, "RANDOM_SEED", 21),
                    tol=1e-5,
                )

                model.fit(training_data, training_labels)

                train_eval_data = np.vstack(
                    (
                        comb_selected_sens_gtr,
                        comb_selected_sens_itr,
                    )
                )

                train_eval_labels = np.concatenate(
                    (
                        Prep3.get_labels(comb_selected_sens_gtr, 1),
                        Prep3.get_labels(comb_selected_sens_itr, -1),
                    )
                )

                tr_far, tr_frr, tr_hter = evaluate_with_threshold(
                    model,
                    train_eval_data,
                    train_eval_labels,
                    threshold,
                )

                zero_eval_data = np.vstack(
                    (
                        comb_selected_sens_gts,
                        comb_selected_sens_its,
                    )
                )

                zero_eval_labels = np.concatenate(
                    (
                        Prep3.get_labels(comb_selected_sens_gts, 1),
                        Prep3.get_labels(comb_selected_sens_its, -1),
                    )
                )

                tsz_far, tsz_frr, tsz_hter = evaluate_with_threshold(
                    model,
                    zero_eval_data,
                    zero_eval_labels,
                    threshold,
                )

                current_feature_result = {
                    "num_features": num_features,
                    "model": model,
                    "best_params": best_params,
                    "grid_score": search.best_score_,
                    "tr_far": tr_far,
                    "tr_frr": tr_frr,
                    "tr_hter": tr_hter,
                    "tsz_far": tsz_far,
                    "tsz_frr": tsz_frr,
                    "tsz_hter": tsz_hter,
                    "comb_selected_sens_gts": comb_selected_sens_gts,
                    "comb_selected_sens_its": comb_selected_sens_its,
                    "normalizers": normalizers,
                    "feature_selectors": feature_selectors,
                }

                # Important:
                # Select feature count by CV score only.
                # Do not select it using test FAR/FRR/HTER.
                if best_feature_result is None:
                    best_feature_result = current_feature_result
                elif current_feature_result["grid_score"] > best_feature_result["grid_score"]:
                    best_feature_result = current_feature_result

            if best_feature_result is None:
                continue

            num_features = best_feature_result["num_features"]
            model = best_feature_result["model"]
            best_params = best_feature_result["best_params"]

            tr_far = best_feature_result["tr_far"]
            tr_frr = best_feature_result["tr_frr"]
            tr_hter = best_feature_result["tr_hter"]

            tsz_far = best_feature_result["tsz_far"]
            tsz_frr = best_feature_result["tsz_frr"]
            tsz_hter = best_feature_result["tsz_hter"]

            comb_selected_sens_gts = best_feature_result["comb_selected_sens_gts"]
            comb_selected_sens_its = best_feature_result["comb_selected_sens_its"]
            normalizers = best_feature_result["normalizers"]
            feature_selectors = best_feature_result["feature_selectors"]

            if debug_threshold_sweep:
                print_threshold_sweep(
                    model,
                    user,
                    curr_comb,
                    comb_selected_sens_gts,
                    comb_selected_sens_its,
                )

            zero_row = {
                "user": user,
                "attacktype": "zero-effort",
                "attacker": "rest",
                "attacker_user": None,
                "trial_id": None,
                "trial_code": None,
                "trial_label": None,
                "trial_group": None,
                "sensors": curr_comb,
                "nfeatures": num_features,
                "threshold": threshold,
                "frr_weight": frr_weight,
                "grid_score": best_feature_result["grid_score"],
                "cval": best_params["C"],
                "gamma": best_params["gamma"],
                "kernel": best_params["kernel"],
                "class_weight": best_params["class_weight"],
                "far": tsz_far,
                "frr": tsz_frr,
                "hter": tsz_hter,
                "ref_far": tr_far,
                "ref_frr": tr_frr,
                "refhter": tr_hter,
            }

            user_rows.append(zero_row)
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
                    feature_selectors,
                    normalizers,
                )

                dict_eval_data = np.vstack(
                    (
                        comb_selected_sens_gts,
                        dict_attack_data,
                    )
                )

                dict_eval_labels = np.concatenate(
                    (
                        Prep3.get_labels(comb_selected_sens_gts, 1),
                        Prep3.get_labels(dict_attack_data, -1),
                    )
                )

                tsd_far, tsd_frr, tsd_hter = evaluate_with_threshold(
                    model,
                    dict_eval_data,
                    dict_eval_labels,
                    threshold,
                )

                if abs(tsd_frr - tsz_frr) > 1e-12:
                    raise ValueError(
                        "False reject rates should be the same between zero-effort and dict-effort "
                        "because the same genuine test set is used."
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
                    "sensors": curr_comb,
                    "nfeatures": num_features,
                    "threshold": threshold,
                    "frr_weight": frr_weight,
                    "grid_score": best_feature_result["grid_score"],
                    "cval": best_params["C"],
                    "gamma": best_params["gamma"],
                    "kernel": best_params["kernel"],
                    "class_weight": best_params["class_weight"],
                    "far": tsd_far,
                    "frr": tsd_frr,
                    "hter": tsd_hter,
                    "ref_far": tsz_far,
                    "ref_frr": tsz_frr,
                    "refhter": tsz_hter,
                }

                user_rows.append(dict_row)
                result_rows.append(dict_row)

            print("\n" + "=" * 100)
            print(f"USER: {user} | SENSORS: {curr_comb} | SELECTED FEATURES: {num_features}")
            print(
                f"MODEL: threshold={threshold:.2f} | frr_weight={frr_weight:.2f} | "
                f"grid_score={best_feature_result['grid_score']:.4f}"
            )
            print(
                f"PARAMS: C={best_params['C']} | gamma={best_params['gamma']} | "
                f"kernel={best_params['kernel']} | class_weight={best_params['class_weight']}"
            )
            print(
                f"TRAIN TESTING-LIKE CHECK -> "
                f"FAR: {tr_far:.4f} | FRR: {tr_frr:.4f} | HTER: {tr_hter:.4f}"
            )
            print(
                f"ZERO-EFFORT TEST       -> "
                f"FAR: {tsz_far:.4f} | FRR: {tsz_frr:.4f} | HTER: {tsz_hter:.4f}"
            )
            print(
                f"DICT-EFFORT TEST      -> "
                f"MAX FAR: {max_far_all:.4f} [{best_all}] | "
                f"WORST HTER: {worst_dict_hter_all:.4f} [{worst_dict_hter_attacker}] | "
                f"DICT FRR: {worst_dict_frr:.4f}"
            )
            print(
                f"DICT NORMAL/CHANGED   -> "
                f"NORMAL MAX FAR: {max_far_normal:.4f} [{best_normal}] | "
                f"CHANGED MAX FAR: {max_far_changed:.4f} [{best_changed}]"
            )
            print("=" * 100, flush=True)

        cuser_df = pd.DataFrame(user_rows)

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

    save_summaries(result_df, result_location)


def main() -> None:
    args = parse_args()

    # If the user supplied --n-jobs, keep loky aligned with that worker count where possible.
    if args.n_jobs and args.n_jobs > 0:
        os.environ["LOKY_MAX_CPU_COUNT"] = str(args.n_jobs)

    print(f"#__Detected os.cpu_count(): {os.cpu_count()}", flush=True)
    print(f"#__SLURM_CPUS_PER_TASK: {os.environ.get('SLURM_CPUS_PER_TASK', 'not set')}", flush=True)
    print(f"#__IMUGAIT_N_JOBS: {os.environ.get('IMUGAIT_N_JOBS', 'not set')}", flush=True)
    print(f"#__LOKY_MAX_CPU_COUNT: {os.environ.get('LOKY_MAX_CPU_COUNT', 'not set')}", flush=True)
    print(f"#__GridSearchCV parallel workers n_jobs: {args.n_jobs}", flush=True)
    print(f"#__Dataset choice: {args.dataset} ({DATASET_CHOICES.get(args.dataset)})", flush=True)
    print(f"#__Threshold: {args.threshold}", flush=True)
    print(f"#__FRR weight: {args.frr_weight}", flush=True)
    print(f"#__CV folds: {args.cv}", flush=True)

    datasets = list_available_genuine_datasets()

    if not datasets:
        raise FileNotFoundError(
            f"No genuine feature datasets found under {DefaultParam.FEATURE_PATH}"
        )

    selected_dataset = resolve_dataset_choice(args.dataset, datasets)

    dict_attacker_metas = list_dictionary_attackers()

    if not dict_attacker_metas:
        raise ValueError(
            f"No dictionary attacker folders found under {DefaultParam.DICT_FEATUREFILE_PATH}"
        )

    print(f"#__Available genuine datasets: {datasets}", flush=True)
    print(f"#__Selected dataset to evaluate: {selected_dataset}", flush=True)
    print(f"#__Dictionary attackers: {len(dict_attacker_metas)}", flush=True)

    print(f"\n#__======== Dataset: {selected_dataset} ========", flush=True)

    evaluate_dataset(
        dataset_name=selected_dataset,
        dict_attacker_metas=dict_attacker_metas,
        threshold=args.threshold,
        frr_weight=args.frr_weight,
        n_jobs=args.n_jobs,
        cv=args.cv,
        debug_threshold_sweep=args.debug_threshold_sweep,
    )


if __name__ == "__main__":
    main()