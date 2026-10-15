########################################################################################################################
# Zero-effort + dictionary-effort attack evaluation for the current 12-trial protocol.
#
# Adapted from the original kNNZDH flow to this repo's layout:
# - Genuine features: Storage/FeatureData/Genuine/{Dataset}/UserX/{Training,Testing}/...
# - Dictionary features: Storage/FeatureData/Dictionary/Ux_Ty/...
#
# Trial grouping:
# - Normal trials:  T1, T4, T7, T10
# - Changed trials: T2, T3, T5, T6, T8, T9, T11, T12
########################################################################################################################

# !/usr/bin/env python -W ignore::DeprecationWarning
def warn(*args, **kwargs):
    pass


import os
import re
import sys
import warnings


def get_parallel_worker_count() -> int:
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


N_JOBS = get_parallel_worker_count()

# Cap loky's worker pool; skips its physical-core detection
# (noisy/broken on some Windows setups).
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(N_JOBS))

# Pin BLAS to a single thread per worker so N_JOBS processes don't
# oversubscribe cores.
for thread_env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(thread_env_var, "1")

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
MODEL_NAME = "kNN"
RESULT_ROOT = os.path.join(PROJECT_ROOT, "Storage", "PerformanceResult")

# adding the project root so imports work regardless of current working directory
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
    8: "Arm low",
    9: "Arm high",
    10: "Normal",
    11: "Arm in",
    12: "Arm out",
}


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
    trial_code = f"T{trial_id}"
    trial_label = TRIAL_NAME_MAP.get(trial_id, "Unknown")
    trial_group = "normal" if trial_id in NORMAL_TRIAL_IDS else "changed"
    return {
        "attacker": attacker,
        "attacker_user": attacker_user,
        "trial_id": trial_id,
        "trial_code": trial_code,
        "trial_label": trial_label,
        "trial_group": trial_group,
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


def list_genuine_users() -> list[str]:
    all_dirs = os.listdir(DefaultParam.FEATUREFILE_PATH)
    user_names = sorted(
        [
            d
            for d in all_dirs
            if d.startswith("User") and os.path.isdir(os.path.join(DefaultParam.FEATUREFILE_PATH, d))
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
        if any(
            d.startswith("User") and os.path.isdir(os.path.join(dataset_path, d))
            for d in os.listdir(dataset_path)
        ):
            datasets.append(dataset_name)
    return sorted(datasets)


def configure_genuine_dataset(dataset_name: str) -> None:
    dataset_path = os.path.join(str(DefaultParam.FEATURE_PATH), dataset_name)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Genuine feature dataset not found: {dataset_path}")

    DefaultParam.FEATUREFILE_PATH = dataset_path


def list_dictionary_attackers() -> list[dict]:
    """Return every dictionary attacker trial Ux_Ty, sorted by user then trial."""
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


def save_summaries(result_df: pd.DataFrame, result_location: str) -> None:
    dict_df = result_df[result_df["attacktype"] == "dict-effort"].copy()
    if dict_df.empty:
        print("No dictionary rows available; skipped summary exports.")
        return

    idx = dict_df.groupby(["user", "sensors", "trial_group"])["far"].idxmax()
    worst_by_group = dict_df.loc[idx].sort_values(["user", "sensors", "trial_group"])
    worst_by_group.to_csv(os.path.join(result_location, "kNN_dictionary_worst_by_group.csv"), index=False)

    idx_overall = dict_df.groupby(["user", "sensors"])["far"].idxmax()
    worst_overall = dict_df.loc[idx_overall].sort_values(["user", "sensors"])
    worst_overall.to_csv(os.path.join(result_location, "kNN_dictionary_worst_overall.csv"), index=False)

    group_summary = (
        dict_df.groupby(["trial_group", "sensors"])
        .agg(
            n_attacks=("far", "size"),
            mean_far=("far", "mean"),
            median_far=("far", "median"),
            max_far=("far", "max"),
            mean_hter=("hter", "mean"),
            median_hter=("hter", "median"),
            max_hter=("hter", "max"),
        )
        .reset_index()
        .sort_values(["sensors", "trial_group"])
    )
    group_summary.to_csv(os.path.join(result_location, "kNN_dictionary_group_summary.csv"), index=False)

    trial_summary = (
        dict_df.groupby(["trial_id", "trial_label", "sensors"])
        .agg(
            n_attacks=("far", "size"),
            mean_far=("far", "mean"),
            median_far=("far", "median"),
            max_far=("far", "max"),
            mean_hter=("hter", "mean"),
            median_hter=("hter", "median"),
            max_hter=("hter", "max"),
        )
        .reset_index()
        .sort_values(["sensors", "trial_id"])
    )
    if "trial_id" in trial_summary.columns:
        trial_summary["trial_id"] = trial_summary["trial_id"].astype("Int64")
    trial_summary.to_csv(os.path.join(result_location, "kNN_dictionary_trial_summary.csv"), index=False)

    print("Saved summary CSV files:")
    print(" - kNN_dictionary_worst_by_group.csv")
    print(" - kNN_dictionary_worst_overall.csv")
    print(" - kNN_dictionary_group_summary.csv")
    print(" - kNN_dictionary_trial_summary.csv")


def evaluate_dataset(dataset_name: str, dict_attacker_metas: list[dict]) -> None:
    """Evaluate every genuine user of one dataset against zero-effort and dictionary attacks."""
    configure_genuine_dataset(dataset_name)

    gen_user_list = list_genuine_users()
    ze_imp_user_list = list_genuine_users()
    if not gen_user_list:
        raise ValueError(f"No genuine user feature folders found in dataset: {dataset_name}")

    sensor_configurations = Utilities.getallcombinations(DefaultParam.SENSOR_LIST)
    list_num_features = [int(item) for item in np.linspace(start=30, stop=30, num=1)]
    fsselect_method = Supervised.fs_using_MI
    result_location = build_results_dir(dataset_name)

    print(f"#__Running for sensor configurations: {sensor_configurations}")
    if DefaultParam.CLASS_BALANCING:
        print("#__Applying SMOTE for CLASS BALANCING")
    if DefaultParam.FEATURE_DOMAIN_LIST:
        print(f"Using features from: {DefaultParam.FEATURE_DOMAIN_LIST}")
    print(f"Using genuine features from: {DefaultParam.FEATUREFILE_PATH}")

    print(f"#__Total users to evaluate: {len(gen_user_list)}")
    print(f"#__Total dictionary attackers: {len(dict_attacker_metas)}")
    print(f"Genuine user list: {gen_user_list}")

    result_rows: list[dict] = []

    for user in gen_user_list:
        user_rows: list[dict] = []

        for curr_comb in sensor_configurations:
            max_far_all = -1.0
            max_far_normal = -1.0
            max_far_changed = -1.0
            best_all = "---"
            best_normal = "---"
            best_changed = "---"

            for num_features in list_num_features:
                (
                    comb_selected_sens_gtr,
                    comb_selected_sens_gts,
                    comb_selected_sens_itr,
                    comb_selected_sens_its,
                    normalizers,
                    feature_selectors,
                ) = Prep3.get_feature_matrices(
                    curr_comb, ze_imp_user_list, user, fsselect_method, num_features
                )

                training_data = np.vstack((comb_selected_sens_gtr, comb_selected_sens_itr))
                training_labels = np.concatenate(
                    (
                        Prep3.get_labels(comb_selected_sens_gtr, 1),
                        Prep3.get_labels(comb_selected_sens_itr, -1),
                    )
                )

                scorer_frr = make_scorer(Utilities.scoringFRR, greater_is_better=False)
                param_grid = {
                    "n_neighbors": [int(x) for x in range(3, 10, 2)],
                    "metric": ["manhattan", "euclidean","cosine"],
                }
                search = GridSearchCV(
                    estimator=KNeighborsClassifier(),
                    param_grid=param_grid,
                    cv=5,
                    scoring=scorer_frr,
                    n_jobs=N_JOBS,
                    verbose=1,
                )
                search.fit(training_data, training_labels)
                best_nn = search.best_params_["n_neighbors"]
                best_dist = search.best_params_["metric"]

                model = KNeighborsClassifier(n_neighbors=best_nn, metric=best_dist)
                model.fit(training_data, training_labels)

                pred_labels_tr = np.concatenate(
                    (
                        model.predict(comb_selected_sens_gtr),
                        model.predict(comb_selected_sens_itr),
                    )
                )
                actual_labels_tr = np.concatenate(
                    (
                        Prep3.get_labels(comb_selected_sens_gtr, 1),
                        Prep3.get_labels(comb_selected_sens_itr, -1),
                    )
                )
                tr_far, tr_frr, tr_hter = compute_far_frr_hter(actual_labels_tr, pred_labels_tr)

                pred_labels_zts = np.concatenate(
                    (
                        model.predict(comb_selected_sens_gts),
                        model.predict(comb_selected_sens_its),
                    )
                )
                actual_labels_zts = np.concatenate(
                    (
                        Prep3.get_labels(comb_selected_sens_gts, 1),
                        Prep3.get_labels(comb_selected_sens_its, -1),
                    )
                )
                tsz_far, tsz_frr, tsz_hter = compute_far_frr_hter(actual_labels_zts, pred_labels_zts)

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
                    "nn": best_nn,
                    "dist": best_dist,
                    "far": tsz_far,
                    "frr": tsz_frr,
                    "hter": tsz_hter,
                    "refhter": tr_hter,
                }
                user_rows.append(zero_row)
                result_rows.append(zero_row)

                for meta in dict_attacker_metas:
                    dict_attack_data = Prep3.get_feature_matrix_for_dict_attack(
                        curr_comb, meta["attacker"], feature_selectors, normalizers
                    )
                    pred_labels_dts = np.concatenate(
                        (
                            model.predict(comb_selected_sens_gts),
                            model.predict(dict_attack_data),
                        )
                    )
                    actual_labels_dts = np.concatenate(
                        (
                            Prep3.get_labels(comb_selected_sens_gts, 1),
                            Prep3.get_labels(dict_attack_data, -1),
                        )
                    )
                    tsd_far, tsd_frr, tsd_hter = compute_far_frr_hter(actual_labels_dts, pred_labels_dts)

                    if abs(tsd_frr - tsz_frr) > 1e-12:
                        raise ValueError(
                            "False reject rates should be the same between zero-effort and dict-effort."
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
                        "nn": best_nn,
                        "dist": best_dist,
                        "far": tsd_far,
                        "frr": tsd_frr,
                        "hter": tsd_hter,
                        "refhter": tsz_hter,
                    }
                    user_rows.append(dict_row)
                    result_rows.append(dict_row)

            print(
                f"{user} | {curr_comb} | tr_hter:{tr_hter:.3f} | "
                f"ze_hter:{tsz_hter:.3f} | "
                f"max_dict_far(all):{max_far_all:.3f} [{best_all}] | "
                f"max_dict_far(normal):{max_far_normal:.3f} [{best_normal}] | "
                f"max_dict_far(changed):{max_far_changed:.3f} [{best_changed}]"
            )

        cuser_df = pd.DataFrame(user_rows)
        for int_col in ("attacker_user", "trial_id"):
            if int_col in cuser_df.columns:
                cuser_df[int_col] = pd.to_numeric(cuser_df[int_col], errors="coerce").astype("Int64")
        cuser_df.to_csv(os.path.join(result_location, user + ".csv"), index=False)

    result_df = pd.DataFrame(result_rows)
    for int_col in ("attacker_user", "trial_id"):
        if int_col in result_df.columns:
            result_df[int_col] = pd.to_numeric(result_df[int_col], errors="coerce").astype("Int64")
    combined_path = os.path.join(
        result_location, os.path.basename(__file__)[:-3] + "_combined.csv"
    )
    result_df.to_csv(combined_path, index=False)
    print(f"Saved combined results to: {combined_path}")

    save_summaries(result_df, result_location)


def main() -> None:
    print(f"#__Detected os.cpu_count(): {os.cpu_count()}", flush=True)
    print(f"#__SLURM_CPUS_PER_TASK: {os.environ.get('SLURM_CPUS_PER_TASK', 'not set')}", flush=True)
    print(f"#__IMUGAIT_N_JOBS: {os.environ.get('IMUGAIT_N_JOBS', 'not set')}", flush=True)
    print(f"#__GridSearchCV parallel workers N_JOBS: {N_JOBS}", flush=True)

    datasets = list_available_genuine_datasets()
    if not datasets:
        raise FileNotFoundError(
            f"No genuine feature datasets found under {DefaultParam.FEATURE_PATH}"
        )

    dict_attacker_metas = list_dictionary_attackers()
    if not dict_attacker_metas:
        raise ValueError(
            f"No dictionary attacker folders found under {DefaultParam.DICT_FEATUREFILE_PATH}"
        )

    print(f"#__Genuine datasets to evaluate: {datasets}", flush=True)
    print(f"#__Dictionary attackers: {len(dict_attacker_metas)}", flush=True)

    for dataset_name in datasets:
        print(f"\n#__======== Dataset: {dataset_name} ========", flush=True)
        evaluate_dataset(dataset_name, dict_attacker_metas)


if __name__ == "__main__":
    main()
