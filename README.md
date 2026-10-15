# IMUGait — Dictionary Attacks on Smartwatch-Based Gait Authentication

Code for **"Evaluating Smartwatch-Based Gait Authentication Under Dictionary
Attacks"** (Zi Wang, Rajesh Kumar; IJCB 2026).

Two genuine smartwatch gait datasets (Kumar2016, WISDM) are used to enroll
per-user binary classifiers, which are then attacked with (1) zero-effort
impostors and (2) a 420-entry smartwatch gait dictionary (35 participants ×
12 controlled walking trials) collected for this work. Performance is
reported as FAR / FRR / HTER (zero-effort) and Dict-FAR / Mean-Dict-FAR /
Max-Dict-FAR (dictionary attack).

**This repository contains code only — no data.** The genuine datasets must
be downloaded from their original sources, and the dictionary dataset is not
distributed (see §2).

---

## 0. Setup

Tested with Python 3.11.

```bash
git clone https://github.com/ZacharyD37/IMUGait.git
cd IMUGait
pip install -r requirements.txt
```

- Keep the project folder named **`IMUGait`**: `DataCleaning/cleandata_wisdm_25Hz.py`
  finds the project root by searching for a folder with that name.
- Run every command below **from the project root** with plain `python`.
- Every script finds its paths relative to the project root, so you don't
  need to edit any paths. All data lives under `Storage/` (git-ignored; you
  create it locally).

---

## 1. Repository layout

| Path | Purpose |
|---|---|
| `GlobalParameters.py` | Shared config (`DefaultParam`): `Storage/` paths, 8 s window / 4 s step, histogram bins, SMOTE neighbours, random seed, feature-selection switch |
| `requirements.txt` | Python dependencies (versions the paper's results were produced with) |
| `DataCleaning/` | Raw dataset → cleaned per-user `clean_acc.txt` / `clean_gyr.txt` files |
| `Segmentation/` | Sliding-window segmentation (8 s windows, 4 s step) for genuine and dictionary data |
| `PreProcessing/` | Moving-average smoothing of the segmented windows |
| `FeatureExtraction_Kumar/` | Time- and frequency-domain feature extraction (`TimeDomain.py`, `FrequencyDomain.py` + two driver scripts) |
| `FeatureSelection/` | `Supervised.py`: mutual-information feature selection used by all ML scripts. `Unsupervised.py` (PCA) is not used in the paper and is kept for reference. |
| `Preprocess/` | Library code imported by `ML/` (not run directly): feature loading, train/test assembly, class balancing, normalization (`Prep3.py`, `Utilities.py`) |
| `ML/` | **The 5 classifiers and attack evaluation** (see §4) |
| `Storage/` | All data and results — **not included**, created locally (see below) |

> `PreProcessing/` (signal smoothing, run as a pipeline step) and
> `Preprocess/` (feature-matrix helpers imported by `ML/`) are different
> folders despite the similar names.

Data flows through `Storage/` in stages:

```
Storage/RawData           →  DataCleaning/
Storage/CleanData         →  Segmentation/
Storage/SegmentedData     →  PreProcessing/
Storage/PreprocessedData  →  FeatureExtraction_Kumar/
Storage/FeatureData       →  ML/
Storage/PerformanceResult ←  evaluation results
```

Each stage has a `Genuine/<dataset>/UserX/{Training,Testing}/` branch and a
`Dictionary/` branch. Genuine feature data ends up in
`Storage/FeatureData/Genuine/{Kumar2015, wisdm_25Hz}/UserX/{Training,Testing}/{TimeFeatures,FrequencyFeatures}/feat_clean_{acc,gyr}.csv`
and dictionary features in
`Storage/FeatureData/Dictionary/Ux_Ty/{TimeFeatures,FrequencyFeatures}/feat_clean_{acc,gyr}.csv`
(attacker user `x`, trial `y`).

---

## 2. Datasets

### 2.1 Genuine datasets (not included — download separately)

| Dataset | Folder name in this repo | Users | Rate | Source |
|---|---|---|---|---|
| **Kumar2016** | `Kumar2015` | 40 | 25 Hz | R. Kumar, V. V. Phoha, R. Raina. *Authenticating users through their arm movement patterns*, 2016. |
| **WISDM** | `wisdm_25Hz` | 51 | 20 Hz → resampled to 25 Hz | G. M. Weiss, K. Yoneda, T. Hayajneh. *Smartphone and smartwatch-based biometrics using activities of daily living*, IEEE Access, 2019. Public dataset: [WISDM Smartphone and Smartwatch Activity and Biometrics Dataset (UCI ML Repository)](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset). |

These are third-party published datasets; please cite the sources above.
Obtain them from their original sources and place the raw files as follows:

```
Storage/RawData/Genuine/Kumar2015/UserX/{Training,Testing}/
    Accelerometer_Data.txt          # comma-separated: timestamp, x, y, z
    Gyroscope_Data.txt

Storage/RawData/Genuine/wisdm-dataset/raw/watch/
    accel/data_<id>_accel_watch.txt
    gyro/data_<id>_gyro_watch.txt
```

(The WISDM path is the `raw/` folder from the unzipped UCI download.)

### 2.2 Dictionary dataset (not included)

The dictionary consists of 420 raw smartwatch gait entries (35 participants ×
12 trials) collected for this paper under an approved university IRB
protocol, using a Samsung Galaxy Watch Ultra at 25 Hz. See the paper
(§3.2.2) for full collection details. **To protect participant privacy, this
data is not distributed with the repository.**

If you have your own recordings in the same format, the pipeline expects one
CSV per trial:

```
Storage/RawData/Dictionary/UserX/Ux_Ty.csv
```

Lines starting with `#` at the top of the file are treated as metadata and
skipped. Required columns: a sensor-type column (`sensor_type`, values such
as `accelerometer` / `gyroscope`), a timestamp column (`timestamp`), and
`x`, `y`, `z`.

Trial → factor mapping:

| Factor | Normal | Modified levels |
|---|---|---|
| Speed | T1 | T2 Slow, T3 Fast |
| Step Length | T4 | T5 Short, T6 Long |
| Arm Arc | T7 | T8 Low, T9 High |
| Arm Rotation | T10 | T11 Inward, T12 Outward |

Trials 1/4/7/10 form the **normal** group, the other eight the **changed**
group; result summaries are reported per group and per trial.

---

## 3. Data preparation pipeline

Only needed when (re)building feature data from raw recordings. If
`Storage/FeatureData/` is already populated, skip to §4.

For the genuine-data scripts, `--dataset` is optional: without it, every
dataset folder found is processed.

### 3.1 Data cleaning

```bash
# Kumar2015: copies Accelerometer/Gyroscope_Data.txt to clean_acc/clean_gyr.txt,
# dropping malformed rows; keeps the dataset's own Training/Testing split.
# Optional flags: --input-root, --output-root, --keep-profile
# NOTE: unless --keep-profile is given, this deletes Profile.txt from the RAW
# input folders (Storage/RawData/Genuine/Kumar2015/...). Keep a backup copy.
python DataCleaning/cleandata_kumar.py

# WISDM: watch accel + gyro, walking rows only (activity code A), linearly
# resampled 20 Hz → 25 Hz, users renumbered User1..UserN, and split
# chronologically 70% Training / 30% Testing per user.
python DataCleaning/cleandata_wisdm_25Hz.py

# Dictionary: RawData/Dictionary/UserX/Ux_Ty.csv → CleanData/Dictionary/UserX/Ux_Ty/clean_{acc,gyr}.txt
python DataCleaning/csv_converter.py
```

### 3.2 Segmentation (8 s windows, 4 s step)

```bash
# Genuine data (sampling rate is estimated per file from the timestamps)
python Segmentation/segment_genuine.py --dataset Kumar2015
python Segmentation/segment_genuine.py --dataset wisdm_25Hz

# Dictionary data (all Ux_Ty trials, 25 Hz)
python Segmentation/segmentation_dictionary.py
```

Writes `acc.csv` / `gyr.csv` (one row per sample, with `frame_id` /
`sample_id`) plus a `segmentation_summary_*.csv` under `Storage/SegmentedData/`.

### 3.3 Preprocessing (moving-average smoothing)

Moving average with window `ceil(0.05 · fs + 1)` samples (3 samples at 25 Hz).

```bash
python PreProcessing/PreprocessSegmentedGenuine.py --dataset Kumar2015
python PreProcessing/PreprocessSegmentedGenuine.py --dataset wisdm_25Hz

python PreProcessing/PreprocessSegmentedDictionary.py
```

### 3.4 Feature extraction (time + frequency domain)

Following Kumar et al. [19], features are computed per window on each of the
x, y, z axes and the magnitude `m = √(x²+y²+z²)`:

- **Time domain (30 per channel):** 14 statistical descriptors (mean, 1st/2nd/3rd
  quartile, mean absolute change, mean absolute deviation, longest strike
  below/above mean, std, skewness, kurtosis, mean energy, mean-crossings,
  number of peaks) + a 16-bin normalized histogram.
- **Frequency domain (4 per channel):** 1st/2nd/3rd quartile and std of the
  FFT magnitude spectrum of the window.

That is 136 features per sensor (4 × 34), 272 when accelerometer and
gyroscope are fused.

```bash
# Genuine: writes Storage/FeatureData/Genuine/<dataset>/UserX/{Training,Testing}/
python FeatureExtraction_Kumar/ExtractFromGenuineDataset.py --dataset Kumar2015
python FeatureExtraction_Kumar/ExtractFromGenuineDataset.py --dataset wisdm_25Hz

# Dictionary: writes Storage/FeatureData/Dictionary/Ux_Ty/
python FeatureExtraction_Kumar/ExtractFromDictionaryEffort.py
```

---

## 4. ML performance evaluation (main part)

All evaluation scripts live in `ML/`: `kNN.py`, `SVM.py`, `LReg.py`
(logistic regression), `Rand.py` (random forest), `MLP.py` (PyTorch MLP).
For every genuine user, each script:

1. builds a **per-user binary classifier**: the user's Training windows are the
   genuine class; a fixed set of `NUM_SAMPLE_IMP` (= 5) windows from each
   other genuine user's Training data is the impostor class;
2. balances classes with SMOTE, applies min-max normalization, and selects
   features by mutual information (`FeatureSelection/Supervised.fs_using_MI`);
3. tunes hyperparameters with `GridSearchCV`;
4. evaluates each sensor configuration (`acc`, `gyr`, `acc+gyr`) and attacks it
   with
   - **zero-effort**: `NUM_SAMPLE_IMP` Testing windows from each other genuine user
   - **dict-effort**: every dictionary trial `Ux_Ty` (35 attackers × 12 trials).

**Decision thresholds.**

- `LReg`, `Rand` and `MLP` (`--threshold-mode validation`) split each user's
  original Testing data into a validation part (`--validation-ratio`) and a
  held-out final test part. They pick the probability threshold on the
  validation part only, minimising `(FAR + w·FRR) / (1 + w)` with `w = --frr-weight`. Either one
  threshold is shared per dataset+sensor or each user+sensor gets its own
  (`--threshold-scope`). FAR / FRR / HTER are reported on the held-out part.
- `SVM` thresholds the `decision_function` at a fixed `--threshold`, with FRR
  weighting (`--frr-weight`) inside the grid-search scorer.
- `kNN` uses the plain `predict` output (majority vote), so it has no threshold.

### 4.1 Exact commands used for the paper's reported results

`--dataset 1` = Kumar2015, `--dataset 2` = wisdm_25Hz. Each classifier except
kNN is run once per dataset. `kNN.py` takes no arguments; a single run
evaluates **every** dataset folder found under `Storage/FeatureData/Genuine/`.

```bash
# LReg
python ML/LReg.py --dataset 1 --threshold-mode validation --validation-ratio 0.40 --threshold-scope dataset-sensor --threshold 0.60 --threshold-min 0.45 --threshold-max 0.99 --threshold-steps 55 --frr-weight 1.0 --n-jobs -1
python ML/LReg.py --dataset 2 --threshold-mode validation --validation-ratio 0.40 --threshold-scope dataset-sensor --threshold 0.60 --threshold-min 0.45 --threshold-max 0.99 --threshold-steps 55 --frr-weight 1.0 --n-jobs -1

# Rand (random forest)
python ML/Rand.py --dataset 1 --threshold-mode validation --threshold-scope dataset-sensor --validation-ratio 0.40 --frr-weight 1.5 --n-jobs -1
python ML/Rand.py --dataset 2 --threshold-mode validation --threshold-scope dataset-sensor --validation-ratio 0.40 --frr-weight 1.5 --n-jobs -1

# kNN — no CLI flags, runs both datasets
python ML/kNN.py

# MLP
python ML/MLP.py --dataset 1 --threshold-mode validation --threshold-scope user-sensor --validation-ratio 0.40 --threshold-min 0.01 --threshold-max 0.95 --threshold-steps 95 --frr-weight 2.0
python ML/MLP.py --dataset 2 --threshold-mode validation --threshold-scope user-sensor --validation-ratio 0.40 --threshold-min 0.01 --threshold-max 0.95 --threshold-steps 95 --frr-weight 2.0

# SVM
python ML/SVM.py --dataset 1 --threshold -0.60 --frr-weight 3.0 --n-jobs -1
python ML/SVM.py --dataset 2 --threshold -0.60 --frr-weight 3.0 --n-jobs -1
```

Run `python ML/<script>.py --help` for every option and its default
(including `--cv` and `--feature-counts`).

**Parallelism.** The number of parallel workers is taken, in order, from
`--n-jobs` (where available), `SLURM_CPUS_PER_TASK`, the `IMUGAIT_N_JOBS`
environment variable, or CPU count − 1. BLAS libraries are pinned to one
thread per worker.

### 4.2 Outputs

Each run writes to `Storage/PerformanceResult/<dataset>/<model>/`, where
`<model>` is `kNN`, `LReg`, `Rand`, `MLP` or `svm`:

| File | Content |
|---|---|
| `UserX.csv` | Per-user rows for each sensor configuration: FRR, zero-effort FAR/HTER, and one dict-effort row per `Ux_Ty` trial |
| `<script>_combined.csv` | All users concatenated (`kNN_combined.csv`, `LReg_combined.csv`, `Rand_combined.csv`, `MLP_combined.csv`, `SVM_combined.csv`) |
| `<prefix>_dictionary_worst_by_group.csv` | Most successful attacker per user × sensor × trial group (normal / changed) |
| `<prefix>_dictionary_worst_overall.csv` | Most successful attacker per user × sensor overall (Max-Dict-FAR) |
| `<prefix>_dictionary_group_summary.csv` | Normal vs changed trial-group statistics |
| `<prefix>_dictionary_trial_summary.csv` | Per-trial-type statistics (T1–T12) |
| `<prefix>_threshold_calibration.csv` | Chosen thresholds and validation scores (`LReg`, `RF`, `MLP` only) |

`<prefix>` is `kNN`, `LReg`, `RF` (random forest), `MLP` or `svm`.

---

## 5. Configuration

`GlobalParameters.py` (`DefaultParam`) holds the settings shared by all
stages. The ones that affect results:

| Setting | Value | Used by |
|---|---|---|
| `WINLENGTH` / `WINSTEP` | 8 s / 4 s | feature extraction |
| `NUM_BINS_FOR_TD` | 16 | time-domain histogram |
| `NUM_SAMPLE_IMP` | 5 | impostor windows taken per other user (training and zero-effort test) |
| `CLASS_BALANCING`, `NNEIGHBORS_FOR_OVERSAMPLING` | `True`, 7 | SMOTE in `Prep3.py` |
| `NORMALIZE_FEATURES` | `True` | min-max scaling in `Prep3.py` |
| `APPLY_SUPERVISED_FSELECTION` | `True` | mutual-information feature selection |
| `RANDOM_SEED` | 21 | SMOTE, validation splits and model initialisation |
| `BAD_USERS` | `[]` | users excluded from evaluation |

The segmentation scripts set their own 8 s / 4 s window constants. Some
entries in `GlobalParameters.py` (e.g. `HIGH_FEATUREFILE_PATH`,
`HIGHEFFORT_USER_LIST`, `FFTLENGTH`) are left over from an earlier
high-effort attack setup and do not affect the results reported in the paper.

---

## 6. Citation

If you use this code, please cite:

```
Z. Wang and R. Kumar, "Evaluating Smartwatch-Based Gait Authentication Under
Dictionary Attacks," in Proc. IEEE International Joint Conference on
Biometrics (IJCB), 2026.
```

Please also cite the Kumar2016 and WISDM dataset papers listed in §2.1.
