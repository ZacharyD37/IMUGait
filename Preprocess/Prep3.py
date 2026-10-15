import os
import sys
#!/usr/bin/env python -W ignore::DeprecationWarning
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
# Adding the path, else import wouldnt work
# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
sys.path.insert(0, os.path.dirname(os.getcwd()))
from GlobalParameters import DefaultParam
import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def get_imp_data(filepath, feat_folder, sfile_suffix, candidate_user, valid_user_list, num_samples_imp):
    # select specified number of samples of imposter from every other user than the candidate
    possible_imp_set = [u for u in valid_user_list if u != candidate_user]
    count_imp = -1

    # We could pick the imp samples from other users
    # From any index, however, the first few feature vectors may not be great choice
    pick_start = 3
    for imp in possible_imp_set:
        tr_file = os.path.join(filepath, imp, 'Training', feat_folder, 'feat_clean_' + sfile_suffix + '.csv')
        ts_file = os.path.join(filepath, imp, 'Testing', feat_folder, 'feat_clean_' + sfile_suffix + '.csv')

        tr_data = pd.read_csv(tr_file, delimiter=',', index_col=0)
        ts_data = pd.read_csv(ts_file, delimiter=',', index_col=0)
        tr_data = tr_data.iloc[pick_start:pick_start + num_samples_imp, :]
        ts_data = ts_data.iloc[pick_start:pick_start + num_samples_imp, :]

        count_imp = count_imp + 1  # I had forgotten this
        if count_imp == 0:
            imp_data_tr = tr_data
            imp_data_ts = ts_data
        else:
            imp_data_tr = pd.concat([imp_data_tr, tr_data], ignore_index=True)
            imp_data_ts = pd.concat([imp_data_ts, ts_data], ignore_index=True)

        if imp_data_tr.isnull().any().any() or imp_data_ts.isnull().any().any():
            raise ValueError('Found nan in current imp', imp)

    return imp_data_tr, imp_data_ts


# Generating label column for the given data matrix
def get_labels(data_matrix, label):
    if data_matrix.shape[0] > 1:
        label_column = np.empty(data_matrix.shape[0])
        label_column.fill(label)
    else:
        print('Warning! user data contains only one sample')
    return label_column


def get_normalized_data_using_minmax(gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data):
    scaler = MinMaxScaler()
    # Fit only on FULL (gen and imp) training data
    training_data = np.vstack((gen_tr_data, imp_tr_data))
    scaler.fit(training_data)

    gen_tr_data = scaler.transform(gen_tr_data)
    gen_ts_data = scaler.transform(gen_ts_data)
    # apply same transformation to test.csv data

    imp_tr_data = scaler.transform(imp_tr_data)
    imp_ts_data = scaler.transform(imp_ts_data)
    # returning as numpy arrays
    return gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data, scaler


def get_normalized_data_using_standard_scaler(gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data):
    ############## Normalization or scaling ##################
    scaler = StandardScaler()
    # Fit only on FULL (gen and imp) training data
    training_data = np.vstack((gen_tr_data, imp_tr_data))
    scaler.fit(training_data)

    gen_tr_data = scaler.transform(gen_tr_data)
    gen_ts_data = scaler.transform(gen_ts_data)
    # apply same transformation to test.csv data

    imp_tr_data = scaler.transform(imp_tr_data)
    imp_ts_data = scaler.transform(imp_ts_data)

    return gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data


def get_balanced_data(gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data):
    ### IT DOES NOT MAKE SENSE TO RESAMPLE THE TESTING DATA === AND YOU MUST NOT BECAUSE THE INFORMATION YOU WILL USE
    ### WONT BE AVAILABLE AT THE TESTING TIME ie. IMPSAMPLES...! EVEN IF YOU HAVE TO DO IT YOU SHOULD DO IT WITH TRAINING IMP SAMPLES
    ### JUST RESAMPLE THE TRAINING SO THE CLASSIFICATION MODEL IS NOT BIASED

    train_X = np.vstack((gen_tr_data, imp_tr_data))
    train_Y = np.concatenate((get_labels(gen_tr_data, 1), get_labels(imp_tr_data, -1)))
    DefaultParam.NNEIGHBORS_FOR_OVERSAMPLING = min(DefaultParam.NNEIGHBORS_FOR_OVERSAMPLING, gen_tr_data.shape[
        0] - 1)  # number of feat maybe less than nn in rarest cases

    # print(f'Using {DefaultParam.NNEIGHBORS_FOR_OVERSAMPLING} NN for SMOTE')
    oversampling = SMOTE(sampling_strategy=1.0, random_state=DefaultParam.RANDOM_SEED,
                         k_neighbors=DefaultParam.NNEIGHBORS_FOR_OVERSAMPLING)
    # The fit_resample function resamples the given data and makes the classes balanced same as fit_sample --backward compatibility

    train_X_resampled, train_Y_resampled = oversampling.fit_resample(train_X, train_Y)

    # Separating genuine and impostor feature vectors, this can be done in numpy format
    training_data = np.column_stack((train_X_resampled, train_Y_resampled))

    gen_tr_data_new = training_data[training_data[:, -1] == 1]
    imp_tr_data_new = training_data[training_data[:, -1] == -1]

    gen_tr_data = gen_tr_data_new[:, :-1]
    imp_tr_data = imp_tr_data_new[:, :-1]

    return gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data


def get_data(DATA_PATH, FEATURE_FOLDER, SENSOR_PREFIX, VALID_USER_LIST, USER, NUM_SAMPLE_IMP):
    candidate_user = USER
    tr_file = os.path.join(DATA_PATH, candidate_user, 'Training', FEATURE_FOLDER,
                           'feat_clean_' + SENSOR_PREFIX + '.csv')
    ts_file = os.path.join(DATA_PATH, candidate_user, 'Testing', FEATURE_FOLDER, 'feat_clean_' + SENSOR_PREFIX + '.csv')

    # Preparing genuine train and test.csv data
    gen_tr_data = pd.read_csv(tr_file, delimiter=',', index_col=0)
    gen_ts_data = pd.read_csv(ts_file, delimiter=',', index_col=0)

    # Getting impostor data
    imp_tr_data, imp_ts_data = get_imp_data(DATA_PATH, FEATURE_FOLDER, SENSOR_PREFIX, USER, VALID_USER_LIST,
                                            NUM_SAMPLE_IMP)

    return gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data


def get_feature_matrices(curr_comb, ze_imp_user_list, user, fsselect_method, num_features):
    all_features = []
    selected_features = []
    # to save the training objects
    normalizers = []
    # balancers = [] we need not to save this as this is required only at the training time
    feature_selectors = []

    for sensor in curr_comb:
        if 'comb_gtr_feature' in locals(): # Just for precaution
            del comb_gtr_feature
            del comb_gts_feature
            del comb_itr_feature
            del comb_its_feature
        for ind, feature_domain in enumerate(DefaultParam.FEATURE_DOMAIN_LIST):
            if 'comb_gtr_feature' not in locals(): # check for the first assignment
                comb_gtr_feature, comb_gts_feature, comb_itr_feature, comb_its_feature = get_data(
                    DefaultParam.FEATUREFILE_PATH, feature_domain, sensor, ze_imp_user_list, user,
                    DefaultParam.NUM_SAMPLE_IMP)
            else:
                gen_tr, gen_ts, imp_tr, imp_ts = get_data(DefaultParam.FEATUREFILE_PATH, feature_domain, sensor,
                                                          ze_imp_user_list, user, DefaultParam.NUM_SAMPLE_IMP)
                comb_gtr_feature = pd.concat([comb_gtr_feature, gen_tr], axis=1)
                comb_gts_feature = pd.concat([comb_gts_feature, gen_ts], axis=1)
                comb_itr_feature = pd.concat([comb_itr_feature, imp_tr], axis=1)
                comb_its_feature = pd.concat([comb_its_feature, imp_ts], axis=1)
###############################################################################################################
            all_features = all_features + [str(sensor)+"_"+DefaultParam.FEATURE_DOMAIN_LIST_SHORT[ind]+"_"+feat for feat in comb_gtr_feature.columns]
###############################################################################################################
        # Performing feature selection right here for the current sensor

        # We have to normalize the data before we perform feature selection
        # The normalizer does not throw any error for NaN
        norm_gtr_fm, norm_gts_fm, norm_itr_fm, norm_its_fm, normalizer = get_normalized_data_using_minmax(comb_gtr_feature, comb_gts_feature, comb_itr_feature, comb_its_feature)
        normalizers.append(normalizer)

        # Then we have to perform class balancing using SMOTE
        if DefaultParam.CLASS_BALANCING:
            bal_norm_gtr_fm, bal_norm_gts_fm, bal_norm_itr_fm, bal_norm_its_fm = get_balanced_data(norm_gtr_fm, norm_gts_fm, norm_itr_fm, norm_its_fm)

        # Then the feature selection
        comb_selected_sens_gtr, comb_selected_sens_gts, comb_selected_sens_itr, comb_selected_sens_its, fselectobj = fsselect_method(bal_norm_gtr_fm, bal_norm_gts_fm, bal_norm_itr_fm, bal_norm_its_fm, num_features)
        selected_features.append(fselectobj.get_support(indices=False)) # Saving the mask instead of the ids
        feature_selectors.append(fselectobj)
        # The selected feature id can be obtained front he respective feature_selectors object using get_support()
        ###############################################################################################################
        if 'sens_gtr' not in locals():  # If one exists all would do
            sens_gtr = comb_selected_sens_gtr
            sens_gts = comb_selected_sens_gts
            sens_itr = comb_selected_sens_itr
            sens_its = comb_selected_sens_its
        else:
            # damage control, just in case due to the error in sampling rate calculation the num of
            # rows varies across sensors
            min_sense_gtr = min(sens_gtr.shape[0], comb_selected_sens_gtr.shape[0])
            min_sense_gts = min(sens_gts.shape[0], comb_selected_sens_gts.shape[0])
            min_sense_itr = min(sens_itr.shape[0], comb_selected_sens_itr.shape[0])
            min_sense_its = min(sens_its.shape[0], comb_selected_sens_its.shape[0])

            sens_gtr = np.hstack((sens_gtr[0:min_sense_gtr, :], comb_selected_sens_gtr[0:min_sense_gtr, :]))
            sens_gts = np.hstack((sens_gts[0:min_sense_gts, :], comb_selected_sens_gts[0:min_sense_gts, :]))
            sens_itr = np.hstack((sens_itr[0:min_sense_itr, :], comb_selected_sens_itr[0:min_sense_itr, :]))
            sens_its = np.hstack((sens_its[0:min_sense_its, :], comb_selected_sens_its[0:min_sense_its, :]))

    # print('all_features',all_features)
    return sens_gtr, sens_gts, sens_itr, sens_its, normalizers, feature_selectors


def get_feature_matrix_for_dict_attack(curr_comb, attacker, feature_selectors, normalizers):
    all_features = []
    selected_features = []
########################################################################################################################
    for senseid, sensor in enumerate(curr_comb):
        if 'comb_attack_feature' in locals(): # Just for precaution
            del comb_attack_feature
        for ind, feature_domain in enumerate(DefaultParam.FEATURE_DOMAIN_LIST):
            if 'comb_attack_feature' not in locals(): # check for the first assignment
                comb_attack_feature = pd.read_csv(os.path.join(DefaultParam.DICT_FEATUREFILE_PATH, attacker, feature_domain,
                                                           'feat_clean_' + sensor + '.csv'), delimiter=',', index_col=0)
            else:
                temp = pd.read_csv(os.path.join(DefaultParam.DICT_FEATUREFILE_PATH, attacker, feature_domain,
                                 'feat_clean_' + sensor + '.csv'), delimiter=',', index_col=0)
                comb_attack_feature = pd.concat([comb_attack_feature, temp], axis=1)

            all_features = all_features + [DefaultParam.FEATURE_DOMAIN_LIST_SHORT[ind] + feat for feat in
                                           comb_attack_feature.columns]

########################################################################################################################
        # Performing feature selection right here for the current sensor

        # We have to normalize the data before we perform feature selection
        # The normalizer does not throw any error for NaN
        comb_attack_feature = normalizers[senseid].transform(comb_attack_feature)
        # Then the feature selection
        comb_attack_feature = feature_selectors[senseid].transform(comb_attack_feature)
        selected_features.append(feature_selectors[senseid].get_support(indices=False)) # Saving the mask instead of the ids

        # The selected feature id can be obtained front he respective feature_selectors object using get_support()
        ###############################################################################################################
        if 'comb_sens_dattack_data' not in locals():  # If one exists all would do
            comb_sens_dattack_data = comb_attack_feature
        else:
            # damage control, just in case due to the error in sampling rate calculation the num of
            # rows varies across sensors
            min_sense_gtr = min(comb_sens_dattack_data.shape[0], comb_attack_feature.shape[0])
            comb_sens_dattack_data = np.hstack((comb_sens_dattack_data[0:min_sense_gtr, :], comb_attack_feature[0:min_sense_gtr, :]))
    return comb_sens_dattack_data


def get_feature_matrix_for_high_attack(curr_comb, high_attacker, attempt, feature_selectors,normalizers):
    all_features = []
    selected_features = []
########################################################################################################################
    for senseid, sensor in enumerate(curr_comb):
        if 'comb_attack_feature' in locals(): # Just for precaution
            del comb_attack_feature
        for ind, feature_domain in enumerate(DefaultParam.FEATURE_DOMAIN_LIST):
            if 'comb_attack_feature' not in locals(): # check for the first assignment
                comb_attack_feature = pd.read_csv(os.path.join(DefaultParam.HIGH_FEATUREFILE_PATH, high_attacker, attempt, feature_domain,
                                                           'feat_clean_' + sensor + '.csv'), delimiter=',', index_col=0)
                # print(f'{high_attacker}, {attempt}, {sensor}, {feature_domain}, {comb_attack_feature}')
            else:
                temp = pd.read_csv(os.path.join(DefaultParam.HIGH_FEATUREFILE_PATH, high_attacker, attempt, feature_domain,
                                                           'feat_clean_' + sensor + '.csv'), delimiter=',', index_col=0)
                comb_attack_feature = pd.concat([comb_attack_feature, temp], axis=1)

            all_features = all_features + [DefaultParam.FEATURE_DOMAIN_LIST_SHORT[ind] + feat for feat in
                                           comb_attack_feature.columns]

########################################################################################################################
        # Performing feature selection right here for the current sensor

        # We have to normalize the data before we perform feature selection
        # The normalizer does not throw any error for NaN
        comb_attack_feature = normalizers[senseid].transform(comb_attack_feature)
        # Then the feature selection
        comb_attack_feature = feature_selectors[senseid].transform(comb_attack_feature)
        selected_features.append(feature_selectors[senseid].get_support(indices=False)) # Saving the mask instead of the ids

        # The selected feature id can be obtained front he respective feature_selectors object using get_support()
        ###############################################################################################################
        if 'comb_sens_dattack_data' not in locals():  # If one exists all would do
            comb_sens_dattack_data = comb_attack_feature
        else:
            # damage control, just in case due to the error in sampling rate calculation the num of
            # rows varies across sensors
            min_sense_gtr = min(comb_sens_dattack_data.shape[0], comb_attack_feature.shape[0])
            comb_sens_dattack_data = np.hstack((comb_sens_dattack_data[0:min_sense_gtr, :], comb_attack_feature[0:min_sense_gtr, :]))
    return comb_sens_dattack_data
