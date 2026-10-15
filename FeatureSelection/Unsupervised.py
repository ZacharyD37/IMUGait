import os
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Adding the path, else import wouldnt work
# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
sys.path.insert(0, os.path.dirname(os.getcwd()))
from GlobalParameters import DefaultParam


# ___________________________________________________________________________________________________________________#
def ft_using_pca(gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data, n_components):
    pca = PCA(n_components=n_components)  # returns the min number of features that explain the PCA_PER_VAR% variance
    training_data = np.vstack((gen_tr_data, imp_tr_data))  # this is for numpy
    pca = pca.fit(training_data)
    # print('Number of selected components from PCA: ', pca.n_components_)
    gen_tr_data = pca.transform(gen_tr_data)
    gen_ts_data = pca.transform(gen_ts_data)
    imp_tr_data = pca.transform(imp_tr_data)
    imp_ts_data = pca.transform(imp_ts_data)
    # print('Printing the training and testing samples inside PCA')
    # print(gen_tr_data.shape, gen_ts_data.shape, imp_tr_data.shape, imp_ts_data.shape)
    return gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data


# ___________________________________________________________________________________________________________________#
# The most useful, explainable, and working .. suiting to the proposal
def fs_using_corr(gen_tr_data, imp_tr_data, corr_threshold, column_names):
    # training_data = gen_tr_data.append(imp_tr_data, ignore_index=True)  # this is for pandas
    training_data = np.vstack((gen_tr_data, imp_tr_data))  # this is for numpy
    # https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
    corr_matrix = np.corrcoef(training_data.T)
    # print('gen_tr_data shape', gen_tr_data.shape)
    # print('corr_matrix:',corr_matrix.shape)
    selected_feature_indices = fs_using_corr_helper2(corr_matrix, column_names, corr_threshold)
    return selected_feature_indices

def fs_using_corr_helper2(dataset, column_names, threshold):
    data_frame = pd.DataFrame(dataset, columns=column_names)
    col_corr = set()  # Set of all the names of deleted columns
    corr_matrix = data_frame.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (abs(corr_matrix.iloc[i, j]) >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
                if colname in data_frame.columns:
                    del data_frame[colname]  # deleting the column from the dataset
    selected = list(data_frame.columns)
    return get_selected_indices(column_names, selected)

def get_selected_indices(original, selected):
    indices_in_original = []
    for i, item1 in enumerate(selected):
        for j, item2 in enumerate(original):
            if item1 == item2:
                indices_in_original.append(j)
    return indices_in_original

# FOLLOWING IS A COMPLETELY FLAWED CODE - FIX IT
def fs_using_corr_helper(corr_matrix, threshold):  # tested
    indices_useless_features = set()  # indices_useless_features
    original_feature_indices = list(
        range(0, corr_matrix.shape[1]))  # number of columns is number of features, assume all are useful
    useful_feature_indices = original_feature_indices
    # print('original_feature_indices',original_feature_indices)
    for i in original_feature_indices:
        for j in range(i):
            # print(f'[{i},{j}], :{corr_matrix[i, j]}')
            if (abs(corr_matrix[i, j]) >= threshold) and (j not in indices_useless_features):
                # print(f'[{i},{j}], :{corr_matrix[i, j]}')
                indices_useless_features.add(i)
                if i in useful_feature_indices:
                    useful_feature_indices.remove(i)
                    # print(f'removing feature{i}')
    return useful_feature_indices
