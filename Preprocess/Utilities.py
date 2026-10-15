import os
from itertools import chain, combinations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def getallcombinations(sensor_list):
    # takes a list of sensors and returns a list of tuples, tuples contains the combinations of sensors
    # generating and returning the combination of all sensors exclusing those who has no sensors
    # filter for removing the empty tuple from the list
    combs = [(item) for item in chain.from_iterable(combinations(sensor_list, r) for r in range(len(sensor_list) + 1)) if item!=()]
    return combs

def getallcombinations_sortened(combs):
# returns a list of combs with shortened names
    ls = []
    for comb in combs:
        name = ""
        for sensor in comb:
            if "LAcc" in sensor:
                name = name+"a+"
            elif "Gyr" in sensor:
                name = name + "g+"
            elif "Mag" in sensor:
                name = name + "m+"
            elif "RVec" in sensor:
                name = name+"r+"
            else:
                raise ValueError("sensor unknown")
        ls.append(name[:-1])
    return ls

def getCombinations(number):
    str = format(number, '04b')
    combination = [int(str[0]),int(str[1]),int(str[2]),int(str[3])]
    return combination

def getCombinationsOnlySig(number):
    str = format(number, '04b')
    combination = [int(str[0]),int(str[1]),int(str[2]),int(str[3])]
    return combination

def get_sampling_rate(raw_data):
    num_samples = raw_data.shape[0]
    # print('num_samples',str(num_samples))
    timestamps = raw_data['timestamps']
    num_unique_stamps = np.unique(timestamps).shape[0]
    # print('num_unique_stamps',str(num_unique_stamps))
    return int(num_samples / num_unique_stamps)


def moving_average(Signal, window):
    ModifiedSignal = np.zeros(len(Signal))
    i = 0
    while (i <= (len(Signal) - window)):
        ModifiedSignal[i] = np.mean(Signal[i:i + window])
        i += 1
    j = i
    while (j < (i + window - 1)):
        ModifiedSignal[j] = Signal[j]
        j += 1
    return ModifiedSignal


def scoringHTER(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    far = fp / (fp + tn)
    frr = fn / (fn + tp)
    hter = (far + frr) / 2
    if hter < 0:
        raise ValueError('ERROR: HTER CANT BE NEATIVE')
    return hter

def scoringFAR(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    far = fp / (fp + tn)
    frr = fn / (fn + tp)
    hter = (far + frr) / 2
    if hter < 0:
        raise ValueError('ERROR: HTER CANT BE NEATIVE')
    return far

def scoringFRR(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    far = fp / (fp + tn)
    frr = fn / (fn + tp)
    hter = (far + frr) / 2
    if hter < 0:
        raise ValueError('ERROR: HTER CANT BE NEATIVE')
    return frr


def get_best_params_max_features(Training_stats, reverse_order):
    # Assuming that the last column is training score and the first number of featues
    # We will select the params as the best which uses least number of features and give the best accuracy
    # Ocam's razor
    sorted_stats = sorted(Training_stats, key=lambda x: x[-1], reverse=reverse_order)
    top_stats = []
    max_score = sorted_stats[0][
        -1]  # Top stats -- highest or lowest depending on the sorted_stats is True or False resp
    for row in sorted_stats:
        if abs(max_score - row[-1]) <= 0.01:
            top_stats.append(row)

    max = top_stats[0][0]  # minimum number of features, first column is num_features, assume the first is min
    top_score_max_feature = top_stats[0]  ## Assume the first is min
    for row in top_stats:
        if max < row[0]:
            max = row[0]
            top_score_max_feature = row

    return top_score_max_feature


def get_best_params_min_features(Training_stats, reverse_order):
    # Assuming that the last column is training score and the first number of featues
    # We will select the params as the best which uses least number of features and give the best accuracy
    # Ocam's razor
    sorted_stats = sorted(Training_stats, key=lambda x: x[-1], reverse=reverse_order)
    top_stats = []
    max_score = sorted_stats[0][
        -1]  # Top stats -- highest or lowest depending on the sorted_stats is True or False resp
    for row in sorted_stats:
        if abs(max_score - row[-1]) <= 0.01:
            top_stats.append(row)

    min = top_stats[0][0]  # minimum number of features, first column is num_features, assume the first is min
    top_score_min_feature = top_stats[0]  ## Assume the first is min
    for row in top_stats:
        if min > row[0]:
            min = row[0]
            top_score_min_feature = row

    return top_score_min_feature


def get_best_params_median_features(Training_stats, reverse_order):
    # Assuming that the last column is training score and the first number of featues
    # We will select the params as the best which uses least number of features and give the best accuracy
    # Ocam's razor
    sorted_stats = sorted(Training_stats, key=lambda x: x[-1], reverse=reverse_order)
    top_stats = []
    max_score = sorted_stats[0][-1]
    # Top stats -- highest or lowest depending on the sorted_stats is True or False resp
    for row in sorted_stats:
        if abs(max_score - row[-1]) <= 0.01:  ## Checking within 3% fo accuracy distance and taking their median
            top_stats.append(row)

    # sorting the top stats based on the number of features x[0] and then returning the median
    top_sorted_stats = sorted(top_stats, key=lambda x: x[0],
                              reverse=reverse_order)  # reverse_order doe snot matter cause we are finding the mid element

    num_rows = len(top_sorted_stats)
    # print(top_sorted_stats)
    # print('check median', top_sorted_stats[int(num_rows/2)])

    return top_sorted_stats[int(num_rows / 2)]
