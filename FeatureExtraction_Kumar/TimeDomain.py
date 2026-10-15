import itertools
from inspect import getmembers

import numpy as np
import pandas as pd
from GlobalParameters import DefaultParam
from scipy import stats

# https://softwareengineering.stackexchange.com/questions/182093/why-store-a-function-inside-a-python-dictionary
# https://tsfresh.readthedocs.io/en/latest/_modules/tsfresh/feature_extraction/feature_calculators.html#mean_abs_change
# feat_names will consists of names of the features that has to be computed
TDFeatureDictionary = {}


#################### Time domain features############


def amean(x):
    if len(x) == 0:
        return 0
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.mean(x)


def mean_abs_change(x):
    # Returns the mean over the absolute differences between subsequent time series values which is
    return np.mean(np.abs(np.diff(x)))


def mean_absolute_deviation(x):
    # Mean absolute distance from the mean; preserve Series.mad's NaN handling.
    x = pd.Series(x)
    return (x - x.mean()).abs().mean()


def _get_length_sequences_where(x):
    # This method calculates the length of all sub-sequences where the array x is either True or 1.
    """    Examples
    #--------
    #>>> x = [0,1,0,0,1,1,1,0,0,1,0,1,1]
    #>>> _get_length_sequences_where(x)
    #>>> [1, 3, 1, 2]

    #>>> x = [0,True,0,0,True,True,True,0,0,True,0,True,True]
    #>>> _get_length_sequences_where(x)
    #>>> [1, 3, 1, 2]

    #>>> x = [0,True,0,0,1,True,1,0,0,True,0,1,True]
    #>>> _get_length_sequences_where(x)
    #>>> [1, 3, 1, 2]

    :param x: An iterable containing only 1, True, 0 and False values
    :return: A list with the length of all sub-sequences where the array is either True or False. If no ones or Trues
    contained, the list [0] is returned.
    """
    if len(x) == 0:
        return [0]
    else:
        res = [len(list(group)) for value, group in itertools.groupby(x) if value == 1]
        return res if len(res) > 0 else [0]


def longest_strike_below_mean(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.max(_get_length_sequences_where(x <= np.mean(x))) if x.size > 0 else 0


def longest_strike_above_mean(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.max(_get_length_sequences_where(x >= np.mean(x))) if x.size > 0 else 0


# def abs_gmean(x):
#     if len(x) == 0:
#         return 1
#     if not isinstance(x, (np.ndarray, pd.Series)):
#         x = np.asarray(x)
#     abs_x = [abs(k) for k in x]
#     return stats.gmean(abs_x)


def std_dev(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.std(x)


def skewness(x):
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return pd.Series.skew(x)


def kurtosis(x):
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return pd.Series.kurtosis(x)


def quantile(x, qt):
    # returns the value of x greater than qt% of the ordered values from x.
    x = pd.Series(x)
    return pd.Series.quantile(x, qt)


def fquantile(x):
    # returns the value of x greater than qt% of the ordered values from x.
    x = pd.Series(x)
    return pd.Series.quantile(x, 0.25)


def squantile(x):
    # returns the value of x greater than qt% of the ordered values from x.
    x = pd.Series(x)
    return pd.Series.quantile(x, 0.50)


def tquantile(x):
    # returns the value of x greater than qt% of the ordered values from x.
    x = pd.Series(x)
    return pd.Series.quantile(x, 0.75)


def mean_energy(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.dot(x, x)/len(x)


def number_crossing_mean(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    # https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
    positive = x > np.mean(x)
    return np.where(np.bitwise_xor(positive[1:], positive[:-1]))[0].size


# Calculates the number of peaks of at least support n in the time series x.
# A peak of support n is defined as a subsequence of x where a value occurs,
# which is bigger than its n neighbours to the left and to the right.
# Hence in the sequence  n here in our specific case depends upon the sampling rate
# so n must be set to < samplingRate*10/6  == n == samplingRate/2
def number_peaks(x):
    n = 4
    x_reduced = x[n:-n]
    res = None
    for i in range(1, n + 1):
        result_first = (x_reduced > np.roll(x, i)[n:-n])
        if res is None:
            res = result_first
        else:
            res &= result_first
        res &= (x_reduced > np.roll(x, -i)[n:-n])
    return np.sum(res)


# # Capture the speed of walking
# def cid_ce(x):
#     # This function calculator is an estimate for a time series complexity [1] (A more complex time series has more peaks,
#     #    valleys etc.). It calculates the value of
#     #    |  [1] Batista, Gustavo EAPA, et al (2014).
#     #    |  CID: an efficient complexity-invariant distance for time series.
#     #    |  Data Mining and Knowledge Discovery 28.3 (2014): 634-669.
#     # param normalize: should the time series be z-transformed?
#     # type normalize: bool
#     normalize = False
#     if not isinstance(x, (np.ndarray, pd.Series)):
#         x = np.asarray(x)
#
#     if normalize:
#         s = np.std(x)
#         if s != 0:
#             x = (x - np.mean(x)) / s
#         else:
#             return 0.0
#
#     x = np.diff(x)
#     return np.sqrt(np.dot(x, x))


def bin_counts(x):
    hist_counts, bin_edges = np.histogram(x, DefaultParam.NUM_BINS_FOR_TD)
    return list(hist_counts / sum(hist_counts))

TDFeatureDictionary['amean'] = amean
TDFeatureDictionary['fquantile'] = fquantile
TDFeatureDictionary['squantile'] = squantile
TDFeatureDictionary['tquantile'] = tquantile
TDFeatureDictionary['meanabschange'] = mean_abs_change
TDFeatureDictionary['mad'] = mean_absolute_deviation
TDFeatureDictionary['strikebelowmean'] = longest_strike_below_mean
TDFeatureDictionary['strikeabovemean'] = longest_strike_above_mean
TDFeatureDictionary['stddev'] = std_dev
TDFeatureDictionary['skewness'] = skewness
TDFeatureDictionary['kurtosis'] = kurtosis
TDFeatureDictionary['mean_energy'] = mean_energy
TDFeatureDictionary['ncmean'] = number_crossing_mean
TDFeatureDictionary['npeaks'] = number_peaks
# TDFeatureDictionary['cidce'] = cid_ce
TDFeatureDictionary['bin_counts'] = bin_counts


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


def getall_td_feature(Signal):
    # smooth signal before extracting the time series features
    feature_values = []
    feature_names = []
    for fname, function in TDFeatureDictionary.items():
        # print(fname)
        if fname == 'bin_counts':
            temp = TDFeatureDictionary[fname](Signal)
            feature_values = feature_values + temp
            for i, v in enumerate(temp):
                feature_names.append('bin_counts'+str(i))
        else:
            feature_values.append(TDFeatureDictionary[fname](Signal))
            feature_names.append(fname)

        # print(feature_values)
    return feature_names, feature_values
