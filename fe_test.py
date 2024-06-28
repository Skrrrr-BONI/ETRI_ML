import numpy as np
import pickle
import pandas as pd
import scipy
from tqdm import tqdm
import statsmodels.api as sm

def hjorth_parameters(data):
    activity = np.var(data)
    diff1 = np.diff(data)
    mobility = np.sqrt(np.var(diff1) / activity)
    diff2 = np.diff(diff1)
    complexity = np.sqrt(np.var(diff2) / np.var(diff1)) / mobility
    return activity, mobility, complexity


def maximum_peaks(diff):
    cnt = 0
    for i in range(len(diff) - 1):
        if (diff[i + 1] < 0) and (diff[i] > 0):
            cnt += 1
        else:
            continue
    return cnt


def minimum_peaks(diff):
    cnt = 0
    for i in range(len(diff) - 1):
        if (diff[i] < 0) and (diff[i + 1] > 0):
            cnt += 1
        else:
            continue
    return cnt


def feat_ext(X):
    df_list = []
    for x in tqdm(X):

        feat_dict = {}

        feat_dict['Mean'] = np.mean(x)
        feat_dict['Var'] = np.var(x)
        feat_dict['Std'] = np.std(x)
        feat_dict['Skew'] = scipy.stats.skew(x)
        feat_dict['Kurt'] = scipy.stats.kurtosis(x)

        feat_dict['Median'] = np.median(x)
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        feat_dict['Q1'] = q1
        feat_dict['Q3'] = q3
        feat_dict['IQR'] = q3 - q1

        max_v = np.max(x)
        min_v = np.min(x)
        feat_dict['Max'] = max_v
        feat_dict['Min'] = min_v
        feat_dict['P2P'] = np.abs(max_v - min_v)
        feat_dict['Total_energy'] = np.sum(x ** 2)
        feat_dict['RMS'] = np.sqrt(np.mean(x ** 2))
        feat_dict['Mean_abs_dev'] = np.mean(np.abs(x ** 2 - np.mean(x)))

        diff = np.diff(x)
        feat_dict['Mean_abs_diff'] = np.mean(np.abs(diff))
        feat_dict['Median_abs_diff'] = np.median(np.abs(diff))
        feat_dict['Min_abs_diff'] = np.min(np.abs(diff))
        feat_dict['Max_abs_diff'] = np.max(np.abs(diff))
        feat_dict['Mean_diff'] = np.mean(diff)
        feat_dict['Median_diff'] = np.median(diff)
        feat_dict['Min_diff'] = np.min(diff)
        feat_dict['Max_diff'] = np.max(diff)
        feat_dict['Sum_abs_diff'] = np.sum(np.abs(diff))
        feat_dict['Distance'] = np.sum(np.sqrt(1 + diff ** 2))
        feat_dict['Max_peaks'] = maximum_peaks(diff)
        feat_dict['Min_peaks'] = minimum_peaks(diff)

        feat_dict['Entropy'] = scipy.stats.entropy(scipy.stats.norm.cdf(x, 0, 1))
        feat_dict['Slope'] = np.polyfit(np.arange(len(x)), np.array(x), 1)[0]
        feat_dict['ACF'] = np.mean(sm.tsa.stattools.acf(x, nlags=int(len(x) * 0.1), fft=False))
        feat_dict['AUC'] = scipy.integrate.simpson(x, x=np.arange(len(x)))

        activity, mobility, complexity = hjorth_parameters(x)
        feat_dict['Activity'] = activity
        feat_dict['Mobility'] = mobility
        feat_dict['Complexity'] = complexity

        feat_dict['ZC'] = np.nonzero(np.diff(x > np.mean(x)))[0].size
        df_list.append(pd.DataFrame(feat_dict, index=[0]))
    return pd.concat(df_list, ignore_index=True)

with open('./data/test.pkl', 'rb') as f:
    test = pickle.load(f)
X_test = test['X']

X_test_feat = feat_ext(X_test)
with open('./data/feat_test.pkl', 'wb') as f:
    pickle.dump(X_test_feat, f)