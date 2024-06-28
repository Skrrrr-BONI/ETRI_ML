import pandas as pd
import numpy as np
import pickle
import scipy
import random
import os
from sklearn.metrics import f1_score

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def eval_metric(gt, pred):
    label = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
    wei_f1 = 0
    for i in range(7):
        if i != 2:
            f1 = f1_score(gt[:, i], pred[:, i], average='macro')
            wei_f1 += f1 * 1.5
            print("{}'s f1-score:".format(label[i]), f1)
        else:
            f1 = f1_score(gt[:, i], pred[:, i], average='macro')
            wei_f1 += f1 * 1.0
            print("{}'s f1-score:".format(label[i]), f1)
    print('Weighted F1-score =', wei_f1)
    return wei_f1