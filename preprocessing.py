import pandas as pd
import numpy as np
import glob
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

def val_test_loader(base_path, mode='val'):
    df_hr = pd.read_parquet(base_path + 'ch2024_val__w_heart_rate.parquet.gzip')
    df_hr['date'] = df_hr['timestamp'].dt.strftime('%Y-%m-%d')
    subjs = np.unique(df_hr['subject_id'].values)
    if mode == 'val':
        labels = pd.read_csv('./data/val_label.csv', encoding='utf-8')
    label_col = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
    X, y = [], []
    for s in subjs:
        s_df = df_hr.loc[df_hr['subject_id'] == s]
        dates = np.unique(s_df['date'].values)
        for d in dates:
            d_df = s_df.loc[(s_df['date'] == d) & (s_df['heart_rate'] != 0)]
            if mode == 'val':
                label = labels.loc[(labels['date'] == d) & (labels['subject_id'] == s)][label_col].values
                if label.shape[0] == 0:
                    print('Not Match Datetime between Data and Label')
                    continue
                X.append(d_df['heart_rate'].values)
                y.append(label[0])
            elif mode == 'test':
                X.append(d_df['heart_rate'].values)
    if mode == 'val':
        with open('./valid.pkl', 'wb') as f:
            pickle.dump({'X': X, 'y': y}, f)
    elif mode == 'test':
        with open('./test.pkl', 'wb') as f:
            pickle.dump({'X': X, 'y': []}, f)


train_users = ['user01-06', 'user07-10', 'user11-12', 'user21-25', 'user26-30']

for tu in train_users:
    base = './data/train dataset/{}'.format(tu)
    users = glob.glob(base+'/*')
    labels = pd.read_csv('./data/train_label.csv', encoding='utf-8')
    label_col = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
    X, y = [], []
    for u in users:
        print('User path:', u)
        user = u.replace(base, '')[1:] # user id 추출
        days = glob.glob(u+'/*') # day path 읽기
        for d in tqdm(days):
            dt = datetime.datetime.fromtimestamp(int(d.replace(u, '')[1:])).strftime('%Y-%m-%d') # unixtime -> datetime str
            label = labels.loc[(labels['date'] == dt) & (labels['subject_id'] == user)][label_col].values # label 파일에서 datetime으로 label 읽기
            if label.shape[0] == 0: # label csv에 date가 없으면 넘어가기
                print('Not Match Datetime between Data and Label:', dt, d.replace(u, '')[1:])
                continue
            mins = glob.glob(d+'/e4Hr/*') # day안에 1분짜리 HR csv 데이터 path 읽기
            if len(mins) == 0: # 해당 day에 HR 파일이 없으면 넘어가기
                print('Not Found HR Data:', dt)
                continue
            cnt = 0
            df_list = []
            shape_list = []
            X_epoch = []
            for m in mins:
                data = pd.read_csv(m, encoding='utf-8')['hr'] # hr 데이터 읽기
                if data.shape[0] == 0: # 데이터 없으면 넘어가기
                    print("Data is not complete:", data.shape)
                    continue
                X_epoch.append(np.mean(data.values)) # validation set sampling rate 맞추기 위해 1분 데이터 average, validation에서는 average 안해도됨
            if len(X_epoch) == 0: # 읽힌 데이터 없으면 넘어가기
                continue
            X.append(np.array(X_epoch))
            y.append(label[0])
    with open('./train_{}.pkl'.format(tu.replace('-', '_')), 'wb') as f:
        pickle.dump({'X': X, 'y': y}, f)

val_test_loader('./data/val dataset/', 'val')
val_test_loader('./data/test dataset/', 'test')

