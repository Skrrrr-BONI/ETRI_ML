from sklearn.feature_selection import f_classif
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from utils import *
import warnings
warnings.filterwarnings("ignore")
seed_everything(42)

def fs_annova(X, y):
    pv_list = []
    for i in range(y.shape[1]):
        _, p = f_classif(X, y[:, i])
        pv_list.append(pd.DataFrame(p.reshape(1, -1), columns=X.columns, index=[0]))
    pv_df = pd.concat(pv_list, ignore_index=True)
    return pv_df

def fs_selection(df):
    sel_col = []
    for c in df.columns:
        for p in df[c].values:
            if p < 0.05:
                sel_col.append(c)
                break
    return sel_col

with open('./data/feat.pkl', 'rb') as f:
    data = pickle.load(f)
with open('./data/feat_test.pkl', 'rb') as f:
    X_test = pickle.load(f)

X_train, y_train, X_valid, y_valid = data['train'][0], data['train'][1], data['valid'][0], data['valid'][1]

df_fs = fs_annova(X_train.fillna(0), y_train)
sel_col = fs_selection(df_fs)
X_train, X_valid, X_test = X_train[sel_col].fillna(0), X_valid[sel_col].fillna(0), X_test[sel_col].fillna(0)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train.values)
X_valid_sc = scaler.transform(X_valid.values)
X_test_sc = scaler.transform(X_test.values)

clf = ClassifierChain(AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, min_samples_split=2, min_samples_leaf=4),
                                            n_estimators=200, learning_rate=0.1, random_state=42, algorithm='SAMME.R'), order=(2, 1, 3, 4, 5, 0, 6), chain_method='predict', random_state=42).fit(X_train_sc, y_train)
y_pred = clf.predict(X_valid_sc)

score = eval_metric(y_valid, y_pred)

y_test_pred = clf.predict(X_test_sc)

# answer = pd.read_csv('./answer_sample.csv', encoding='utf-8')
# for i, l in enumerate(['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']):
#     answer[l] = y_test_pred[:, i]
# answer.to_csv('./answer.csv', encoding='utf-8', index=False)


