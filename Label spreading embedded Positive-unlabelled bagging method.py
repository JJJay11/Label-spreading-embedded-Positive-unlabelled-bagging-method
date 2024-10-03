from sklearn.semi_supervised import LabelSpreading
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold

#####################
# python3.6.5
# Label spreading embedded Positive-unlabelled bagging method
#####################


# Modeled with the following factors
save_columns = [
    "DRI",
    "TWI",
    "DGL",
    'Lithology',
    "Altitude",
    "Slope",
    "Aspect",
    "Qulv_ping",
    "Qulv_pou",
    "Undulation",
    "NDVI",
    "DRO",
    "DSE",
]

# Landslide data
ZQ_point = pd.read_csv(u".\data_1.txt",sep=",",header=0,encoding="utf-8")
data_1 = ZQ_point.iloc[:,:]
for i in data_1.columns:
    if i not in save_columns:
        del data_1[i]
x_1 = np.array(data_1)
y_1 = np.ones((len(x_1),))

# unlabelled data
MRD_point = pd.read_csv(u".\data_0.txt",sep=",",header=0,encoding="utf-8")
data_0 = MRD_point.iloc[:,:]
for i in data_0.columns:
    if i not in save_columns:
        del data_0[i]
x_0 = np.array(data_0)
y_0 = np.zeros((len(x_0),))

# All data
ALL_point = pd.read_csv(u".\ALL.txt",sep=",",header=0,encoding="utf-8")
ALL = ALL_point.iloc[:,:]
for i in ALL.columns:
    if i not in save_columns:
        del ALL[i]
ALL = np.array(ALL)

# Standard normolization
scaler = StandardScaler()
scaler.fit(ALL)
x_1d = scaler.transform(x_1)
x_0d = scaler.transform(x_0)

# Label spreading embedded Positive-unlabelled bagging method
X = x_0d
y = y_0
# Parallelized 5 times
n_splits = 5
sss = StratifiedShuffleSplit(n_splits=n_splits, train_size=0.1)
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
T = []
dict={}
result_1 = []
x_1d_proba = np.zeros((len(x_1d),))

# PU-bagging
for train_idx, test_idx in sss.split(X,y):
    print(train_idx.shape)
    # shuffle split
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    T.extend(train_idx)
    # Label spreading
    ls_model = LabelSpreading(kernel='rbf',gamma=3)
    # model training
    ls_model.fit(np.concatenate((x_1d, X_train), axis=0), np.concatenate((y_1, y_train), axis=0))
    # Labeling the OOB unlabeled point samples
    unlabeled_proba = ls_model.predict_proba(X_test)[:, 1]
    unlabeled_proba2 = ls_model.predict_proba(X_train)[:, 1]

    x_1d_proba += ls_model.predict_proba(x_1d)[:, 1]/n_splits

    # Plotting probability cumulative curves (unlabeled samples)
    sorted_unlabeled_proba0 = np.sort(unlabeled_proba)
    cumulative_freq0 = np.arange(1, len(sorted_unlabeled_proba0)+1) / len(sorted_unlabeled_proba0)
    plt.plot(sorted_unlabeled_proba0, cumulative_freq0, marker='.', linestyle='-', color='b')
    # Frequency plot
    num_bins = 20
    hist, bins = np.histogram(sorted_unlabeled_proba0, range=(0,1),bins=num_bins, density=True)
    bin_centers0 = (bins[:-1] + bins[1:]) / 2
    frequency0 = hist * np.diff(bins)  # 计算频率
    plt.bar(bin_centers0, frequency0, width=0.025, color='g', alpha=0.5, label='Frequency')   # 频率图


    # Plotting probability cumulative curves (landslide samples)
    sorted_unlabeled_proba1 = np.sort(ls_model.predict_proba(x_1d)[:, 1])
    cumulative_freq1 = np.arange(1, len(sorted_unlabeled_proba1) + 1) / len(sorted_unlabeled_proba1)
    plt.plot(sorted_unlabeled_proba1, cumulative_freq1, marker='.', linestyle='-', color='b')
    # Frequency plot
    num_bins = 20
    hist, bins = np.histogram(sorted_unlabeled_proba1, range=(0,1),bins=num_bins, density=True)
    bin_centers1 = (bins[:-1] + bins[1:]) / 2
    frequency1 = hist * np.diff(bins)  # 计算频率
    plt.bar(bin_centers1, frequency1, width=0.025, color='r', alpha=0.5, label='Frequency')  # 频率图
    plt.show()

    plt.hist(ls_model.predict_proba(x_1d)[:, 1],bins=100)
    plt.show()
    plt.hist(unlabeled_proba,bins=100)
    plt.show()

    # Aggregation of LP_proba
    for key in test_idx:
        if key not in dict:
            dict[key] = []
            dict[key].append(unlabeled_proba[list(test_idx).index(key)])
        else:
            dict[key].append(unlabeled_proba[list(test_idx).index(key)])

    # for key in train_idx:
    #     if key not in dict:
    #         dict[key] = []
    #         dict[key].append(unlabeled_proba2[list(train_idx).index(key)])
    #     else:
    #         dict[key].append(unlabeled_proba2[list(train_idx).index(key)])

    # save the threshold
    result_1.append(ls_model.predict_proba(x_1d)[:, 1].min())

# print threshold for data sampling
Threshold = np.array(result_1).mean(axis=0)
print("Sampling Threshold:",Threshold)

# Match the landslide transition probability to each sample
for key in dict.keys():
    dict[key] = np.mean(dict[key])
LP_proba = []
for i in range(len(data_0)):
    LP_proba.append(dict[i])
LP_proba = pd.DataFrame(LP_proba)
LP_proba.columns = ["LP_proba"]


## output txt
kk = pd.concat([MRD_point,LP_proba],axis=1)
kk.to_csv(r".\sample_with_LP_proba.txt",index=False,header=True)


