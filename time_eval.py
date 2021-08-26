from sklearn import metrics
import lightgbm as lgb
import matplotlib.pyplot as plt

# Utilities for preprocessing
import pandas as pd
from preprocessing import generate_types

train_file = "new_train.csv"
df_train = pd.read_csv(
    train_file,
    dtype=generate_types(train_file),
    engine="python",
)

test_file = "new_test.csv"
df_test = pd.read_csv(test_file, dtype=generate_types(test_file), engine="python")

eval_train = pd.read_csv(
    'train.csv',
    dtype=generate_types(test_file),
    engine="python",
)

eval_test = pd.read_csv(
    'test.csv',
    dtype=generate_types(test_file),
    engine="python",
)

# Take pre-time-cut data as training set, post-time-cut data as test set.

time_cuts = range(2000,2021)
time_cuts = list(map(str,time_cuts))

recall_scores = []
accu_scores = []

y_test = df_test["IsMalware"]
X_test = df_test.drop("IsMalware", axis=1)

for cut in time_cuts:

    df_train_1 = df_train[eval_train['TimeDateStamp'] <= cut]

    y_train = df_train_1["IsMalware"]
    X_train = df_train_1.drop("IsMalware", axis=1)

    ############################################


    clf = lgb.LGBMClassifier(
        n_estimators=1000,
        max_depth=11,
        min_child_weight=9,
        subsample=0.954239,
        reg_lambda=0.33991,
    )

    eval_set = [(X_train, y_train), (X_test, y_test)]

    # Fit the classsifier
    clf.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        eval_metric="rmse",
        early_stopping_rounds=20,
        verbose=True,
    )

    pred = clf.predict(X_test)
    thres = 0.8
    # Can alter threshold and sort of performance metrics here
    recall = metrics.recall_score(y_test, pred > thres)
    accu_scores.append(recall)
    print("The ", cut, " cut recall score is: ", recall)
    del clf

plt.figure(figsize=(12,4))
plt.plot(time_cuts, accu_scores)
plt.title('Recall scores of model with different time-cuts')
plt.xlabel('time-cut')
plt.ylabel('recall score')