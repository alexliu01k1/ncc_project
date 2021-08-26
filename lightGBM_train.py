import pickle
from sklearn import metrics
import lightgbm as lgb
import matplotlib.pyplot as plt

# Utilities for preprocessing
import pandas as pd
from preprocessing import generate_types

train_file = "train.csv"
df_train = pd.read_csv(
    train_file,
    dtype=generate_types(train_file),
    engine="python",
)
df_train.set_index(["SampleName"], inplace=True)

test_file = "test.csv"
df_test = pd.read_csv(test_file, dtype=generate_types(test_file), engine="python")
df_test.set_index(["SampleName"], inplace=True)

df_train = df_train[df_train['TimeDateStamp'] <= '2000']
df_test = df_test[df_test['TimeDateStamp'] > '2000']

y_train = df_train["IsMalware"]
X_train = df_train.drop("IsMalware", axis=1)


y_test = df_test["IsMalware"]
X_test = df_test.drop("IsMalware", axis=1)


with open("encoder.pickle", "rb") as f:
    column_trans = pickle.load(f, encoding="bytes")

X_train = column_trans.transform(X_train)
X_test = column_trans.transform(X_test)


with open("selector.pickle", "rb") as f:
    selector = pickle.load(f, encoding="bytes")

X_train = selector.transform(X_train)
X_test = selector.transform(X_test)


with open("scale.pickle", "rb") as f:
    scale_transform = pickle.load(f, encoding="bytes")

X_train = scale_transform.transform(X_train)
X_test = scale_transform.transform(X_test)
############################################


clf = clf = lgb.LGBMClassifier(
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
    early_stopping_rounds=10,
    verbose=True,
)

# Predict on Cross Validation data
pred = clf.predict(X_test)

# Threshold setting
thres = 0.5

# Calculate our Metric - accuracy
accuracy = metrics.accuracy_score(y_test, pred > thres)
print("accuracy: ", accuracy)

# Precision and recall
recall = metrics.recall_score(y_test, pred > thres)
print("recall score is: ", recall)

# F1-score
f1s = metrics.f1_score(y_test, pred > thres)
print("f1 score is: ", f1s)

# Print the confusion matrix
cm = metrics.confusion_matrix(y_test, pred > thres)
print(cm)

# ROC AUC
auc = metrics.roc_auc_score(y_test, pred)
print("AUC score is: ", auc)
'''
metrics.plot_roc_curve(clf, X_test, y_test) 
plt.savefig(fname="ROC_LGBM.png",figsize=[10,10])
'''

booster = clf.booster_
# plot tree
'''
lgb.plot_tree(booster)
plt.savefig(fname="lgb_tree.jpg", dpi=1000)
'''

# plot metrics
'''
lgb.plot_metric(clf,metric='auc')
plt.savefig(fname="accuracy_lgb.jpg", dpi=1000)
'''
# Save the trained model
# clf.booster_.save_model("pe_lgbm_2.txt")