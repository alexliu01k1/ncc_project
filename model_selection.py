from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
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


##########################################
'''
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
'''
##########################################
y_train = df_train["IsMalware"]
X_train = df_train.drop("IsMalware", axis=1)

y_test = df_test["IsMalware"]
X_test = df_test.drop("IsMalware", axis=1)

############################################
# Set the probablity threhold for performance evaluation here, 0.5 as default
thres = 0.5

###########################
# Decision Tree
DT_clf = DecisionTreeClassifier(
    max_depth=12,
    min_samples_leaf=9,
)

# Fit the classsifier
DT_clf.fit(X_train, y_train)

# Peformance metrics based on predicted prob. and threhold
DT_pred = DT_clf.predict_proba(X_test)[:,1]
DT_accuracy = metrics.accuracy_score(y_test, DT_pred > thres)
print("Decision Tree accuracy is: ", DT_accuracy)

DT_recall = metrics.recall_score(y_test, DT_pred > thres)
print("Decision Tree recall score is: ", DT_recall)

DT_f1s = metrics.f1_score(y_test, DT_pred > thres)
print("Decision Tree f1 score is: ", DT_f1s)

DT_auc = metrics.roc_auc_score(y_test, DT_pred)
print("Decision Tree AUC score is: ", DT_auc)

fpr1, tpr1, thres1 = metrics.roc_curve(y_test, DT_pred, pos_label=1)
plt.plot(fpr1, tpr1, label='Decision Tree')
###########################
# SVM
SVM_clf = SGDClassifier(
    alpha=0.121
    )
SVM_clf.fit(X_train,y_train)

SVM_pred = SVM_clf.decision_function(X_test)
SVM_accuracy = metrics.accuracy_score(y_test, SVM_pred > thres)
print("SVM accuracy is: ", SVM_accuracy)

SVM_recall = metrics.recall_score(y_test, SVM_pred > thres)
print("SVM recall score is: ", SVM_recall)

SVM_f1s = metrics.f1_score(y_test, SVM_pred > thres)
print("SVM f1 score is: ", SVM_f1s)

SVM_auc = metrics.roc_auc_score(y_test, SVM_pred)
print("SVM AUC score is: ", SVM_auc)

fpr2, tpr2, thres2 = metrics.roc_curve(y_test, SVM_pred, pos_label=1)
plt.plot(fpr2, tpr2, label='SVM')
###########################
# Random Forest

RF_clf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=15,
        min_samples_leaf=2,
        max_samples=0.991
    )
RF_clf.fit(X_train,y_train)

RF_pred = RF_clf.predict_proba(X_test)[:,1]
RF_accuracy = metrics.accuracy_score(y_test, RF_pred > thres)
print("RF accuracy is: ", RF_accuracy)

RF_recall = metrics.recall_score(y_test, RF_pred > thres)
print("RF recall score is: ", RF_recall)

RF_f1s = metrics.f1_score(y_test, RF_pred > thres)
print("RF f1 score is: ", RF_f1s)

RF_auc = metrics.roc_auc_score(y_test, RF_pred)
print("RF AUC score is: ", RF_auc)

fpr3, tpr3, thres3 = metrics.roc_curve(y_test, RF_pred, pos_label=1)
plt.plot(fpr3, tpr3, label='Random Forest')
###########################
# XGBoost
XGB_clf = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.3,
        max_depth=13,
        min_child_weight=4,
        subsample=0.765,
        reg_lambda=0.540
    )
eval_set = [(X_train, y_train), (X_test, y_test)]
XGB_clf.fit(X_train,y_train,
        eval_set=eval_set,
        eval_metric="rmse",
        early_stopping_rounds=10,
        verbose=True,)

XGB_pred = XGB_clf.predict_proba(X_test)[:,1]
XGB_accuracy = metrics.accuracy_score(y_test, XGB_pred > thres)
print("XGB accuracy is: ", XGB_accuracy)

XGB_recall = metrics.recall_score(y_test, XGB_pred > thres)
print("XGB recall score is: ", XGB_recall)

XGB_f1s = metrics.f1_score(y_test, XGB_pred > thres)
print("XGB f1 score is: ", XGB_f1s)

XGB_auc = metrics.roc_auc_score(y_test, XGB_pred)
print("XGB AUC score is: ", XGB_auc)

fpr4, tpr4, thres4 = metrics.roc_curve(y_test, XGB_pred, pos_label=1)
plt.plot(fpr4, tpr4, label='XGBoost')
###########################
# lightGBM
LGB_clf = lgb.LGBMClassifier(
        n_estimators=1000,
        max_depth=8,
        min_child_weight=3,
        subsample=0.812,
        reg_lambda=0.412
    )
eval_set = [(X_train, y_train), (X_test, y_test)]
LGB_clf.fit(X_train,y_train,
        eval_set=eval_set,
        eval_metric= "logloss",
        early_stopping_rounds=20,
        verbose=True)

LGB_pred = LGB_clf.predict_proba(X_test)[:,1]
LGB_accuracy = metrics.accuracy_score(y_test, LGB_pred > thres)
print("LGB accuracy is: ", LGB_accuracy)

LGB_recall = metrics.recall_score(y_test, LGB_pred > thres)
print("LGB recall score is: ", LGB_recall)

LGB_f1s = metrics.f1_score(y_test, LGB_pred > thres)
print("LGB f1 score is: ", LGB_f1s)

LGB_auc = metrics.roc_auc_score(y_test, LGB_pred)
print("LGB AUC score is: ", LGB_auc)

fpr5, tpr5, thres5 = metrics.roc_curve(y_test, LGB_pred, pos_label=1)
plt.plot(fpr5, tpr5, label='LightGBM')
plt.legend(shadow=True)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves of the machine learning models')