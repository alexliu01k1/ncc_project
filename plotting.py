from sklearn import metrics
import matplotlib.pyplot as plt
import lightgbm as lgb

def get_metrics(pred, y_test, thres=0.5):
    # Calculate our Metric - accuracy
    accuracy = metrics.accuracy_score(y_test, pred > thres)
    print("accuracy: ", accuracy)
    # Precision and recall
    recall = metrics.recall_score(y_test, pred > thres)
    print("recall score is: ", recall)
    # F1-score
    f1s = metrics.f1_score(y_test, pred > thres)
    print("f1 score is: ", f1s)
    # auc score
    auc = metrics.roc_auc_score(y_test, pred)
    print("AUC score is: ", auc)
    # Print the confusion matrix
    cm = metrics.confusion_matrix(y_test, pred > thres)
    print(cm)

def roc_auc_check(clf, X_test, y_test, pred):
    # Get the ROC curve and AUC score
    auc = metrics.roc_auc_score(y_test, pred)
    print("AUC score is: ", auc)
    metrics.plot_roc_curve(clf, X_test, y_test) 

def plot_tree(clf):
    # plot tree
    booster = lgb._booster(clf)
    lgb.plot_tree(booster)