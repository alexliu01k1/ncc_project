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

y_train = df_train["IsMalware"]
X_train = df_train.drop("IsMalware", axis=1)

# Choose the malwares before/after 2000 as a cut
y_test = df_test["IsMalware"]
X_test = df_test.drop("IsMalware", axis=1)

#####################

clf = lgb.Booster(model_file='pe_lgbm_total.txt')

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

# det curve

fpr, fnr, thresholds = metrics.det_curve(y_test, pred)
plt.plot(thresholds, fpr)
plt.plot(thresholds, fnr)
plt.legend(['fpr','fnr'])

# Print the feature importance analysis

feature_names = df_train.columns.values.tolist()
feature_importance = clf.feature_importance(importance_type='gain')
df_importance = pd.DataFrame({
        'feature': feature_names[0:-1],
        'importance': clf.feature_importance(),
    }).sort_values(by='importance')
print(df_importance)

# Plot the histogram of feature importance of top-10 important features
t10_features = df_importance[-10:]['feature']
t10_values = df_importance[-10:]['importance']
plt.figure(figsize=(12,4))
plt.bar(t10_features, t10_values,width=0.4,color='r')
plt.title('Feature Importance of 10 most important features based on count of splits')
plt.xticks(rotation=-15)
plt.xlabel('feature')
plt.ylabel('importance')

# Select the zero/low-importance features
trunc = df_importance[df_importance['importance'] >= 10]
trunc_features = list(trunc['feature'])
trunc_features.append('IsMalware')
trunc_train = df_train[trunc_features]
trunc_test = df_test[trunc_features]
# Save the truncated dataframe for future training/testing
save_trunc = False
if save_trunc:
    trunc_train.to_csv('trunc_train.csv',index=False)
    trunc_test.to_csv('trunc_test.csv',index=False)
###########################
'''
eval_test = pd.read_csv(
    'test.csv',
    dtype=generate_types(test_file),
    engine="python",
)

errors = eval_test[((pred > 0.5) != y_test)]
fn = errors[errors['IsMalware'] == 1]
fp = errors[errors['IsMalware'] == 0]

times = ['1970-1990','1990-2000','2000-2010','2010-2020','2020-']
counts = [0,0,0,0,0]
time_records = fn['TimeDateStamp']
for record in time_records:
    if record <= '1990':
        counts[0] += 1
    elif record <= '2000':
        counts[1] += 1
    elif record <= '2010':
        counts[2] += 1
    elif record <= '2020':
        counts[3] += 1
    else:
        counts[4] += 1

plt.figure()
plt.bar(times,counts)
'''