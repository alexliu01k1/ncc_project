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


# Read evaluation set that contains extra information such as the time stamps.

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

#####################

# Load the pre-trained LGBM model
clf = lgb.Booster(model_file='pe_lgbm_new_2000.txt')

years = range(2015,2021)
records = []

for year in years:
    
    df_test_temp = df_test[eval_test['TimeDateStamp'] == str(year)]
    y_test = df_test_temp["IsMalware"]
    X_test = df_test_temp.drop("IsMalware", axis=1)
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
    
    records.append(recall)
    ###########################
    
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

plt.plot(years, records)