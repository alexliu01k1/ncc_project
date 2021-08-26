import lightgbm as lgb
import matplotlib.pyplot as plt
from plotting import get_metrics
# Utilities for preprocessing
import pandas as pd
from preprocessing import generate_types

train_file = "trunc_train.csv"
df_train = pd.read_csv(
    train_file,
    dtype=generate_types(train_file),
    engine="python",
)

test_file = "trunc_test.csv"
df_test = pd.read_csv(test_file, dtype=generate_types(test_file), engine="python")

##########################################
# Dataframe used for evaluation, containing extra information besides input features
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

def time_cut(cut, df_train, df_test):
    df_train = df_train[eval_train['TimeDateStamp'] <= cut]
    df_test = df_test[eval_test['TimeDateStamp'] > cut]
    return df_train, df_test
    
y_train = df_train["IsMalware"]
X_train = df_train.drop("IsMalware", axis=1)

y_test = df_test["IsMalware"]
X_test = df_test.drop("IsMalware", axis=1)

############################################

clf = lgb.LGBMClassifier(
        n_estimators=2000,
        max_depth=13,
        min_child_weight=8,
        subsample=0.884271,
        reg_lambda=0.13966,
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

# Predict on Cross Validation data
pred = clf.predict(X_test)
booster = clf.booster_

# Save the trained model
save_model = False
if save_model:
    clf.booster_.save_model("pe_lgbm_total.txt")
    
get_metrics(pred, y_test, 0.9)
