from sklearn.metrics import accuracy_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.ensemble import RandomForestClassifier
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

y_train = df_train["IsMalware"]
X_train = df_train.drop("IsMalware", axis=1)


y_test = df_test["IsMalware"]
X_test = df_test.drop("IsMalware", axis=1)


############################################


def objective(space):
    # Instantiate the classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=int(space["max_depth"]),
        min_samples_leaf=int(space["min_child_weight"]),
        max_samples=space["subsample"],
    )

    # Fit the classsifier
    clf.fit(
        X_train,
        y_train,
    )
    # Predict on Cross Validation data
    pred = clf.predict(X_test)

    # Calculate our Metric - accuracy
    accuracy = accuracy_score(y_test, pred > 0.5)
    # return needs to be in this below format. We use negative of accuracy since we want to maximize it.
    return {"loss": -accuracy, "status": STATUS_OK}


space = {
    "max_depth": hp.quniform("x_max_depth", 4, 16, 1),
    "min_child_weight": hp.quniform("x_min_child", 1, 10, 1),
    "subsample": hp.uniform("x_subsample", 0.7, 1),
}

trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)
print(best)

