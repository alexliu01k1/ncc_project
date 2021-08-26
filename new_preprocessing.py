# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 13:44:53 2021

@author: Alex_
"""
import pickle
from sklearn import metrics
import lightgbm as lgb
import matplotlib.pyplot as plt
import math

# Utilities for preprocessing
import pandas as pd
from preprocessing import generate_types

train_file = "train.csv"
df_train = pd.read_csv(
    train_file,
    dtype=generate_types(train_file),
    engine="python",
)

test_file = "test.csv"
df_test = pd.read_csv(test_file, dtype=generate_types(test_file), engine="python")

features = list(df_train.columns)
for feature in features:
    if type(df_train[feature][1]) == str or math.isnan(df_train[feature][1]):
        df_train = df_train.drop(feature, axis=1)
        df_test = df_test.drop(feature, axis=1)

df_train.to_csv('new_train.csv', index=False)
df_test.to_csv('new_test.csv', index=False)