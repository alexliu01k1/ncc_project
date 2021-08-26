# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 08:02:01 2021

@author: Alex_
"""
import pandas as pd

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

'''
df_train = df_train.drop('Unnamed: 0', axis=1)

df_train.to_csv('train.csv', index=False)
'''

df_train_early = df_train[df_train['TimeDateStamp'] <= '2000']
df_train_early.to_csv('early_train.csv', index=False)

df_train_new = df_train[df_train['TimeDateStamp'] > '2000']
df_train_new.to_csv('new_train.csv', index=False)

df_test_early = df_test[df_test['TimeDateStamp'] <= 2000]
df_test_early.to_csv('early_test.csv', index=False)

df_test_new = df_test[df_test['TimeDateStamp'] > 2000]
df_test_new.to_csv('new_test.csv', index=False)
