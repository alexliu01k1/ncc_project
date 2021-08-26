# -*- coding: utf-8 -*-
import pandas as pd
from preprocessing import generate_types
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb

def plot_time_stat(train_file='train.csv', test_file='test.csv'):
    # Plot the time distribution graph of train and test set
    eval_train = pd.read_csv(
    train_file,
    dtype=generate_types(train_file),
    engine="python",
    )
    
    eval_test = pd.read_csv(
    test_file,
    dtype=generate_types(test_file),
    engine="python",
    )
    
    times = pd.concat([eval_train['TimeDateStamp'],eval_test['TimeDateStamp']])
    times = np.array(times)
    labels = pd.concat([eval_train['IsMalware'],eval_test['IsMalware']])
    labels = np.array(labels)
    
    bars = ['1970-1990','1990-2000','2000-2010','2010-2020','2020-']
    m_counts = [0,0,0,0,0]
    b_counts = [0,0,0,0,0]
    
    # Count the no. of samples of different time intervals
    # malicious and benign counted separately
    for i in range(len(labels)):
        if times[i] <= '1990':
            if labels[i] == 0:
                b_counts[0] += 1
            else:
                m_counts[0] += 1
        elif times[i] <= '2000':
            if labels[i] == 0:
                b_counts[1] += 1
            else:
                m_counts[1] += 1
        elif times[i] <= '2010':
            if labels[i] == 0:
                b_counts[2] += 1
            else:
                m_counts[2] += 1
        elif times[i] <= '2020':
            if labels[i] == 0:
                b_counts[3] += 1
            else:
                m_counts[3] += 1
        else:
            if labels[i] == 0:
                b_counts[4] += 1
            else:
                m_counts[4] += 1
        
    plt.figure()
    plt.bar(bars,m_counts,label='Malwares')
    plt.bar(bars,b_counts,label='Benignwares',bottom=m_counts)
    plt.title('Statistic of time distribution of all files')
    plt.legend(shadow=True)
    
def plot_fn_vs_time(model, X_test, y_test, eval_test):
    clf = lgb.Booster(model_file=model)
    pred = clf.predict(X_test) > 0.5
    error_check = (pred!=y_test)
    error_instance_time = eval_test[error_check]['TimeDateStamp']
    error_instance_time = error_instance_time[error_instance_time>='2000']
    error_instance_time = error_instance_time[error_instance_time<='2020']
    from collections import Counter
    c_dict = dict(Counter(error_instance_time))
    c_dict = dict(sorted(c_dict.items()))
    years = list(c_dict.keys())
    counts = list(c_dict.values())
    fig = plt.figure(figsize=(12,4))
    plt.bar(years, counts)
    plt.title('Distribution of mis-classified samples')
    plt.xlabel('year')
    plt.ylabel('no. of samples')
    