#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 13:07:13 2019

@author: lukeholbrook
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#filename = "/Users/lukeholbrook/Desktop/Full Test Mirai/DNN/Full Test Results Mirai.csv"
filename = "/Users/lukeholbrook/Desktop/Journal Paper Data Sets/Provision Security Camera/mirai/full test mirai results Provision.csv"

df = pd.read_csv(filename, header=None,low_memory=False)

X = df.iloc[:, 2].values

X_values = np.array(X)

print(X_values)

labels_benign = np.zeros(96)
labels_attack = np.ones(1)
labels_benign_2 = np.zeros(65)
#labels_attack_2 = np.ones(1)
#labels_benign_3 = np.zeros(51)
#labels_attack_3 = np.ones(1)
#labels_benign_4 = np.zeros(59)
#labels_attack_4 = np.ones(1)
#labels_benign_5 = np.zeros(5)


labels = np.concatenate((labels_benign, labels_attack, labels_benign_2))

y = labels

print('Accuracy: %.7f' % accuracy_score(X_values, y))

print('MSE train: test: %.7f' % (
        mean_squared_error(X_values, y)))

#MSE train: 1.642, test: 11.052

print('R^2 train: test: %.7f' % (
        r2_score(X_values, y)))

cm = pd.DataFrame(confusion_matrix(X_values, y))
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15})

plt.show()

