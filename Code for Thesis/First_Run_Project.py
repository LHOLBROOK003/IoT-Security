#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 18:14:30 2018

@author: lukeholbrook
"""

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_decision_regions(X, y, classifier, test_idx=None,
                          resolution=0.02):
    
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl,
                    edgecolor='black')
        
    # highlight test samples
    if test_idx:
    # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='test set')


filename = "/Users/lukeholbrook/Desktop/Controls Research/MAT 5983 NN and Data Analysis/bengin_traffic Danmini_Doorbell.csv"
filename_attack = "/Users/lukeholbrook/Desktop/Controls Research/MAT 5983 NN and Data Analysis/mirai_udp.csv"
df_benign = pd.read_csv(filename,header=None,low_memory=False)
df_attack = pd.read_csv(filename_attack,header=None,low_memory=False)
#df = pd.read_csv(filename,header=None)


dfar_benign = df_benign.iloc[101:300,0:2] # attempt 2, double ':' since we need to grab everything
dfar_attack = df_attack.iloc[101:300,0:2] 


data_benign = np.array(dfar_benign)
data_attack = np.array(dfar_attack)

data = np.concatenate((data_benign, data_attack))

#labels = df.iloc[:,4] #this is where the lables are in the iris data
#labels = df.iloc[0,:] #this is where the lables are in the doorbell data
labels_benign = np.zeros(199)
labels_attack = np.ones(199)
labels = np.concatenate((labels_benign, labels_attack))

#dfar = df.iloc[1:301,0:3]
#
#data = np.array(dfar)
#
#X = data[:,0:2]
#
#y = data[:,2]

X = data
y = labels

print('Class labels:', np.unique(y))

#print(data.std())

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(n_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

lr = LogisticRegression(C=100.0, random_state=1)

lr.fit(X_train_std, y_train)

knn = KNeighborsClassifier(n_neighbors=5, p=2,
                           metric='minkowski')
knn.fit(X_train_std, y_train)

svm = SVC(kernel='linear', C=1.0, random_state=1)

svm.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier=lr,
                      test_idx=range(105, 150))

plt.xlabel('X1 LR [standardized]')
plt.ylabel('X2 LR [standardized]')
plt.legend(loc='upper left')
plt.show()

plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier=ppn,
                      test_idx=range(105, 150))

plt.xlabel('X1 PPN [standardized]')
plt.ylabel('X2 PPN [standardized]')
plt.legend(loc='upper left')
plt.show()

plot_decision_regions(X_combined_std, y_combined,
                      classifier=knn, test_idx=range(105,150))
plt.xlabel('X1 KNN [standardized]')
plt.ylabel('X2 KNN [standardized]')
plt.legend(loc='upper left')
plt.show()

plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier=svm,
                      test_idx=range(105, 150))
plt.xlabel('X1 SVM [standardized]')
plt.ylabel('X2 SVM [standardized]')
plt.legend(loc='upper left')
plt.show()