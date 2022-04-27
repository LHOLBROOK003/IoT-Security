#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 19:45:37 2018

@author: lukeholbrook
"""
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
import seaborn as sns


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

#filename = "/Users/lukeholbrook/Desktop/full_test.csv"
#filename = "/Users/lukeholbrook/Desktop/Full Test Mirai/Random Forest/full_test_mirai.csv"
filename = "/Users/lukeholbrook/Desktop/Journal Paper Data Sets/Ecobee Thermostat/mirai/full test mirai/1head.csv"

df = pd.read_csv(filename, header=None,low_memory=False)

pca = PCA(n_components=2)


X = df.iloc[:, 0:3].values
print(X)

X_values = np.array(X)


#The below changes per 1head.csv of the device
labels_benign = np.zeros(7368)
labels_attack = np.ones(32)
labels_benign_2 = np.zeros(5745)
labels = np.concatenate((labels_benign, labels_attack, labels_benign_2))

#y = df['MEDV'].values
y = labels

X_train, X_test, y_train, y_test = train_test_split(
        X_values, y, test_size=0.7, random_state=1, stratify=y)
 

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

forest = RandomForestRegressor(n_estimators=100,
                               criterion='mse',
                               random_state=1,
                               n_jobs=-1)

forest.fit(X_train_pca, y_train)
#forest.fit(X_train, y_train)


#print('X_test_pca:', X_test_pca)
#print(forest.score(X_test_pca, y_test))
#print(forest.predict(X_train_pca))

X_combined = np.vstack((X_train_pca, X_test_pca))
#X_combined = np.vstack((X_train, X_test))

y_combined = np.hstack((y_train, y_test))
#
y_pred = forest.predict(X_test_pca)
y_train_pred = forest.predict(X_train_pca)
#
#print('y_pred:', y_pred)
#print('y_test:', y_test)
#
print('Accuracy: %.8f' % accuracy_score(y_train, y_train_pred.round()))

#accuracy = accuracy_score(y_test, y_pred)
#print('Accuracy:', accuracy)

print('MSE train: %.7f, test: %.7f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_pred)))

##MSE train: 1.642, test: 11.052
#
print('R^2 train: %.7f, test: %.7f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_pred)))

#tn, fp, fn, tp = confusion_matrix(y_test, y_pred.round()).ravel()
#
#print('tn:',tn)
#print('fp:',fp)
#print('fn:',fn)
#print('tp:',tp)

#cm = pd.DataFrame(confusion_matrix(y_test, y_pred.round()))
#print(cm)
#hm = sns.heatmap(cm,
#                 cbar=True,
#                 annot=True,
#                 square=True,
#                 fmt='.2f',
#                 annot_kws={'size': 15})
#
#plt.show()

#R^2 train: 0.979, test: 0.878

#plt.scatter(y_train_pred,
#    y_train_pred - y_train,
#    c='steelblue',
#    edgecolor='white',
#    marker='x',
#    s=35,
#    alpha=0.9,
#    label='Training data')
#
#plt.scatter(y_test_pred,
#    y_test_pred - y_test,
#    c='limegreen',
#    edgecolor='white',
#    marker='s',
#    s=35,
#    alpha=0.9,
#    label='Test data')

plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(50,150))

plt.xlabel('X1 RF [standardized]')
plt.ylabel('X2 RF [standardized]')
plt.legend(loc='upper right')
#plt.hlines(y=0, xmin=-0.5, xmax=1.5, lw=2, color='black')
#plt.xlim([-0.5, 1.5])
plt.show()
