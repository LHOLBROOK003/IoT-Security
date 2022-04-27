#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 14:47:41 2018

@author: lukeholbrook
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('/Users/lukeholbrook/Desktop/Bidirectional LSTM line_user_red_loss.csv')

#df = pd.read_csv('./housing.data.txt'), sep='\s+')


df.columns = ['Line', 'User', 'Red', 'Loss']

#df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
#              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
#              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']


#cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

cols = ['Line', 'User', 'Red', 'Loss']

sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
plt.show()