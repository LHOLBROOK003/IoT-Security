#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 14:47:41 2018

@author: lukeholbrook
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('/Users/lukeholbrook/Desktop/Controls Research/SECURITY/Research Notes/Data Plots 3.csv')

#df = pd.read_csv('./housing.data.txt'), sep='\s+')


df.columns = ['HH_L1_mean', 'HH_jit_L1_mean', 'Mirai_HH_L1_mean', 'Mirai_HH_jit_L1_mean']

cols = ['HH_L1_mean', 'HH_jit_L1_mean', 'Mirai_HH_L1_mean', 'Mirai_HH_jit_L1_mean']

sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
plt.show()