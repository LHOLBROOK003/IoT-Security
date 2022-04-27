#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 19:18:50 2018

@author: lukeholbrook
"""

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/lukeholbrook/Desktop/Controls Research/SECURITY/Research Notes/Data Plots 3.csv')


df.columns = ['HH_L1_mean', 'HH_jit_L1_mean', 'Mirai_HH_L1_mean', 'Mirai_HH_jit_L1_mean']

cols = ['HH_L1_mean', 'HH_jit_L1_mean', 'Mirai_HH_L1_mean', 'Mirai_HH_jit_L1_mean']

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)
plt.show()