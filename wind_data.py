# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:39:09 2024

@author: nri
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

PATH = r'C:\Users\NRI\OneDrive - EE\Skrivebord\Research Projects\Trackers'
FILE = r'\Agersted_Wind_and_Angle_2023.csv'

df = pd.read_csv(PATH+FILE, delimiter=';', skiprows=[1], index_col='Date Time')

cols = df.columns

drop_cols = [c for c in cols if 'Windspeed' not in c]

df = df.drop(columns=drop_cols)

cols_new = list(np.linspace(1, 9, 9, dtype=int))

df.columns = cols_new

df_maxmin = df.max(axis=1) - df.min(axis=1)


colors = ['r', 'b', 'g', 'orange', 'c', 'm', 'y', 'black', 'grey']


fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

df.plot(ax=axes[0], lw=1, x_compat=True)
df_maxmin.plot(ax=axes[1], x_compat=True)

# axes[0].xaxis.set_major_locator(mpl.dates.DayLocator(interval=7))
# axes[0].xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%m-%d"))

axes[0].legend(title='Sensor #', loc='upper right')
axes[0].set_ylabel('Wind speed (m/s)', fontsize=14)
axes[0].grid(linestyle='dashed')
axes[1].set_ylabel('Range of wind speeds (m/s)', fontsize=14)
axes[1].grid(linestyle='dashed')
axes[1].set_xlabel('Datetime', fontsize=14)
