# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:22:07 2024

@author: nri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors

plt.rcParams.update({'font.size': 12})

PATH = r'C:\Users\NRI\OneDrive - EE\Skrivebord\IEA PVPS\Task 13'
FILE = '\ITRPV Bifacial Forecasts.csv'

df = pd.read_csv(PATH+FILE)

years = df.Report.unique()
year_min, year_max = min(years), max(years)
year_ticks = np.arange(year_min-1, year_max+12, 5)

jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=year_min, vmax=year_max)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)


fig, ax = plt.subplots(1)
for i in years:
    colorVal = scalarMap.to_rgba(i)
    sub = df.loc[df.Report == i]
    ax.plot(sub.Year, sub['Bifacial Cell Market (%)'], c=colorVal, label=i)
    ax.scatter(sub.Year, sub['Bifacial Cell Market (%)'], c=colorVal,)
ax.set_ylabel('Bifacial Cell Market (%)')
ax.grid(linestyle='dashed')
ax.legend(loc='upper left', title='ITRPV Report')
ax.set_xticks(year_ticks)
