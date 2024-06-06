# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 20:04:46 2024

@author: nri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pvlib as pvl

PATH = r'C:\Users\NRI\EE\PV Technology - Documents\General\Trackers\EE Site Data\Risø\Data 30_10_2023'

take_subset = True
t_start = '2023-10-05 06:30:00'
t_end = '2023-10-05 17:30:00'

ts = pd.date_range(t_start, t_end, freq='min', tz='Etc/GMT-1')
solpos = pvl.solarposition.get_solarposition(ts, 55.6, 12.1)
tilt = pvl.tracking.singleaxis(solpos.zenith,
                               solpos.azimuth, axis_azimuth=180,
                               max_angle=49.5, gcr=0.23)

FILES = {'tcu': '\\tcu.csv'}

data = {}

for f in FILES:
    data[f] = pd.read_csv(PATH+FILES[f], index_col='datetime')
    data[f].index = pd.to_datetime(data[f].index)

# take subset
if take_subset:
    for d in data:
        data[d] = data[d].loc[t_start:t_end]

tilt_measured = data['tcu'].loc[data['tcu']['tcu_id'] == 2]
tilt_measured = tilt_measured.tz_localize('Etc/GMT-1')
tilt_measured = tilt_measured.rename(columns={'currentanglex': 'measured'})

tilt = tilt.rename(columns={'tracker_theta': 'modeled'})
tilt = tilt[['modeled']].join(tilt_measured['measured'])

# %% discontinuous tracking of EE tracker
# half_step = (max(tilt['modeled'].dropna()) -
#              min(tilt['modeled'].dropna())) / 120

# labels = np.round(np.linspace(start=min(tilt['modeled'].dropna(
# )) + half_step, stop=max(tilt['modeled'].dropna()) - half_step,
#     num=120), 1)

# tilt['discontinuous'], bins = pd.cut(tilt['modeled'], labels=labels,
#                                      bins=120,
#                                      retbins=True, precision=1)

# %%

tilt['error (°)'] = tilt['measured'] - tilt['modeled']

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

tilt[['modeled', 'measured']].plot(ax=axes[0], lw=2, x_compat=True)
tilt[['error (°)']].plot(ax=axes[1], x_compat=True)

# axes[0].xaxis.set_major_locator(mpl.dates.DayLocator(interval=7))
# axes[0].xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%m-%d"))

axes[0].legend(title='Tracker Tilt', loc='upper left')
axes[0].set_ylabel('Inclination (°)', fontsize=12)
axes[0].grid(linestyle='dashed')
axes[1].set_ylabel('Measured — Modeled (°)', fontsize=12)
axes[1].grid(linestyle='dashed')
axes[1].set_xlabel('Datetime (MM-DD HH)', fontsize=12)
