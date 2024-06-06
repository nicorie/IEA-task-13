# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 20:47:38 2023

Analyze bifacial_radiance results

@author: nri
"""
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from intersect import intersection
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import string
import seaborn as sns

# distances between y sensors
l1, l2, l3 = 2.1, 2.1*2, 2.1*2+0.145
d1, d2, d3 = l1/22, l2/22, l3/22  # ysensors = 21, so there are 22 segments
sensorsy = np.arange(1, 22, 1)  # dimensionless
xaxis_geoloop = np.arange(0, 23, 1)  # dimensionless
# xaxis_percent = np.array([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
xaxis_percent = np.array(
    [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
# xaxis_percent = np.array(
#     [-1, -0.9, -0.7, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 1])
nsegments = 22/2

sensorsy_percent = []  # distance of each sensor relative to TT
for i in xaxis_geoloop:
    if i == 0:
        x = -1
    elif i < 11:
        x = -(nsegments-i)/nsegments  #
        # label = f'{x:.0f}%W'
    elif i == 11:
        # label = 'TT'
        x = 0
    elif i < 22:
        x = (i-nsegments)/nsegments
        # label = f'{x:.0f}%E'
    elif i == 22:
        x = 1
    sensorsy_percent.append(x)

# now find the sensor # corresponding to desired xaxis
xaxis_sensor_nums, x_labels = [], []
for i in xaxis_percent:
    x = np.interp(i, sensorsy_percent, xaxis_geoloop)
    if i == -1:
        label = 'West'
    elif i == 0:
        label = 'TT'
    elif i == 1:
        label = 'East'
    else:
        label = f'{100*i:.0f}%'

    x_labels.append(label)
    xaxis_sensor_nums.append(x)


# %% open gendaylit pickle results
path = r'C:\Users\NRI\OneDrive - EE\Skrivebord\IEA PVPS\Task 13\HPC_Results\Pickles Final'
all_dir = os.listdir(path)
all_dir = [i for i in all_dir if '.pkl' in i]
dict_rpoa_tot, dict_rpoa_tot_tt, dict_avgs = {}, {}, {}


for file in tqdm(all_dir):

    pkl = pd.read_pickle(path+'\\'+file)
    syst = file.split('_')[1]
    lat = file.split('_')[2]
    lon = file.split('_')[3].replace('.pkl', '')

    for i in list(pkl.items()):
        dt = i[0].split('_')
        i[1]['date'], i[1]['time'] = dt[0], dt[1]
        i[1]['sensor'] = np.linspace(1, 23, i[1].shape[0], dtype=int)

    dfs = list(pkl.values())
    results = pd.concat(dfs)
    results['datetime'] = pd.to_datetime(results['date']+' '+results['time'])
    results.set_index('datetime', inplace=True)
    # results['Wm2Back'].plot()
    rtrace = {'back': results.rearMat.unique(),
              'front': results.mattype.unique()}

    on_tt = [11, 22, 23]

    results_tt = results.loc[results['sensor'].isin(on_tt)]
    results = results.loc[~results['sensor'].isin(on_tt)]

    # df = df.loc[df['rearMat'].str.contains('Big-Module')]

    # yearly total at each sensor pos.
    rpoa_tot = results.groupby(['sensor'])[
        ['Wm2Back', 'Wm2Front']].sum()/1000

    rpoa_tot_tt = results_tt.groupby(['sensor'])[
        ['Wm2Back', 'Wm2Front']].sum()/1000

    rpoa_mean = rpoa_tot[['Wm2Back', 'Wm2Front']].mean()

    rpoa_tot['rpoa_mean'] = rpoa_mean['Wm2Back']
    rpoa_tot['gpoa_mean'] = rpoa_mean['Wm2Front']

    # find where the average is between two rpoa values
    x1 = rpoa_tot.index.values
    x2 = x1
    y1, y2 = rpoa_tot['Wm2Back'].values, rpoa_tot['rpoa_mean'].values
    x, y = intersection(x1, y1, x2, y2)

    dict_rpoa_tot[f'{syst}_{lat}_{lon}'] = rpoa_tot
    dict_rpoa_tot_tt[f'{syst}_{lat}_{lon}'] = rpoa_tot_tt
    dict_avgs[f'{syst}_{lat}_{lon}_x'] = list(x)
    dict_avgs[f'{syst}_{lat}_{lon}_y'] = list(y)

# %% visualize annual totals and averages

colors = ['blue', 'red', 'green']  # set the colors for TT
labels = ['TT Center', 'TT South', 'TT North']  # labels for TT sensors
systems = {'1': '1P Tracker',
           '2': '2P Tracker',
           '3': '2P_gap Tracker'}

top = 0.905
bottom = 0.115
left = 0.125
right = 0.9
hspace = 0.12
wspace = 0.13

nrows, ncols = 2, 3

for syst_type in systems:
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            sharey=False, sharex=False)
    count = 0
    for i, sim in enumerate(dict_rpoa_tot.keys()):
        if count < ncols:
            row, col = 0, count
        else:
            row, col = 1, count-3

        if sim.split('_')[0] == syst_type:
            print(sim)
            lat = sim.split('_')[1]
            lon = sim.split('_')[2]
            df_tot = dict_rpoa_tot[sim]
            df_tot_tt = dict_rpoa_tot_tt[sim]
            x = dict_avgs[f'{sim}_x']
            y = dict_avgs[f'{sim}_y']

            axs[row, col].plot(df_tot.index, df_tot.Wm2Back,
                               color='orange', label=r'PV POA')
            axs[row, col].scatter(df_tot.index, df_tot.Wm2Back, color='orange')
            axs[row, col].hlines(y=y[0], xmin=min(df_tot.index), xmax=max(df_tot.index),
                                 colors='black', linestyles='dashed')
            axs[row, col].vlines(x=x, ymin=0, ymax=y[0],
                                 colors='black', linestyles='dashed')
            for c, s in enumerate(df_tot_tt.index):
                ys = df_tot_tt.loc[df_tot_tt.index == s]['Wm2Back'].values
                axs[row, col].scatter(x=11, y=ys, c=colors[c], label=labels[c])
            axs[row, col].grid(linestyle='dashed')
            # axs[row, col].set_xlabel('Sensor #', fontsize=12)
            axs[row, col].set_ylim([0, 425])
            axs[row, col].set_xticks(xaxis_sensor_nums, labels=x_labels)
            axs[row, col].xaxis.set_minor_locator(
                plt.MultipleLocator(xaxis_sensor_nums[1]/2))
            axs[row, col].legend(loc='upper right')
            # axs[row, col].set_title(
            #     f'{float(lat):.1f}°, {float(lon):.1f}°', fontsize=12)
            axs[row, col].text(1, 400, f'{float(lat):.1f}°, {float(lon):.1f}°',
                               fontsize=12,
                               bbox={'facecolor': 'white',
                                     'alpha': 0.6, 'pad': 2})
            count += 1
    axs[0, 0].set_ylabel(r'R$_{\rm POA}$ (kWh/m²)', fontsize=12)
    axs[1, 0].set_ylabel(r'R$_{\rm POA}$ (kWh/m²)', fontsize=12)
    plt.suptitle(f'{systems[syst_type]}', fontsize=14, y=0.93, x=0.51)
    plt.subplots_adjust(left=left, bottom=bottom, right=right,
                        top=top, wspace=wspace, hspace=hspace)

# %% create box plots of the avg RPOA locations
# https://towardsdatascience.com/scattered-boxplots-graphing-experimental-results-with-matplotlib-seaborn-and-pandas-81f9fa8a1801

loc_1P_W, loc_1P_E, loc_2P_W, loc_2P_E, loc_2Pg_W1, loc_2Pg_E1, loc_2Pg_W2, loc_2Pg_E2 = [
    [] for i in range(8)]

for i in dict_avgs:
    if i[0] == '1':
        if 'x' in i:
            x1 = abs(np.interp(dict_avgs[i][0],
                     xaxis_geoloop, sensorsy_percent))
            x2 = abs(np.interp(dict_avgs[i][1],
                     xaxis_geoloop, sensorsy_percent))
            loc_1P_W.append(x1*100)
            loc_1P_E.append(x2*100)
    elif i[0] == '2':
        if 'x' in i:
            x1 = abs(np.interp(dict_avgs[i][0],
                     xaxis_geoloop, sensorsy_percent))
            x2 = abs(np.interp(dict_avgs[i][1],
                     xaxis_geoloop, sensorsy_percent))
            loc_2P_W.append(x1*100)
            loc_2P_E.append(x2*100)
    elif i[0] == '3':
        if ('x' in i) & (len(dict_avgs[i]) == 2):
            x1 = abs(np.interp(dict_avgs[i][0],
                     xaxis_geoloop, sensorsy_percent))
            x2 = abs(np.interp(dict_avgs[i][1],
                     xaxis_geoloop, sensorsy_percent))
            loc_2Pg_W1.append(x1*100)
            loc_2Pg_E1.append(x2*100)
        elif ('x' in i) & (len(dict_avgs[i]) == 4):
            x1 = abs(np.interp(dict_avgs[i][0],
                     xaxis_geoloop, sensorsy_percent))
            x2 = abs(np.interp(dict_avgs[i][1],
                     xaxis_geoloop, sensorsy_percent))
            x3 = abs(np.interp(dict_avgs[i][2],
                     xaxis_geoloop, sensorsy_percent))
            x4 = abs(np.interp(dict_avgs[i][3],
                     xaxis_geoloop, sensorsy_percent))
            loc_2Pg_W1.append(x1*100)
            loc_2Pg_W2.append(x2*100)
            loc_2Pg_E2.append(x3*100)
            loc_2Pg_E1.append(x4*100)

data = [list(x) for x in [loc_1P_W, loc_1P_E, loc_2P_W,
                          loc_2P_E, loc_2Pg_W1, loc_2Pg_E1]]
# loc_2Pg_W2, loc_2Pg_E2]]

plt.boxplot(data, labels=['1P West', '1P East',
            '2P West', '2P East', r'2P$_{\rm gap}$ West', r'2P$_{\rm gap}$ East'])

palette = ['r', 'b', 'r', 'b', 'r', 'b']
labels = ['West', 'East', None, None, None, None]
xs = list(range(1, 7))
for i, val, c, l in zip(xs, data, palette, labels):
    x = [i]*len(val)
    #val = [v*100 for v in val]
    plt.scatter(x, val, alpha=0.4, color=c, label=l)

plt.hlines(np.average([loc_1P_W, loc_1P_E]), 1,
           2, colors='green', linestyles='dashed')
plt.hlines(np.average([loc_2P_W, loc_2P_E]), 3,
           4, colors='green', linestyles='dashed')
plt.hlines(np.average([loc_2Pg_W1, loc_2Pg_E1]), 5,
           6, colors='green', linestyles='dashed')
plt.ylabel("Distance from TT to E/W edge (%)", fontsize=12)
plt.xlabel("Tracker type and side", fontsize=12)
plt.grid(linestyle='dashed')
plt.legend(title='PV array side')
plt.show()

# %% calculate error summary

# %% sensitivity analysis genCumSky
# path where the sensitivity files are
# path = r'C:\Users\NRI\Bifacial_Radiance_Files\results\Geoloop_Sensitivity_R1'
# path = r'C:\Users\NRI\Bifacial_Radiance_Files\results\Geoloop_Sensitivity_R2'
path = r'C:\Users\NRI\Bifacial_Radiance_Files\results\Geoloop_Sensitivity_R3'

all_dir = os.listdir(path)
li = []

# NOTE: always check the indexing on the 'info' string
# R2 format: f'_{height}_{gcr}_{shape}_{albedo}_{structure}_{tilt}.csv'
# R3 f'_{height}_{gcr}_{shape}_{albedo}_{structure}_{loc[0]}_{loc[1]}'
for dirindex in tqdm(range(len(all_dir))):
    path_temp = path + '\\' + all_dir[dirindex]
    df = pd.read_csv(path_temp, index_col=None)

    info = path_temp.split('_')
    df['angle'] = float(info[13].replace('.csv', ''))
    df['height'] = float(info[5])
    df['gcr'] = float(info[6])
    df['TT'] = info[7]
    df['albedo'] = float(info[8])
    df['structure'] = info[9]
    df['sensor'] = np.linspace(1, 23, df.shape[0], dtype=int)

    li.append(df)

results = pd.concat(li, axis=0, ignore_index=True)

# position 11 is bottom of torque tube
# position 22 is south outboard sensor
# position 23 is north outboard sensor
on_tt = [11, 22, 23]

results_tt = results.loc[results['sensor'].isin(on_tt)]
results = results.loc[~results['sensor'].isin(on_tt)]

rtrace = {'main': results.rearMat.unique(),
          'outboard': results_tt.rearMat.unique()}

rpoa_tot = results.groupby(['height', 'gcr', 'TT', 'albedo', 'structure',
                            'sensor'])[['Wm2Back', 'Wm2Front']].sum()/1000

rpoa_tot_tt = results_tt.groupby(['height', 'gcr', 'TT', 'albedo', 'structure',
                                  'sensor'])[['Wm2Back', 'Wm2Front']].sum()/1000

rpoa_mean = rpoa_tot.groupby(['height', 'gcr', 'TT', 'albedo',
                              'structure'])[['Wm2Back', 'Wm2Front']].mean()

rpoa_mean = rpoa_mean.rename(columns={'Wm2Back': 'rpoa_mean',
                                      'Wm2Front': 'gpoa_mean'})

rpoa_tot = rpoa_tot.join(rpoa_mean)

group = rpoa_tot.groupby(['height', 'gcr', 'TT', 'albedo', 'structure'])

group.size().shape[0]

avgs = []
for i, idx in enumerate(group):
    df = idx[1]
    df.reset_index(inplace=True)
    # find where the average is between two rpoa values
    x1 = df['sensor'].values
    x2 = x1
    y1, y2 = df['Wm2Back'].values, df['rpoa_mean'].values
    x, y = intersection(x1, y1, x2, y2)
    avgs.append(list(x))

results2 = pd.DataFrame(avgs, index=group.groups.keys())
results2.reset_index(inplace=True)
results2 = results2.loc[results2['level_4'] != 'True']
results2.columns = ['height', 'gcr', 'TT', 'albedo',
                    'structure', 'x1', 'x2']  # 'x3', 'x4', 'x5']
results2[['x1', 'x2']].hist(bins=20)

"""
albedo is chart
gcr is x axis
height is line color
TT is symbol/line style
"""
colors = ['r', 'b', 'g']
symbols = ['o', 's']
sub1 = results2[results2['albedo'] == 0.1]
sub2 = results2[results2['albedo'] == 0.2]
sub3 = results2[results2['albedo'] == 0.4]
sub4 = results2[results2['albedo'] == 0.8]

fig, axs = plt.subplots(2, 2)

for i, h in enumerate(sub1.height.unique()):
    sub11 = sub1.loc[sub1.height == h]
    for j, tt in enumerate(sub11.TT.unique()):
        sub111 = sub11.loc[sub11.TT == tt]
        axs[0, 0].plot(sub111.gcr, sub111.x1, color=colors[i],
                       marker=symbols[j], label=h)

for i, h in enumerate(sub2.height.unique()):
    sub21 = sub2.loc[sub2.height == h]
    for j, tt in enumerate(sub21.TT.unique()):
        sub211 = sub21.loc[sub21.TT == tt]
        axs[0, 1].plot(sub211.gcr, sub211.x1, color=colors[i],
                       marker=symbols[j], label=h)

for i, h in enumerate(sub3.height.unique()):
    sub31 = sub3.loc[sub3.height == h]
    for j, tt in enumerate(sub31.TT.unique()):
        sub311 = sub31.loc[sub31.TT == tt]
        axs[1, 0].plot(sub311.gcr, sub311.x1, color=colors[i],
                       marker=symbols[j], label=h)

for i, h in enumerate(sub4.height.unique()):
    sub41 = sub4.loc[sub4.height == h]
    for j, tt in enumerate(sub41.TT.unique()):
        sub411 = sub41.loc[sub41.TT == tt]
        axs[1, 1].plot(sub411.gcr, sub411.x1, color=colors[i],
                       marker=symbols[j], label=h)

axs[0, 0].grid(linestyle='dashed')
axs[0, 1].grid(linestyle='dashed')
axs[1, 0].grid(linestyle='dashed')
axs[1, 1].grid(linestyle='dashed')
axs[0, 0].set_ylabel('West side average position', fontsize=12)
axs[1, 0].set_ylabel('West side average position', fontsize=12)
axs[0, 1].legend(loc='upper right', title='Hub height (m)')
axs[1, 1].set_xlabel('Ground cover ratio (GCR)', fontsize=12)
axs[1, 0].set_xlabel('Ground cover ratio (GCR)', fontsize=12)
axs[0, 0].title.set_text('10% albedo')
axs[0, 1].title.set_text('20% albedo')
axs[1, 0].title.set_text('40% albedo')
axs[1, 1].title.set_text('80% albedo')

# east sensors
fig, axs = plt.subplots(2, 2)

for i, h in enumerate(sub1.height.unique()):
    sub11 = sub1.loc[sub1.height == h]
    for j, tt in enumerate(sub11.TT.unique()):
        sub111 = sub11.loc[sub11.TT == tt]
        axs[0, 0].plot(sub111.gcr, sub111.x2, color=colors[i],
                       marker=symbols[j], label=h)

for i, h in enumerate(sub2.height.unique()):
    sub21 = sub2.loc[sub2.height == h]
    for j, tt in enumerate(sub21.TT.unique()):
        sub211 = sub21.loc[sub21.TT == tt]
        axs[0, 1].plot(sub211.gcr, sub211.x2, color=colors[i],
                       marker=symbols[j], label=h)

for i, h in enumerate(sub3.height.unique()):
    sub31 = sub3.loc[sub3.height == h]
    for j, tt in enumerate(sub31.TT.unique()):
        sub311 = sub31.loc[sub31.TT == tt]
        axs[1, 0].plot(sub311.gcr, sub311.x2, color=colors[i],
                       marker=symbols[j], label=h)

for i, h in enumerate(sub4.height.unique()):
    sub41 = sub4.loc[sub4.height == h]
    for j, tt in enumerate(sub41.TT.unique()):
        sub411 = sub41.loc[sub41.TT == tt]
        axs[1, 1].plot(sub411.gcr, sub411.x2, color=colors[i],
                       marker=symbols[j], label=h)

axs[0, 0].grid(linestyle='dashed')
axs[0, 1].grid(linestyle='dashed')
axs[1, 0].grid(linestyle='dashed')
axs[1, 1].grid(linestyle='dashed')
axs[0, 0].set_ylabel('East side average position', fontsize=12)
axs[1, 0].set_ylabel('East side average position', fontsize=12)
axs[0, 1].legend(loc='upper right', title='Hub height (m)')
axs[1, 1].set_xlabel('Ground cover ratio (GCR)', fontsize=12)
axs[1, 0].set_xlabel('Ground cover ratio (GCR)', fontsize=12)
axs[0, 0].title.set_text('10% albedo')
axs[0, 1].title.set_text('20% albedo')
axs[1, 0].title.set_text('30% albedo')
axs[1, 1].title.set_text('40% albedo')


colors = ['black', 'red', 'green']  # set the colors for TT
labels = ['TT_center', 'TT_south', 'TT_north']  # labels for TT sensors

# plots 144 plots
rpoa_tt = rpoa_tot_tt.reset_index()
keys = group.groups.keys()

letters = list(string.ascii_lowercase)
letters26x = []

for let in letters:
    for j in range(26):
        letters26x.append(let+letters[j])

for i, k in enumerate(keys):
    df = group.get_group(k)
    df.reset_index(inplace=True)
    df_tt = rpoa_tt.loc[(rpoa_tt['height'] == k[0]) & (rpoa_tt['gcr'] == k[1])
                        & (rpoa_tt['TT'] == k[2]) & (rpoa_tt['albedo'] == k[3])]

    plt.figure()
    plt.ylim(0, max(max(rpoa_tot['Wm2Back']), max(rpoa_tot_tt['Wm2Back'])))
    plt.plot(df['sensor'], df['Wm2Back'], marker='o', label='linescan')
    for c, s in enumerate(df_tt.sensor):
        y = df_tt.loc[df_tt.sensor == s]['Wm2Back'].values
        plt.scatter(x=11, y=y, c=colors[c], label=labels[c])
    plt.suptitle(str(k))
    plt.grid(linestyle='dashed')
    plt.ylabel('Rpoa [W/m2]', fontsize=14)  # check units
    plt.xlabel('Sensor #', fontsize=14)
    plt.xticks(np.arange(0, 25, 5))
    plt.legend(loc='upper right', title='Sensor')
    plt.show()
    plt.savefig(f'{letters26x[i]}.png')

# # plt.scatter(x, y, marker='*', s=100, color='black')
# plt.plot(df['sensor'], df[['Wm2Back', 'rpoa_mean']], marker='o')
