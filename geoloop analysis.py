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
import string

# %% open pvpmc 2023 analysis
# path_pvpmc = r'C:\Users\NRI\OneDrive - EE\Skrivebord\IEA PVPS\Task 13\HPC_Results'
# file_pvpmc = '\TT3_all.csv'

# df = pd.read_csv(path_pvpmc+file_pvpmc)

# %%
# path where the sensitivity files are
#path = r'C:\Users\NRI\Bifacial_Radiance_Files\results\Geoloop_Sensitivity_R1'
path = r'C:\Users\NRI\Bifacial_Radiance_Files\results\Geoloop_Sensitivity_R2'

all_dir = os.listdir(path)
li = []

# NOTE: always check the indexing on the 'info' string
# R2 format: f'_{height}_{gcr}_{shape}_{albedo}_{structure}_{tilt}.csv'
for dirindex in tqdm(range(len(all_dir))):
    path_temp = path + '\\' + all_dir[dirindex]
    df = pd.read_csv(path_temp, index_col=None)

    info = path_temp.split('_')
    df['angle'] = float(info[12].replace('.csv', ''))
    df['height'] = float(info[6])
    df['gcr'] = float(info[7])
    df['TT'] = info[8]
    df['albedo'] = float(info[9])
    df['structure'] = info[10]
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
sub3 = results2[results2['albedo'] == 0.3]
sub4 = results2[results2['albedo'] == 0.4]

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
axs[1, 0].title.set_text('30% albedo')
axs[1, 1].title.set_text('40% albedo')

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
    plt.ylabel('Rpoa [W/m2]', fontsize=14)
    plt.xlabel('Sensor #', fontsize=14)
    plt.xticks(np.arange(0, 25, 5))
    plt.legend(loc='upper right', title='Sensor')
    plt.show()
    plt.savefig(f'{letters26x[i]}.png')

# # plt.scatter(x, y, marker='*', s=100, color='black')
# plt.plot(df['sensor'], df[['Wm2Back', 'rpoa_mean']], marker='o')
