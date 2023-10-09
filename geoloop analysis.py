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

# %% open pvpmc 2023 analysis
# path_pvpmc = r'C:\Users\NRI\OneDrive - EE\Skrivebord\IEA PVPS\Task 13\HPC_Results'
# file_pvpmc = '\TT3_all.csv'

# df = pd.read_csv(path_pvpmc+file_pvpmc)

# %%
# path where the sensitivity files are
path = r'C:\Users\NRI\Bifacial_Radiance_Files\results\Geoloop_Sensitivity'

all_dir = os.listdir(path)
li = []

for dirindex in tqdm(range(len(all_dir))):
    path_temp = path + '\\' + all_dir[dirindex]
    df = pd.read_csv(path_temp, index_col=None)

    info = path_temp.split('_')
    df['angle'] = float(info[5])
    df['height'] = float(info[6])
    df['gcr'] = float(info[7])
    df['TT'] = info[8]
    df['albedo'] = info[9]
    df['structure'] = info[10].replace('.csv', '')
    df['sensor'] = np.linspace(1, 21, df.shape[0], dtype=int)
    li.append(df)

results = pd.concat(li, axis=0, ignore_index=True)

rtrace = results.rearMat.unique()

rpoa_tot = results.groupby(['height', 'gcr', 'TT', 'albedo', 'structure',
                            'sensor'])[['Wm2Back', 'Wm2Front']].sum()/1000

rpoa_mean = rpoa_tot.groupby(['height', 'gcr', 'TT', 'albedo',
                              'structure'])[['Wm2Back', 'Wm2Front']].mean()

rpoa_mean = rpoa_mean.rename(columns={'Wm2Back': 'rpoa_mean',
                                      'Wm2Front': 'gpoa_mean'})

rpoa_tot = rpoa_tot.join(rpoa_mean)

group = rpoa_tot.groupby(['height', 'gcr', 'TT', 'albedo', 'structure'])

for i, idx in enumerate(group):
    df = idx[1]
    # find where the average is between two rpoa values
