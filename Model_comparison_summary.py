# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 14:21:37 2023
-Pull RPOA stored results from bifacial_radiance
-Regenerate tracker angles
-Calculate GPOA
-Calculate Tmod at each module
-Calculate string power
@author: nri
"""
import pandas as pd
import numpy as np
import pvlib as pvl
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


PATH = r'C:\Users\NRI\OneDrive - EE\Skrivebord\IEA PVPS\Task 13\Blind Modeling Comparison'
FILE = '\Phase2_meteo_hourly_psm3format.csv'

meteo = pd.read_csv(PATH+FILE, skiprows=2,
                    usecols=list(np.linspace(0, 10, 11, dtype=int)))

meteo.index = pd.to_datetime(meteo[['Year', 'Month', 'Day', 'Hour', 'Minute']])
meteo.index = meteo.index.tz_localize('Etc/GMT+7')

# site specs
lat = 35.05
lon = -106.54
tz = -7
elev = 1600

# tracker specs
rot = 55

# params that vary across scenarios
gcrs = {1: 0.4,
        2: 0.25,
        3: 0.4,
        4: 0.4,
        5: 0.4,
        6: 0.4,
        7: 0.4}

hub_heights = {1: 1.5,
               2: 1.5,
               3: 1.5,
               4: 3.5,
               5: 1.5,
               6: 1.5,
               7: 3.5}

albedos = {1: 0.2,
           2: 0.2,
           3: 0.5,
           4: 0.2,
           5: 0.2,
           6: 0.2,
           7: 0.2}

p_up = {1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 1,
        6: 1,
        7: 2}

# from the .pan file
Width = 1.303
Height = 2.384
Depth = 0.040
NCelS = 66
NCelP = 2
NDiode = 3
GRef = 1000
TRef = 25.0
PNom = 650.0
PNomTolLow = 0.00
PNomTolUp = 3.00
BifacialityFactor = 0.700
Isc = 18.350
Voc = 45.50
Imp = 17.270
Vmp = 37.70
muISC = 7.32
muVocSpec = -114.0
muPmpReq = -0.340
RShunt = 20000
Rp_0 = 80000
Rp_Exp = 5.50
RSerie = 0.168
Gamma = 1.020
muGamma = -0.0002
VMaxIEC = 1500
VMaxUL = 1500
Absorb = 0.90
ARev = 3.200
BRev = 10.360
RDiode = 0.010
VRevDiode = -0.70
AirMassRef = 1.500
CellArea = 220.5
SandiaAMCorr = 50.000

eff = (PNom/(1000*Height*Width))

AOIs = [0, 40, 50, 60, 70, 75, 80, 85, 90]
IAMs = [1, 1, 0.998, 0.992, 0.983, 0.961, 0.933, 0.853, 0]

# %% import RPOA
# some how, the 2022-12-27 didn't get simulated
PATH2 = r'C:\Users\NRI\Bifacial_Radiance_Files\results\IEA'

all_dir = os.listdir(PATH2)
li = []

for dirindex in tqdm(range(len(all_dir))):
    path_temp = PATH2 + '\\' + all_dir[dirindex]

    if 'Back' in path_temp:
        df = pd.read_csv(path_temp, index_col=None)
        info = path_temp.split('_')
        date = info[4]
        time = info[5][:4]
        df['datetime'] = pd.to_datetime(date+' '+time)
        df['module'] = int(info[8])
        df['scenario'] = int(info[6])
        li.append(df)

rpoa = pd.concat(li, axis=0, ignore_index=True)

# drop RPOA not on a module
rpoa = rpoa.loc[rpoa['rearMat'].str.contains('Trina')]

rpoa_avg = rpoa.groupby(
    ['module', 'scenario', 'datetime']).mean().reset_index()

rpoa_avg['N Sensors'] = rpoa.groupby(
    ['module', 'scenario', 'datetime']).size().values

rpoa_avg.set_index('datetime', inplace=True)

# %%  tracking angles

solpos = pvl.solarposition.get_solarposition(meteo.index, lat, lon)

tracking_angles = pvl.tracking.singleaxis(solpos.apparent_zenith,
                                          solpos.azimuth, axis_azimuth=180,
                                          backtrack=True, gcr=0.4,
                                          max_angle=rot)

tracking_angles_s2 = pvl.tracking.singleaxis(solpos.apparent_zenith,
                                             solpos.azimuth, axis_azimuth=180,
                                             backtrack=True, gcr=0.25,
                                             max_angle=rot)

axis_tilt_s5 = pvl.tracking.calc_axis_tilt(slope_azimuth=90, slope_tilt=5.71,
                                           axis_azimuth=180)

axis_tilt_s6 = pvl.tracking.calc_axis_tilt(slope_azimuth=225, slope_tilt=5.71,
                                           axis_azimuth=180)

cross_axis_s5 = pvl.tracking.calc_cross_axis_tilt(slope_azimuth=90,
                                                  slope_tilt=5.71,
                                                  axis_azimuth=180,
                                                  axis_tilt=axis_tilt_s5)

cross_axis_s6 = pvl.tracking.calc_cross_axis_tilt(slope_azimuth=225,
                                                  slope_tilt=5.71,
                                                  axis_azimuth=180,
                                                  axis_tilt=axis_tilt_s6)


tracking_angles_s5 = pvl.tracking.singleaxis(solpos.apparent_zenith,
                                             solpos.azimuth,
                                             axis_tilt=axis_tilt_s5,
                                             axis_azimuth=180,
                                             backtrack=True, gcr=0.4,
                                             max_angle=rot,
                                             cross_axis_tilt=cross_axis_s5)

tracking_angles_s6 = pvl.tracking.singleaxis(solpos.apparent_zenith,
                                             solpos.azimuth,
                                             axis_tilt=axis_tilt_s6,
                                             axis_azimuth=180,
                                             backtrack=True, gcr=0.4,
                                             max_angle=rot,
                                             cross_axis_tilt=cross_axis_s6)

tracking_angles_all = pd.concat([tracking_angles['tracker_theta'],
                                 tracking_angles_s2['tracker_theta'],
                                 tracking_angles_s5['tracker_theta'],
                                 tracking_angles_s6['tracker_theta']], axis=1)

tracking_angles_all.columns = [
    'reference_theta', 'A1_theta', 'A4_theta', 'A5_theta']
tracking_angles_all.plot(marker='o', grid=True)

# %% frontside irradiance

dni_extra = pvl.irradiance.get_extra_radiation(solpos.index)

poa = pvl.irradiance.get_total_irradiance(tracking_angles.surface_tilt,
                                          tracking_angles.surface_azimuth,
                                          solpos.zenith, solpos.azimuth,
                                          meteo.DNI, meteo.GHI, meteo.DHI,
                                          dni_extra=dni_extra, albedo=0.2,
                                          model='perez')

poa_s2 = pvl.irradiance.get_total_irradiance(tracking_angles_s2.surface_tilt,
                                             tracking_angles_s2.surface_azimuth,
                                             solpos.zenith, solpos.azimuth,
                                             meteo.DNI, meteo.GHI, meteo.DHI,
                                             dni_extra=dni_extra, albedo=0.2,
                                             model='perez')

poa_s3 = pvl.irradiance.get_total_irradiance(tracking_angles_s2.surface_tilt,
                                             tracking_angles_s2.surface_azimuth,
                                             solpos.zenith, solpos.azimuth,
                                             meteo.DNI, meteo.GHI, meteo.DHI,
                                             dni_extra=dni_extra, albedo=0.4,
                                             model='perez')

poa_s5 = pvl.irradiance.get_total_irradiance(tracking_angles_s5.surface_tilt,
                                             tracking_angles_s5.surface_azimuth,
                                             solpos.zenith, solpos.azimuth,
                                             meteo.DNI, meteo.GHI, meteo.DHI,
                                             dni_extra=dni_extra, albedo=0.2,
                                             model='perez')

poa_s6 = pvl.irradiance.get_total_irradiance(tracking_angles_s6.surface_tilt,
                                             tracking_angles_s6.surface_azimuth,
                                             solpos.zenith, solpos.azimuth,
                                             meteo.DNI, meteo.GHI, meteo.DHI,
                                             dni_extra=dni_extra, albedo=0.2,
                                             model='perez')

poa_all = pd.concat([poa['poa_global'], poa_s2['poa_global'], poa_s3['poa_global'],
                     poa_s5['poa_global'], poa_s6['poa_global']], axis=1)

poa_all.columns = ['reference_poa', 'A1_poa', 'A2_poa', 'A4_poa', 'A5_poa']
poa_all.plot(marker='o', grid=True)

# %% module temperature

tmod = pvl.temperature.pvsyst_cell(poa_all.reference_poa, meteo.Temperature,
                                   meteo['Wind Speed'], u_c=25.7, u_v=1.1)

tmod_s2 = pvl.temperature.pvsyst_cell(poa_all.A1_poa, meteo.Temperature,
                                      meteo['Wind Speed'], u_c=25.7, u_v=1.1)

tmod_s3 = pvl.temperature.pvsyst_cell(poa_all.A2_poa, meteo.Temperature,
                                      meteo['Wind Speed'], u_c=25.7, u_v=1.1)

tmod_s5 = pvl.temperature.pvsyst_cell(poa_all.A4_poa, meteo.Temperature,
                                      meteo['Wind Speed'], u_c=25.7, u_v=1.1)

tmod_s6 = pvl.temperature.pvsyst_cell(poa_all.A5_poa, meteo.Temperature,
                                      meteo['Wind Speed'], u_c=25.7, u_v=1.1)

tmod_all = pd.concat([tmod, tmod_s2, tmod_s3, tmod_s5, tmod_s6], axis=1)

tmod_all.columns = ['reference_tmod',
                    'A1_tmod', 'A2_tmod', 'A4_tmod', 'A5_tmod']
tmod_all.plot(marker='o', grid=True)

# %% results summaries and pv array power

columns = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'GHI', 'DNI', 'DHI',
           'Temperature', 'Relative Humidity', 'Wind Speed', 'Gpoa South',
           'Rpoa South', 'Tmod South', 'Gpoa Mid', 'Rpoa Mid', 'Tmod Mid',
           'Gpoa North', 'Rpoa North', 'Tmod North', 'Theta']


def getRpoa(scenario):
    """
    scenario is an int value
    """
    south = rpoa_avg.loc[(rpoa_avg.scenario == scenario) &
                         (rpoa_avg.module == 1)]['Wm2Back']
    mid = rpoa_avg.loc[(rpoa_avg.scenario == scenario) &
                       (rpoa_avg.module == 13)]['Wm2Back']
    north = rpoa_avg.loc[(rpoa_avg.scenario == scenario) &
                         (rpoa_avg.module == 25)]['Wm2Back']
    results = pd.concat([south, mid, north], axis=1)
    results.columns = ['Rpoa South', 'Rpoa Mid', 'Rpoa North']
    results.index = results.index - pd.Timedelta(minutes=30)
    results.index = results.index.tz_localize('Etc/GMT+7')
    return results


s1, s2, s3, s4, s5, s6, s7 = [getRpoa(s+1) for s in range(7)]

fig, axs = plt.subplots(nrows=7, sharex=True)
for i, c in enumerate([s1, s2, s3, s4, s5, s6, s7]):
    c.plot(ax=axs[i], marker='o')
    axs[i].set_ylabel(f'scenario {i+1}')
    axs[i].legend()

# %%s1
s1 = s1.join([poa_all.reference_poa, tmod_all.reference_tmod,
             tracking_angles_all.reference_theta])

s1['Gpoa Mid'], s1['Gpoa North'] = s1.reference_poa, s1.reference_poa
s1['Tmod Mid'], s1['Tmod North'] = s1.reference_tmod, s1.reference_tmod

s1 = s1.rename(columns={'reference_poa': 'Gpoa South',
                        'reference_tmod': 'Tmod South',
                        'reference_theta': 'Theta'})

s1 = s1.join(meteo)
s1 = s1[columns]
# %%s2
s2 = s2.join([poa_all.A1_poa, tmod_all.A1_tmod,
             tracking_angles_all.A1_theta])

s2['Gpoa Mid'], s2['Gpoa North'] = s2.A1_poa, s2.A1_poa
s2['Tmod Mid'], s2['Tmod North'] = s2.A1_tmod, s2.A1_tmod

s2 = s2.rename(columns={'A1_poa': 'Gpoa South',
                        'A1_tmod': 'Tmod South',
                        'A1_theta': 'Theta'})

s2 = s2.join(meteo)
s2 = s2[columns]
# %%s3
s3 = s3.join([poa_all.A2_poa, tmod_all.A2_tmod,
             tracking_angles_all.reference_theta])

s3['Gpoa Mid'], s3['Gpoa North'] = s3.A2_poa, s3.A2_poa
s3['Tmod Mid'], s3['Tmod North'] = s3.A2_tmod, s3.A2_tmod

s3 = s3.rename(columns={'A2_poa': 'Gpoa South',
                        'A2_tmod': 'Tmod South',
                        'reference_theta': 'Theta'})

s3 = s3.join(meteo)
s3 = s3[columns]
# %%s4
s4 = s4.join([poa_all.reference_poa, tmod_all.reference_tmod,
             tracking_angles_all.reference_theta])

s4['Gpoa Mid'], s4['Gpoa North'] = s4.reference_poa, s4.reference_poa
s4['Tmod Mid'], s4['Tmod North'] = s4.reference_tmod, s4.reference_tmod

s4 = s4.rename(columns={'reference_poa': 'Gpoa South',
                        'reference_tmod': 'Tmod South',
                        'reference_theta': 'Theta'})

s4 = s4.join(meteo)
s4 = s4[columns]
# %%s5
s5 = s5.join([poa_all.A4_poa, tmod_all.A4_tmod,
             tracking_angles_all.A4_theta])

s5['Gpoa Mid'], s5['Gpoa North'] = s5.A4_poa, s5.A4_poa
s5['Tmod Mid'], s5['Tmod North'] = s5.A4_tmod, s5.A4_tmod

s5 = s5.rename(columns={'A4_poa': 'Gpoa South',
                        'A4_tmod': 'Tmod South',
                        'A4_theta': 'Theta'})

s5 = s5.join(meteo)
s5 = s5[columns]
# %% s6
s6 = s6.join([poa_all.A5_poa, tmod_all.A5_tmod,
             tracking_angles_all.A5_theta])

s6['Gpoa Mid'], s6['Gpoa North'] = s6.A5_poa, s6.A5_poa
s6['Tmod Mid'], s6['Tmod North'] = s6.A5_tmod, s6.A5_tmod

s6 = s6.rename(columns={'A5_poa': 'Gpoa South',
                        'A5_tmod': 'Tmod South',
                        'A5_theta': 'Theta'})

s6 = s6.join(meteo)
s6 = s6[columns]
# %%s7
s7 = s7.join([poa_all.reference_poa, tmod_all.reference_tmod,
             tracking_angles_all.reference_theta])

s7['Gpoa Mid'], s7['Gpoa North'] = s7.reference_poa, s7.reference_poa
s7['Tmod Mid'], s7['Tmod North'] = s7.reference_tmod, s7.reference_tmod

s7 = s7.rename(columns={'reference_poa': 'Gpoa South',
                        'reference_tmod': 'Tmod South',
                        'reference_theta': 'Theta'})

s7 = s7.join(meteo)
s7 = s7[columns]

# %% max power determination

coeff = (0.85914, -0.020880, -0.0058853,
         0.12029, 0.026814, -0.0017810)  # monoSi FSLR coeffs


def calcPower(df, scenario, nMods=25):
    """
    applies IAM, spectral and soiling losses

    """
    df = df.join(solpos[['zenith', 'azimuth']])
    if (scenario == 1) | (scenario == 3) | (scenario == 4) | (scenario == 7):
        df = df.join(tracking_angles['surface_azimuth'])
    elif scenario == 2:
        df = df.join(tracking_angles_s2['surface_azimuth'])
    elif scenario == 5:
        df = df.join(tracking_angles_s5['surface_azimuth'])
    elif scenario == 6:
        df = df.join(tracking_angles_s6['surface_azimuth'])
    else:
        print('select 1 to 7')
    df['aoi'] = pvl.irradiance.aoi(
        abs(df.Theta), df.surface_azimuth, df.zenith, df.azimuth)
    df['iam'] = pvl.iam.martin_ruiz(df.aoi, a_r=0.15)
    df['iam-diff'] = pvl.iam.martin_ruiz_diffuse(abs(s1.Theta), a_r=0.15)[0]
    df['pw'] = pvl.atmosphere.gueymard94_pw(
        df['Temperature'], df['Relative Humidity'])
    df['am'] = pvl.atmosphere.get_relative_airmass(df.zenith)
    df['smmf'] = (coeff[0] + coeff[1]*df['am'] + coeff[2]*df['pw'] + coeff[3]*np.sqrt(df['am']) +
                  coeff[4]*np.sqrt(df['pw']) + coeff[5]*df['am']/np.sqrt(df['pw']))
    # iam should be applied to diffuse and beam seperately,
    # apply iam to global (for simplicity),
    # but... during clear sky, apply beam iam, and during cloudy apply diffuse iam
    # 0.5% for soiling losses
    df['Ee'] = np.where(df.DHI/df.GHI < 0.5,
                        df['Gpoa South']*df['smmf']*df['iam']*(1-0.005),
                        df['Gpoa South']*df['smmf']*df['iam-diff']*(1-0.005))

    df['Ee'] = df['Ee'] + df['Rpoa Mid']*BifacialityFactor  # add the mid Rpoa

    # 2) calculate 5 parameters
    iv_params = pvl.pvsystem.calcparams_pvsyst(df['Ee'], 25, muISC/1000, Gamma,
                                               muGamma, Isc, 2.5e-10, 20000, Rp_0, RSerie, NCelS*nMods)

    df['Iph'], df['I0'], df['Rs'], df['Rp'], df['nNsVth'] = iv_params[0], iv_params[1], iv_params[2], iv_params[3], iv_params[4]

    # 3) calculate Pmax
    df['Pmax'] = pvl.pvsystem.max_power_point(
        df['Iph'], df['I0'], df['Rs'], df['Rp'], df['nNsVth'])['p_mp']

    # adjust for temperature
    df['Pmax'] = df['Pmax']*(1 + (muPmpReq/100)*(df['Tmod South'] - 25))

    plt.scatter(x=df['Gpoa South'].values, y=df['Pmax'], c=df['Tmod South'])
    plt.colorbar()
    plt.ylabel(f'Power (W) for S{scenario}')

    return df['Pmax']


s1['Pmax'] = calcPower(s1, 1)
s2['Pmax'] = calcPower(s2, 2)
s3['Pmax'] = calcPower(s3, 3)
s4['Pmax'] = calcPower(s4, 4)
s5['Pmax'] = calcPower(s5, 5)
s6['Pmax'] = calcPower(s6, 6)
s7['Pmax'] = calcPower(s7, 7, nMods=50)

for i, df in enumerate([s1, s2, s3, s4, s5, s6, s7]):
    df.to_csv(PATH+f'\scenario_{i+1}.csv', index=True)

# s1['Eff'] = s1['Pmax']/((s1['Gpoa South']+s1['Rpoa South'])*Height*Width*25)

# plt.scatter(x=s1['Gpoa South'], y=s1.Eff, c=s1['Tmod South'])
