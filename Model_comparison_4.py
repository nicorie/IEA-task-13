# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 14:13:44 2023
IEA task 13 bifacial tracking modeling exercise
@author: nri
"""
import pandas as pd
import numpy as np
import pvlib as pvl
import bifacial_radiance
import os
import matplotlib.pyplot as plt
import pprint

PATH = r'C:\Users\NRI\OneDrive - EE\Skrivebord\IEA PVPS\Task 13\Blind Modeling Comparison'
FILE = '\Phase2_meteo_hourly_psm3format.csv'

scenario = 4
simulationName = 'IEA-A3'

days = ['2022-01-03', '2022-12-27', '2022-03-24', '2022-03-22', '2022-06-15',
        '2022-06-21']

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

AOIs = [0, 40, 50, 60, 70, 75, 80, 85, 90]
IAMs = [1, 1, 0.998, 0.992, 0.983, 0.961, 0.933, 0.853, 0]
# unrealistic iam

# plt.plot(AOIs, IAMs, marker='o', linestyle='-')
# plt.grid()

meteo = pd.read_csv(PATH+FILE, skiprows=2,
                    usecols=list(np.linspace(0, 10, 11, dtype=int)))

meteo.index = pd.to_datetime(meteo[['Year', 'Month', 'Day', 'Hour', 'Minute']])
meteo.index = meteo.index.tz_localize('Etc/GMT+7')

solpos = pvl.solarposition.get_solarposition(meteo.index, lat, lon)

# plt.scatter(solpos.azimuth, solpos.elevation)
# solpos['elevation'].plot()

# %% compare tracking angles
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

tracking_angles = pd.concat([tracking_angles['tracker_theta'],
                             tracking_angles_s2['tracker_theta'],
                             tracking_angles_s5['tracker_theta'],
                             tracking_angles_s6['tracker_theta']], axis=1)

# tracking_angles.columns = ['reference', 'A1', 'A4', 'A5']
# tracking_angles.plot(marker='o', grid=True)

# %% bifacial_radiance modeling

moduletype = 'Trina-Vertex'
testfolder = os.path.abspath(r'C:\Users\NRI\Bifacial_Radiance_Files')

TMY_file = r'EPWs\Phase2_meteo_hourly_tmyformat.csv'

albedo = albedos[scenario]
sensorsy_back = 10
sensorsy_front = 1

nMods = 25
nRows = 5
panelx = Width
panely = Height

hub_height = hub_heights[scenario]  # meters
gcr = gcrs[scenario]
pitch = panely/gcr

sceneDict = {'pitch': pitch, 'hub_height': hub_height,
             'nMods': nMods, 'nRows': nRows}

BeamsDict = {'length': Height*0.5, 'width': 0.02, 'thickness': 0.02}

# Tracking parameters
cumulativesky = False
limit_angle = rot  # tracker rotation limit angle
backtrack = True

xgap = 0.002  #
ygap = 0  # no torque tube gap in 1P
zgap = 0.02  # thickness of "transverse beams

numpanels = p_up[scenario]
torquetube = True
diameter = 0.150
tubetype = 'Round'  # Options: 'Square', 'Round' (default), 'Hex' or 'Oct'.


tracker = bifacial_radiance.RadianceObj(simulationName, path=testfolder)

tracker.setGround(albedo)

metdata = tracker.readWeatherFile(weatherFile=TMY_file, coerce_year=2022)

mymodule = tracker.makeModule(name=moduletype, x=panelx, y=panely, xgap=xgap,
                              ygap=ygap, zgap=zgap, numpanels=numpanels,
                              modulematerial='pvblue')

mymodule.addTorquetube(diameter=diameter, tubetype=tubetype,
                       material='Metal_Grey2')

mymodule.addFrame(frame_z=Depth, frame_material='Alu')

mymodule.addOmega(y_omega=Height*0.5, x_omega1=0.04,
                  omega_material='Metal_Grey2')

# cellLevelModuleParams=cellLevelModuleParams

trackerdict = tracker.set1axis(limit_angle=rot, backtrack=backtrack, gcr=gcr,
                               cumulativesky=cumulativesky)

# make sky files
trackerdict = tracker.gendaylit1axis(metdata=metdata, trackerdict=trackerdict)

trackerdict = tracker.makeScene1axis(trackerdict=trackerdict,
                                     moduletype=moduletype,
                                     sceneDict=sceneDict)

trackerdict['2022-04-30_1200']['scene'].showScene()

trackerkeys = sorted(trackerdict.keys())

t = []
for i in trackerkeys:  # is there a clever one liner for this?
    for j in days:
        if j in i:
            t.append(i)

trackerdict_sub = dict((k, trackerdict[k]) for k in t if k in trackerdict)


# %%
# make .oct files and run sim
# it takes ~1 min per oct file (not realistic for 7 scenarios)
trackerdict_sub = tracker.makeOct1axis(trackerdict=trackerdict_sub)

mod01 = tracker.analysis1axis(trackerdict=trackerdict_sub, accuracy='low',
                              customname=f'Scenario_{scenario}_Module_01',
                              modWanted=1, rowWanted=3,
                              sensorsy=[sensorsy_front, sensorsy_back])

mod13 = tracker.analysis1axis(trackerdict=trackerdict_sub, accuracy='low',
                              customname=f'Scenario_{scenario}_Module_13',
                              modWanted=13, rowWanted=3,
                              sensorsy=[sensorsy_front, sensorsy_back])

mod25 = tracker.analysis1axis(trackerdict=trackerdict_sub, accuracy='low',
                              customname=f'Scenario_{scenario}_Module_25',
                              modWanted=25, rowWanted=3,
                              sensorsy=[sensorsy_front, sensorsy_back])
