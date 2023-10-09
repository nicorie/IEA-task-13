# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 12:52:48 2023

@author: nri
"""

import bifacial_radiance
import pandas as pd
import numpy as np
import os

shapes = ['Square', 'Round']
heights = [1.0, 1.5, 2.0]
gcrs = [0.3, 0.4, 0.5]
albedos = [0.1, 0.2, 0.3, 0.4]

lat, lon = 55.6, 12.1  # DTU Risø

sensorsy = 21  # number of points to sample

limit_angle = 60
angledelta = 5  # sampling between the limit angles.
diameter = 0.145  # torque tube diameter
backtrack = True
cumulativesky = True

nMods, nRows = 22, 5
panelx, panely = 1.3, 2.1
moduletype = 'Big-Module'
zgap, ygap, xgap = 0.115, 0, 0

materialpav = 'Tracker_Steel'
Rrefl, Grefl, Brefl = 0.3, 0.25, 0.2

testfolder = os.path.abspath(r'C:\Users\NRI\Bifacial_Radiance_Files')


# %% add piles


# %%
structure = False

for shape in shapes:
    for height in heights:
        for gcr in gcrs:
            for albedo in albedos:
                print(f'simulating {shape} TT, \n'
                      f'{height} m height, \n'
                      f'{gcr} gcr, \n'
                      f'{albedo} albedo.')

                tracker = bifacial_radiance.RadianceObj('IEA13_Sensitivity',
                                                        path=testfolder)

                epwfile = tracker.getEPW(lat=lat, lon=lon)

                metdata = tracker.readWeatherFile(weatherFile=epwfile)

                tracker.addMaterial(material=materialpav, Rrefl=Rrefl,
                                    Grefl=Grefl, Brefl=Brefl, specularity=0.03,
                                    roughness=0.15)

                mymodule = tracker.makeModule(name=moduletype, x=panelx,
                                              y=panely, xgap=xgap, ygap=ygap,
                                              zgap=zgap, numpanels=1,
                                              modulematerial='pvblue')

                mymodule.addTorquetube(diameter=diameter, tubetype=shape,
                                       material='Tracker_Steel')

                if structure:
                    mymodule.addFrame(frame_z=0.035, frame_material='Alu')

                    mymodule.addOmega(y_omega=panely*0.5, x_omega1=0.04,
                                      omega_material='Alu')

                pitch = panely/gcr

                tracker.setGround(albedo)

                trackerdict = tracker.set1axis(metdata=metdata,
                                               limit_angle=limit_angle,
                                               backtrack=backtrack,
                                               gcr=gcr,
                                               cumulativesky=cumulativesky,
                                               angledelta=angledelta)

                trackerdict = tracker.genCumSky1axis(trackerdict)  # create sky

                sceneDict = {'pitch': pitch, 'hub_height': height,
                             'nMods': nMods, 'nRows': nRows}

                trackerdict = tracker.makeScene1axis(trackerdict=trackerdict,
                                                     module=mymodule,
                                                     sceneDict=sceneDict)

                trackerkeys = sorted(trackerdict.keys())

                # %% piles
                if structure:
                    tubelength = mymodule.scenex * nMods
                    pile_offsetsy = [-0.45*tubelength, -0.25*tubelength,
                                     0, 0.25 * tubelength, 0.45*tubelength]
                    pile_offsetsx = [pitch*-2, pitch*-1, 0, pitch, pitch*2]

                    name = 'Piles'
                    text = '! genbox Tracker_Steel pile{}row{} 0.18 0.08 {} | xform -t {} {} 0'.format(
                        1, 1, height, pile_offsetsx[0]-0.09, pile_offsetsy[0])
                    text += '\r\n! genbox Tracker_Steel pile{}row{} 0.18 0.08 {} | xform -t {} {} 0'.format(
                        1, 2, height, pile_offsetsx[1]-0.09, pile_offsetsy[0])
                    text += '\r\n! genbox Tracker_Steel pile{}row{} 0.18 0.08 {} | xform -t {} {} 0'.format(
                        1, 3, height, pile_offsetsx[2]-0.09, pile_offsetsy[0])
                    text += '\r\n! genbox Tracker_Steel pile{}row{} 0.18 0.08 {} | xform -t {} {} 0'.format(
                        1, 4, height, pile_offsetsx[3]-0.09, pile_offsetsy[0])
                    text += '\r\n! genbox Tracker_Steel pile{}row{} 0.18 0.08 {} | xform -t {} {} 0'.format(
                        1, 5, height, pile_offsetsx[4]-0.09, pile_offsetsy[0])

                    for t in trackerkeys:
                        for i in range(1, 6):  # tracker row wise (x-direction)
                            for j in range(1, 5):  # tracker pile wise (y-direction)
                                text += '\r\n! genbox Tracker_Steel pile{}row{} 0.18 0.08 {} | xform -t {} {} 0'.format(
                                    j+1, i, height, pile_offsetsx[i-1], pile_offsetsy[j])

                        customObject = tracker.makeCustomObject(name, text)
                        tracker.appendtoScene(
                            trackerdict[t]['radfile'], customObject, '!xform -rz 0')

                # %% make oct files and run sim

                customname = f'_{height}_{gcr}_{shape}_{albedo}_{structure}'

                # trackerdict[30.0]['scene'].showScene()

                trackerdict = tracker.makeOct1axis(trackerdict=trackerdict)

                trackerdict = tracker.analysis1axis(trackerdict, modWanted=11,
                                                    rowWanted=3,
                                                    customname=customname,
                                                    sensorsy=sensorsy)
