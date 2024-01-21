# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 12:52:48 2023

This script is modified from geoloop_sensitivity.py
The main change is to run gendaylit instead of gencumsky
Loop through: 
    system types
    locations

TODO: be crystal clear about what 0-100% means
    eg., does the 5th point from west really mean 50% between TT and edge
    BR doesnt put sensors on the absolute module edges
    eg., for 21 sensors and module len 2.1m, distance between sensors is 2.1/22 = 0.095 between points
    so the first sensor is 0.095 m from the edge, not on the module edge.
    

@author: nri
"""

import bifacial_radiance
import numpy as np
import os

shape = 'Square'
gcr = 0.43
albedo = 0.2
system_setups = [1, 2, 3]
locations = [(31.2, 29.95), (36.72, 3.25), (28.08, 30.73),
             (23.97, 32.78), (27.05, 31.02)]

# lat, lon = 55.6, 12.1  # DTU Risø

sensorsy = 21  # number of points to sample

limit_angle = 60
diameter = 0.145  # torque tube diameter
backtrack = True
cumulativesky = False

nMods, nRows = 22, 5
panelx, panely = 1.3, 2.1
moduletype = 'Big-Module'
zgap, ygap, xgap = 0.115, 0, 0

testfolder = os.path.abspath(r'C:\Users\NRI\Bifacial_Radiance_Files')

# %%
structure = False
for loc in locations:
    for system_setup in system_setups:
        if system_setup == 1:  # 1P with no TT gap
            nMods, numpanels, zgap, ygap, xgap = 22, 1, 0.115, 0, 0
            height = 1.65-0.115-(0.5*0.145)

        if system_setup == 2:  # 2P with no TT gap
            nMods, numpanels, zgap, ygap, xgap = 44, 2, 0.115, 0, 0
            height = 2.55-0.115-(0.5*0.145)

        if system_setup == 3:
            nMods, numpanels, zgap, ygap, xgap = 44, 2, 0.115, 0.145, 0
            height = 2.61-0.115-(0.5*0.145)

        print(f'simulating system {system_setup}, \n'
              f'lat: {loc[0]}, lon: {loc[1]}, \n'
              f'{shape} TT, \n'
              f'{height} m height, \n'
              f'{gcr} gcr, \n'
              f'{albedo} albedo.')

        tracker = bifacial_radiance.RadianceObj('IEA13_Sensitivity',
                                                path=testfolder)

        epwfile = tracker.getEPW(lat=loc[0], lon=loc[1])

        metdata = tracker.readWeatherFile(weatherFile=epwfile)

        mymodule = tracker.makeModule(name=moduletype, x=panelx,
                                      y=panely, xgap=xgap, ygap=ygap,
                                      zgap=zgap, numpanels=numpanels,
                                      modulematerial='pvblue')

        mymodule.addTorquetube(diameter=diameter, tubetype=shape,
                               material='Tracker_Steel')

        # extend the torque tube by 1m in north and 1m in south
        rad_file = testfolder+'\objects'+f'\{moduletype}.rad'

        with open(rad_file, 'r') as file:
            data = file.readlines()
            # extend the TT by 1 m in N-S direction
            if shape == 'Square':
                data[1] = f'! genbox Tracker_Steel tube1 {panelx+2} {diameter} {diameter} | xform -t {-panelx/2 - 1} {-diameter/2} {-diameter/2}'
            elif shape == 'Round':
                data[1] = f'! genrev Tracker_Steel tube1 t*{panelx+2} {diameter/2} 32 | xform -ry 90 -t {-panelx/2 - 1} 0 0'

        with open(rad_file, 'w') as file:
            file.writelines(data)

        if structure:
            mymodule.addFrame(frame_z=0.035, frame_material='Alu')

            mymodule.addOmega(y_omega=panely*0.5, x_omega1=0.04,
                              omega_material='Tracker_Steel')

        pitch = panely*numpanels/gcr

        tracker.setGround(albedo)

        trackerdict = tracker.set1axis(metdata=metdata,
                                       limit_angle=limit_angle,
                                       backtrack=backtrack,
                                       gcr=gcr,
                                       cumulativesky=cumulativesky)

        trackerdict = tracker.gendaylit1axis(metdata=metdata,
                                             trackerdict=trackerdict)  # create sky

        sceneDict = {'pitch': pitch, 'hub_height': height,
                     'nMods': nMods, 'nRows': nRows}

        trackerdict = tracker.makeScene1axis(trackerdict=trackerdict,
                                             module=mymodule,
                                             sceneDict=sceneDict)

        trackerkeys = sorted(trackerdict.keys())

        # %% piles
        tubelength = mymodule.scenex * nMods

        pile_offsetsy = [-0.45*tubelength, -0.25*tubelength,
                         0, 0.25 * tubelength, 0.45*tubelength]

        pile_offsetsx = [pitch*-2, pitch*-1, 0, pitch, pitch*2]

        if structure:

            name = 'Piles'
            text = '! genbox Tracker_Steel pile{}row{} 0.14 0.08 {} | xform -t {} {} 0'.format(
                1, 1, height, pile_offsetsx[0]-0.07, pile_offsetsy[0])
            text += '\r\n! genbox Tracker_Steel pile{}row{} 0.14 0.08 {} | xform -t {} {} 0'.format(
                1, 2, height, pile_offsetsx[1]-0.07, pile_offsetsy[0])
            text += '\r\n! genbox Tracker_Steel pile{}row{} 0.14 0.08 {} | xform -t {} {} 0'.format(
                1, 3, height, pile_offsetsx[2]-0.07, pile_offsetsy[0])
            text += '\r\n! genbox Tracker_Steel pile{}row{} 0.14 0.08 {} | xform -t {} {} 0'.format(
                1, 4, height, pile_offsetsx[3]-0.07, pile_offsetsy[0])
            text += '\r\n! genbox Tracker_Steel pile{}row{} 0.14 0.08 {} | xform -t {} {} 0'.format(
                1, 5, height, pile_offsetsx[4]-0.07, pile_offsetsy[0])

            for i in range(1, 6):  # tracker row wise (x-direction)
                for j in range(1, 5):  # tracker pile wise (y-direction)
                    text += '\r\n! genbox Tracker_Steel pile{}row{} 0.14 0.08 {} | xform -t {} {} 0'.format(
                        j+1, i, height, pile_offsetsx[i-1]-0.07, pile_offsetsy[j])

            customObject = tracker.makeCustomObject(name, text)

            for t in trackerkeys:
                tracker.appendtoScene(
                    trackerdict[t]['radfile'], customObject, '!xform -rz 0')

        # %% make oct files and run sim
        #customname = f'_{height}_{gcr}_{shape}_{albedo}_{structure}'
        customname = f'_system_{system_setup}_{loc[0]}_{loc[1]}'
        customname = customname + '_outboard_gendaylit'

        # trackerdict[-10.0]['scene'].showScene()
        # trackerdict['2021-04-30_1200']['scene'].showScene()

        trackerdict = tracker.makeOct1axis(trackerdict=trackerdict)

        for t in trackerkeys:
            analysis = bifacial_radiance.AnalysisObj(
                trackerdict[t]['octfile'], tracker.basename)

            frontscan, backscan = analysis.moduleAnalysis(
                trackerdict[t]['scene'], sensorsy=sensorsy)

            linepts = analysis._linePtsMakeDict(backscan)
            linepts_front = analysis._linePtsMakeDict(frontscan)
            # modify the line scans
            # custom function modifyLineScan() in AnalysisObj class
            linepts_back_mod, linepts_front_mod = analysis.modifyLineScan(
                linepts, linepts_front, sensorsy, trackerdict[t])

            # do the rtrace, backDict contains the rgb results
            backDict = analysis._irrPlot(trackerdict[t]['octfile'],
                                         linepts_back_mod, customname+'_Back')

            frontDict = analysis._irrPlot(trackerdict[t]['octfile'],
                                          linepts_front_mod, customname+'_Front')

            # saves results to analysis object and to csv
            if frontDict is not None:
                if len(frontDict['Wm2']) != len(backDict['Wm2']):
                    analysis.Wm2Front = np.mean(frontDict['Wm2'])
                    analysis.Wm2Back = np.mean(backDict['Wm2'])
                    analysis.backRatio = analysis.Wm2Back / \
                        (analysis.Wm2Front + .001)
                    analysis._saveResults(
                        frontDict, reardata=None, savefile='irr%s.csv' % (customname+f'_Front_{str(t)}'))
                    analysis._saveResults(
                        data=None, reardata=backDict, savefile='irr%s.csv' % (customname+f'_Back_{str(t)}'))
                else:
                    analysis._saveResults(
                        frontDict, backDict, 'irr%s.csv' % (customname+f'_{str(t)}'))
