# -*- coding: utf-8 -*-
"""
Created on Thu May 30 20:35:05 2024

@author: nri
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

PATH = r'C:\Users\NRI\OneDrive - EE\Skrivebord\IEA PVPS\Task 13\Peer Review'
FILE = '\Locations.xlsx'

df = pd.read_excel(PATH+FILE)

m = Basemap(projection='merc', resolution='i')
# llcrnrlon=-11, llcrnrlat=35,
# urcrnrlon=33, urcrnrlat=60,
# lat_0=48, lon_0=14)


# convert lat and long to map projection coordinates
lons, lats = m(df.lon.values, df.lat.values)

m.scatter(lons, lats, marker='o', color='r', zorder=5)
m.drawcoastlines()
m.drawcountries()
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='white', lake_color='#46bcec')

plt.show()
