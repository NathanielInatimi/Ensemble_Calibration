# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:19:48 2022

A scrip to read in Matt Lang's processed MAVEN data and position'

@author: mathewjowens
"""


import pandas as pd
import datetime as datetime
import numpy as np

import helio_time as htime

datafilepath = 'D:\\Dropbox\\Data\\MAVEN\\maven1min4.csv'
ephemfilepath = 'D:\\Dropbox\\Data\\MAVEN\\mavenEarthDiff2013_2022.lst'

maven = pd.read_csv(datafilepath)
maven_ephem = pd.read_csv(ephemfilepath)

maven['mjd'] = htime.doyyr2mjd(maven['DoY'], maven['Year']) + maven['Hour']/24 \
    + maven['Min']/(24*60) + maven['Sec']/(24*60*60)
maven['datetime'] = htime.mjd2datetime(maven['mjd'].values)

#run a 10-min smooth through the maven data, then take the max value in 6-hour window
maven['Vsmooth'] = maven['Vr'].rolling(10).mean()
maven['Vsw'] = maven['Vsmooth'].rolling(6*60).min()

#remove low values and high values
mask = maven['Vsw'] > -200
maven.loc[mask, 'Vsw'] = np.nan
mask = maven['Vsw'] < -800
maven.loc[mask, 'Vsw'] = np.nan

#average up to 6 hour resolution
maven_6h = maven.resample('6H', on='datetime').min() 
maven_6h['datetime'] = maven_6h.index
maven_6h.reset_index(drop=True, inplace=True)