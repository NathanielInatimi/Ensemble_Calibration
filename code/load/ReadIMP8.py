# -*- coding: utf-8 -*-
"""
A reader for IMP8 data

@author: mathewjowens
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
# these are custom functions which need to be in the same directory or the python path
os.chdir(os.path.abspath(os.environ['DBOX'] + '\\python'))
import helio_coords as hcoord
import helio_time as htime 

imp8dir='D:\\Dropbox\\Data\\IMP8\\'
startyr=1973
stopyr=2006

imp8_1hour = pd.DataFrame()

yr = 1983

for yr in range(startyr,stopyr+1):
    print('Loading ' + str(yr))
    
    plasmafile = imp8dir + 'plasma\\imp8.' + str(yr) + '.hravg.subset'
    
    
    temp = pd.read_csv(plasmafile,
                         skiprows = 4, delim_whitespace=True, index_col=False,
                         names=['year','doy', 'hr', 'dec',
                                'V', 'Va', 'Vb',
                                'Vx','Vxerr','Vy','Vyerr', 'Vz', 'Vzerr',
                                'Vth','Vtha','Vthb',
                                'n'])
    
    
    
    plasma = pd.DataFrame()
    
    plasma['datetime'] = pd.to_datetime(temp['year'] * 1000 + temp['doy'], format='%Y%j')
    plasma['datetime'] = plasma['datetime'] + pd.to_timedelta(temp['hr'], unit='h')
    #plasma['datetime'] = htime.mjd2datetime(plasma['mjd'].to_numpy())
    
    plasma['Vx'] = temp['Vx']
    plasma['Vy'] = temp['Vy']
    plasma['Vz'] = temp['Vz']
    plasma['Vth'] = temp['Vth']
    plasma['n'] = temp['n']
    
    #remove bad data
    mask = (plasma['Vx'] > -1.0)
    plasma['Vx'][mask] = np.nan
    plasma['Vy'][mask] = np.nan
    plasma['Vz'][mask] = np.nan
    plasma['Vth'][mask] = np.nan
    plasma['n'][mask] = np.nan
    
    
    
    
    magfile = imp8dir + 'mag\\imp_mag_hr_sw' + str(yr) + '.asc'
    
    temp = pd.read_csv(magfile,
                         skiprows = 0, delim_whitespace=True, index_col=False,
                         names=['year','doy', 'hr', 'dec',
                                '5', '6', '7',
                                '8','9','B','11', '12', '13',
                                'Bx','By','Bz'])
    
    mag = pd.DataFrame()
    #mag['datetime'] = htime.mjd2datetime(mag['mjd'].to_numpy())
    mag['datetime'] = pd.to_datetime(temp['year'] * 1000 + temp['doy'], format='%Y%j')
    mag['datetime'] = mag['datetime'] + pd.to_timedelta(temp['hr'], unit='h')
    
    mag['B'] = temp['B']
    mag['Bx'] = temp['Bx']
    mag['By'] = temp['By']
    mag['Bz'] = temp['Bz']
    
    #remove bad data
    mask = (mag['B'] > 999)
    mag['B'][mask] = np.nan
    mag['Bx'][mask] = np.nan
    mag['By'][mask] = np.nan
    mag['Bz'][mask] = np.nan
    
    
    merged = pd.merge_asof(mag, plasma, on='datetime', tolerance=pd.Timedelta("1H"))
    merged = merged.set_index('datetime')
    merged = merged.resample('1H').mean()
    #merged = merged.reset_index()
    
    #add it to the existing dataframe
    imp8_1hour = pd.concat([imp8_1hour,merged])

plt.plot(-imp8_1hour['Vx'])

