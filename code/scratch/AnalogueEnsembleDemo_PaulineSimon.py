# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:53:08 2023

@author: vy902033
"""
import astropy.units as u
from astropy.time import Time, TimeDelta
import numpy as np
import datetime
import os as os
from scipy import interpolate
from scipy.ndimage.filters import uniform_filter1d
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import percentileofscore

import helio_time as htime

datadir = 'C:\\Users\\vy902033\\Dropbox\\Data_hdf5\\'




omni_1hour = pd.read_hdf(datadir + 'omni_1hour.h5')

# <codecell> plot the single CR
CR = 2231

smjd = htime.crnum2mjd(CR)
fmjd = smjd + 27.27
mask_thisCR = (omni_1hour['mjd'] >= smjd) & (omni_1hour['mjd'] < fmjd)

v_thisCR = omni_1hour['V'][mask_thisCR].values
v_nonans = omni_1hour['V'][omni_1hour['V'].notna()].values
v_L = len(v_nonans)

fig = plt.figure()

time = omni_1hour['mjd'][mask_thisCR].values
time = time - time[0]

vlims = (250,750)
blims = (-10,10)

ax = plt.subplot(211)
ax.plot(time, v_thisCR, 'k', )
ax.set_title('CR' +str(CR))
ax.set_ylim((250,750))
#ax.legend(loc = 'upper right')
ax.set_ylabel(r'$V$ [km/s]')
ax.text(0.05,0.9,'(a)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')

vpercentiles = np.ones((len(v_thisCR),1))

for n in range(0, len(v_thisCR)):
    nbelow = np.sum(v_nonans < v_thisCR[n])
    vpercentiles[n] = 100*nbelow/v_L

ax = plt.subplot(212)
ax.plot(time, vpercentiles, 'k', )
ax.set_title('CR' +str(CR))
#ax.legend(loc = 'upper right')
ax.set_ylabel(r'$V$ precentiles')
ax.text(0.05,0.9,'(a)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
