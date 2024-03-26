# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 13:27:08 2021

@author: mathewjowens
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:00:35 2017

Script to produce a heliospheric flux summary

@author: vy902033
"""

import datetime as datetime
import os as os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
os.chdir(os.path.abspath(os.environ['DBOX'] + '\\python'))
import helio_coords as hcoord
import helio_time as htime 

data_dir = os.environ['DBOX'] + 'Data_hdf5\\'
save_dir = os.environ['DBOX'] + 'Papers_WIP\\_coauthor\\ManuelaTemmer\\'
br_window = '20H'
rolling_window_days_long = 365
rolling_window_days_short = 27


plt.rcParams.update({'font.size': 16})
hfont = {'fontname':'Tahoma'}


AU=149598000

#load the data
omni_1hour = pd.read_hdf(data_dir + 'omni_1hour.h5')

omni_brwindow_nans = omni_1hour.resample(br_window, on='datetime').mean() 
#omni_brwindow_nans['datetime'] = htime.mjd2datetime(omni_brwindow_nans['mjd'].to_numpy())
#omni_brwindow_nans.reset_index(drop=True, inplace=True)

#compute the constant to convert |Br| at 1 AU to total HMF
Fconst=(1e-3)*4*np.pi*AU*AU/(1e14)

omni_brwindow_nans['OSF (x10^14 Wb)'] = np.abs(omni_brwindow_nans['Bx_gse']) * Fconst

osf_1d = pd.DataFrame(index = omni_brwindow_nans.index)
osf_1d['OSF (x10^14 Wb)'] = omni_brwindow_nans['OSF (x10^14 Wb)']
osf_1d['MJD'] = omni_brwindow_nans['mjd']




#dowload the sunspot data from http://www.sidc.be/silso/DATA/SN_m_tot_V2.0.csv
filepath= os.environ['DBOX'] + 'Data\\SN_m_tot_V2.0.csv'
col_specification =[(0, 4), (5, 7), (8,16),(17,23),(24,29),(30,35)]
ssn=pd.read_fwf(filepath, colspecs=col_specification,header=None)
dfdt=[]
for i in range(0,len(ssn)):
    dfdt.append(datetime.datetime(ssn[0][i],ssn[1][i],15))
#replace the index with the datetime objects
ssn.index=dfdt
ssn['ssn']=ssn[3]

#TSO data from https://spot.colorado.edu/~koppg/TSI/TSI_Composite-SIST.txt
filepath= os.environ['DBOX'] + 'Data\\TSI_Composite-SIST.txt'
tsi = pd.read_csv(filepath, delim_whitespace=True, header = 35, names=['TSI', 'dTSI'])  
dfdt = htime.jd2datetime(tsi.index.to_numpy()) 
#replace the index with the datetime objects
tsi.index=dfdt



hour_str_long = str(rolling_window_days_long*24) +'H'
hour_str_short = str(rolling_window_days_short*24) +'H'

omniavg_long=osf_1d.rolling(hour_str_long, min_periods=int(rolling_window_days_long/4)).mean()
omniavg_long = omniavg_long.reset_index()
omniavg_long['datetime'] = omniavg_long['datetime'] - datetime.timedelta(days=rolling_window_days_long/2)

omniavg_short=osf_1d.rolling(hour_str_short, min_periods=int(rolling_window_days_short/4)).mean()
omniavg_short= omniavg_short.reset_index()
omniavg_short['datetime'] = omniavg_short['datetime'] - datetime.timedelta(days=rolling_window_days_short/2)

ssnavg=ssn.rolling(hour_str_long, min_periods=int(rolling_window_days_long//30/4)).mean()
ssnavg.index = ssnavg.index - datetime.timedelta(days=rolling_window_days_long/2)

tsiavg=tsi.rolling(hour_str_long, min_periods=int(rolling_window_days_long//30/4)).mean()
tsiavg.index = tsiavg.index - datetime.timedelta(days=rolling_window_days_long/2)


import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator




years = mdates.YearLocator() 
yrs5 = mdates.YearLocator(5) 
minor_locator = AutoMinorLocator(2)

#plot total heliospheric flux

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

d = ssnavg.index.values
ax1.fill_between(d,0,ssnavg['ssn'],facecolor='silver')
ax1.set_ylabel('Sunspot number', fontsize=16,**hfont)
ax1.set_xlim(omniavg_long['datetime'][0], 
             omniavg_long['datetime'][len(omniavg_long)-1] + datetime.timedelta(days=rolling_window_days_long/2))
ax1.set_ylim(0,250)

ax1.yaxis.set_label_position('left')
ax1.yaxis.set_ticks_position('left')
#ax1.minorticks_on()


plotcol='blue'
plotcol_short = 'dodgerblue'
ax2.plot(omniavg_short['datetime'],omniavg_short['OSF (x10^14 Wb)'],plotcol_short)
ax2.plot(omniavg_long['datetime'],omniavg_long['OSF (x10^14 Wb)'],plotcol)
ax2.set_ylabel('Total Heliospheric Flux \n[$x10^{14}$ Wb]', color=plotcol, fontsize=16,**hfont)
ax2.tick_params('y', colors=plotcol)
ax2.yaxis.set_ticks_position('right')
ax2.yaxis.set_label_position('right')
ax2.set_xlim(omniavg_long['datetime'][0], 
             omniavg_long['datetime'][len(omniavg_long)-1] + datetime.timedelta(days=rolling_window_days_long/2))
ax2.minorticks_on()
ax2.xaxis.set_minor_locator(years)
#ax2.xaxis.set_major_locator(yrs5)
ax2.yaxis.set_minor_locator(minor_locator)
ax2.set_ylim(0,15)

#ax2.grid(True,  which='major')
#ax1.xaxis.grid(True)
#ax1.set_axisbelow(True)
#ax2.set_axisbelow(True)


#ax2.set_xlabel('Year', fontsize=16,**hfont)
#ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#fig.autofmt_xdate()
plt.tight_layout()


plt.savefig(save_dir + 'Hflux.png', format='png', dpi=600)
omniavg_short.to_csv(save_dir + 'OMNI_OSF.dat')

#
#
#
##Plot field magnitude
#
#
#
#fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()
#
#d = ssnavg.index.values
#ax1.fill_between(d,0,ssnavg['ssn'],facecolor='silver')
#ax1.set_ylabel('Sunspot number', fontsize=16,**hfont)
#ax1.set_xlim(starttime,  datetime(2018,6,1))
#ax1.set_ylim(0,230)
#
#ax1.yaxis.set_label_position('right')
#ax1.yaxis.set_ticks_position('right')
#ax1.minorticks_on()
#
#
#plotcol='blue'
#ax2.plot(omniavg['Bmag'],plotcol)
#ax2.set_ylabel('B [nT]', color=plotcol, fontsize=16,**hfont)
#ax2.tick_params('y', colors=plotcol)
#ax2.yaxis.set_ticks_position('left')
#ax2.yaxis.set_label_position('left')
#ax2.set_xlim(starttime, datetime(2018,6,1))
#ax2.minorticks_on()
#ax2.xaxis.set_minor_locator(years)
#ax2.xaxis.set_major_locator(yrs5)
#ax2.yaxis.set_minor_locator(minor_locator)
#ax2.set_xlabel('Year', fontsize=16,**hfont)
##fig.autofmt_xdate()
#
#plt.savefig('Bmag.png', format='png', dpi=600)
#
##insituplot.quickplot(starttime,endtime,spacecraft,res)


fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

d = ssnavg.index.values
ax1.fill_between(d,0,ssnavg['ssn'],facecolor='silver')
ax1.set_ylabel('Sunspot number', fontsize=16,**hfont)
ax1.set_xlim(tsiavg.index[0] - datetime.timedelta(days=rolling_window_days_long/2), 
             tsiavg.index[len(tsiavg)-1] + datetime.timedelta(days=rolling_window_days_long/2))
ax1.set_ylim(0,250)

ax1.yaxis.set_label_position('left')
ax1.yaxis.set_ticks_position('left')
#ax1.minorticks_on()


plotcol='blue'
plotcol_short = 'dodgerblue'
ax2.plot(tsiavg['TSI'],plotcol)
ax2.set_ylabel(r'Total solar irradiance [W m$^{-2}$]', color=plotcol, fontsize=16,**hfont)
ax2.tick_params('y', colors=plotcol)
ax2.yaxis.set_ticks_position('right')
ax2.yaxis.set_label_position('right')
ax2.set_xlim(tsiavg.index[0] - datetime.timedelta(days=rolling_window_days_long/2), 
             tsiavg.index[len(tsiavg)-1] + datetime.timedelta(days=rolling_window_days_long/2))
ax2.minorticks_on()
ax2.xaxis.set_minor_locator(years)
#ax2.xaxis.set_major_locator(yrs5)
ax2.yaxis.set_minor_locator(minor_locator)
#ax2.set_ylim(0,15)
 
plt.tight_layout()


plt.savefig(save_dir + 'TSI_zoom.png', format='png', dpi=600)




fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

d = ssnavg.index.values
ax1.fill_between(d,0,ssnavg['ssn'],facecolor='silver')
ax1.set_ylabel('Sunspot number', fontsize=16,**hfont)
ax1.set_xlim(tsiavg.index[0] - datetime.timedelta(days=rolling_window_days_long/2), 
             tsiavg.index[len(tsiavg)-1] + datetime.timedelta(days=rolling_window_days_long/2))
ax1.set_ylim(0,250)

ax1.yaxis.set_label_position('left')
ax1.yaxis.set_ticks_position('left')
#ax1.minorticks_on()


plotcol='blue'
plotcol_short = 'dodgerblue'
ax2.plot(tsiavg['TSI'],plotcol)
ax2.set_ylabel(r'Total solar irradiance [W m$^{-2}$]', color=plotcol, fontsize=16,**hfont)
ax2.tick_params('y', colors=plotcol)
ax2.yaxis.set_ticks_position('right')
ax2.yaxis.set_label_position('right')
ax2.set_xlim(tsiavg.index[0] - datetime.timedelta(days=rolling_window_days_long/2), 
             tsiavg.index[len(tsiavg)-1] + datetime.timedelta(days=rolling_window_days_long/2))
ax2.minorticks_on()
ax2.xaxis.set_minor_locator(years)
#ax2.xaxis.set_major_locator(yrs5)
ax2.yaxis.set_minor_locator(minor_locator)
ax2.set_ylim(0,1400)
 
plt.tight_layout()


plt.savefig(save_dir + 'TSI.png', format='png', dpi=600)
