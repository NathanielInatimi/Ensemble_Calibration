# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:08:13 2017

A script to load and plot the Reading thunder data supplied by Roger Brugge

@author: mathewjowens
"""


from datetime import datetime
import os as os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



filepath= os.environ['DBOX'] + 'Data\\Climate_Met_data\\RogerBrugge\\reading_thunder_data.csv'

readingth=pd.read_csv(filepath,header=None)

#convert yr/mn/dy to datetime object
dfdt=[]
for i in range(0,len(readingth)):
    dfdt.append(datetime(int(readingth[0][i]),int(readingth[1][i]),int(readingth[2][i])))

readingth.index=dfdt

#remove old date info
del(readingth[0])
del(readingth[1])
del(readingth[2])

#deal with datagaps: 'x'
readingth[3]=readingth[3].astype(str)
readingth[3].replace({'x' : np.nan}, inplace=True)
readingth[3]=readingth[3].astype(float)


thavg=readingth.resample('A',label='center').mean()

plt.figure(figsize=(8,6), dpi=100)
plt.plot(thavg[3])
plt.ylim([0, thavg[3].max()+thavg[3].max()/10])
plt.ylabel('Mean annual thunder rate [day$^{-1}$]')
plt.xlabel('Year')

thavg=readingth.rolling('96360H',min_periods=3500).mean()
plt.plot(thavg[3])








#dowload the sunspot data from http://www.sidc.be/silso/DATA/SN_m_tot_V2.0.csv
filepath= os.environ['DBOX'] + 'Data\\SN_m_tot_V2.0.txt'
col_specification =[(0, 4), (5, 7), (8,16),(17,23),(24,29),(30,35)]
ssn=pd.read_fwf(filepath, colspecs=col_specification,header=None)
dfdt=[]
for i in range(0,len(ssn)):
    dfdt.append(datetime(ssn[0][i],ssn[1][i],15))
#replace the index with the datetime objects
ssn.index=dfdt
ssn['ssn']=ssn[3]

ssnavg=ssn.rolling('96360H',min_periods=100).mean()



import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator


hfont = {'fontname':'Tahoma'}

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

plt.rc('font', **font)

years = mdates.YearLocator() 
yrs5 = mdates.YearLocator(5) 
minor_locator = AutoMinorLocator(2)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

d = ssnavg.index.values
ax1.fill_between(d,0,ssnavg['ssn'],facecolor='silver')
ax1.set_ylabel('Sunspot number', fontsize=16,**hfont)
ax1.set_xlim(datetime(1900,1,1), datetime(2017,1,1))
ax1.set_ylim(30,150)

ax1.yaxis.set_label_position('right')
ax1.yaxis.set_ticks_position('right')
ax1.minorticks_on()


plotcol='blue'
ax2.plot(thavg[3],plotcol)
#ax2.plot(tsi['tsi'].resample('96360H').mean(),'k')
ax2.set_ylabel('Thunder [day$^{-1}$]', color=plotcol, fontsize=16,**hfont)
ax2.tick_params('y', colors=plotcol)
ax2.yaxis.set_ticks_position('left')
ax2.yaxis.set_label_position('left')
ax2.set_xlim(datetime(1900,1,1), datetime(2017,1,1))
ax2.minorticks_on()
ax2.xaxis.set_minor_locator(years)
ax2.xaxis.set_major_locator(yrs5)
ax2.yaxis.set_minor_locator(minor_locator)
ax2.set_xlabel('Year', fontsize=16,**hfont)
plt.tight_layout()