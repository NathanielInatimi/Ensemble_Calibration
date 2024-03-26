# -*- coding: utf-8 -*-
"""
Created on Wed May 24 16:31:33 2023

@author: mathewjowens
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 12:57:58 2022


ESA L5 project 15 CME analysis for 3 observers. Script to compare transit speeds 
to CME and ICME speeds, as sanity check on 

@author: vy902033
"""


import astropy.units as u
from astropy.time import Time, TimeDelta
import numpy as np
import datetime
import h5py
import os as os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from astropy.io import fits
import shutil
from scipy import interpolate
import glob
import pandas as pd

#from HUXt
import huxt_inputs as Hin
import huxt as H
import huxt_analysis as HA

#from HUXt_tools
import huxt_ensembles as Hens

#from BRaVDA
import startBravda


#directory for the cone files. Can be the same or different from rootdir
conerootdir = os.environ['DBOX'] + 'Papers_WIP\\_coauthor\\SiegfriedGonzi\\C2-KIRK-24h-v2\\'
#conerootdir = os.environ['DBOX'] + 'Papers_WIP\\_coauthor\\SiegfriedGonzi\\C2-MICHAEL-24h-v2\\'
#conerootdir = os.environ['DBOX'] + 'Papers_WIP\\_coauthor\\SiegfriedGonzi\\C2-REBECCA-24h-v2\\'

conefile_number = 0 #zero is unperturbed from forecaster

#load the 64-second data in
data_dir = os.environ['DBOX'] + 'Data_hdf5\\'
omni = pd.read_hdf(data_dir + 'omni_1hour.h5')


# <codecell> CME data
#CAT tool output is in cone2bc.in files

#ICME arrival times and forecast times (i.e. timestep of the WSA input file)
obstimes=['2010-04-11T13:04:00', '2020-11-01T11:54:00', '2021-11-03T19:49:00',
          '2010-05-28T02:58:00', '2009-12-19T10:00:00', '2011-06-17T02:41:00',
          '2010-03-23T22:33:00', '2010-04-05T08:26:00', '2021-10-31T09:13:00',
          '2020-12-10T02:10:00', '2020-10-05T06:52:00', '2021-02-15T18:58:00',
          '2011-05-28T00:14:00', '2011-07-14T12:00:00', '2011-11-01T08:09:00']
foretimes = ['20100409', '20201028', '20211103',
             '20100525', '20091217', '20110615',
             '20100321', '20100404', '20211029',
             '20201208', '20201002', '20210212',
             '20110526', '20110712', '20111030']





# <codecell> work out file paths
cme_props = np.ones((15,4))
for event_number in range(1,16):
    
    #find the cone file
    coneeventdir = glob.glob(conerootdir + '*' + 'CME-'+str(event_number))[0]
    coneworkingdir = glob.glob(coneeventdir +'\\swcx\\*')[0] + '\\'
    conefilepath = glob.glob(coneworkingdir + 'cone2bc*' )[conefile_number]
    
    #read in the cone data 
    cme_params = Hin.import_cone2bc_parameters(conefilepath)
    for cme_id, cme_val in cme_params.items():
        # CME initialisation date
        t_cme = Time(cme_val['ldates'])

        # Get lon, lat and speed
        lon = cme_val['lon'] * u.deg
        lat = cme_val['lat'] * u.deg
        v_cme = cme_val['vcld'] * u.km / u.s

        # Get full angular width, cone2bc specifies angular half width under rmajor
        wid = 2 * cme_val['rmajor'] * u.deg
        
        # Set the initial height to be 21.5 rS, the default for WSA
    
    #find the travel time
    t_icme = Time(obstimes[event_number-1], format='isot')
    dt = (t_icme - t_cme).jd
    
    
    #compute the transit speed
    v_transit = (215-21.5)*696340/(dt*24*60*60)
    
    #get the 1-AU speed
    mask = ((omni['mjd'] >= t_icme.mjd) & (omni['mjd'] <= (t_icme.mjd+0.25)))
    datachunk = omni.loc[mask] 
    v_icme = np.nanmean(datachunk['Vr'])
    
    cme_props[event_number-1,0] = event_number
    cme_props[event_number-1,1] = v_cme.value
    cme_props[event_number-1,2] = v_icme
    cme_props[event_number-1,3] = v_transit
    

# <codecell>
#produce plots of DA experiment results      

dt_kirk = np.array([[np.nan,	np.nan,	np.nan],
[np.nan,	np.nan,	np.nan],
[-0.508,	-1.283,	-0.917],
[-0.123,	-0.369,	-0.386],
[-0.330,	0.235,	0.148],
[np.nan,	np.nan,	np.nan],
[np.nan,	np.nan,	np.nan],
[-0.361,	-0.638,	-0.57],
[0.608,	0.061,	0.204],
[-0.337,	-0.365,	0.055],
[np.nan,	np.nan,	np.nan],
[np.nan,	np.nan,	np.nan],
[0.190,	-0.043,	-0.101],
[-0.474,	-0.327,	-0.247],
[-0.287,	-0.178,	-0.074]])

dt_michael = np.array([[np.nan,	np.nan,	np.nan],
[np.nan,	np.nan,	np.nan],
[-0.574,	-1.482,	-1.071],
[-0.078,	-0.31,	-0.324],
[np.nan,	np.nan,	np.nan],
[np.nan,	np.nan,	np.nan],
[np.nan,	np.nan,	np.nan],
[-0.204,	-0.515,	-0.429],
[0.207,	-0.395,	-0.24],
[-0.65,	-0.604,	-0.141],
[np.nan,	np.nan,	np.nan],
[np.nan,	np.nan,	np.nan],
[0.412,	0.157,	0.096],
[np.nan,	np.nan,	np.nan],
[np.nan,	np.nan,	np.nan]])


dt_rebecca = np.array([[0.429,	0.224,	-0.032],
[np.nan,	np.nan,	np.nan],
[-0.644,	-1.168,	-0.924],
[-0.287,	-0.531,	-0.553],
[0.095,	0.479,	0.454],
[-0.309,	-0.739,	-0.781],
[np.nan,	np.nan,	np.nan],
[-0.382,	-0.627,	-0.566],
[0.613,	-0.008,	0.157],
[-0.299,	-0.331,	0.077],
[-0.282,	-0.013,	0.073],
[np.nan,	np.nan,	np.nan],
[0.568,	0.333,	0.278],
[-0.663,	-0.547,	-0.487],
[np.nan,	np.nan,	np.nan]])

goodevents_kirk = np.array([2,3,4,5,6,9,10,12,15]) - 1
goodevents_michael = np.array([2,3,4,5,6,7,9,10,11,12,13]) - 1
goodevents_rebecca = np.array([1,2,5,9,10,13,15]) - 1 

#compute stats
print('OBSERVER KIRK. N = ' + str(sum(~np.isnan(dt_kirk[:,0]))))
print(' ')
print('<|dt|>, no DA:    ' + str(np.nanmean(abs(dt_kirk[:,0]))))
print('<|dt|>, L1 DA:    ' + str(np.nanmean(abs(dt_kirk[:,1]))))
print('<|dt|>, L1+L5 DA: ' + str(np.nanmean(abs(dt_kirk[:,2]))))
print('Mdn |dt|, no DA:    ' + str(np.nanmedian(abs(dt_kirk[:,0]))))
print('Mdn |dt|, L1 DA:    ' + str(np.nanmedian(abs(dt_kirk[:,1]))))
print('Mdn |dt|, L1+L5 DA: ' + str(np.nanmedian(abs(dt_kirk[:,2]))))
print(' ')
print('removing bad events. N = ' + str(sum(~np.isnan(dt_kirk[goodevents_kirk,0]))))
print('<|dt|>, no DA:    ' + str(np.nanmean(abs(dt_kirk[goodevents_kirk,0]))))
print('<|dt|>, L1 DA:    ' + str(np.nanmean(abs(dt_kirk[goodevents_kirk,1]))))
print('<|dt|>, L1+L5 DA: ' + str(np.nanmean(abs(dt_kirk[goodevents_kirk,2]))))
print('Mdn |dt|, no DA:    ' + str(np.nanmedian(abs(dt_kirk[goodevents_kirk,0]))))
print('Mdn |dt|, L1 DA:    ' + str(np.nanmedian(abs(dt_kirk[goodevents_kirk,1]))))
print('Mdn |dt|, L1+L5 DA: ' + str(np.nanmedian(abs(dt_kirk[goodevents_kirk,2]))))


print(' ')
print('OBSERVER MICHAEL. N = ' + str(sum(~np.isnan(dt_michael[:,0]))))
print(' ')
print('<|dt|>, no DA:    ' + str(np.nanmean(abs(dt_michael[:,0]))))
print('<|dt|>, L1 DA:    ' + str(np.nanmean(abs(dt_michael[:,1]))))
print('<|dt|>, L1+L5 DA: ' + str(np.nanmean(abs(dt_michael[:,2]))))
print('Mdn |dt|, no DA:    ' + str(np.nanmedian(abs(dt_michael[:,0]))))
print('Mdn |dt|, L1 DA:    ' + str(np.nanmedian(abs(dt_michael[:,1]))))
print('Mdn |dt|, L1+L5 DA: ' + str(np.nanmedian(abs(dt_michael[:,2]))))
print(' ')
print('removing bad events. N = ' + str(sum(~np.isnan(dt_michael[goodevents_michael,0]))))
print('<|dt|>, no DA:    ' + str(np.nanmean(abs(dt_michael[goodevents_michael,0]))))
print('<|dt|>, L1 DA:    ' + str(np.nanmean(abs(dt_michael[goodevents_michael,1]))))
print('<|dt|>, L1+L5 DA: ' + str(np.nanmean(abs(dt_michael[goodevents_michael,2]))))
print('Mdn |dt|, no DA:    ' + str(np.nanmedian(abs(dt_michael[goodevents_michael,0]))))
print('Mdn |dt|, L1 DA:    ' + str(np.nanmedian(abs(dt_michael[goodevents_michael,1]))))
print('Mdn |dt|, L1+L5 DA: ' + str(np.nanmedian(abs(dt_michael[goodevents_michael,2]))))

print(' ')
print('OBSERVER REBECCA. N = ' + str(sum(~np.isnan(dt_rebecca[:,0]))))
print(' ')
print('<|dt|>, no DA:    ' + str(np.nanmean(abs(dt_rebecca[:,0]))))
print('<|dt|>, L1 DA:    ' + str(np.nanmean(abs(dt_rebecca[:,1]))))
print('<|dt|>, L1+L5 DA: ' + str(np.nanmean(abs(dt_rebecca[:,2]))))
print('Mdn |dt|, no DA:    ' + str(np.nanmedian(abs(dt_rebecca[:,0]))))
print('Mdn |dt|, L1 DA:    ' + str(np.nanmedian(abs(dt_rebecca[:,1]))))
print('Mdn |dt|, L1+L5 DA: ' + str(np.nanmedian(abs(dt_rebecca[:,2]))))
print(' ')
print('removing bad events. N = ' + str(sum(~np.isnan(dt_rebecca[goodevents_rebecca,0]))))
print('<|dt|>, no DA:    ' + str(np.nanmean(abs(dt_rebecca[goodevents_rebecca,0]))))
print('<|dt|>, L1 DA:    ' + str(np.nanmean(abs(dt_rebecca[goodevents_rebecca,1]))))
print('<|dt|>, L1+L5 DA: ' + str(np.nanmean(abs(dt_rebecca[goodevents_rebecca,2]))))
print('Mdn |dt|, no DA:    ' + str(np.nanmedian(abs(dt_rebecca[goodevents_rebecca,0]))))
print('Mdn |dt|, L1 DA:    ' + str(np.nanmedian(abs(dt_rebecca[goodevents_rebecca,1]))))
print('Mdn |dt|, L1+L5 DA: ' + str(np.nanmedian(abs(dt_rebecca[goodevents_rebecca,2]))))

# <codecell> generate some plots

def cdf_nans(data, norm='True'):
    data_nonans = data[~np.isnan(data)]
    count, bins_count = np.histogram(data_nonans, bins=10000)
    if norm:
        count = count / sum(count)
    cdf = np.cumsum(count)
    return bins_count[1:], cdf

fig1, axs = plt.subplots(3,2, figsize=(8, 10), tight_layout=True)



bins, cdf = cdf_nans(abs(dt_kirk[:,0]), norm=False)
axs[0,0].plot(bins, cdf, 'k', label="No DA")

bins, cdf = cdf_nans(abs(dt_kirk[:,1]), norm=False)
axs[0,0].plot(bins, cdf, 'b', label="L1 DA")

bins, cdf = cdf_nans(abs(dt_kirk[:,2]), norm=False)
axs[0,0].plot(bins, cdf, 'r', label="L1+L5 DA")

#axs[0,0].set_xlabel(r'$|\Delta T|$')
axs[0,0].set_ylabel('Cumulative number')
axs[0,0].set_title('Fcst 1, all CMEs, N = ' + str(sum(~np.isnan(dt_kirk[:,0]))), 
                   fontsize = 14)

axs[0,0].legend()




bins, cdf = cdf_nans(abs(dt_kirk[goodevents_kirk,0]), norm=False)
axs[0,1].plot(bins, cdf, 'k', label="No DA")

bins, cdf = cdf_nans(abs(dt_kirk[goodevents_kirk,1]), norm=False)
axs[0,1].plot(bins, cdf, 'b', label="L1 DA")

bins, cdf = cdf_nans(abs(dt_kirk[goodevents_kirk,2]), norm=False)
axs[0,1].plot(bins, cdf, 'r', label="L1+L5 DA")

#axs[0,1].set_xlabel(r'$|\Delta T|$')
axs[0,1].set_title('Fcst 1, "good" CMEs, N = ' + str(sum(~np.isnan(dt_kirk[goodevents_kirk,0]))), 
                   fontsize = 14)



bins, cdf = cdf_nans(abs(dt_michael[:,0]), norm=False)
axs[1,0].plot(bins, cdf, 'k', label="No DA")

bins, cdf = cdf_nans(abs(dt_michael[:,1]), norm=False)
axs[1,0].plot(bins, cdf, 'b', label="L1 DA")

bins, cdf = cdf_nans(abs(dt_michael[:,2]), norm=False)
axs[1,0].plot(bins, cdf, 'r', label="L1+L5 DA")

#axs[1,0].set_xlabel(r'$|\Delta T|$')
axs[1,0].set_ylabel('Cumulative number')
axs[1,0].set_title('Fcst 2, all CMEs, N = ' + str(sum(~np.isnan(dt_michael[:,0]))), 
                   fontsize = 14)




bins, cdf = cdf_nans(abs(dt_michael[goodevents_michael,0]), norm=False)
axs[1,1].plot(bins, cdf, 'k', label="No DA")

bins, cdf = cdf_nans(abs(dt_michael[goodevents_michael,1]), norm=False)
axs[1,1].plot(bins, cdf, 'b', label="L1 DA")

bins, cdf = cdf_nans(abs(dt_michael[goodevents_michael,2]), norm=False)
axs[1,1].plot(bins, cdf, 'r', label="L1+L5 DA")


#axs[1,1].set_xlabel(r'$|\Delta T|$')
axs[1,1].set_title('Fcst 2, "good" CMEs, N = ' + str(sum(~np.isnan(dt_michael[goodevents_michael,0]))), 
                   fontsize = 14)







bins, cdf = cdf_nans(abs(dt_rebecca[:,0]), norm=False)
axs[2,0].plot(bins, cdf, 'k', label="No DA")

bins, cdf = cdf_nans(abs(dt_rebecca[:,1]), norm=False)
axs[2,0].plot(bins, cdf, 'b', label="L1 DA")

bins, cdf = cdf_nans(abs(dt_rebecca[:,2]), norm=False)
axs[2,0].plot(bins, cdf, 'r', label="L1+L5 DA")

axs[2,0].set_xlabel(r'$|\Delta T|$')
axs[2,0].set_ylabel('Cumulative number')
axs[2,0].set_title('Fcst 3, all CMEs, N = ' + str(sum(~np.isnan(dt_rebecca[:,0]))), 
                   fontsize = 14)




bins, cdf = cdf_nans(abs(dt_rebecca[goodevents_rebecca,0]), norm=False)
axs[2,1].plot(bins, cdf, 'k', label="No DA")

bins, cdf = cdf_nans(abs(dt_rebecca[goodevents_rebecca,1]), norm=False)
axs[2,1].plot(bins, cdf, 'b', label="L1 DA")

bins, cdf = cdf_nans(abs(dt_rebecca[goodevents_rebecca,2]), norm=False)
axs[2,1].plot(bins, cdf, 'r', label="L1+L5 DA")


axs[2,1].set_xlabel(r'$|\Delta T|$')
axs[2,1].set_title('Fcst 3, "good" CMEs, N = ' + str(sum(~np.isnan(dt_rebecca[goodevents_rebecca,0]))), 
                   fontsize = 14)


#tidy up the plots
for x in range(0,2):
    for y in range(0,3):
        axs[y,x].yaxis.set_major_locator(MaxNLocator(integer=True))
        axs[y,x].set_xlim((0,1.5))
        
for x in range(0,2):
    for y in range(0,2):
        axs[y,x].set_xticklabels([])