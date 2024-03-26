# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:22:25 2020

@author: mathewjowens
"""

import os as os
from scipy.io import netcdf
import pandas as pd
import matplotlib.pyplot as plt

discovrfile = os.environ['DBOX'] + 'Data\\DISCOVR\\oe_m1m_dscovr_s20200408000000_e20200408235959_p20200410031649_pub.nc'

nc = netcdf.netcdf_file(discovrfile,'r')

for x in nc.variables:
    print(x)
     
#store the variables in a dataframe
discovr=pd.DataFrame()  
print('Time units = '+str(nc.variables['time'].units))
discovr['datetime'] = pd.to_datetime(nc.variables['time'][:].flatten(),unit='ms')
#discovr['time']=pd.Series(nc.variables['time'][:].flatten())
nc.variables['time'].units
discovr['bx_gse']=pd.Series(nc.variables['bx_gse'][:].flatten())
discovr['by_gse']=pd.Series(nc.variables['by_gse'][:].flatten())
discovr['bz_gse']=pd.Series(nc.variables['bz_gse'][:].flatten())

#Use the datetime as teh index
discovr.set_index('datetime', inplace=True)


##remove bad data 
#baddata=obs_df['mlat'] < -180
#obs_df.drop(obs_df[baddata].index, inplace=True)
#baddata=obs_df['mlt'] < -180
#obs_df.drop(obs_df[baddata].index, inplace=True)
##remove data where the position is a NaN
#obs_df.dropna(how='any', subset=['mlat', 'mlt'], inplace=True)

#close the netCDF file
nc.close()


#Export the data as an ascii file, for import to Matlab, etc
datetime= discovr.index       
discovr['Year']=datetime.year
discovr['Month']=datetime.month
discovr['Day']=datetime.day
discovr['Hour']=datetime.hour
discovr['Min']=datetime.minute
discovr.to_csv('discovr.csv', index=False)

#plot the data
plotstart='2020-04-08 00:00'
plotend='2020-04-09 00:00'
fig1, axs=plt.subplots(3,1,constrained_layout=True) 
       
axs[0].plot(discovr[plotstart:plotend]['bx_gse'])
axs[0].plot(discovr[plotstart:plotend]['bx_gse']*0.0) #zero line
axs[0].set_ylabel('B_X [nT]')

axs[1].plot(discovr[plotstart:plotend]['by_gse'])  
axs[1].plot(discovr[plotstart:plotend]['by_gse']*0.0) #zero line
axs[1].set_ylabel('B_Y [nT]')
       
axs[2].plot(discovr[plotstart:plotend]['bz_gse']) 
axs[2].plot(discovr[plotstart:plotend]['bz_gse']*0.0) #zero line   
axs[2].set_ylabel('B_Z [nT]')   