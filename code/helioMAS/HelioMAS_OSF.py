# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:52:10 2022

@author: vy902033
"""
"""
a Script to compute the OSF from HelioMAS


Created on Mon Dec 20 11:48:49 2021

@author: mathewjowens
"""

import os
from pyhdf.SD import SD, SDC  
import numpy as np
import astropy.units as u
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

import helio_time as htime


plt.rcParams.update({'font.size': 16})
downloadnow = False #flag to determine whether HelioMAS data needs to be obtained. Set to False after first use

#datadir = os.environ['DBOX'] + 'python_repos\\SolarWindVariability\data\\'
#heliomasdir = datadir + 'HelioMAS\\'
heliomasdir = os.environ['DBOX'] + 'Data\\HelioMAS\\'
dlat_pole = 20*np.pi/180 *u.rad #for defining N-pole, S-pole and Equatorial regions 
dlat_eq = 15*np.pi/180 *u.rad

def read_HelioMAS(filepath):
    """
    A function to read in the HelioMAS data cubes

    Parameters
    ----------
    directory_path : INT
        Carrington rotation number

    Returns
    -------
    MAS_vr : NP ARRAY (NDIM = 2)
        Solar wind speed at 30rS, in km/s
    MAS_vr_Xa : NP ARRAY (NDIM = 1)
        Carrington longitude of Vr map, in rad
    MAS_vr_Xm : NP ARRAY (NDIM = 1)
        Latitude of Vr as angle down from N pole, in rad
    MAS_vr_Xr : NP ARRAY (NDIM = 1)
        Radial distance of Vr, in solar radii

    """
    
    assert os.path.exists(filepath)
    
    file = SD(filepath, SDC.READ)
        
    sds_obj = file.select('fakeDim0')  # select sds
    MAS_vr_Xa = sds_obj.get()  # get sds data
    sds_obj = file.select('fakeDim1')  # select sds
    MAS_vr_Xm = sds_obj.get()  # get sds data
    sds_obj = file.select('fakeDim2')  # select sds
    MAS_vr_Xr = sds_obj.get()  # get sds data
    sds_obj = file.select('Data-Set-2')  # select sds
    MAS_vr = sds_obj.get()  # get sds data
    
    # # Convert from model to physicsal units
    # MAS_vr = MAS_vr*481.0 * u.km/u.s
    MAS_vr_Xa = MAS_vr_Xa * u.rad
    MAS_vr_Xm = MAS_vr_Xm * u.rad
    MAS_vr_Xr = MAS_vr_Xr * u.solRad
    
    
    return MAS_vr, MAS_vr_Xa, MAS_vr_Xm, MAS_vr_Xr



#Data reader functions
def LoadSSN(filepath='null'):
    #(dowload from http://www.sidc.be/silso/DATA/SN_m_tot_V2.0.csv)
    if filepath == 'null':
        filepath= os.environ['DBOX'] + 'Data\\SN_m_tot_V2.0.txt'
        
    col_specification =[(0, 4), (5, 7), (8,16),(17,23),(24,29),(30,35)]
    ssn_df=pd.read_fwf(filepath, colspecs=col_specification,header=None)
    dfdt=np.empty_like(ssn_df[0],dtype=datetime)
    for i in range(0,len(ssn_df)):
        dfdt[i] = datetime(int(ssn_df[0][i]),int(ssn_df[1][i]),15)
    #replace the index with the datetime objects
    ssn_df['datetime']=dfdt
    ssn_df['ssn']=ssn_df[3]
    ssn_df['mjd'] = htime.datetime2mjd(dfdt)
    #delete the unwanted columns
    ssn_df.drop(0,axis=1,inplace=True)
    ssn_df.drop(1,axis=1,inplace=True)
    ssn_df.drop(2,axis=1,inplace=True)
    ssn_df.drop(3,axis=1,inplace=True)
    ssn_df.drop(4,axis=1,inplace=True)
    ssn_df.drop(5,axis=1,inplace=True)
    
    #add the 13-month running smooth
    window = 13*30
    temp = ssn_df.rolling(str(window)+'D', on='datetime').mean()
    ssn_df['smooth'] = np.interp(ssn_df['mjd'],temp['mjd'],temp['ssn'],
                                              left =np.nan, right =np.nan)
    
    #add in a solar activity index, which normalises the cycle magnitude
    #approx solar cycle length, in months
    nwindow = int(11*12)
    
    #find maximum value in a 1-solar cycle bin centred on current time
    ssn_df['rollingmax'] = ssn_df.rolling(nwindow, center = True).max()['smooth']
    
    #fill the max value at the end of the series
    fillval = ssn_df['rollingmax'].dropna().values[-1]
    ssn_df['rollingmax'] = ssn_df['rollingmax'].fillna(fillval) 
    
    #create a Solar Activity Index, as SSN normalised to the max smoothed value in
    #1-sc window centred on current tim
    ssn_df['sai'] = ssn_df['smooth']/ssn_df['rollingmax']
    
    return ssn_df

# <codecell> Process HelioMAS data

#loop through the HelioMAS solutions and compute OSF 

ssn_df = LoadSSN()

CRstart = 1625
CRstop = 2232

dlat = 15*np.pi/180 *u.rad #for defining N-pole, S-pole and Equatorial regions 


observatories = ['hmi', 'mdi', 'solis', 'gong', 'mwo', 'wso', 'kpo']

lats = np.arange(2.5,180-2.499999,5) * np.pi/180
Nlats = len(lats)-1

df = pd.DataFrame()


for obs in observatories: 
    
    print('Processing ' + obs)
    
    OSF = np.ones((CRstop-CRstart))*np.nan
    Br_N = np.ones((CRstop-CRstart))*np.nan
    Br_S = np.ones((CRstop-CRstart))*np.nan
    Br_Eq = np.ones((CRstop-CRstart))*np.nan
    nCR = np.ones((CRstop-CRstart))*np.nan
    Br_lat = np.ones((CRstop-CRstart,Nlats))*np.nan
    Br_norm_lat = np.ones((CRstop-CRstart,Nlats))*np.nan
    ssn = np.ones((CRstop-CRstart))*np.nan
    sai = np.ones((CRstop-CRstart))*np.nan
    
    counter = 0 
    for CR in range(CRstart,CRstop):
        brfilepath = heliomasdir + obs + '\\CR' + str(CR) + '\\br002.hdf'
        
        nCR[counter] = CR 
        
        smjd = htime.crnum2mjd(CR-1)
        fmjd = htime.crnum2mjd(CR+2)
        
        mask = (ssn_df['mjd'] >= smjd) & (ssn_df['mjd'] < fmjd)
        ssn[counter] = np.nanmean(ssn_df.loc[mask,'ssn'])
        sai[counter] = np.nanmean(ssn_df.loc[mask,'sai'])
        
        
        if (os.path.exists(brfilepath) == True):
            
            
            #load the Br data
            Br, BXa, BXm, BXr = read_HelioMAS(brfilepath) 
            Br = Br *  2.2e5
            

            
            #take the slice at r = 215 rS           
            id_r_b = np.argmin(abs(BXr - 215*u.solRad))
            
            
            
            #now find teh differences at each lat bin
            for ilat in range(0,Nlats):
                mask_b = ((BXm >= lats[ilat]*u.rad) & (BXm < lats[ilat+1]*u.rad))
            
                #compute the mean abs Br at the given lat band
                Br_lat[counter,ilat] = np.nanmean(abs(Br[:, mask_b, id_r_b]))
                #normalise to the max value
                Br_norm_lat[counter,:] = Br_lat[counter,:] / np.nanmax(Br_lat[counter,:])
            
            #compute the OSF from full lat resolution
            OSF[counter] = 0.0
            for ilat in range(0,len(BXm)):
                theta = BXm[ilat].value
                absBr = np.nanmean(abs(Br[:, ilat]))
                
                OSF[counter] = OSF[counter] + np.sin(theta) * absBr
                
            OSF[counter] = OSF[counter] / len(BXm)
            
            
            #take averages over the three regions of interest
            mask_N = (BXm <= dlat_pole) 
            mask_S = (BXm >= np.pi*u.rad - dlat_pole) 
            mask_Eq = ((BXm >= np.pi/2*u.rad - dlat_eq) & (BXm <= np.pi/2*u.rad + dlat_eq)) 
            Br_N[counter] = np.nanmean(abs(Br[:, mask_N, id_r_b])) 
            Br_S[counter] = np.nanmean(abs(Br[:, mask_S, id_r_b])) 
            Br_Eq[counter] = np.nanmean(abs(Br[:, mask_Eq, id_r_b])) 
        

        else:
            OSF[counter] = np.nan
            Br_N[counter] = np.nan
            Br_S[counter] = np.nan
            Br_Eq[counter] = np.nan
            Br_lat[counter,:] = np.nan
            Br_norm_lat[counter,:] = np.nan
    
        counter = counter + 1
    
    #add the data to the data frame
    df['CR'] = nCR
    df['SSN'] = ssn
    df['SAI'] = sai
    df[obs + '_OSF'] = OSF
    df[obs + '_Br_N'] = Br_N
    df[obs + '_Br_S'] = Br_S
    df[obs + '_Br_Eq'] = Br_Eq
    for ilat in range(0,Nlats):
        df[obs + '_Br_lat_' + str(ilat)] = Br_lat[:,ilat]
        df[obs + '_Br_norm_lat_' + str(ilat)] = Br_norm_lat[:,ilat]

#convert to datetime
df['datetime'] = htime.mjd2datetime(htime.crnum2mjd(df['CR'].values))

#take averages across all observatories
#observatories = ['hmi', 'mdi', 'solis', 'gong', 'mwo', 'wso', 'kpo']
df['OSF'] = df[['hmi_OSF','mdi_OSF','solis_OSF',
               'gong_OSF','mwo_OSF','wso_OSF', 
               'kpo_OSF']].mean(axis = 1, skipna=True)
df['Br_N'] = df[['hmi_Br_N','mdi_Br_N','solis_Br_N',
               'gong_Br_N','mwo_Br_N','wso_Br_N', 
               'kpo_Br_N']].mean(axis = 1, skipna=True)
df['Br_S'] = df[['hmi_Br_S','mdi_Br_S','solis_Br_S',
               'gong_Br_S','mwo_Br_S','wso_Br_S', 
               'kpo_Br_S']].mean(axis = 1, skipna=True)
df['Br_Eq'] = df[['hmi_Br_Eq','mdi_Br_Eq','solis_Br_Eq',
               'gong_Br_Eq','mwo_Br_Eq','wso_Br_Eq', 
               'kpo_Br_Eq']].mean(axis = 1, skipna=True)

for ilat in range(0,Nlats):
    l = str(ilat)
    df['Br_lat_' + str(ilat)] = df[['hmi_Br_lat_' + l,'mdi_Br_lat_' + l,'solis_Br_lat_' + l,
                   'gong_Br_lat_' + l,'mwo_Br_lat_' + l,'wso_Br_lat_' + l, 
                   'kpo_Br_lat_' + l]].mean(axis = 1, skipna=True)
    
    
    df['Br_norm_lat_' + str(ilat)] = df[['hmi_Br_norm_lat_' + l,'mdi_Br_norm_lat_' + l,'solis_Br_norm_lat_' + l,
                   'gong_Br_norm_lat_' + l,'mwo_Br_norm_lat_' + l,'wso_Br_norm_lat_' + l, 
                   'kpo_Br_norm_lat_' + l]].mean(axis = 1, skipna=True)

    
#add a pole average
df['Br_pole'] = (df['Br_S'] + df['Br_N'])/2

mask_max = ((df['SAI'] >= 0.5))
mask_min = ((df['SAI'] < 0.5)) 

Nall = len(np.isfinite(df['OSF']))   
Nmin = len(np.isfinite(df.loc[mask_min,'OSF']))
Nmax = len(np.isfinite(df.loc[mask_max,'OSF']))


# <codecell> plot the HelioMAS results

#plt.rcParams.update({'font.size': 14})
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color='bgkymcr')

plot_observatories = ['hmi', 'mdi', 'solis', 'gong', 'mwo', 'wso', 'kpo']


xx = (datetime(1973,1,1), datetime(2024,1,1))  


#import OMNI data and compute OSF from 20-hour averages of Br
datadir = os.environ['DBOX'] + 'Data_hdf5\\'
omni_1hour = pd.read_hdf(datadir + 'omni_1hour.h5')

br = omni_1hour.copy()
br_20H = br.resample('20H', on='datetime').mean() 

br_20H['datetime'] = br_20H.index
br_20H.reset_index(drop=True, inplace=True)
br_20H['absBr'] = abs(br_20H['Bx_gse'])

br_CR = br_20H.resample('27D', on='datetime').mean() 




#time series plots
fig = plt.figure()

ax = plt.subplot(411)
plt.plot(df['datetime'],df['SSN']/200, 'k', label = 'SSN/200')
plt.plot(df['datetime'],df['SAI'], 'r', label = 'SAI')
plt.ylabel('SSN')
plt.legend(fontsize = 16)
ax.text(0.05,0.9,'(a)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax.set_xlim(xx)


ax = plt.subplot(412)
plt.plot(br_CR['absBr'])
ax.set_xlim(xx)
ax.set_ylabel('Br, OMNI ' + '\n[nT]', fontsize = 16)
ax.text(0.05,0.9,'(b)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')



ax = plt.subplot(413)
for obs in plot_observatories: 
    plt.plot(df['datetime'], df[obs + '_OSF'], label = obs)
#plt.legend(fontsize = 16)
ax.set_ylabel(r'OSF, HelioMAS ' + '\n[arb units]', fontsize = 16)
#ax.set_ylim((0,230))
ax.text(0.05,0.9,'(c)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax.set_xlim(xx)

# ax = plt.subplot(313)
# for obs in plot_observatories: 
#     plt.plot(df['datetime'], df[obs + '_dV_eclip'], label = obs)
# #plt.legend(fontsize = 16)
# ax.set_ylabel(r'$<|\Delta V|>_{CR}$ [km/s]' + '\n(Ecliptic)', fontsize = 16)
# ax.set_ylim((0,230))
# ax.text(0.05,0.9,'(c)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')






ax = plt.subplot(414)

for obs in plot_observatories: 
    plt.plot(df['datetime'], df[obs + '_Br_lat_16'], label = obs)
#plt.legend(fontsize = 16)
ax.set_ylabel('Br_eclip, HelioMAS ' + '\n[nT]', fontsize = 16)
#ax.set_ylim((0,230))
ax.text(0.05,0.9,'(d)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax.set_xlim((datetime(1973,1,1), datetime(2024,1,1)))

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right', framealpha=1)


# <codecell> plot the HelioMAS results as function of lat

dtheta = (lats[1]-lats[0])*180/np.pi
lat_centres = 90 - lats[0:len(lats)-1]*180/np.pi -dtheta/2

#extract data as an array for ploitting purposes
B_lat = np.ones((CRstop-CRstart,Nlats))*np.nan
B_norm_lat = np.ones((CRstop-CRstart,Nlats))*np.nan
for ilat in range(0,Nlats):
    B_lat[:,ilat] = df['Br_lat_' + str(ilat)].to_numpy()
    B_norm_lat[:,ilat] = df['Br_norm_lat_' + str(ilat)].to_numpy()
    
xx = (datetime(1973,1,1), datetime(2024,1,1))                                      
                                      
fig = plt.figure()

ax = plt.subplot(311)
plt.plot(df['datetime'],df['SSN']/200, 'k', label = 'SSN/200')
plt.plot(df['datetime'],df['SAI'], 'r', label = 'SAI')
plt.ylabel('SSN')
plt.legend(fontsize = 16)
ax.get_xaxis().set_ticklabels([])
ax.text(0.05,0.9,'(a)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax.set_xlim(xx)


ax = plt.subplot(312)
im_v = ax.pcolor(df['datetime'], lat_centres, B_lat.T)#,norm=plt.Normalize(0,2))
ax.set_yticks([-90, -45, 0, 45, 90])
ax.get_xaxis().set_ticklabels([])
ax.text(0.02,1.05,'(b)' + r'$<|B_R|>$ [nT]                                                            ',
        fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax.set_xlim(xx)
ax.set_ylim((-90,90))
ax.set_ylabel('Latitude [deg]')

#ax.plot([0, 360],[7.5, 7.5],'w--'); ax.plot([0, 360],[-7.5, -7.5],'w--');
#cb = plt.colorbar(im_v); cb.ax.tick_params(labelsize=12)
#cb.ax.set_title(r'$<|\Delta V|>_{CR}$ [km/s]', fontsize = 14)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
axins = inset_axes(ax,
                    width="100%",  # width = 50% of parent_bbox width
                    height="10%",  # height : 5%
                    loc='upper right',
                    bbox_to_anchor=(0.45, 0.65, 0.5, 0.5),
                    bbox_transform=ax.transAxes,
                    borderpad=0,)

cb = fig.colorbar(im_v, cax = axins, orientation = 'horizontal',  pad = -0.1)
cb.ax.tick_params(labelsize=12)

#plot the polar and equatorial bands
plotlat = (dlat_pole*180/np.pi).value
ax.plot(xx, (87,87),'r')
ax.plot(xx, (90 - plotlat,90 - plotlat),'r')
ax.plot(xx, (-87,-87),'b')
ax.plot(xx, (-90 + plotlat, -90 + plotlat),'b')
plotlat = (dlat_eq*180/np.pi).value
ax.plot(xx, (plotlat, plotlat),'k')
ax.plot(xx, (-plotlat, -plotlat),'k')


ax = plt.subplot(313)
im_v = ax.pcolor(df['datetime'], lat_centres, B_norm_lat.T,norm=plt.Normalize(0,1))
ax.set_yticks([-90, -45, 0, 45, 90])
ax.text(0.02,1.05,'(c)' + r'$<|B_R|>/max(|B_R|)$ [nT]                                                            ',
        fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax.set_xlim(xx)
ax.set_ylim((-90,90))
ax.set_ylabel('Latitude [deg]')

#ax.plot([0, 360],[7.5, 7.5],'w--'); ax.plot([0, 360],[-7.5, -7.5],'w--');
#cb = plt.colorbar(im_v); cb.ax.tick_params(labelsize=12)
#cb.ax.set_title(r'$<|\Delta V|>_{CR}$ [km/s]', fontsize = 14)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
axins = inset_axes(ax,
                    width="100%",  # width = 50% of parent_bbox width
                    height="10%",  # height : 5%
                    loc='upper right',
                    bbox_to_anchor=(0.45, 0.65, 0.5, 0.5),
                    bbox_transform=ax.transAxes,
                    borderpad=0,)

cb = fig.colorbar(im_v, cax = axins, orientation = 'horizontal',  pad = -0.1)
cb.ax.tick_params(labelsize=12)

plotlat = (dlat_pole*180/np.pi).value
ax.plot(xx, (87,87),'r')
ax.plot(xx, (90 - plotlat,90 - plotlat),'r')
ax.plot(xx, (-87,-87),'b')
ax.plot(xx, (-90 + plotlat, -90 + plotlat),'b')
plotlat = (dlat_eq*180/np.pi).value
ax.plot(xx, (plotlat, plotlat),'k')
ax.plot(xx, (-plotlat, -plotlat),'k')

# <codecell> Plot the pole-to-equator ratio as a function of time


                                    
                                      
fig = plt.figure()

ax = plt.subplot(311)
plt.plot(df['datetime'],df['SSN']/200, 'k', label = 'SSN/200')
plt.plot(df['datetime'],df['SAI'], 'r', label = 'SAI')
plt.ylabel('SSN')
plt.legend(fontsize = 16)
ax.get_xaxis().set_ticklabels([])
ax.text(0.05,0.9,'(a)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax.set_xlim((datetime(1973,1,1), datetime(2024,1,1)))


ax = plt.subplot(312)
plt.plot(df['datetime'],df['Br_N'], 'r', label = 'North pole')
plt.plot(df['datetime'],df['Br_S'], 'b', label = 'South pole')
plt.plot(df['datetime'],df['Br_Eq'], 'k', label = 'Equator')
plt.ylabel(r'$<|B_R|>$ [nT]')
plt.legend(fontsize = 14)
ax.get_xaxis().set_ticklabels([])
ax.text(0.05,0.9,'(b)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax.set_xlim((datetime(1973,1,1), datetime(2024,1,1)))

ax = plt.subplot(313)
plt.plot(df['datetime'],
         2*(df['Br_Eq'] - df['Br_pole'] )/ (df['Br_Eq'] + df['Br_pole']), 
         'k', label = 'HelioMAS')
plt.ylabel(r'$(<|B_R|>_{EQ} - <|B_R|>_{POLE})/<|B_R|> $')
#ax.get_xaxis().set_ticklabels([])
ax.text(0.05,0.9,'(c)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax.set_xlim((datetime(1973,1,1), datetime(2024,1,1)))   
ax.plot((datetime(1973,1,1), datetime(2024,1,1)),(0,0),'k--')
    
# <codecell> Load OMNI and ULysses data to compute high/low latitude Br ratio

dlat_uly = dlat = 20*np.pi/180 *u.rad #for defining N-pole, S-pole and Equatorial regions 
R_max = 2

#import OMNI data and compute OSF from 20-hour averages of Br
datadir = os.environ['DBOX'] + 'Data_hdf5\\'
omni_1hour = pd.read_hdf(datadir + 'omni_1hour.h5')
ulysses_1hour = pd.read_hdf(datadir + 'ulysses_1hour.h5')

#compute r^2 Br for 
ulysses_1hour['rsqBr'] = ulysses_1hour['SC_R_AU'] * ulysses_1hour['SC_R_AU'] * ulysses_1hour['Br']

#ditch the omni data outside the ulysses mission lifetime
mask = ((omni_1hour['mjd'] >= ulysses_1hour['mjd'][0]) &
        (omni_1hour['mjd'] <= ulysses_1hour['mjd'][len(ulysses_1hour)-1]))
omni_1hour = omni_1hour[mask]


#take 20H means of Br, then 27D means of |Br|
#omni_20H = omni_1hour.resample('20H', on='datetime').mean() 
#omni_20H['datetime'] = omni_20H.index
#omni_20H.reset_index(drop=True, inplace=True)
#omni_20H['absBr'] = abs(omni_20H['Bx_gse'])
omni_1hour['absBr'] = abs(omni_1hour['Bx_gse'])
omni_CR = omni_1hour.resample('27D', on='datetime').mean()   
omni_CR['datetime'] = omni_CR.index
omni_CR.reset_index(drop=True, inplace=True) 

#ulysses_20H = ulysses_1hour.resample('20H', on='datetime').mean() 
#ulysses_20H['datetime'] = ulysses_20H.index
#ulysses_20H.reset_index(drop=True, inplace=True)
#ulysses_20H['absBr'] = abs(ulysses_20H['rsqBr'])
ulysses_1hour['absBr'] = abs(ulysses_1hour['rsqBr'])
ulysses_CR = ulysses_1hour.resample('27D', on='datetime').mean()   
ulysses_CR['datetime'] = ulysses_CR.index
ulysses_CR.reset_index(drop=True, inplace=True)

#find the high-latitude Ulysses observations
mask = ((abs(ulysses_CR['SC_hlat']) >= 90 - dlat_uly.to(u.deg).value) &
        (abs(ulysses_CR['SC_R_AU']) <= R_max))

#add to the previous figure
plt.plot(omni_CR['datetime'][mask], 2*(omni_CR['absBr'][mask]
         - ulysses_CR['absBr'][mask])/(omni_CR['absBr'][mask]
                  + ulysses_CR['absBr'][mask]),'ro', label = 'OMNI/ULysses')
plt.legend(fontsize = 14)         



# <codecell> Plot for MIke Frontiers article


dtheta = (lats[1]-lats[0])*180/np.pi
lat_centres = 90 - lats[0:len(lats)-1]*180/np.pi -dtheta/2

#extract data as an array for ploitting purposes
B_lat = np.ones((CRstop-CRstart,Nlats))*np.nan
B_norm_lat = np.ones((CRstop-CRstart,Nlats))*np.nan
for ilat in range(0,Nlats):
    B_lat[:,ilat] = df['Br_lat_' + str(ilat)].to_numpy()
    B_norm_lat[:,ilat] = df['Br_norm_lat_' + str(ilat)].to_numpy()
    
xx = (datetime(1973,1,1), datetime(2024,1,1))                                      
                                      
fig = plt.figure()

ax = plt.subplot(311)
plt.plot(df['datetime'],df['SSN']/200, 'k', label = 'SSN/200')
plt.plot(df['datetime'],df['SAI'], 'r', label = 'SAI')
#plt.ylabel('SSN')
plt.legend(fontsize = 16)
ax.get_xaxis().set_ticklabels([])
ax.text(0.05,1.05,'(a)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax.set_xlim(xx)


ax = plt.subplot(312)
im_v = ax.pcolor(df['datetime'], lat_centres, B_lat.T, vmin = 0)#,norm=plt.Normalize(0,2))
ax.set_yticks([-90, -45, 0, 45, 90])
ax.get_xaxis().set_ticklabels([])
ax.text(0.05,1.05,'(b)' + r'$<|B_R|>$ [nT]                                                                      ',
        fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax.set_xlim(xx)
ax.set_ylim((-90,90))
ax.set_ylabel('Latitude [deg]')

#ax.plot([0, 360],[7.5, 7.5],'w--'); ax.plot([0, 360],[-7.5, -7.5],'w--');
#cb = plt.colorbar(im_v); cb.ax.tick_params(labelsize=12)
#cb.ax.set_title(r'$<|\Delta V|>_{CR}$ [km/s]', fontsize = 14)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
axins = inset_axes(ax,
                    width="100%",  # width = 50% of parent_bbox width
                    height="10%",  # height : 5%
                    loc='upper right',
                    bbox_to_anchor=(0.35, 0.65, 0.5, 0.5),
                    bbox_transform=ax.transAxes,
                    borderpad=0,)

cb = fig.colorbar(im_v, cax = axins, orientation = 'horizontal',  pad = -0.1)
cb.ax.tick_params(labelsize=12)

#plot the polar and equatorial bands
plotlat = (dlat_pole*180/np.pi).value
ax.plot(xx, (87,87),'r')
ax.plot(xx, (90 - plotlat,90 - plotlat),'r')
ax.plot(xx, (-87,-87),'b')
ax.plot(xx, (-90 + plotlat, -90 + plotlat),'b')
plotlat = (dlat_eq*180/np.pi).value
ax.plot(xx, (plotlat, plotlat),'k')
ax.plot(xx, (-plotlat, -plotlat),'k')
  


ax = plt.subplot(313)
plt.plot(df['datetime'],
         2*(df['Br_Eq'] - df['Br_pole'] )/ (df['Br_Eq'] + df['Br_pole']), 
         'k', label = 'HelioMAS')
#plt.ylabel(r'$(<|B_R|>_{EQ} - <|B_R|>_{P})$' + '\n' + r'$/<|B_R|> $')
#ax.get_xaxis().set_ticklabels([])
ax.text(0.05,1.05,'(c) ' + r'$(<|B_R|>_{EQ} - <|B_R|>_{P})$ / $<|B_R|> $', 
        fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax.set_xlim((datetime(1973,1,1), datetime(2024,1,1)))   
ax.plot((datetime(1973,1,1), datetime(2024,1,1)),(0,0),'k--')    
ax.set_ylim((-1,1))
#plt.tight_layout()               