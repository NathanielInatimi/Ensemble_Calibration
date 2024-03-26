# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 10:30:55 2023

@author: mathewjowens

A script to compute streamer belt width consistent with the OSF continuity equation
"""


# script to compute OSF from R. Uses Geomagnetic and OMNI OSF estimates to 
#first compute OSF loss term.

import numpy as np
import pandas as pd
from datetime import datetime
import os as os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import helio_time as htime
import sunspots 

plt.rcParams.update({'font.size': 12})
hfont = {'fontname':'Tahoma'}

nphase = 11
startOSF = 0.1 #x10^14 Wb
source_ref = 1 # the form of teh OSF source term

#start and stop years for loss rate calculation
startyr = 1878.9
stopyr = 2019.96

#OSF computed from annual B and V requires a correction - see OSF_strahl_Brmod_BVgeo.py
coefficients = np.array([ 0.66544633, -1.15354585])


osfRfilepath = os.path.join(os.environ['DBOX'], 'Papers_WIP','_archive','2017_GlobalVsw1600','JGR_Lockwood_OSF3_supplementarydata.txt' )
osfGEOfilepath = os.path.join(os.environ['DBOX'], 'Data','Geomag_B_V_2023.txt' )
omnipath = os.path.join(os.environ['DBOX'],'Data_hdf5','omni_1hour.h5')

figdir = os.path.join(os.environ['DBOX'], 'Apps','Overleaf','SSN from OSF','figures')

#=============================================================================
#read in the solar minimum times
solarmintimes_df = sunspots.LoadSolarMinTimes()

#=============================================================================
#Read in the sunspot data 
ssn_df = sunspots.LoadSSN(download_now = True)

#create annual SSN record
ssn_1y = ssn_df.resample('1Y', on='datetime').mean() 
ssn_1y['datetime'] = htime.mjd2datetime(ssn_1y['mjd'].to_numpy())
ssn_1y.reset_index(drop=True, inplace=True)
#recompute phase, as it will have been averaged over 0/2pi
ssn_1y['phase'] = sunspots.solar_cycle_phase(ssn_1y['mjd'])



#Use annual SILSO SSN back to 1700
#==============================================================================
ssn_1y = sunspots.LoadSSNyr(download_now = True)
ssn_1y['smooth'] = ssn_1y['ssn']
#==============================================================================


# alter the SILSO annual SSN
#==============================================================================
car_yr = np.array([1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1726]) +0.5
car_ssn= np.array([14.8, 26.1, 14.6, 35.7, 23.0, 16.9, np.nan, np.nan, np.nan, 17.3, 77.0])

mask = (ssn_1y['fracyear'] >= 1716) & (ssn_1y['fracyear'] <= 1727)

car_ratio = np.nanmedian(car_ssn/ssn_1y.loc[mask, 'smooth'])

#reduce SILSO before 1755 by the required ratio
mask  = (ssn_1y['fracyear'] <= 1755)
ssn_1y.loc[mask,'smooth'] = ssn_1y.loc[mask,'smooth'] * car_ratio

#==============================================================================

nt_ssn = len(ssn_1y)


#Read in the BB GSN data
gsn_df = sunspots.LoadBBgsn()
gsn_df['phase'] = sunspots.solar_cycle_phase(gsn_df['mjd'])


# #load OSF(R) and associated streamer belt width
# osfR = pd.read_csv(osfRfilepath, header = 27,
#                      delim_whitespace=True, index_col=False,
#                      names = ['year', 'Fs_min', 'Fs', 'Fs_max', 'Fs_group', 'Fch', 'Fsb','sinLSB'])
#load OSF(GEO) and associated streamer belt width
osfG = pd.read_csv(osfGEOfilepath, header = 24,
                     delim_whitespace=True, index_col=False,
                     names = ['year', 'R', 'aaH', 'IDV1d', 'IDV', 'IHV', 
                              'Bmin','B','Bmax', 'Vmin', 'V', 'Vmax',
                              'OSFmin','OSF','OSFmax'])
#compute MJD
osfG['mjd'] = htime.doyyr2mjd(366/2,np.floor(osfG['year']).astype(int))

#Double Mike's OSF values, as he does signed flux
osfG['OSF'] = 2*osfG['OSF']
osfG['OSFmin'] = 2*osfG['OSFmin']
osfG['OSFmax'] = 2*osfG['OSFmax']

#compute the constant to convert |Br| at 1 AU to total HMF
AU=149598000
Tsid = sidereal_period = 25.38 * 24*60*60
Fconst=(1e-3)*4*np.pi*AU*AU/(1e14)
#compute the OSF from ideal Parker angle
vrot_geo = 2 * np.pi * (osfG['mjd']*0 +1) * AU / Tsid
phi_geo = np.arctan(vrot_geo/osfG['V'])
Br_geo = osfG['B'] * np.cos(phi_geo)
osfG['OSF_BV'] = Br_geo * Fconst 

#apply the correction
osfG['OSF_BV_corr'] = np.polyval(coefficients, osfG['OSF_BV'])


#compute upper and lower limits in the same way.
phi_geo = np.arctan(vrot_geo/osfG['Vmin'])
Br_geo = osfG['Bmin'] * np.cos(phi_geo)
osfG['OSF_BVmin'] = Br_geo * Fconst 
osfG['OSF_BV_corr_min'] = np.polyval(coefficients, osfG['OSF_BVmin'])

phi_geo = np.arctan(vrot_geo/osfG['Vmax'])
Br_geo = osfG['Bmax'] * np.cos(phi_geo)
osfG['OSF_BVmax'] = Br_geo * Fconst 
osfG['OSF_BV_corr_max'] = np.polyval(coefficients, osfG['OSF_BVmax'])



#interpolate silso sunspot numebr on to the OSF timestep
Nt = len(osfG)
fracyears = osfG['year'].to_numpy()
intyears = np.floor(fracyears).astype(int)
osfG['mjd'] = htime.doyyr2mjd(365.25*(fracyears - intyears),  intyears)
osfG['datetime'] = htime.mjd2datetime(osfG['mjd'].to_numpy())
osfG['Rsilso'] = np.interp(osfG['mjd'],ssn_1y['mjd'],ssn_1y['ssn'])
osfG['GSN'] = np.interp(osfG['mjd'],gsn_df['mjd'],gsn_df['ssn'], left = np.nan, right=np.nan)
#compute the source term 



#load and process in the OMNI data to compute OSF 

AU=149598000
br_window = '20H'

omni_1hour = pd.read_hdf(omnipath)
omni_brwindow_nans = omni_1hour.resample(br_window, on='datetime').mean() 
#omni_brwindow_nans['datetime'] = htime.mjd2datetime(omni_brwindow_nans['mjd'].to_numpy())
#omni_brwindow_nans.reset_index(drop=True, inplace=True)

#compute the constant to convert |Br| at 1 AU to total HMF
Fconst=(1e-3)*4*np.pi*AU*AU/(1e14)

omni_brwindow_nans['OSF'] = np.abs(omni_brwindow_nans['Bx_gse']) * Fconst

omni_1d = pd.DataFrame(index = omni_brwindow_nans.index)
omni_1d['OSF'] = omni_brwindow_nans['OSF']
omni_1d['mjd'] = omni_brwindow_nans['mjd']
omni_1d['datetime'] = omni_brwindow_nans.index

#make annual means
omni_1y = omni_1d.resample('1Y', on='datetime').mean() 
omni_1y['datetime'] = htime.mjd2datetime(omni_1y['mjd'].to_numpy())
omni_1y.reset_index(drop=True, inplace=True)

#compute the source term over the OMNI interval
omni_1y['Rsilso'] = np.interp(omni_1y['mjd'],ssn_1y['mjd'],ssn_1y['ssn'], left = np.nan, right=np.nan)



#compute the source term 
osfG['OSF_source'] = sunspots.compute_osf_source(osfG['Rsilso'])
osfG['OSF_source_GSN'] = sunspots.compute_osf_source(osfG['GSN'])
omni_1y['OSF_source'] = sunspots.compute_osf_source(omni_1y['Rsilso'])
ssn_1y['source'] =  sunspots.compute_osf_source(ssn_1y['smooth'])
gsn_df['source'] =  sunspots.compute_osf_source(gsn_df['ssn'])





# <codecell> get the average sunspot variation over the solar cycle and make
# an analogue forecast

solarmin_mjd = solarmintimes_df['mjd']

SPE_SSN_phase, SPE_SSN_NORM_AVG, SPE_SSN_NORM_STD = sunspots.ssn_analogue_forecast(ssn_df, 
                        solarmin_mjd,  nphase = 11*12, peak_ssn = 140, plotnow = True)



# <codecell> compute the loss term and generate the SPE with phase
import sunspots 
 
 
#create a phase series   
osfG['phase'] = sunspots.solar_cycle_phase(osfG['mjd'] )

mask = (osfG['year'] >= startyr) & (osfG['year'] <= stopyr)

#compte the loss term
osf = osfG.loc[mask,'OSF_BV_corr'].to_numpy() 
mjd =  osfG.loc[mask,'mjd'].to_numpy() 
ssn = osfG.loc[mask,'Rsilso'].to_numpy() 

# osf = omni_1y['OSF'] 
# mjd =  omni_1y['mjd'] 
# source = omni_1y['OSF_source'] 

SPE_phase, SPE_LOSS_AVG, SPE_LOSS_STD, SPE_LOSS  = sunspots.compute_osfloss_SPE(mjd, osf, ssn, 
                       nphase = 11, plotnow = True, source_ref = source_ref)


#compte the loss term for GSN
# ssn = osfG.loc[mask,'GSN'].to_numpy() 
# SPE_GSN_phase, SPE_GSN_LOSS_AVG, SPE_GSN_LOSS_STD, SPE_GSN_LOSS  = sunspots.compute_osfloss_SPE(mjd, osf, ssn, 
#                        nphase = 11, plotnow = True)
# <codecell> Compute OSF using SPE loss term

#find the solar cycle phase over the whole sunspot record
ssn_1y['phase'] = sunspots.solar_cycle_phase(ssn_1y['mjd'] )
            
#compute the fractional loss term from the phase SPE
ssn_1y['loss_frac'] = np.interp(ssn_1y['phase'], 
                                SPE_phase, SPE_LOSS_AVG, period = 2*np.pi)

# gsn_df['loss_frac'] = np.interp(gsn_df['phase'], 
#                                 SPE_GSN_phase, SPE_GSN_LOSS_AVG, period = 2*np.pi)

# plt.figure()
# plt.plot(ssn_df['datetime'],ssn_df['ssn'])
# plt.plot(ssn_1y['datetime'],ssn_1y['ssn'])


ssn_1y.loc[0,'OSF'] = startOSF

ssn_1y['F_SB'] = 0
ssn_1y['F_CH'] = startOSF/4
ssn_1y['S_CH'] = 0
ssn_1y['chi_SB'] = 0

weights = [0.1, 0.5, 0.3, 0.075, 0.025]
chi_CH = 0.20
#compute OSF using average loss profile
for n in range(0, nt_ssn-1):
    #compute the absolute loss rate
    ssn_1y.loc[n,'loss_abs'] = ssn_1y.loc[n,'OSF'] * ssn_1y.loc[n,'loss_frac']
    
    #compute the OSF for the next time step
    ssn_1y.loc[n+1,'OSF'] = ssn_1y.loc[n,'OSF'] + ssn_1y.loc[n,'source'] - ssn_1y.loc[n,'loss_abs'] 
    
    #osf can't be negative
    if ssn_1y.loc[n+1,'OSF'] < 0.0:
        ssn_1y.loc[n+1,'OSF'] = 0.0
    
    #the fractional loss of SB flux + CH must equal the total
    ssn_1y.loc[n,'chi_SB'] = ssn_1y.loc[n,'loss_frac'] - chi_CH
    
    #some of the SB flux evolves into CH flux

    S_CH = 0 # (ssn_1y.loc[n,'source'] - ssn_1y.loc[n,'OSF'] * ssn_1y.loc[n,'chi_SB']) * weights[0]
    if n>1:
        S_CH = S_CH + (ssn_1y.loc[n-1,'source'] - ssn_1y.loc[n-1,'OSF'] * ssn_1y.loc[n-1,'chi_SB']) * weights[0]
    if n>2:
        S_CH = S_CH + (ssn_1y.loc[n-2,'source'] - ssn_1y.loc[n-2,'OSF'] * ssn_1y.loc[n-2,'chi_SB']) * weights[1]
    if n>3:
        S_CH = S_CH + (ssn_1y.loc[n-3,'source'] - ssn_1y.loc[n-3,'OSF'] * ssn_1y.loc[n-3,'chi_SB']) * weights[2]
    if n>4:
        S_CH = S_CH + (ssn_1y.loc[n-4,'source'] - ssn_1y.loc[n-4,'OSF'] * ssn_1y.loc[n-4,'chi_SB']) * weights[3]
    if n>5:
        S_CH = S_CH + (ssn_1y.loc[n-5,'source'] - ssn_1y.loc[n-5,'OSF'] * ssn_1y.loc[n-5,'chi_SB']) * weights[4]
    
    ssn_1y.loc[n,'S_CH'] = S_CH

    #new flux emerges into the streamer belt, there disconnection loss and CH loss
    ssn_1y.loc[n+1,'F_SB'] = ssn_1y.loc[n,'F_SB'] \
                            + ssn_1y.loc[n,'source'] \
                            - ssn_1y.loc[n,'chi_SB'] * ssn_1y.loc[n,'OSF'] \
                                - ssn_1y.loc[n,'S_CH']
    if ssn_1y.loc[n+1,'F_SB'] < 0:
        ssn_1y.loc[n+1,'F_SB'] = 0
      
    #CH flux has a source from the SB and decays with a constant time
    ssn_1y.loc[n+1,'F_CH'] = ssn_1y.loc[n,'F_CH'] - chi_CH*ssn_1y.loc[n,'OSF'] + ssn_1y.loc[n,'S_CH']
    
    if ssn_1y.loc[n+1,'F_CH'] < 0:
        ssn_1y.loc[n+1,'F_CH'] = 0
    
    ssn_1y.loc[n,'SBwidth'] = np.sin(1.0 - ssn_1y.loc[n,'F_CH']/ssn_1y.loc[n,'OSF'])
    
    
fig = plt.figure(figsize=(9,6))

ax = plt.subplot(4,1,1)
ax.plot(ssn_1y['datetime'], ssn_1y['OSF'], 'b', label = 'SSN-based model')
ax.plot(osfG['datetime'], osfG['OSF_BV_corr'],'r' , label = 'Geomagnetic')
ax.plot(omni_1y['datetime'], omni_1y['OSF'],'k', label = 'OMNI')

ax.fill_between(ssn_1y['datetime'], ssn_1y['ssn']*0, ssn_1y['ssn']/100, label = 'SSN/100', facecolor ='grey')
#plt.plot(osfG['datetime'], osfG['OSFmin'],'r--')
#plt.plot(osfG['datetime'], osfG['OSFmax'],'r--')
ax.set_ylim((0,14.5))
ax.set_ylabel(r'OSF [x$10^14$ Wb]', fontsize = 12)
ax.set_xlabel(r'Year', fontsize = 12)
ax.legend(ncol = 4, fontsize = 12, loc = 'upper left')

#interpolate the SSN OSF to the GEO time step to compuate MAE and correlation
osfG['OSF_SSN'] = np.interp(osfG['mjd'], ssn_1y['mjd'], ssn_1y['OSF'])
mae = np.nanmean(abs(osfG['OSF_SSN'] - osfG['OSF_BV_corr']))

print('MAE = ' + str(mae))
rl =  np.corrcoef(osfG['OSF_SSN'], osfG['OSF_BV_corr'])

ax.set_title(r'$r_L$ = ' + "{:.2f}".format(rl[0,1]), fontsize=12) 

# ax = plt.subplot(4,1,2)
# ax.plot(ssn_1y['datetime'], ssn_1y['loss_frac'], 'k', label = 'f')
# ax.plot(ssn_1y['datetime'], ssn_1y['chi_SB'], 'b', label = 'f SB')
# ax.legend(ncol = 3, fontsize = 12, loc = 'upper left')

ax = plt.subplot(4,1,2)
ax.plot(ssn_1y['datetime'], ssn_1y['OSF'], 'k', label = 'OSF')
ax.plot(ssn_1y['datetime'], ssn_1y['F_SB'], 'b', label = 'F_SB')
ax.plot(ssn_1y['datetime'], ssn_1y['F_CH'], 'r', label = 'F_CH')
ax.legend(ncol = 3, fontsize = 12, loc = 'upper left')

ax = plt.subplot(4,1,3)
ax.plot(ssn_1y['datetime'], ssn_1y['source'], 'k', label = 'source')
ax.plot(ssn_1y['datetime'], ssn_1y['S_CH'], 'r', label = 'S_CH')
ax.legend(ncol = 3, fontsize = 12, loc = 'upper left')

ax = plt.subplot(4,1,4)
#ax.plot(ssn_1y['datetime'], np.arcsin(ssn_1y['SBwidth'])*180/np.pi, 'k', label = 'SBwidth')
ax.plot(ssn_1y['datetime'], ssn_1y['SBwidth'], 'k', label = 'SBwidth')
ax.legend(ncol = 3, fontsize = 12, loc = 'upper left')


# <codecell> Use teh sunspot.py function with Rsilso

yrs = ssn_1y['fracyear'].to_numpy()
ssn = ssn_1y['smooth'].to_numpy()
min_yrs = solarmintimes_df['fracyear'].to_numpy()

OSF_model, SBwidth = sunspots.osf_and_streamer_belt_width(yrs, ssn, min_yrs, 
                                                                        SPE_phase, SPE_LOSS_AVG, 
                                startOSF = 8, startFch = 2, startFsb = 0,
                                S_CH_weights = [0.1, 0.5, 0.3, 0.05, 0.05],
                                chi_CH = 0.20, plotnow = False, source_ref = source_ref)

inclination = sunspots.HCSinclination(yrs, min_yrs)

plt.figure()

ax = plt.subplot(4,1,1)
ax.plot(yrs, OSF_model, label = 'model');
ax.plot(osfG['year'], osfG['OSF_BV_corr'], label = 'geomag');
ax.set_ylabel('OSF');

ax = plt.subplot(4,1,2)
ax.plot(yrs, ssn, label = 'run');
ax.set_ylabel('SSN');

ax = plt.subplot(4,1,3)
ax.plot(yrs, SBwidth, label = 'run');
ax.set_ylabel('sin(SBW)');
#ax.plot(osfG['year'], geo_concensus_sbw[:,1], 'r', label = 'Forward model')

# <codecell> plot results


yrlines = [1724, 1733]
colours = ['b', 'g', 'k']

startyr = yrlines[0] - 20
stopyr = yrlines[-1] + 20

fig = plt.figure()

gs = gridspec.GridSpec(5, 2)

ax = fig.add_subplot(gs[0, 0:2])
ax.plot(yrs, OSF_model, 'r', label = 'OSF model'); 
ax.plot(osfG['year'], osfG['OSF_BV_corr'], 'k', label = 'OSF geomag');   
ax.set_ylabel(r'OSF [x10$^{14}$ Wb]')
ax.set_xlim([startyr, stopyr])
ax.legend(loc = 'upper left')
ax.set_ylim([0, 10])
yy=ax.get_ylim()
for count, yrline in enumerate(yrlines):
    ax.plot([yrline, yrline], yy, color=colours[count])
ax.set_xlabel('Year (CE)');
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

ax = fig.add_subplot(gs[1, 0:2])
ax.plot(yrs, ssn, 'k', label = 'SILSOv2'); 
ax.set_ylabel('SSN');
ax.set_xlim([startyr, stopyr])
ax.legend(loc = 'upper left')
ax.set_ylim([0, 100])
yy=ax.get_ylim()
for count, yrline in enumerate(yrlines):
    ax.plot([yrline, yrline], yy, color=colours[count])
ax.set_xticklabels([])

ax = fig.add_subplot(gs[2, 0:2])
ax.plot(yrs, np.arcsin(SBwidth)*180/np.pi, 'r', label = 'OSF model')
ax.set_ylabel('SB width [deg]')
ax.set_xlim([startyr, stopyr])
ax.legend(loc = 'upper left')
yy=ax.get_ylim()
for count, yrline in enumerate(yrlines):
    ax.plot([yrline, yrline], yy, color=colours[count])
ax.set_xticklabels([])

ax = fig.add_subplot(gs[3, 0:2])
ax.plot(yrs, inclination*180/np.pi, 'r', label = 'OSF model')
ax.set_ylabel('Inclination [deg]')
ax.set_xlim([startyr, stopyr])
ax.legend(loc = 'upper left')
yy=ax.get_ylim()
for count, yrline in enumerate(yrlines):
    ax.plot([yrline, yrline], yy, color=colours[count])
ax.set_xticklabels([])






for count, yrline in enumerate(yrlines):
   
    rot_angle = 0*np.pi/180
    #get the SBW and inclination at the required time
    sbw = np.interp(yrline, yrs, np.arcsin(SBwidth))
    inc = np.interp(yrline, yrs, np.arcsin(inclination))
    
    
    ax = fig.add_subplot(gs[4, count])
    
    
    # Circle parameters
    center = (0, 0)  # Center coordinates (x, y)
    radius = 1       # Radius of the circle
    # Generate theta values for the circle
    theta = np.linspace(0, 2*np.pi, 100)
    # Calculate x and y coordinates of the circle points
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    
    # Plot the circle
    ax.plot(x, y, color = 'k')
    
    # Set axis limits
    ax.set_xlim(center[0] - radius - 1, center[0] + radius + 1)
    ax.set_ylim(center[1] - radius - 1, center[1] + radius + 1)
    
    # Add labels and legend
    ax.set_aspect('equal')
    
    
    #draw the rotational axis
    
    line_length=1.2
    line_x = np.array([-radius*line_length, radius*line_length]) * np.sin(rot_angle)
    line_y = np.array([-radius*line_length, radius*line_length]) * np.cos(rot_angle)
    ax.plot(line_x, line_y, 'k')   
       
    #draw teh equator 
    eq_angle = rot_angle + np.pi/2
    line_length=1
    line_x = np.array([-radius*line_length, radius*line_length]) * np.sin(eq_angle)
    line_y = np.array([-radius*line_length, radius*line_length]) * np.cos(eq_angle)
    ax.plot(line_x, line_y, 'k--') 
    
    #sbw = 45*np.pi/180
    #inc = 10*np.pi/180
    
    
    #plot the upper and lower edges from the inclination
    
    #first the RHS
    th_start = eq_angle - sbw - inc -np.pi/2
    th_end = eq_angle + sbw - inc  -np.pi/2
    # Define the vertices of the sector
    line_length = 1.6
    vertices = np.array([[0, 0]])
    for th in np.linspace(th_start, th_end, 100):
        x = np.cos(th) * line_length
        y = np.sin(th) * line_length
        vertices = np.append(vertices, [[x, y]], axis=0)
    
    # Shade in the sector
    ax.fill(vertices[:, 0], vertices[:, 1], color=colours[count], alpha=0.5)
    
    th_start = eq_angle - sbw + inc  -np.pi/2
    th_end = eq_angle + sbw + inc  -np.pi/2
    # Define the vertices of the sector
    line_length = 1.8
    vertices = np.array([[0, 0]])
    for th in np.linspace(th_start, th_end, 100):
        x = np.cos(th) * line_length
        y = np.sin(th) * line_length
        vertices = np.append(vertices, [[x, y]], axis=0)
    
    # Shade in the sector
    ax.fill(vertices[:, 0], vertices[:, 1], color=colours[count], alpha=0.5)
    
    #then the LHS
    th_start = eq_angle - sbw - inc   +np.pi -np.pi/2
    th_end = eq_angle + sbw - inc   +np.pi -np.pi/2
    # Define the vertices of the sector
    line_length = 1.6
    vertices = np.array([[0, 0]])
    for th in np.linspace(th_start, th_end, 100):
        x = np.cos(th) * line_length
        y = np.sin(th) * line_length
        vertices = np.append(vertices, [[x, y]], axis=0)
    
    # Shade in the sector
    ax.fill(vertices[:, 0], vertices[:, 1], color=colours[count], alpha=0.5)
    
    th_start = eq_angle - sbw+ inc +np.pi -np.pi/2
    th_end = eq_angle + sbw + inc +np.pi -np.pi/2
    # Define the vertices of the sector
    line_length = 1.8
    vertices = np.array([[0, 0]])
    for th in np.linspace(th_start, th_end, 100):
        x = np.cos(th) * line_length
        y = np.sin(th) * line_length
        vertices = np.append(vertices, [[x, y]], axis=0)
    
    # Shade in the sector
    ax.fill(vertices[:, 0], vertices[:, 1], color=colours[count], alpha=0.5)


    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Remove axis spines
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    #ax.set_xlabel(runtypes[count])
    
    
    circle = plt.Circle((0, 0), 1, fill=True, color='white')
    ax.add_patch(circle)

# <codecell> Use the 14C estimates
#load in the data from the 

fileheader = 'SSN_14C_1190_to_1260_' 
runtypes = ['best', 'upper', 'lower']

osf_save_dir = os.path.join(os.getenv('DBOX'),'Papers_WIP','_coauthor','HisashiHayakawa')


startyr = 1205#min_years[0]
stopyr = 1255#min_years[-1]


# min_yrs = []
yrs = []
ssn = []
#OSF_model = np.empty((stopyr-startyr,len(runtypes)))
OSF_orig = []
SBwidth = []
inclination = []
min_yrs = []
for count, run in enumerate(runtypes):
    filename = fileheader + run + '_minyears.dat'
    df_minyears = pd.read_csv(os.path.join(osf_save_dir, filename), delimiter=',')
    min_yrs.append(df_minyears['Cycle start year'].to_numpy())
    
    filename = fileheader + run + '.dat'
    df_data = pd.read_csv(os.path.join(osf_save_dir,filename), delimiter=',')
    thisyrs = df_data['Year'].to_numpy()
    
    mask = (thisyrs >=startyr) & (thisyrs <= stopyr)
    yrs.append(thisyrs[mask])
    ssn.append(df_data['SSN_recon'].loc[mask].to_numpy())
    OSF_orig.append(df_data['OSF_orig'].loc[mask].to_numpy())
    SBwidth.append(df_data['SBwidth'].loc[mask].to_numpy())
    inclination.append(df_data['HCSinc'].loc[mask].to_numpy())
    
    
#     OSF_model[:, count], SBwidth[:, count] = sunspots.osf_and_streamer_belt_width(yrs[count], ssn[count], min_yrs[count], 
#                                                                             SPE_phase, SPE_LOSS_AVG, 
#                                     startOSF = 8, startFch = 2, startFsb = 0,
#                                     S_CH_weights = [0.1, 0.5, 0.3, 0.1, 0],
#                                     chi_CH = 0.22, plotnow = False)
    
    
#     #Vband, lats, inclination = sunspots.Vlat_from_SBW(yrs, min_yrs, SBwidth, plotnow = False)



# <codecell> plot results
yrline = 1230
colours = ['k', 'r', 'b']

fig = plt.figure()

gs = gridspec.GridSpec(5, 3)

ax = fig.add_subplot(gs[0, 0:3])
for count, run in enumerate(runtypes):
    ax.plot(yrs[count], OSF_orig[count], color = colours[count], label = run);   
ax.plot()
ax.set_ylabel(r'OSF [x10$^{14}$ Wb]')
ax.set_xlim([startyr, stopyr])
ax.legend(loc = 'upper left')
yy=ax.get_ylim()
ax.plot([yrline, yrline], yy, 'k--')
ax.set_xlabel('Year (CE)');
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

ax = fig.add_subplot(gs[1, 0:3])
for count, run in enumerate(runtypes):
    ax.plot(yrs[count], ssn[count], color = colours[count], label = run)
ax.set_ylabel('SSN');
ax.set_xlim([startyr, stopyr])
ax.legend(loc = 'upper left')
yy=ax.get_ylim()
ax.plot([yrline, yrline], yy, 'k--')
ax.set_xticklabels([])

ax = fig.add_subplot(gs[2, 0:3])
for count, run in enumerate(runtypes):
    ax.plot(yrs[count], np.arcsin(SBwidth[count])*180/np.pi, color = colours[count], label = run)
ax.set_ylabel('SB width [deg]')
ax.set_xlim([startyr, stopyr])
ax.legend(loc = 'upper left')
yy=ax.get_ylim()
ax.plot([yrline, yrline], yy, 'k--')
ax.set_xticklabels([])

ax = fig.add_subplot(gs[3, 0:3])
for count, run in enumerate(runtypes):
    ax.plot(yrs[count], inclination[count]*180/np.pi, color = colours[count], label = run)
ax.set_ylabel('Inclination [deg]')
ax.set_xlim([startyr, stopyr])
ax.legend(loc = 'upper left')
yy=ax.get_ylim()
ax.plot([yrline, yrline], yy, 'k--')
ax.set_xticklabels([])









for count in range(0,3):
    #get the SBW and inclination at the required time
    sbw = np.interp(yrline, yrs[count], np.arcsin(SBwidth[count]))
    inc = np.interp(yrline, yrs[count], inclination[count])
    rot_angle = - 20*np.pi/180
    
    ax = fig.add_subplot(gs[4, count])
    
    
    # Circle parameters
    center = (0, 0)  # Center coordinates (x, y)
    radius = 1       # Radius of the circle
    # Generate theta values for the circle
    theta = np.linspace(0, 2*np.pi, 100)
    # Calculate x and y coordinates of the circle points
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    
    # Plot the circle
    ax.plot(x, y, color = colours[count])
    
    # Set axis limits
    ax.set_xlim(center[0] - radius - 1, center[0] + radius + 1)
    ax.set_ylim(center[1] - radius - 1, center[1] + radius + 1)
    
    # Add labels and legend
    ax.set_aspect('equal')
    
    
    #draw the rotational axis
    
    line_length=1.2
    line_x = np.array([-radius*line_length, radius*line_length]) * np.sin(rot_angle)
    line_y = np.array([-radius*line_length, radius*line_length]) * np.cos(rot_angle)
    ax.plot(line_x, line_y, 'k')   
       
    #draw teh equator 
    eq_angle = rot_angle + np.pi/2
    line_length=1
    line_x = np.array([-radius*line_length, radius*line_length]) * np.sin(eq_angle)
    line_y = np.array([-radius*line_length, radius*line_length]) * np.cos(eq_angle)
    ax.plot(line_x, line_y, 'k--') 
    
    #sbw = 45*np.pi/180
    #inc = 10*np.pi/180
    
    
    #plot the upper and lower edges from the inclination
    
    #first the RHS
    th_start = eq_angle - sbw - inc  -np.pi/4
    th_end = eq_angle + sbw - inc   -np.pi/4
    # Define the vertices of the sector
    line_length = 1.6
    vertices = np.array([[0, 0]])
    for th in np.linspace(th_start, th_end, 100):
        x = np.cos(th) * line_length
        y = np.sin(th) * line_length
        vertices = np.append(vertices, [[x, y]], axis=0)
    
    # Shade in the sector
    ax.fill(vertices[:, 0], vertices[:, 1], color = colours[count], alpha=0.5)
    
    th_start = eq_angle - sbw + inc  -np.pi/4
    th_end = eq_angle + sbw + inc  -np.pi/4
    # Define the vertices of the sector
    line_length = 1.8
    vertices = np.array([[0, 0]])
    for th in np.linspace(th_start, th_end, 100):
        x = np.cos(th) * line_length
        y = np.sin(th) * line_length
        vertices = np.append(vertices, [[x, y]], axis=0)
    
    # Shade in the sector
    ax.fill(vertices[:, 0], vertices[:, 1], color = colours[count], alpha=0.5)
    
    #then the LHS
    th_start = eq_angle - sbw - inc  + np.pi -np.pi/4
    th_end = eq_angle + sbw - inc  + np.pi -np.pi/4
    # Define the vertices of the sector
    line_length = 1.6
    vertices = np.array([[0, 0]])
    for th in np.linspace(th_start, th_end, 100):
        x = np.cos(th) * line_length
        y = np.sin(th) * line_length
        vertices = np.append(vertices, [[x, y]], axis=0)
    
    # Shade in the sector
    ax.fill(vertices[:, 0], vertices[:, 1], color = colours[count], alpha=0.5)
    
    th_start = eq_angle - sbw + inc  + np.pi -np.pi/4
    th_end = eq_angle + sbw + inc  + np.pi -np.pi/4
    # Define the vertices of the sector
    line_length = 1.8
    vertices = np.array([[0, 0]])
    for th in np.linspace(th_start, th_end, 100):
        x = np.cos(th) * line_length
        y = np.sin(th) * line_length
        vertices = np.append(vertices, [[x, y]], axis=0)
    
    # Shade in the sector
    ax.fill(vertices[:, 0], vertices[:, 1], color = colours[count], alpha=0.5)
    
    
    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Remove axis spines
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    ax.set_xlabel(runtypes[count])
    
    
    circle = plt.Circle((0, 0), 1, fill=True, color='white')
    ax.add_patch(circle)
  
#plt.tight_layout()
# ax = plt.subplot(4,1,2)
# ax.plot(yrs, inclination*180/np.pi , 'k')
# ax.set_ylabel('HCS tilt [degrees]')
# ax.set_xlim([startyr, stopyr])

# ax = plt.subplot(4,1,3)
# ax.plot(yrs, np.arcsin(SBwidth) , 'k');
# ax.set_xlim([startyr, stopyr]);
# ax.set_ylabel('sin(SBW)');

# ax = plt.subplot(4,1,4)
# im = ax.pcolor(yrs, lats*180/np.pi, np.rot90(Vband),  vmin = 300, vmax = 750)
# ax.set_ylim([-90, 90]); #set(gca,'YTick',[-90 -60 -30 0 30 60 90]);
# #set(gca,'CLim',[250 850]);                               
# ax.set_xlim([startyr, stopyr])
# # colorbar; 
# # title('V_{SW} [kms/]');
# ax.set_ylabel('Helio lat [deg]');
# ax.set_xlabel('Year')

# #offset colourbar
# axins = inset_axes(ax,                                       
#                     width="100%",  # width = 50% of parent_bbox width
#                     height="100%",  # height : 5%
#                     loc='upper right',
#                     bbox_to_anchor=(1.03, 0.0, 0.02, 1),
#                     bbox_transform=ax.transAxes,
#                     borderpad=0,)
# cb = fig.colorbar(im, cax = axins, orientation = 'vertical',  pad = -0.1)
# cb.ax.tick_params(labelsize=10)
# axins.text(0.99,1.07,r'V$_{SW}$ [km/s]' , 
#         fontsize = 12          , transform=ax.transAxes, backgroundcolor = 'w')


