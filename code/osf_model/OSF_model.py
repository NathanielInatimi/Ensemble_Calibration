# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 17:17:41 2023

@author: mathewjowens
"""

# script to compute OSF from R. Uses Geomagnetic and OMNI OSF estimates to 
#first compute OSF loss term.

import numpy as np
import pandas as pd
from datetime import datetime
import os as os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

import helio_time as htime
import sunspots 
import mplot

plt.rcParams.update({'font.size': 12})
hfont = {'fontname':'Tahoma'}

nphase = 11
startOSF = 8 #x10^14 Wb

#start and stop years for loss rate calculation
startyr = 1878.9
stopyr = 2019.96

#OSF computed from annual B and V requires a correction - see OSF_strahl_Brmod_BVgeo.py
#coefficients = np.array([ 0.66544633, -1.15354585]) # OLS
coefficients = np.array([0.665485855512134, -1.154618324677229]) # ODR

#osfRfilepath = os.environ['DBOX'] + 'Papers_WIP\\_archive\\2017_GlobalVsw1600\\JGR_Lockwood_OSF3_supplementarydata.txt' 
osfGEOfilepath = os.environ['DBOX'] + 'Data\\Geomag_B_V_2023.txt' 
omnipath = os.environ['DBOX'] + '\\Data_hdf5\omni_1hour.h5'

figdir = os.environ['DBOX'] + 'Apps\\Overleaf\\SSN from OSF\\figures\\'

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
nt_ssn = len(ssn_1y)
#recompute phase, as it will have been averaged over 0/2pi
ssn_1y['phase'] = sunspots.solar_cycle_phase(ssn_1y['mjd'])


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


#adjust the mean SSN variation to make it hit zero at solar min.
ssnmin = np.nanmin(SPE_SSN_NORM_AVG)
SPE_SSN_NORM_AVG = SPE_SSN_NORM_AVG - ssnmin/3
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
                       nphase = 11, plotnow = True)


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

#compute OSF using average loss profile
for n in range(0, nt_ssn-1):
    #compute the absolute loss rate
    ssn_1y.loc[n,'loss_abs'] = ssn_1y.loc[n,'OSF'] * ssn_1y.loc[n,'loss_frac']
    
    #compute the OSF for the next time step
    ssn_1y.loc[n+1,'OSF'] = ssn_1y.loc[n,'OSF'] + ssn_1y.loc[n,'source'] - ssn_1y.loc[n,'loss_abs'] 
    
    #osf can't be negative
    if ssn_1y.loc[n+1,'OSF'] < 0.0:
        ssn_1y.loc[n+1,'OSF'] = 0.0
    
fig = plt.figure(figsize = (12,5))
gs = gridspec.GridSpec(1, 3)


#interpolate this onto the Geomagnetic dataframe
osfG['OSF_SSN'] = np.interp(osfG['mjd'],ssn_1y['mjd'],ssn_1y['OSF'])

#scatter plot
#====================
plotlims = [0, 14.5]
ax = fig.add_subplot(gs[0, 2])
x = osfG['OSF_SSN'].to_numpy()
y = osfG['OSF_BV_corr'].to_numpy()
xrange = np.array([np.nanmin(x), np.nanmax(x)])
bvfit_slope, bvfit_intercept, bvconf_slope, bvconf_intercept =  mplot.odr_lin_regress_bootstrap(x, y)
ax.fill_between(xrange, bvconf_slope[0] * xrange + bvconf_intercept[0], 
                  bvconf_slope[1] * xrange + bvconf_intercept[1], 
                  color='lightskyblue', label='Confidence Interval')
ax.plot(x, y,'ko')
ax.plot(x, bvfit_slope * x + bvfit_intercept, 'b')
# omni_1y['OSF_BV_corr'] = omni_1y['OSF_BV'] *bvfit_slope + bvfit_intercept
# p, regressparams = mplot.lin_regress(x,y, plotnow = True, ax = ax,
#                                      color = 'lightskyblue', alpha =1, linecolor = 'b')
ax.set_xlim(plotlims); ax.set_ylim(plotlims); ax.plot(plotlims,  plotlims,'k--')
ax.set_xlabel(r'SSN OSF [x$10^{14}$ Wb]', fontsize = 12)
ax.set_ylabel(r'GEO $OSF_{PS}*$ [x$10^{14}$ Wb]', fontsize = 12)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.text(0.02, 0.95, '(b)', transform=plt.gca().transAxes, fontsize=12)
# ax.text(0.02, 0.85, r'$OSF_{BR}$ = ' + "{:.2f}".format(p[0]) + r' $OSF_{PS}$ ' +
#          "{:.2f}".format(p[1]), 
#         transform=plt.gca().transAxes, fontsize=12, color = 'b')
mae = np.nanmean(abs(osfG['OSF_SSN'] - osfG['OSF_BV_corr']))
rl =  np.corrcoef(osfG['OSF_SSN'], osfG['OSF_BV_corr'])

ax.set_title(r'$r_L$ = ' + "{:.2f}".format(rl[0,1]) + 
             '; MAE = '  + "{:.1f}".format(mae) + r' x $10^{14}$ Wb', fontsize=12) 


#time series
#====================
ax = fig.add_subplot(gs[0, 0:2])

# confid_int, prob_int = mplot.lin_regress_confid_int(ssn_1y['OSF'], regressparams)
ax.fill_between(ssn_1y['datetime'], bvconf_slope[0] * ssn_1y['OSF'] + bvconf_intercept[0], 
                  bvconf_slope[1] * ssn_1y['OSF'] + bvconf_intercept[1], 
                  color='lightskyblue')

ax.plot(ssn_1y['datetime'], ssn_1y['OSF'], 'b', label = 'SSN OSF')
ax.plot(osfG['datetime'], osfG['OSF_BV_corr'],'r' , label = r'GEO $OSF_{PS}*$')
#ax.plot(omni_1y['datetime'], omni_1y['OSF'],'k', label = r'OMNI $OSF_{BR}$')

ax.fill_between(ssn_1y['datetime'], ssn_1y['ssn']*0, ssn_1y['ssn']/100, label = 'SSN/100', facecolor ='grey')
#plt.plot(osfG['datetime'], osfG['OSFmin'],'r--')
#plt.plot(osfG['datetime'], osfG['OSFmax'],'r--')
ax.set_ylim(plotlims)
ax.set_ylabel(r'OSF [x$10^14$ Wb]', fontsize = 12)
ax.set_xlabel(r'Year', fontsize = 12)
ax.legend(ncol = 4, fontsize = 12, loc = 'upper right')
ax.text(0.02, 0.95, '(a)', transform=plt.gca().transAxes, fontsize=12)

#interpolate the SSN OSF to the GEO time step to compuate MAE and correlation
osfG['OSF_SSN'] = np.interp(osfG['mjd'], ssn_1y['mjd'], ssn_1y['OSF'])




fig.savefig(figdir + 'OSF_SSN.pdf')





#Group sunspot number

# gsn_df.loc[0,'OSF'] = startOSF

# #compute OSF using average loss profile
# for n in range(0, len(gsn_df)-1):
#     #compute the absolute loss rate
#     gsn_df.loc[n,'loss_abs'] = gsn_df.loc[n,'OSF'] * gsn_df.loc[n,'loss_frac']
    
#     #compute the OSF for the next time step
#     gsn_df.loc[n+1,'OSF'] = gsn_df.loc[n,'OSF'] + gsn_df.loc[n,'source'] - gsn_df.loc[n,'loss_abs'] 
    
#     #osf can't be negative
#     if gsn_df.loc[n+1,'OSF'] < 0.0:
#         gsn_df.loc[n+1,'OSF'] = 0.0
        
# plt.figure()
# plt.plot(gsn_df['fracyear'], gsn_df['OSF'], 'b', label = 'SSN-based model')
# plt.plot(osfG['year'], osfG['OSF_BV_corr'],'r' , label = 'Geomagnetic')
# #plt.plot(osfG['datetime'], osfG['OSFmin'],'r--')
# #plt.plot(osfG['datetime'], osfG['OSFmax'],'r--')
# plt.ylim((0,13))
# plt.ylabel(r'OSF [x$10^14$ Wb]', fontsize = 12)
# plt.xlabel(r'Year', fontsize = 12)
# plt.legend(fontsize = 12)


# plt.figure()
# plt.plot(gsn_df['OSF'], gsn_df['gsn'], 'ko')
# plt.xlabel(r'OSF from GSN [x$10^14$ Wb]')
# plt.ylabel(r'GSN')

# # <codecell> Monte Carlo sample from the various loss profiles for each Solar Cycle


# #remove solar cycles with no loss data (i.e. prior to GEO OSF reconstruction)
# LOSS_all = SPE_LOSS.copy()
# allnans = []
# for n in range(0,len(SPE_LOSS[0,:])):
#     if np.isnan(SPE_LOSS[:,n]).all():
#         allnans.append(n)
# LOSS_all = np.delete(LOSS_all, allnans, axis=1)        
        





# def interpolate_nans(array):
#     # Find indices of NaN values
#     nan_indices = np.isnan(array)
    
#     # Create an array of indices
#     indices = np.arange(len(array))
    
#     # Interpolate using non-NaN indices and values
#     interpolated_values = np.interp(indices, indices[~nan_indices], array[~nan_indices])
    
#     # Replace NaN values with interpolated values
#     array[nan_indices] = interpolated_values[nan_indices]
    
#     return array

# #plt.figure()

# Nmc = 100
# Nrealisations = len(LOSS_all[0,:])


# osf_mc = np.empty((Nmc,nt_ssn))
# loss_frac = np.empty(nt_ssn)

# osf_mc[:,0] = startOSF
# for n in range(0, Nmc):
#     #compute the loss rate for this phase from each cycle
    
#     #select the random loss terms
#     randi =  [random.randint(0, Nrealisations-1) for _ in range(nt_ssn)]
    
#     for t in range(0,nt_ssn):
#         loss_frac[t] = np.interp(ssn_1y.loc[t,'phase'], 
#                                         SPE_phase, LOSS_all[:,randi[t]], period = 2*np.pi)
#     loss_frac = interpolate_nans(loss_frac)
    
#     #now compute the OSF
#     for t in range(0,nt_ssn-1):
#         ##compute the absolute loss rate for this time step
#         loss_abs = osf_mc[n,t]* loss_frac[t]
    
#         #compute the OSF for the next time step
#         osf_mc[n,t+1] =  osf_mc[n,t] + ssn_1y.loc[t,'source'] - loss_abs
        
#         #osf can't be negative
#         if osf_mc[n,t+1]  < 0.0:
#             osf_mc[n,t+1]  = 0.0
    
#     #plt.plot(loss_frac)


# # <codecell> plot the MC results

# #compute median of MC
# MC_median = np.empty(nt_ssn)
# for t in range(0,nt_ssn):
#     MC_median[t] = np.nanmedian(osf_mc[:,t])
    


  

# confid_intervals = [5]



# fig = plt.figure(figsize=(10,5))
# ax = plt.subplot(1,1,1)
# ax.plot(ssn_1y['datetime'], MC_median, 'b', label = 'SSN-based model')
# ax.plot(osfG['datetime'], osfG['OSF_BV_corr'],'r' , label = 'GEO')
# ax.fill_between(ssn_1y['datetime'], ssn_1y['ssn']*0, ssn_1y['ssn']/100, label = 'SSN/100', facecolor ='grey')
# ax.legend(ncol = 3, fontsize = 12, loc = 'upper left')


# mplot.plotconfidbands(ssn_1y['datetime'], osf_mc, confid_intervals)
# ax.plot(ssn_1y['datetime'], MC_median, 'b')

# #plt.plot(ssn_1y['datetime'], ssn_1y['OSF'], 'b', label = 'SSN-based model')
# ax.plot(osfG['datetime'], osfG['OSF_BV_corr'],'r' , label = 'GEO')
# #plt.plot(omni_1y['datetime'], omni_1y['OSF'],'k', label = 'OMNI')


# #plt.plot(osfG['datetime'], osfG['OSFmin'],'r--')
# #plt.plot(osfG['datetime'], osfG['OSFmax'],'r--')
# ax.set_ylim((0,14.5))
# ax.set_ylabel(r'OSF [x$10^14$ Wb]', fontsize = 12)
# ax.set_xlabel(r'Year', fontsize = 12)


# #compute the correlation of the SSN OSF and the GEO OSF
# osfG['OSF_SSN_MC'] = np.interp(osfG['mjd'],ssn_1y['mjd'],MC_median)
# rl =  np.corrcoef(osfG['OSF_SSN_MC'], osfG['OSF_BV_corr'])
# mae = np.nanmean(abs(osfG['OSF_SSN_MC'] - osfG['OSF_BV_corr']))
# ax.set_title(r'$r_L$ = ' + "{:.2f}".format(rl[0,1]) + 
#              '; MAE = '  + "{:.1f}".format(mae) + r' x $10^{14}$ Wb', fontsize=12) 





# <codecell> SSN - OSF regressions


#create 11yrecord
osf_11y = osfG.resample('11Y', on='datetime').mean()
osf_11y['datetime'] = htime.mjd2datetime(osf_11y['mjd'].to_numpy())
osf_11y.reset_index(drop=True, inplace=True)
#dump the first value
osf_11y = osf_11y.tail(-1)

#also create a rolling 11-yr mean OSF
window = str(365*11) + 'D'
temp = osfG.rolling(str(window), on='datetime').mean()
osfG['OSF_11ysmooth'] = np.interp(osfG['mjd'],temp['mjd'],temp['OSF_BV_corr'],  left =np.nan, right =np.nan)
osfG['OSF_11ysmooth_max'] = np.interp(osfG['mjd'],temp['mjd'],temp['OSF_BV_corr_max'],left =np.nan, right =np.nan)
osfG['OSF_11ysmooth_min'] = np.interp(osfG['mjd'],temp['mjd'],temp['OSF_BV_corr_min'],left =np.nan, right =np.nan)
osfG['SSN_11ysmooth'] = np.interp(osfG['mjd'],temp['mjd'],temp['Rsilso'],
                                          left =np.nan, right =np.nan)
#drop the incomplete window at the start
mask = osfG['mjd'] < osfG['mjd'].loc[0] + 365*11/2
osfG.loc[mask, 'OSF_11ysmooth'] = np.nan
osfG.loc[mask, 'OSF_11ysmooth_max'] = np.nan
osfG.loc[mask, 'OSF_11ysmooth_min'] = np.nan
osfG.loc[mask, 'SSN_11ysmooth'] = np.nan

#fit the smoothed data
#==============================================================================


valid_indices = ~np.isnan(osfG['OSF_11ysmooth']) & ~np.isnan( osfG['SSN_11ysmooth'])
ssnsmooth_from_osfsmooth_coefficients = mplot.odr_lin_regress(osfG.loc[valid_indices,'OSF_11ysmooth'], 
                                                    osfG.loc[valid_indices,'SSN_11ysmooth'])
#coefficient uncertainty from bootstrap
fit_slope, fit_intercept, conf_slope, conf_intercept = \
    mplot.odr_lin_regress_bootstrap(osfG.loc[valid_indices,'OSF_11ysmooth'].to_numpy(), 
                                    osfG.loc[valid_indices,'SSN_11ysmooth'].to_numpy(),
                                    xerr= None, yerr= None, 
                                    num_bootstraps = 1000, plotnow = False)
print('SSN_11 = m OSF_11 + c' )
print('m = ' + str(fit_slope) + ' +/- ' + str(conf_slope) )
print('c = ' + str(fit_intercept) + ' +/- ' + str(conf_intercept) )


#subtract the 11-yr OSF values
#==============================================================================

osfG['dOSF_11'] = osfG['OSF_BV_corr'] - osfG['OSF_11ysmooth']
osfG['dOSF_11_max'] = osfG['OSF_BV_corr_max'] - osfG['OSF_11ysmooth_max']
osfG['dOSF_11_min'] = osfG['OSF_BV_corr_min'] - osfG['OSF_11ysmooth_min']
osfG['dSSN_11'] = osfG['Rsilso'] - osfG['SSN_11ysmooth']

valid_indices = ~np.isnan(osfG['dOSF_11']) & ~np.isnan( osfG['dSSN_11'])
# dssn11_from_dosf11_coefficients = np.polyfit(osfG['dOSF_11'].loc[valid_indices], 
#                                                    osfG['dSSN_11'].loc[valid_indices],1)
dssn11_from_dosf11_coefficients = mplot.odr_lin_regress(osfG['dOSF_11'].loc[valid_indices], 
                                                   osfG['dSSN_11'].loc[valid_indices])

#coefficient uncertainty from bootstrap
# fit_slope, fit_intercept, conf_slope, conf_intercept =odr_lin_regress_bootstrap(x, y, xerr= None, yerr= None, 
#                               num_bootstraps = 1000, plotnow = False)

#recreate SSN using these regressions
ssn_regress_11 = np.polyval(ssnsmooth_from_osfsmooth_coefficients, osfG['OSF_11ysmooth']) + \
    np.polyval(dssn11_from_dosf11_coefficients, osfG['dOSF_11'])
ssn_regress_11_max = np.polyval(ssnsmooth_from_osfsmooth_coefficients, osfG['OSF_11ysmooth_max']) + \
        np.polyval(dssn11_from_dosf11_coefficients, osfG['dOSF_11_max'])
ssn_regress_11_min = np.polyval(ssnsmooth_from_osfsmooth_coefficients, osfG['OSF_11ysmooth_min']) + \
        np.polyval(dssn11_from_dosf11_coefficients, osfG['dOSF_11_min'])
#compute error bands       
ssn_regress_11_lowererror = abs(ssn_regress_11 -  ssn_regress_11_min)
ssn_regress_11_uppererror = abs(ssn_regress_11 -  ssn_regress_11_max)

#coefficient uncertainty from bootstrap
fit_slope, fit_intercept, conf_slope, conf_intercept = \
    mplot.odr_lin_regress_bootstrap(osfG.loc[valid_indices,'dOSF_11'].to_numpy(), 
                                    osfG.loc[valid_indices,'dSSN_11'].to_numpy(),
                                    xerr= None, yerr= None, 
                                    num_bootstraps = 1000, plotnow = False)
print('SSN_11 = m OSF_11 + c' )
print('m = ' + str(fit_slope) + ' +/- ' + str(conf_slope) )
print('c = ' + str(fit_intercept) + ' +/- ' + str(conf_intercept) )


#OSF squared
#==============================================================================

osf_yrs = osfG['year'].to_numpy()
osf = osfG['OSF_BV_corr'].to_numpy()
osf_min = osfG['OSF_BV_corr_min'].to_numpy()
osf_max = osfG['OSF_BV_corr_max'].to_numpy()
osf_ssn = osfG['Rsilso'].to_numpy()
xplotlims = (1845, 2025) 

#fit SSN - OSF^2 relation
ssn_from_osfsq_coefficients = mplot.odr_lin_regress(pow(osf,2),osf_ssn)
ssn_from_osf_regress = np.polyval(ssn_from_osfsq_coefficients, pow(osf,2))
ssn_from_osf_lowererror = abs(ssn_from_osf_regress -  np.polyval(ssn_from_osfsq_coefficients, pow(osf_min,2)))
ssn_from_osf_uppererror = abs(ssn_from_osf_regress -  np.polyval(ssn_from_osfsq_coefficients, pow(osf_max,2)))

#coefficient uncertainty from bootstrap
fit_slope, fit_intercept, conf_slope, conf_intercept = \
    mplot.odr_lin_regress_bootstrap(pow(osf,2), 
                                    osf_ssn,
                                    xerr= None, yerr= None, 
                                    num_bootstraps = 1000, plotnow = False)
print('SSN_11 = m OSF_11 + c' )
print('m = ' + str(fit_slope) + ' +/- ' + str(conf_slope) )
print('c = ' + str(fit_intercept) + ' +/- ' + str(conf_intercept) )


# <codecell> show a few examples of the SSN reconstruction method
osf_yrs = osfG['year'].to_numpy()
osf = osfG['OSF_BV_corr'].to_numpy()
osf_min = osfG['OSF_BV_corr_min'].to_numpy()
osf_max = osfG['OSF_BV_corr_max'].to_numpy()
osf_ssn = osfG['Rsilso'].to_numpy()
xplotlims = (1845, 2025) 

#now generate a random OSF source time series for differen SC properties
window_yrs = 30
n_rand_cycles = 4
cycle_length_mean = 10.5
cycle_length_std = 2
#cycle_amp_mean = 130
cycle_amp_std = 70
N = 50


final_ssn = []
final_yrs = []
final_osf = []
final_mae = []
final_minyrs = []


windowstart = 1920
windowstop = windowstart + window_yrs


    
yrmask = (osf_yrs >= windowstart) & (osf_yrs < windowstop)
osf_window = osf[yrmask]
osf_yrs_window = osf_yrs[yrmask]
ssn_window = osf_ssn[yrmask]
nt= len(osf_window)
    
    
    
    
MC_minyrs = np.empty((N, n_rand_cycles+1))
MC_ssn = np.empty((N, nt))
MC_osf = np.empty((N, nt))
MC_mae = np.empty((N))
#MC_rl = np.empty((N))
    
#do teh required number or realisations and pick the best
for nmc in range(0,N):
    
    
    #compute likely SSN amplitudes from teh regression with OSF
    maxOSF = np.nanmax(osf_window)
    ssn_osf_coeffs = [  29.65267933, -124.76978472]
    cycle_amp_mean = np.polyval(ssn_osf_coeffs, maxOSF)
    

    #compute the random cycle lengths
    cycle_lengths =  np.random.normal(cycle_length_mean, cycle_length_std, n_rand_cycles)
    mask = cycle_lengths < 0
    cycle_lengths[mask] = 0
    #compute the random cycle amplitudes
    cycle_amps =  np.random.normal(cycle_amp_mean, cycle_amp_std, n_rand_cycles)
    mask = cycle_amps < 0
    cycle_amps[mask] = 0
    
    #compute the first cycle start, relative to the first osf year
    cycle_start = np.random_numbers = np.random.uniform(-11, 0, 1)
    
    #generate the min times from the cycle lengths and cycle start
    min_yrs = np.empty((n_rand_cycles+1)) * np.nan
    min_yrs[0] = osf_yrs_window[0] + cycle_start
    for n in range(0,n_rand_cycles):
        min_yrs[n+1] = min_yrs[n] + cycle_lengths[n]
    
    
    #generate solar cycle phase from the min times. give both args in years.
    phase = sunspots.solar_cycle_phase(osf_yrs_window, solarmin_mjd = min_yrs) 
    
    #first generate the normalised SNN profile from the SC phase
    ssn_norm = np.empty(nt)*np.nan
    ssn_norm = np.interp(phase, SPE_SSN_phase, SPE_SSN_NORM_AVG, left = np.nan, right = np.nan)  
    
    #then add the cycle-specific amplitude
    ssn = np.empty((nt))*np.nan
    for n in range(0,len(min_yrs)-1):
        mask = (osf_yrs_window >= min_yrs[n]) & (osf_yrs_window < min_yrs[n+1])
        ssn[mask] = ssn_norm[mask] * cycle_amps[n]
        
    #now compute the OSF source term from the SSN
    osf_source = sunspots.compute_osf_source(ssn)
    
    #compute teh fractional OSF loss from the phase
    loss_frac = np.interp(phase, SPE_phase, SPE_LOSS_AVG, period = 2*np.pi)
    
    #now, finally, compute the OSF
    
    ssn_osf = np.empty((nt))*np.nan
    loss_abs = np.empty((nt))*np.nan
    ssn_osf[0] = osf_window[0]
    #compute OSF using average loss profile
    for n in range(0, nt-1):
        #compute the absolute loss rate
        loss_abs[n] = ssn_osf[n] * loss_frac[n]
        
        #compute the OSF for the next time step
        ssn_osf[n+1] = ssn_osf[n] + osf_source[n] - loss_abs[n]
        
        #osf can't be negative
        if ssn_osf[n+1] < 0.0:
            ssn_osf[n+1] = 0.0   
            
            
    #compute some error stats
    mae = np.nanmean(abs(ssn_osf - osf_window))

    
    #store the value
    #MC_rl[nmc] = rl[0,1]
    MC_mae[nmc] = mae
    MC_ssn[nmc,:] = ssn
    MC_minyrs[nmc,:] = min_yrs   
    MC_osf[nmc,:] = ssn_osf

#fidn the min MAE
i = np.argmin(MC_mae)
imax = np.argmax(MC_mae)

plt.figure(figsize = (7,8))

ax = plt.subplot(2,1,1)
ax.plot(osf_yrs_window,  osf_window, 'k', label = 'GEO')
ax.plot(osf_yrs_window,  MC_osf[i,:], 'r', label = 'MC Model (best)')
ax.legend(ncol = 2, fontsize = 12, loc = 'upper left')
ax.set_ylabel(r'OSF [x10$^{14}$ Wb]', fontsize = 12)
for n in range(0,N):
    weight = np.exp( - 8* (abs(MC_mae[n] - MC_mae[i])) / (MC_mae[imax] - MC_mae[i]))
    ax.plot(osf_yrs_window,  MC_osf[n,:],  alpha = weight)
ax.plot(osf_yrs_window,  osf_window, 'k', label = 'GEO')
ax.plot(osf_yrs_window,  MC_osf[i,:], 'r', label = 'MC Model')

ax = plt.subplot(2,1,2)
ax.plot(osf_yrs_window,  ssn_window, 'k', label = 'SILSOv2')
ax.plot(osf_yrs_window,  MC_ssn[i,:], 'r', label = 'MC Model (best)')
ax.legend(ncol = 2, fontsize = 12, loc = 'upper left')
ax.set_ylabel(r'SSN', fontsize = 12)
for n in range(0,N):
    weight = np.exp( - 8*(abs(MC_mae[n] - MC_mae[i])) / (MC_mae[imax] - MC_mae[i]))
    ax.plot(osf_yrs_window,  MC_ssn[n,:], alpha = weight)
ax.plot(osf_yrs_window,  ssn_window, 'k', label = 'GEO')
ax.plot(osf_yrs_window,  MC_ssn[i,:], 'r', label = 'MC Model')
ax.set_xlabel(r'Year', fontsize = 12)


# <codecell> Alternate deomstration of the method.
#now generate a random OSF source time series for differen SC properties
window_yrs = 22


windowstart = 1920
windowstop = windowstart + window_yrs

yrmask = (osf_yrs >= windowstart) & (osf_yrs < windowstop)
osf_window = osf[yrmask]
osf_yrs_window = osf_yrs[yrmask]
ssn_window = osf_ssn[yrmask]
nt= len(osf_window)


#now generate a random OSF source time series for differen SC properties
window_yrs = 30
n_rand_cycles = 4
cycle_length_mean = 10.5
cycle_length_std = 2
#cycle_amp_mean = 130
cycle_amp_std = 70
N = 50




n_example = 3


fig = plt.figure(figsize = (9,8))

for i_example in range(0, n_example):


    #compute likely SSN amplitudes from teh regression with OSF
    maxOSF = np.nanmax(osf_window)
    ssn_osf_coeffs = [  29.65267933, -124.76978472]
    cycle_amp_mean = np.polyval(ssn_osf_coeffs, maxOSF)
    
    cycle_lengths =  np.random.normal(cycle_length_mean, cycle_length_std, n_rand_cycles)
    mask = cycle_lengths < 0
    cycle_lengths[mask] = 0
    #compute the random cycle amplitudes
    cycle_amps =  np.random.normal(cycle_amp_mean, cycle_amp_std, n_rand_cycles)
    mask = cycle_amps < 0
    cycle_amps[mask] = 0
    
    #compute the first cycle start, relative to the first osf year
    cycle_start = np.random_numbers = np.random.uniform(-11, 0, 1)
    
    
    #hardcode the last example
    if (i_example == n_example - 1):
        cycle_start = [-6.7] 
        cycle_lengths = [9.2, 10.7, 11, 10]
        cycle_amps = [170, 160, 200, 180]
    
    #generate the min times from the cycle lengths and cycle start
    min_yrs = np.empty((n_rand_cycles+1)) * np.nan
    min_yrs[0] = osf_yrs_window[0] + cycle_start
    for n in range(0,n_rand_cycles):
        min_yrs[n+1] = min_yrs[n] + cycle_lengths[n]
    
    
    #generate solar cycle phase from the min times. give both args in years.
    phase = sunspots.solar_cycle_phase(osf_yrs_window, solarmin_mjd = min_yrs) 
    
    #first generate the normalised SNN profile from the SC phase
    ssn_norm = np.empty(nt)*np.nan
    ssn_norm = np.interp(phase, SPE_SSN_phase, SPE_SSN_NORM_AVG, left = np.nan, right = np.nan)  
    
    #then add the cycle-specific amplitude
    ssn = np.empty((nt))*np.nan
    for n in range(0,len(min_yrs)-1):
        mask = (osf_yrs_window >= min_yrs[n]) & (osf_yrs_window < min_yrs[n+1])
        ssn[mask] = ssn_norm[mask] * cycle_amps[n]
        
    #now compute the OSF source term from the SSN
    osf_source = sunspots.compute_osf_source(ssn)
    
    #compute teh fractional OSF loss from the phase
    loss_frac = np.interp(phase, SPE_phase, SPE_LOSS_AVG, period = 2*np.pi)
    
    #now, finally, compute the OSF
    
    ssn_osf = np.empty((nt))*np.nan
    loss_abs = np.empty((nt))*np.nan
    ssn_osf[0] = osf_window[0]
    #compute OSF using average loss profile
    for n in range(0, nt-1):
        #compute the absolute loss rate
        loss_abs[n] = ssn_osf[n] * loss_frac[n]
        
        #compute the OSF for the next time step
        ssn_osf[n+1] = ssn_osf[n] + osf_source[n] - loss_abs[n]
        
        #osf can't be negative
        if ssn_osf[n+1] < 0.0:
            ssn_osf[n+1] = 0.0   
            
            
    #compute some error stats
    mae = np.nanmean(abs(ssn_osf - osf_window))
    
    #create some output strings
    sclen_str = '[' + f'{cycle_lengths[0]:.1f}' +  ', ' + \
    f'{cycle_lengths[1]:.1f}' +  ', ' + \
    f'{cycle_lengths[2]:.1f}' +  ', ' + \
    f'{cycle_lengths[3]:.1f}' +  '] yrs'
    
    ssnmax_str = '[' + f'{cycle_amps[0]:.0f}' +  ', ' + \
    f'{cycle_amps[1]:.0f}' +  ', ' + \
    f'{cycle_amps[2]:.0f}' +  ', ' + \
    f'{cycle_amps[3]:.0f}' +  ']'
    
    scstart_str =  f'{windowstart + cycle_start[0]:.1f}' 
    
    
    dy = 0.07
    ssnlims = [-50, 320]
    osflims = [0, 13]
    
    #SSN plot
    #==========================================================================
    
    panel_num = i_example*2 + 1
    ax = plt.subplot(n_example, 2, panel_num)
    
    ax.plot(osf_yrs_window,  ssn_window, 'k', label = 'SILSOv2')
    ax.plot(osf_yrs_window,  ssn, 'r', label = 'MC Model ')
    ax.set_ylabel(r'SSN', fontsize = 12)
    ax.set_ylim(ssnlims)
    
    ax.text(0.04, 0.92, 'SC lengths = ' + sclen_str, transform=plt.gca().transAxes, fontsize=12)
    ax.text(0.04, 0.92 - dy, 'SSN amps = ' + ssnmax_str, transform=plt.gca().transAxes, fontsize=12)
    ax.text(0.04, 0.92 - 2*dy, 'SC start = ' + scstart_str, transform=plt.gca().transAxes, fontsize=12)
    
    
    ax.text(0.04, dy, '('+ chr(ord('`')+panel_num) +')', transform=plt.gca().transAxes, fontsize=12)
    
    if i_example < n_example-1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Year')
        ax.legend(ncol = 1, fontsize = 12, loc = 'lower right')
    
    
    #OSF plot
    #==========================================================================
    panel_num = i_example*2 + 2
    ax = plt.subplot(n_example, 2, panel_num)
    
    ax.plot(osf_yrs_window,  osf_window, 'k', label = r'GEO $OSF_{PS}*$')
    ax.plot(osf_yrs_window,  ssn_osf, 'r', label = 'MC Model ')
    ax.set_ylabel(r'OSF [x$10^{14}$ Wb]', fontsize = 12)
    ax.set_ylim(osflims)
    
    ax.text(0.04, 0.92, 'MAE = ' + f'{mae:.1f}' + r' [x$10^{14}$ Wb]' , transform=plt.gca().transAxes, fontsize=12)
    
    ax.text(0.04, dy, '('+ chr(ord('`')+panel_num) +')', transform=plt.gca().transAxes, fontsize=12)
    
    if i_example < n_example-1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Year')
        ax.legend(ncol = 1, fontsize = 12, loc = 'lower right')
    

plt.tight_layout()
#fig.savefig(figdir + 'SSN_recon_example.pdf')

# <codecell> create a dataframe with all OSF estimates for export as CSV

osf_df = ssn_1y.copy()
columns_to_drop = ['smooth', 'smooth', 'rollingmax']
osf_df = osf_df.drop(columns=osf_df[columns_to_drop])

#reorder the columns
column_order = ['datetime', 'fracyear', 'mjd', 'phase', 'ssn', 'sai', 'source',
                'loss_frac', 'loss_abs', 'OSF']
osf_df = osf_df[column_order]

#rename the OSF column
osf_df.rename(columns={'OSF': 'OSF_SSN'}, inplace=True)
osf_df.rename(columns={'source': 'OSF_SSN_source'}, inplace=True)
osf_df.rename(columns={'loss_frac': 'OSF_SSN_loss_frac'}, inplace=True)
osf_df.rename(columns={'loss_abs': 'OSF_SSN_loss_abs'}, inplace=True)

#now add the geomagnetic OSF and data
columns_to_copy = ['Bmin', 'B', 'Bmax', 'Vmin', 'V', 'Vmax',
                   'OSF_BV_corr_min', 'OSF_BV_corr', 'OSF_BV_corr_max']
columns_to_rename = ['B_GEO_min', 'B_GEO', 'B_GEO_max', 'V_GEO_min', 'V_GEO', 'V_GEO_max',
                   'OSF_GEO_min', 'OSF_GEO', 'OSF_GEO_max']
for var, var_rename in zip(columns_to_copy, columns_to_rename):
    osf_df[var] = np.interp(osf_df['mjd'],osfG['mjd'],osfG[var],  left =np.nan, right =np.nan)
    osf_df.rename(columns={var: var_rename}, inplace=True)
    
#now add the OMNI columns
osf_df['OSF_OMNI'] = np.interp(osf_df['mjd'],omni_1y['mjd'],omni_1y['OSF'],  left =np.nan, right =np.nan)



#save the data
outputfilepath = os.environ['DBOX'] + 'Data\\OSF_OEL2023.csv'
osf_df.to_csv(outputfilepath, index=False)

#read it back in
osf_df = pd.read_csv(outputfilepath)
osf_df['datetime'] = htime.mjd2datetime(osf_df['mjd'].to_numpy()) #datetime gets messed up?

# plt.figure()
# plt.plot(osf_df['datetime'],osf_df['OSF_SSN'])
# plt.plot(osf_df['datetime'],osf_df['OSF_GEO'])
# plt.plot(osf_df['datetime'],osf_df['OSF_OMNI'])

# <codecell> generate random sunspot series, compute OSF and compare with observed OSF



#justification for reconstruction process - correlation becomes weak in weak cycles






def ssn_from_osf(osf_yrs, osf, 
                 SPE_SSN_phase, SPE_SSN_NORM_AVG,
                 SPE_phase, SPE_LOSS_AVG,
                 window_yrs = 22, dwindow = 1, 
                 cycle_length_mean = 10.5, cycle_length_std = 2, cycle_amp_std = 100,
                 N = 10000):
    #now generate a random OSF source time series for different SC properties
    #via a Monte Carlo process
    

    n_rand_cycles = int(np.ceil(window_yrs/11) + 2)

    final_ssn = []
    final_yrs = []
    final_osf = []
    final_mae = []
    final_minyrs = []
    
    
    windowstart = osf_yrs[0]
    windowstop = windowstart + window_yrs
    
    while windowstart < osf_yrs[len(osf_yrs)-1]-5:
        
        mask = (osf_yrs >= windowstart) & (osf_yrs < windowstop)
        osf_window = osf[mask]
        osf_yrs_window = osf_yrs[mask]
        nt= len(osf_window)
        
        #compute likely SSN amplitudes from the regression with OSF
        maxOSF = np.nanmax(osf_window)
        ssn_osf_coeffs = [  29.65267933, -124.76978472]
        cycle_amp_mean = ssn_osf_coeffs[0] * maxOSF + ssn_osf_coeffs[0]
        
        
        MC_minyrs = np.empty((N, n_rand_cycles+1))
        MC_ssn = np.empty((N, nt))
        MC_osf = np.empty((N, nt))
        MC_mae = np.empty((N))
        
        #do teh required number or realisations and pick the best
        for nmc in range(0,N):
        
            #compute the random cycle lengths
            cycle_lengths =  np.random.normal(cycle_length_mean, cycle_length_std, n_rand_cycles)
            mask = cycle_lengths < 0
            cycle_lengths[mask] = 0
            #compute the random cycle amplitudes
            cycle_amps =  np.random.normal(cycle_amp_mean, cycle_amp_std, n_rand_cycles)
            mask = cycle_amps < 0
            cycle_amps[mask] = 0
            
            #compute the first cycle start, relative to the first osf year
            cycle_start = np.random_numbers = np.random.uniform(-11, 0, 1)
            
            #generate the min times from the cycle lengths and cycle start
            min_yrs = np.empty((n_rand_cycles+1)) * np.nan
            min_yrs[0] = osf_yrs_window[0] + cycle_start
            for n in range(0,n_rand_cycles):
                min_yrs[n+1] = min_yrs[n] + cycle_lengths[n]
            
            
            #generate solar cycle phase from the min times. give both args in years.
            phase = sunspots.solar_cycle_phase(osf_yrs_window, solarmin_mjd = min_yrs) 
            
            #first generate the normalised SNN profile from the SC phase
            ssn_norm = np.empty(nt)*np.nan
            ssn_norm = np.interp(phase, SPE_SSN_phase, SPE_SSN_NORM_AVG, left = np.nan, right = np.nan)  
            
            #then add the cycle-specific amplitude
            ssn = np.empty((nt))*np.nan
            for n in range(0,len(min_yrs)-1):
                mask = (osf_yrs_window >= min_yrs[n]) & (osf_yrs_window < min_yrs[n+1])
                ssn[mask] = ssn_norm[mask] * cycle_amps[n]
                
            #now compute the OSF source term from the SSN
            osf_source = sunspots.compute_osf_source(ssn)
            
            #compute teh fractional OSF loss from the phase
            loss_frac = np.interp(phase, SPE_phase, SPE_LOSS_AVG, period = 2*np.pi)
            
            #now, finally, compute the OSF
            
            ssn_osf = np.empty((nt))*np.nan
            loss_abs = np.empty((nt))*np.nan
            ssn_osf[0] = osf_window[0]
            #compute OSF using average loss profile
            for n in range(0, nt-1):
                #compute the absolute loss rate
                loss_abs[n] = ssn_osf[n] * loss_frac[n]
                
                #compute the OSF for the next time step
                ssn_osf[n+1] = ssn_osf[n] + osf_source[n] - loss_abs[n]
                
                #osf can't be negative
                if ssn_osf[n+1] < 0.0:
                    ssn_osf[n+1] = 0.0   
                    
            
            #compute some error stats
            mae = np.nanmean(abs(ssn_osf - osf_window))
            
            #store the values
            MC_mae[nmc] = mae
            MC_ssn[nmc,:] = ssn
            MC_minyrs[nmc,:] = min_yrs   
            MC_osf[nmc,:] = ssn_osf
        
        #fidn the min MAE
        i = np.argmin(MC_mae)
        
    
        #save this ssn sequence
        final_ssn.append(MC_ssn[i,:])
        final_yrs.append(osf_yrs_window)
        final_osf.append(MC_osf[i,:])
        final_mae.append(MC_mae[i])
        final_minyrs.append(MC_minyrs[i,:])
        
    
        #advance the window
        windowstart = windowstart + dwindow
        windowstop = windowstart + window_yrs
    
    return final_yrs, final_ssn, final_osf, final_mae, final_minyrs


def SBwidth_from_individual_MC(final_yrs, final_ssn, final_minyrs, SPE_phase, SPE_LOSS_AVG):
    #compute SB width and HCS inclination for each MC realisation of SSN and 
    #the min years
    
    final_SBwidth = []
    final_inc = []
    
    for n in range(0,len(final_yrs)):
        OSF, SBwidth = sunspots.osf_and_streamer_belt_width(final_yrs[n], final_ssn[n], final_minyrs[n], 
                                                                                SPE_phase, SPE_LOSS_AVG, 
                                        startOSF = 8, startFch = 2, startFsb = 0,
                                        S_CH_weights = [0.1, 0.5, 0.3, 0.1, 0],
                                        chi_CH = 0.22, plotnow = False)
        final_SBwidth.append(SBwidth)
        
        inclination = sunspots.HCSinclination(final_yrs[n], final_minyrs[n])
        final_inc.append(inclination)
        
    return final_SBwidth, final_inc

def produce_concensus_ssn_osf(osf_yrs, final_yrs, final_osf, final_ssn, final_mae):
    
    # produce the consensus values, year by year
    concensus_osf = np.empty((len(osf_yrs),2))*np.nan
    concensus_ssn = np.empty((len(osf_yrs),2))*np.nan
    
    Nwindows = len(final_osf)
    
    for n in range(0, len(osf_yrs)):
        
        #loop through the windows and find the given year
        sum_ssn = 0
        sum_osf = 0
        sum_ssn_w = 0
        sum_osf_w = 0
        count = 0
        count_w = 0
        
        for nwindow in range(0, Nwindows):
            win_years = final_yrs[nwindow]
            win_ssn = final_ssn[nwindow]
            win_osf = final_osf[nwindow]
            win_mae = final_mae[nwindow]
            
            mask = win_years == osf_yrs[n]
            if mask.any():
                sum_ssn = sum_ssn +  win_ssn[mask]
                sum_osf = sum_osf +  win_osf[mask]
                
                #weight each value by 1-MAE - this should use the MAE computed by ssn_from_osf? 
                #mae = np.abs(win_osf[mask] - osf_obs[n])
                mae = win_mae
                w = np.exp( - mae )
                
                sum_osf_w = sum_osf_w + win_osf[mask] * w
                sum_ssn_w = sum_ssn_w + win_ssn[mask] * w
                
                count = count + 1
                count_w = count_w + w
        
        #save the mean value
        if count > 0:
            concensus_osf[n, 0] = sum_osf / count
            concensus_ssn[n, 0] = sum_ssn / count
            
            concensus_osf[n, 1] = sum_osf_w / count_w
            concensus_ssn[n, 1] = sum_ssn_w / count_w
        #concensus_osf[n, 1] = np.nanstd(sum_ssn / count)
    
    return concensus_ssn, concensus_osf

def produce_concensus_SBwidth(osf_yrs, final_SBwidth, final_inc, final_mae):
    
    # produce the consensus values, year by year
    concensus_sbw = np.empty((len(osf_yrs),2))*np.nan
    concensus_inc = np.empty((len(osf_yrs),2))*np.nan
    
    Nwindows = len(final_SBwidth)
    
    for n in range(0, len(osf_yrs)):
        
        #loop through the windows and find the given year
        sum_sbw = 0
        sum_sbw_w = 0
        sum_inc = 0
        sum_inc_w = 0
        count = 0
        count_w = 0
        
        for nwindow in range(0, Nwindows):
            win_years = final_yrs[nwindow]
            win_sbw = final_SBwidth[nwindow]
            win_inc = final_inc[nwindow]
            win_mae = final_mae[nwindow]
            
            mask = win_years == osf_yrs[n]
            if mask.any():
                if ~np.isnan(win_sbw[mask]):
                    sum_sbw = sum_sbw +  win_sbw[mask]
                    sum_inc = sum_inc +  win_inc[mask]
                    
                    #weight each value by 1-MAE
                    mae = win_mae
                    w = np.exp( - mae )
                    
                    sum_sbw_w = sum_sbw_w + win_sbw[mask] * w
                    sum_inc_w = sum_inc_w + win_inc[mask] * w
                    
                    count = count + 1
                    count_w = count_w + w
        
        #save the mean value
        if count > 0:
            concensus_sbw[n, 0] = sum_sbw / count
            concensus_inc[n, 0] = sum_inc / count
            
            concensus_sbw[n, 1] = sum_sbw_w / count_w
            concensus_inc[n, 1] = sum_inc_w / count_w
        #concensus_osf[n, 1] = np.nanstd(sum_ssn / count)
        
        
    #apply the rough empirical correction
    concensus_sbw[:,1] = concensus_sbw[:,1]*2.7 - 1.5
    
    return concensus_sbw, concensus_inc

def produce_concensus_cycle_starts(osf_yrs, final_mae, final_minyrs,
                                   dy=0.2, kden = 0.05):
        
    
    # produce the min years consensus values, window by window
    Nyrs = osf_yrs[-1] - osf_yrs[0] +1
    N_hires = int(np.ceil(Nyrs/0.2))
    hi_res_yrs = np.linspace(osf_yrs[0], osf_yrs[-1], num = N_hires)
    kde_series = hi_res_yrs*0.0
 
     
    for nwindow in range(0,len(final_minyrs)):
        #discretise the minimum years
        kernel_width = np.exp(final_mae[nwindow]) * kden
        
        kde = gaussian_kde(final_minyrs[nwindow], bw_method = kernel_width)
        kde_series = kde_series + kde(hi_res_yrs)
        
        
    #find the solar min KDE maxima
    peaks, _ = find_peaks(kde_series)
    cycle_start_yrs = hi_res_yrs[peaks]
    
    return cycle_start_yrs, hi_res_yrs, kde_series



# <codecell>  Reconstruction of SSN from GEO

osf_yrs = osfG['year'].to_numpy()
osf = osfG['OSF_BV_corr'].to_numpy()
osf_min = osfG['OSF_BV_corr_min'].to_numpy()
osf_max = osfG['OSF_BV_corr_max'].to_numpy()
osf_ssn = osfG['Rsilso'].to_numpy()
xplotlims = (1845, 2025) 


window_yrs = 22
dwindow = 1
Nmc = 10000

#compute the SSN from geomagnetic OSF
print('Computing SSN for OSF(best)')
final_yrs, final_ssn, final_osf, final_mae, final_minyrs = ssn_from_osf(osf_yrs, osf, 
                                                                        SPE_SSN_phase, SPE_SSN_NORM_AVG,
                                                                        SPE_phase, SPE_LOSS_AVG,
                                                                        window_yrs = window_yrs, dwindow = dwindow,
                                                                        N= Nmc)
print('Computing SSN for OSF(max)')
final_yrs_max, final_ssn_max, final_osf_max, final_mae_max, final_minyrs_max = ssn_from_osf(osf_yrs, osf_max, 
                                                                                            SPE_SSN_phase, SPE_SSN_NORM_AVG,
                                                                                            SPE_phase, SPE_LOSS_AVG,
                                                                                            window_yrs = window_yrs, dwindow = dwindow,
                                                                                            N= Nmc)
print('Computing SSN for OSF(min)')
final_yrs_min, final_ssn_min, final_osf_min, final_mae_min, final_minyrs_min = ssn_from_osf(osf_yrs, osf_min, 
                                                                                            SPE_SSN_phase, SPE_SSN_NORM_AVG,
                                                                                            SPE_phase, SPE_LOSS_AVG,
                                                                                            window_yrs = window_yrs, dwindow = dwindow,
                                                                                            N= Nmc)

#compute the individual SB width estimaes
#final_SBwidth, final_inc = SBwidth_from_individual_MC(final_yrs, final_ssn, final_minyrs, SPE_phase, SPE_LOSS_AVG)
#produce consensus SSN and OSF records
#geo_concensus_sbw, geo_concensus_inc = produce_concensus_SBwidth(osf_yrs, final_SBwidth, final_inc, final_mae)

#produce the consensus cycle start years
model_min, hi_res_yrs, kde_values = produce_concensus_cycle_starts(osf_yrs, final_mae, final_minyrs)
model_min_min, hi_res_yrs, kde_values_min = produce_concensus_cycle_starts(osf_yrs, final_mae_min, final_minyrs_min)
model_min_max, hi_res_yrs, kde_values_max = produce_concensus_cycle_starts(osf_yrs, final_mae_max, final_minyrs_max)


#produce consensus SSN and OSF records
geo_concensus_ssn, geo_concensus_osf = produce_concensus_ssn_osf(osf_yrs,final_yrs, final_osf, final_ssn, final_mae)
geo_concensus_ssn_max, geo_concensus_osf_max = produce_concensus_ssn_osf(osf_yrs, final_yrs_max, final_osf_max, final_ssn_max, final_mae_max)
geo_concensus_ssn_min, geo_concensus_osf_min = produce_concensus_ssn_osf(osf_yrs, final_yrs_min, final_osf_min, final_ssn_min, final_mae_min)




# <codecell> Solar cycle start times

regress_min = [1856.5, 1867.5, 1879.5, 1890.5, 1901.5, 1913.5, 1923.5, 1934.5,
               1944.5, 1954.5, 1965.5, 1976.5, 1987.5, 1996.5, 2009.5 ]

#values determined by eye
# model_min = [1845.7, 1855.8, 1867.1, 1878.7, 1889.7, 1900.7, 1913.1, 1923.2,
#              1934.0, 1944.0, 1954.1, 1964.0, 1975.8, 1986.5, 1995.7, 2008.3, 2019.3]

#sometimes an early peak is detected, which is not in the OSF record
mask = (model_min > 1850) & (model_min < 2015)
model_min = model_min[mask]
mask = (model_min_min > 1850) & (model_min_min < 2015)
model_min_min = model_min_min[mask]
mask = (model_min_max > 1850) & (model_min_max < 2015)
model_min_max = model_min_max[mask]

regress_dt = np.ones((len(regress_min)))
model_dt = np.ones((len(regress_min)))
model_dt_min = np.ones((len(regress_min)))
model_dt_max = np.ones((len(regress_min)))

for n in range(0, len(regress_min)):   
    regress_dt[n] = abs(solarmintimes_df['fracyear'].loc[n+21] - regress_min[n])
    model_dt[n] = abs(solarmintimes_df['fracyear'].loc[n+21] - model_min[n])
    model_dt_min[n] = abs(solarmintimes_df['fracyear'].loc[n+21] - model_min_min[n])
    model_dt_max[n] = abs(solarmintimes_df['fracyear'].loc[n+21] - model_min_max[n])


#compute teh CDF
sorted_data_regress = np.sort(abs(regress_dt))
cdf_regress = np.arange(1, len(sorted_data_regress) + 1) / len(sorted_data_regress)

sorted_data_model = np.sort(abs(model_dt))
cdf_model = np.arange(1, len(sorted_data_model) + 1) / len(sorted_data_model)

sorted_data_model_min = np.sort(abs(model_dt_min))
cdf_model_min = np.arange(1, len(sorted_data_model_min) + 1) / len(sorted_data_model_min)

sorted_data_model_max = np.sort(abs(model_dt_max))
cdf_model_max = np.arange(1, len(sorted_data_model_max) + 1) / len(sorted_data_model_max)

# Compute the interquartile range (IQR)
q25_regress = np.percentile(sorted_data_regress, 25)
q75_regress = np.percentile(sorted_data_regress, 75)
q50_regress = np.percentile(sorted_data_regress, 50)
iqr_regress = q75_regress - q25_regress

q25_model = np.percentile(sorted_data_model, 25)
q75_model = np.percentile(sorted_data_model, 75)
q50_model = np.percentile(sorted_data_model, 50)
iqr_model = q75_model - q25_model


xx = [0-0.05,max(max(regress_dt), max(model_dt))+0.05]

fig = plt.figure()
# Plot the CDF and add shaded area for the IQR
plt.fill_between(xx, [0.25, 0.25], [0.75, 0.75], color='gray', alpha=0.3)
plt.plot(xx, [0.5, 0.5], 'k')


# Plot the CDF
plt.step(sorted_data_regress, cdf_regress, 'b', label='Regression')
plt.step(sorted_data_model, cdf_model, 'r', label='Forward model')
#plt.step(sorted_data_model_max, cdf_model_max, 'r--', label='Forward model, max')
#plt.step(sorted_data_model_min, cdf_model_min, 'r.', label='Forward model, min')
plt.xlabel(r'$\Delta$ t [years]')
plt.ylabel('Cumulative Probability')
plt.xlim(xx)
plt.grid(True)
plt.legend(fontsize =12)

#fig.savefig(figdir + 'SSN_recon_dt.pdf')
   


# <codecell> Plot the GEO OSF - SSN reconstruction results
osf_yrs = osfG['year'].to_numpy()
osf = osfG['OSF_BV_corr'].to_numpy()
osf_min = osfG['OSF_BV_corr_min'].to_numpy()
osf_max = osfG['OSF_BV_corr_max'].to_numpy()
osf_ssn = osfG['Rsilso'].to_numpy()
xplotlims = (1845, 2025) 

#fit the OSF-SSN scatter
#ssn_from_osfsq_coefficients = np.polyfit(pow(osf,power),osf_ssn,1)


#find the interval of overlap between 1y regress, 1+11y regress and osf model.
valid_indices = ~np.isnan(ssn_from_osf_regress) \
    & ~np.isnan( ssn_regress_11) & ~np.isnan( geo_concensus_osf[:,1]) 

#=============================================================================
#scatter plots
#=============================================================================

ssn_max = 270
ssnplotlims = [-100, 400]
fig = plt.figure(figsize = (12,8))


ax = plt.subplot(2,3,1)
ax.plot(osfG['OSF_BV_corr'], osf_ssn, 'ko')
xrange = np.arange(2,14,0.1)
ax.plot(xrange, np.polyval(ssn_from_osfsq_coefficients, pow(xrange,2)), 'b')
ax.set_ylabel(r'SILSOv2 SSN', fontsize=12)
ax.set_xlabel(r'GEO OSF [x$10^{14}$ Wb]', fontsize=12)  
ax.set_ylim((0,ssn_max))
ax.text(0.04, 0.93, '(a)', transform=plt.gca().transAxes, fontsize=12)
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')


ax = plt.subplot(2,3,2)
ax.plot(osfG['OSF_11ysmooth'], osfG['SSN_11ysmooth'], 'ko')
xrange = ax.get_xlim()
ax.plot(xrange, np.polyval(ssnsmooth_from_osfsmooth_coefficients, xrange), 'b')
ax.set_ylabel(r'SILSOv2 <SSN>$_{11}$', fontsize=12)
ax.set_xlabel(r'GEO <OSF>$_{11}$ [x$10^{14}$ Wb]', fontsize=12)  
ax.text(0.04, 0.93, '(b)', transform=plt.gca().transAxes, fontsize=12)
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

ax = plt.subplot(2,3,3)
ax.plot(osfG['dOSF_11'], osfG['dSSN_11'], 'ko')
xrange = ax.get_xlim()
ax.plot(xrange, np.polyval(dssn11_from_dosf11_coefficients, xrange), 'b')
ax.set_ylabel(r'SILSOv2 SSN - <SSN>$_{11}$', fontsize=12)
ax.set_xlabel(r'GEO OSF - <OSF>$_{11}$ [x$10^{14}$ Wb]', fontsize=12)  
ax.text(0.04, 0.93, '(c)', transform=plt.gca().transAxes, fontsize=12)
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

ax = plt.subplot(2,3,4)
plt.errorbar(ssn_from_osf_regress, osf_ssn, 
             xerr=[ssn_from_osf_lowererror,  ssn_from_osf_uppererror], fmt='ko')
plt.plot([0,ssn_max],[0,ssn_max], 'r')
plt.xlabel(r'SSN from GEO OSF$^2$ regression', fontsize = 12)
rl =   mplot.lin_correl(osf_ssn[valid_indices], ssn_from_osf_regress[valid_indices])
mae = np.nanmean(abs(osf_ssn[valid_indices] - ssn_from_osf_regress[valid_indices]))
plt.title(r'$r_L$ = ' + "{:.2f}".format(rl[0]) + '; MAE = ' + "{:.1f}".format(mae), fontsize = 12) 
ax.set_ylim((0,ssn_max))
ax.set_ylabel(r'SILSOv2 SSN ', fontsize=12)
ax.text(0.04, 0.93, '(d)', transform=plt.gca().transAxes, fontsize=12)
ax.set_xlim(ssnplotlims) 


ax = plt.subplot(2,3,5)
plt.errorbar(ssn_regress_11, osf_ssn, 
             xerr=[ssn_regress_11_lowererror,  ssn_regress_11_uppererror], fmt='ko')
plt.plot([0,ssn_max],[0,ssn_max], 'r')
plt.xlabel(r'SSN from GEO OSF regression (1 + 11y)', fontsize = 12)
rl =   mplot.lin_correl(osf_ssn[valid_indices], ssn_regress_11[valid_indices])
mae = np.nanmean(abs(osf_ssn[valid_indices] - ssn_regress_11[valid_indices]))
plt.title(r'$r_L$ = ' + "{:.2f}".format(rl[0]) + '; MAE = ' + "{:.1f}".format(mae), fontsize = 12) 
ax.set_ylim((0,ssn_max))
ax.set_ylabel(r'SILSOv2 SSN', fontsize=12)
#ax.set_yticklabels([])
ax.text(0.04, 0.93, '(e)', transform=plt.gca().transAxes, fontsize=12)
ax.set_xlim(ssnplotlims) 

ax = plt.subplot(2,3,6)
plt.errorbar(geo_concensus_ssn[:,1], osf_ssn, 
             xerr=[abs(geo_concensus_ssn[:,1] - geo_concensus_ssn_min[:,1]),
                   abs(geo_concensus_ssn[:,1] - geo_concensus_ssn_max[:,1])], fmt='ko')
plt.plot([0,ssn_max],[0,ssn_max], 'r')
plt.xlabel(r'SSN from GEO OSF model', fontsize = 12)
rl =  mplot.lin_correl(osf_ssn[valid_indices], geo_concensus_ssn[valid_indices,1])
mae = np.nanmean(abs(osf_ssn[valid_indices] - geo_concensus_ssn[valid_indices,1]))
plt.title(r'$r_L$ = ' + "{:.2f}".format(rl[0]) + '; MAE = ' + "{:.1f}".format(mae), fontsize = 12) 
ax.set_ylim((0,ssn_max))
ax.set_ylabel(r'SILSOv2 SSN', fontsize=12)
#ax.set_yticklabels([])
ax.text(0.04, 0.93, '(f)', transform=plt.gca().transAxes, fontsize=12)
ax.set_xlim(ssnplotlims) 

plt.tight_layout()
#fig.savefig(figdir + 'SSN_recon_scatterplots.pdf')



#=============================================================================
#time series plots
#=============================================================================


fig = plt.figure(figsize = (10,10))

ax = plt.subplot(4,1,1)
ax.plot(osf_yrs, osf_ssn, 'k', label = 'SILSOv2')
ax.plot(osf_yrs, ssn_from_osf_regress, 'b', label = r'Regression, OSF$^2$')
ax.plot(osf_yrs, ssn_regress_11, 'r', label = 'Regression, 1+11yr')
#plt.plot(osf_yrs, osf_ssn, 'k') 
ax.set_ylabel(r'SSN', fontsize = 12)
ax.set_xlim(xplotlims)
ax.text(0.93, 0.89, '(a)', transform=plt.gca().transAxes, fontsize=12)
ax.legend(fontsize = 12, ncol = 3, loc = 'upper left')
ax.set_xticklabels([])
ax.set_ylim((-50,320))
ax.plot(xplotlims, [0,0], 'k')

ax = plt.subplot(4,1,2)
ax.fill_between(osf_yrs, geo_concensus_ssn_min[:,1], geo_concensus_ssn_max[:,1], facecolor = 'pink')
ax.plot(osf_yrs, osf_ssn, 'k', label = 'SILSOv2')
ax.plot(osf_yrs, geo_concensus_ssn[:,1], 'r', label = 'Forward model')
#plt.plot(osf_yrs, osf_ssn, 'k') 
ax.set_ylabel(r'SSN', fontsize = 12)
ax.set_xlim(xplotlims)
ax.text(0.93, 0.89, '(b)', transform=plt.gca().transAxes, fontsize=12)
ax.legend(fontsize = 12, ncol = 3, loc = 'upper left')
ax.set_xticklabels([])
ax.set_ylim((-50,320))
ax.plot(xplotlims, [0,0], 'k')

ax = plt.subplot(4,1,3)
ax.fill_between(osf_yrs, geo_concensus_osf_min[:,1], geo_concensus_osf_max[:,1], facecolor = 'pink')
ax.plot(osf_yrs, osf, 'k', label = 'Geomagnetic') 
ax.plot(osf_yrs, geo_concensus_osf[:,1], 'r', label = 'Forward model')
ax.set_ylabel(r'OSF [x$10^{14}$ Wb]', fontsize = 12)
ax.set_xlim(xplotlims)
ax.set_ylim((3,12))
ax.text(0.93, 0.89, '(c)', transform=plt.gca().transAxes, fontsize=12)
ax.legend(fontsize = 12, ncol =2, loc = 'upper left')
ax.set_xticklabels([])




ax = plt.subplot(4,1,4)
ax.plot(hi_res_yrs, kde_values_min,'pink')
ax.plot(hi_res_yrs, kde_values_max,'pink')
ax.plot(hi_res_yrs, kde_values,'r')
yy = ax.get_ylim()
for n in range(0,len(solarmintimes_df)):
    ax.plot([solarmintimes_df['fracyear'][n], solarmintimes_df['fracyear'][n]],
            yy,'k')
for n in range(0,len(regress_min)):
    ax.plot([regress_min[n], regress_min[n]],
            yy,'b--')
ax.set_xlim(xplotlims)   
ax.set_ylabel('Cycle start kernel' + '\ndensity [Arb. units]', fontsize = 12) 
ax.set_ylim(yy)
ax.text(0.93, 0.89, '(d)', transform=plt.gca().transAxes, fontsize=12)
ax.set_xlabel('Year', fontsize = 12) 

plt.tight_layout()
#fig.savefig(figdir + 'SSN_recon_from_GEO.pdf')




   
    