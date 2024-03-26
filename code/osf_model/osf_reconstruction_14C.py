# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 09:10:58 2023

@author: mathewjowens
"""

#OSF reconstruction and space climate routines

import numpy as np
import pandas as pd
from datetime import datetime
import os as os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import urllib.request

from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

import helio_time as htime
import sunspots 
import mplot


osfGEOfilepath = os.path.join(os.environ['DBOX'],'Data','Geomag_B_V_2023.txt') 

#OSF computed from annual B and V requires a correction - see OSF_strahl_Brmod_BVgeo.py
coefficients = np.array([ 0.66544633, -1.15354585])


#start and stop years for loss rate calculation
startyr = 1878.9
stopyr = 2019.96

# <codecell> functions

#load and process Ilya's 2023 annual OSF estimates from 14C

def load_usoskin_osf(data_dir = None, download_now = False):

    if data_dir is None:
        data_dir = os.path.join(os.environ['DBOX'], 'Data')
        
    osffilepath = os.path.join(data_dir,'Usoskin2023_osf.dat')
    ssnfilepath = os.path.join(data_dir,'Usoskin2023_ssn.dat')
    
    if download_now:
        urllib.request.urlretrieve('http://cdsarc.u-strasbg.fr/ftp/J/A+A/649/A141/osf.dat',
                                    osffilepath)
        urllib.request.urlretrieve('http://cdsarc.u-strasbg.fr/ftp/J/A+A/649/A141/osn.dat',
                                    ssnfilepath)


    usoskin_osf_df = pd.read_csv(osffilepath,
                         delim_whitespace=True,
                         names=['intyear','osf', 'osf_1-sigma','osf_smooth22'])
    
    usoskin_ssn_df = pd.read_csv(ssnfilepath,
                         delim_whitespace=True,
                         names=['intyear','ssn', 'ssn_1-sigma','ssn_smooth22'])
    
    #copy the SSN data to the OSF df
    usoskin_osf_df['ssn'] = usoskin_ssn_df['ssn']
    usoskin_osf_df['ssn_1-sigma'] = usoskin_ssn_df['ssn_1-sigma']
    usoskin_osf_df['ssn_smooth22'] = usoskin_ssn_df['ssn_smooth22']
    
    
    #create fracyear and MJD columns
    usoskin_osf_df['fracyear'] = usoskin_osf_df['intyear'] + 0.5
    
    doy = (usoskin_osf_df['fracyear'] - np.floor(usoskin_osf_df['fracyear']))*364 + 1
    doy = doy.to_numpy()
    yr = np.floor(usoskin_osf_df['fracyear']).to_numpy()
    yr=yr.astype(int)
    usoskin_osf_df['mjd'] = htime.doyyr2mjd(doy,yr)
    
    return usoskin_osf_df





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


# <codecell> sunspot data

#read in the solar minimum times
solarmintimes_df = sunspots.LoadSolarMinTimes()

#Read in the sunspot data 
ssn_df = sunspots.LoadSSN(download_now = True)

#create annual SSN record
ssn_1y = ssn_df.resample('1Y', on='datetime').mean() 
ssn_1y['datetime'] = htime.mjd2datetime(ssn_1y['mjd'].to_numpy())
ssn_1y.reset_index(drop=True, inplace=True)
nt_ssn = len(ssn_1y)
#recompute phase, as it will have been averaged over 0/2pi
ssn_1y['phase'] = sunspots.solar_cycle_phase(ssn_1y['mjd'])


solarmin_mjd = solarmintimes_df['mjd']

SPE_SSN_phase, SPE_SSN_NORM_AVG, SPE_SSN_NORM_STD = sunspots.ssn_analogue_forecast(ssn_df, 
                        solarmin_mjd,  nphase = 11*12, peak_ssn = 140, plotnow = True)


# load teh OSF data to compute the loss rate

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



#create a phase series   
osfG['phase'] = sunspots.solar_cycle_phase(osfG['mjd'] )

mask = (osfG['year'] >= startyr) & (osfG['year'] <= stopyr)

#compte the loss term
osfGEO = osfG.loc[mask,'OSF_BV_corr'].to_numpy() 
mjd =  osfG.loc[mask,'mjd'].to_numpy() 
ssn = osfG.loc[mask,'Rsilso'].to_numpy() 

# osf = omni_1y['OSF'] 
# mjd =  omni_1y['mjd'] 
# source = omni_1y['OSF_source'] 

SPE_phase, SPE_LOSS_AVG, SPE_LOSS_STD, SPE_LOSS  = sunspots.compute_osfloss_SPE(mjd, osfGEO, ssn, 
                       nphase = 11, plotnow = True)





# <codecell>  Reconstruction of SSN from 14C

#load Ilya's OSF from 14C
osfU = sunspots.load_usoskin_osf()

startyr = 950
stopyr = 2020

window_yrs = 22
dwindow = 2

runtype = 'best'
savenow = False



mask = (osfU['fracyear'] >= startyr) & (osfU['fracyear'] <= stopyr)
osf14C_yrs = osfU['fracyear'].loc[mask].to_numpy()
osf14C_ssn = osfU['ssn'].loc[mask].to_numpy()
xplotlims = (startyr+11, stopyr)  




if runtype == 'best':
    osf14C = osfU['osf'].loc[mask].to_numpy()
elif runtype == 'upper':
    osf14C = osfU['osf'].loc[mask].to_numpy() +  osfU['osf_1-sigma'].loc[mask].to_numpy()
elif runtype == 'lower':
    osf14C = osfU['osf'].loc[mask].to_numpy() -  osfU['osf_1-sigma'].loc[mask].to_numpy()
    mask = (osf14C < 0)
    osf14C[mask] = 0


#correct 14C OSF to geomagnetic level
mask = osf14C_yrs >= np.floor(osfG.loc[0, 'year'])
osf_14C_mean = np.nanmean(osf14C[mask])
mask = osfG['year'] <= np.ceil(osf14C_yrs[-1])
osf_GEO_mean = np.nanmean(osfG.loc[mask,'OSF_BV_corr'])

osf14C = osf14C * osf_GEO_mean/osf_14C_mean

#compute the SSN from 14C OSF using the Monte Carlo method
final_yrs, final_ssn, final_osf, final_mae, final_minyrs = ssn_from_osf(osf14C_yrs, osf14C, 
                                                                        SPE_SSN_phase, SPE_SSN_NORM_AVG,
                                                                        SPE_phase, SPE_LOSS_AVG,
                                                                        window_yrs = window_yrs, 
                                                                        dwindow = dwindow, N=10000,
                                                                        cycle_amp_std = 100)
#produce consensus SSN and OSF and min years
concensus_ssn, concensus_osf = produce_concensus_ssn_osf(osf14C_yrs,  final_yrs, final_osf, final_ssn, final_mae)

model_min, hi_res_yrs, kde_values = produce_concensus_cycle_starts(osf14C_yrs, final_mae, final_minyrs)

# <codecell> Plot the results

ylims = [-75, 300]

fig = plt.figure(figsize = (8,8))
gs = gridspec.GridSpec(3, 2)

###############################################################################

ax = fig.add_subplot(gs[0, 0:2])

ax.fill_between(osfU['fracyear'],  osfU['ssn'] - osfU['ssn_1-sigma'], 
                osfU['ssn'] + osfU['ssn_1-sigma'], facecolor = 'blue', alpha = 0.2)
ax.fill_between(ssn_1y['fracyear'],  ssn_1y['smooth']*0, ssn_1y['smooth'], 
                 facecolor = 'silver', alpha = 0.5, edgecolor = 'k', label = 'Observed (SILSOv2)')

ax.plot( osfU['fracyear'],  osfU['ssn'], 'b', label = '14C regression')


# ax.plot(ssn_1y['fracyear'],  ssn_1y['smooth'], 'k',
#                  label = 'Observed (SILSOv2)')

#plt.plot(osf14C_yrs, concensus_ssn[:,1], 'r', label = '14C forward model')
#plt.plot(ssn_1y['fracyear'], ssn_1y['smooth'], 'k', label = 'Observed (SILSOv2)') 
ax.set_ylabel(r'Sunspot number', fontsize = 12)
ax.set_xlim(xplotlims); ax.set_ylim(ylims)
ax.plot(xplotlims,[0,0], 'k')
ax.legend(loc = 'upper center', fontsize = 12)

###############################################################################

ax = fig.add_subplot(gs[1, 0:2])
ax.fill_between(ssn_1y['fracyear'], ssn_1y['smooth']*0, ssn_1y['smooth'], facecolor = 'silver', edgecolor='k',
                 label = 'Observed (SILSOv2)', alpha = 0.5,)
#plt.plot(osf14C_yrs, osf14C_ssn, 'b', label = '14C regression')
ax.plot(osf14C_yrs, concensus_ssn[:,1], 'r', label = '14C forward model')
#plt.plot(ssn_1y['fracyear'], ssn_1y['smooth'], 'k', label = 'Observed (SILSOv2)') 
ax.set_ylabel(r'Sunspot number', fontsize = 12)
ax.set_xlim(xplotlims); ax.set_ylim(ylims)
ax.plot(xplotlims,[0,0], 'k')
ax.legend(loc = 'upper center',fontsize = 12)
ax.set_xlabel('Year',fontsize = 12)


###############################################################################
ax = fig.add_subplot(gs[2, 0])
bins = np.arange(-100,300,10)
bin_centres = (bins[1:]+bins[0:-1])/2

ax.set_title('950-1845')
mask = osf14C_yrs <= 1845
n, x = np.histogram(osf14C_ssn[mask], bins)
ax.step(bin_centres,n,'b', label = '14C regression')
n, x = np.histogram(concensus_ssn[mask,1], bins)
ax.step(bin_centres,n,'r', label = '14C forward model')
ax.legend()
ax.set_ylabel('Count',fontsize = 12)
ax.set_xlabel('Sunspot number',fontsize = 12)
###############################################################################
ax = fig.add_subplot(gs[2, 1])
bins = np.arange(-100,300,20)
bin_centres = (bins[1:]+bins[0:-1])/2

ax.set_title('1845-1900')
mask = (ssn_1y['fracyear'] >= 1845) & (ssn_1y['fracyear'] <= 1901)
n, x = np.histogram(ssn_1y.loc[mask,'smooth'], bins)
#ax.plot(bin_centres,n,'k',label = 'Observed (SILSOv2)')
ax.fill_between(bin_centres,  n*0, n, facecolor = 'silver', edgecolor = 'grey', label = 'Observed (SILSOv2)')

mask = (osf14C_yrs >= 1845) & (osf14C_yrs <= 1901)
n, x = np.histogram(osf14C_ssn[mask], bins)
ax.step(bin_centres, n,  'b', label = '14C regression')
n, x = np.histogram(concensus_ssn[mask,1], bins)
ax.step(bin_centres, n, 'r', label = '14C forward model')
ax.legend()
ax.set_xlabel('Sunspot number',fontsize = 12)

plt.tight_layout()

# #compute the individual SB width estimaes
# final_SBwidth, final_inc = SBwidth_from_individual_MC(final_yrs, final_ssn, final_minyrs, 
#                                                       SPE_phase, SPE_LOSS_AVG)
# concensus_sbw, concensus_inc = produce_concensus_SBwidth(osf_yrs, final_SBwidth, final_inc, final_mae)



# #save the data
# runname = 'SSN_14C_11yr_' + str(startyr) +'_to_' +str(stopyr) + '_' + runtype
# if savenow:
#     osf_save_dir = os.getenv('DBOX') + 'Papers_WIP\\_coauthor\\HisashiHayakawa\\'
#     filename = runname  + '.dat'
#     data = {'Year' : osf14C_yrs, 'OSF_orig' : osf14C, 'SSN_orig': osf14C_ssn, 
#             'OSF_recon': concensus_osf[:,1], 'SSN_recon': concensus_ssn[:,1],
#             'SBwidth': concensus_sbw[:,1], 'HCSinc': concensus_inc[:,1]}
#     df = pd.DataFrame(data)
#     df.to_csv(osf_save_dir + filename, index = False)
    
#     filename = runname  +'_minyears.dat'
#     df = pd.DataFrame({'Cycle start year': model_min})
#     df.to_csv(osf_save_dir + filename, index = False)
    
    

# #summary plot ofhte reconstruction
# plt.figure()
# plt.subplot(4,1,1)
# plt.plot(osf14C_yrs, concensus_osf[:,1], 'r', label = 'OSF, model')
# #plt.plot(osf_yrs, concensus_osf[:,0], 'b', label = 'OSF, model, unweighted')
# plt.plot(osf14C_yrs, osf14C, 'k', label = 'OSF, 14C') 
# plt.ylabel(r'OSF [x$10^{14}$ Wb]', fontsize = 12)
# plt.xlim(xplotlims)
# plt.legend(fontsize = 12)


# plt.subplot(4,1,2)
# plt.plot(osf14C_yrs, osf14C_ssn, 'b', label = 'SSN, orig')
# plt.plot(osf14C_yrs, concensus_ssn[:,1], 'r', label = 'SSN, OSF model')
# plt.plot(ssn_1y['fracyear'], ssn_1y['smooth'], 'k', label = 'SILSO') 
# plt.ylabel(r'SSN', fontsize = 12)
# plt.xlim(xplotlims)
# plt.legend(fontsize = 12)


# plt.subplot(4,1,3)
# plt.plot(osf14C_yrs, np.arcsin(concensus_sbw[:,1])*180/np.pi, 'r', label = 'SBW, OSF model')
# plt.plot(osf14C_yrs, concensus_inc[:,1]*180/np.pi, 'k', label = 'INC, OSF model')
# #plt.plot(osf_yrs, osf_ssn, 'k') 
# plt.ylabel(r'Angle [deg]', fontsize = 12)
# plt.xlim(xplotlims)
# plt.legend(fontsize = 12)

# ax = plt.subplot(4,1,4)
# ax.plot(hi_res_yrs, kde_values,'r')
# yy = ax.get_ylim()
# for n in range(12,len(solarmintimes_df)):
#     ax.plot([solarmintimes_df['fracyear'][n], solarmintimes_df['fracyear'][n]],
#             yy,'k')
# for n in range(0,len(model_min)):
#     ax.plot([model_min[n], model_min[n]],       yy,'r')
# ax.set_xlim(xplotlims)   
# ax.set_ylabel('Prob. cycle start', fontsize = 12) 
   

# <codecell> More compact plot

ylims = [-75, 300]

fig = plt.figure(figsize = (12,4))
gs = gridspec.GridSpec(1, 7)

###############################################################################

ax = fig.add_subplot(gs[0, 0:5])

# ax.fill_between(osfU['fracyear'],  osfU['ssn'] - osfU['ssn_1-sigma'], 
#                 osfU['ssn'] + osfU['ssn_1-sigma'], facecolor = 'blue', alpha = 0.2)
ax.fill_between(ssn_1y['fracyear'],  ssn_1y['smooth']*0, ssn_1y['smooth'], 
                 facecolor = 'silver', alpha = 0.5, edgecolor = 'k')

ax.plot( osfU['fracyear'],  osfU['ssn'], 'b', label = '14C, regression')
ax.plot(osf14C_yrs, concensus_ssn[:,1], 'r', label = '14C, forward model')

ax.plot(ssn_1y['fracyear'],  ssn_1y['smooth'], 'k',
                  label = 'Observed (SILSOv2)')

#plt.plot(osf14C_yrs, concensus_ssn[:,1], 'r', label = '14C forward model')
#plt.plot(ssn_1y['fracyear'], ssn_1y['smooth'], 'k', label = 'Observed (SILSOv2)') 
ax.set_ylabel(r'Sunspot number, R', fontsize = 12)
ax.set_xlim(xplotlims); ax.set_ylim(ylims)
ax.plot(xplotlims,[0,0], 'k')
ax.legend(loc = 'upper center', fontsize = 12)
ax.set_xlabel(r'Year', fontsize = 12)
###############################################################################

# ax = fig.add_subplot(gs[1, 0:2])
# ax.fill_between(ssn_1y['fracyear'], ssn_1y['smooth']*0, ssn_1y['smooth'], facecolor = 'silver', edgecolor='k',
#                  label = 'Observed (SILSOv2)', alpha = 0.5,)
# #plt.plot(osf14C_yrs, osf14C_ssn, 'b', label = '14C regression')
# ax.plot(osf14C_yrs, concensus_ssn[:,1], 'r', label = '14C forward model')
# #plt.plot(ssn_1y['fracyear'], ssn_1y['smooth'], 'k', label = 'Observed (SILSOv2)') 
# ax.set_ylabel(r'Sunspot number', fontsize = 12)
# ax.set_xlim(xplotlims); ax.set_ylim(ylims)
# ax.plot(xplotlims,[0,0], 'k')
# ax.legend(loc = 'upper center',fontsize = 12)
# ax.set_xlabel('Year',fontsize = 12)


###############################################################################




#create variance in 11-yr bins
var_U = np.ones((len(osfU)-12,1))*np.nan
for i in range(0, len(osfU)-12):
    var_U[i] = np.nanmax(osfU.loc[i:i+11, 'ssn']) - np.nanmin(osfU.loc[i:i+11, 'ssn'])
    
var_O = np.ones((len(concensus_ssn[:,1])-12,1))*np.nan
for i in range(0, len(concensus_ssn[:,1])-12):
    var_O[i] = np.nanmax(concensus_ssn[i:i+11,1]) - np.nanmin(concensus_ssn[i:i+11,1])


ax = fig.add_subplot(gs[0, 5:])

bins = np.arange(-20,370,15)
bin_centres = (bins[1:]+bins[0:-1])/2


n, x = np.histogram(var_O, bins)
ax.step(bin_centres,n/np.sum(n),'r', label = '14C, forward model')
plt.fill_between(bin_centres,n/np.sum(n), facecolor = 'r', step="pre", alpha=0.5)

n, x = np.histogram(var_U, bins)
ax.step(bin_centres,n/np.sum(n),'b', label = '14C, regression')
plt.fill_between(bin_centres,n/np.sum(n), facecolor = 'b', step="pre", alpha=0.5)

#ax.legend()
ax.set_ylabel('Probability density',fontsize = 12)
ax.set_xlabel(r'Sunspot cycle amplitude,' +'\n'+ r'$R_{MAX} - R_{MIN}$',fontsize = 12)
#ax. set_xscale("log", base=10)

ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")

plt.tight_layout()

# <codecell> Single panel plot


ylims = [-75, 300]

fig = plt.figure(figsize = (12,4))
ax=plt.subplot(1,1,1)


# ax.fill_between(osfU['fracyear'],  osfU['ssn'] - osfU['ssn_1-sigma'], 
#                 osfU['ssn'] + osfU['ssn_1-sigma'], facecolor = 'blue', alpha = 0.2)
ax.fill_between(ssn_1y['fracyear'],  ssn_1y['smooth']*0, ssn_1y['smooth'], 
                 facecolor = 'silver', alpha = 0.5, edgecolor = 'k')

ax.plot( osfU['fracyear'],  osfU['ssn'], 'b', label = '14C, regression')
ax.plot(osf14C_yrs, concensus_ssn[:,1], 'r', label = '14C, forward model')

ax.plot(ssn_1y['fracyear'],  ssn_1y['smooth'], 'k',
                  label = 'Observed SSN')

#plt.plot(osf14C_yrs, concensus_ssn[:,1], 'r', label = '14C forward model')
#plt.plot(ssn_1y['fracyear'], ssn_1y['smooth'], 'k', label = 'Observed (SILSOv2)') 
ax.set_ylabel(r'Sunspot number, SSN', fontsize = 14)
ax.set_xlim(xplotlims); ax.set_ylim(ylims)
ax.plot(xplotlims,[0,0], 'k')
ax.legend(loc = 'upper center', fontsize = 12)
ax.set_xlabel(r'Year', fontsize = 14)
###############################################################################

# ax = fig.add_subplot(gs[1, 0:2])
# ax.fill_between(ssn_1y['fracyear'], ssn_1y['smooth']*0, ssn_1y['smooth'], facecolor = 'silver', edgecolor='k',
#                  label = 'Observed (SILSOv2)', alpha = 0.5,)
# #plt.plot(osf14C_yrs, osf14C_ssn, 'b', label = '14C regression')
# ax.plot(osf14C_yrs, concensus_ssn[:,1], 'r', label = '14C forward model')
# #plt.plot(ssn_1y['fracyear'], ssn_1y['smooth'], 'k', label = 'Observed (SILSOv2)') 
# ax.set_ylabel(r'Sunspot number', fontsize = 12)
# ax.set_xlim(xplotlims); ax.set_ylim(ylims)
# ax.plot(xplotlims,[0,0], 'k')
# ax.legend(loc = 'upper center',fontsize = 12)
# ax.set_xlabel('Year',fontsize = 12)


###############################################################################


plt.tight_layout()