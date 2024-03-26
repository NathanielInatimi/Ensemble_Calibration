# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:04:07 2023

@author: vy902033
"""
#directory for input and output solar wind speed maps


import astropy.units as u
import numpy as np
import pandas as pd
import datetime
import h5py
import os as os
import matplotlib.pyplot as plt
import xarray as xr
import helio_time as htime
from datetime import datetime
import ReadICMElist_CaneRichardson as ICMElist


datadir = os.environ['DBOX'] + 'Data\\'


# <codecell> Load and process data

#read in Anna's topology data
filepath = datadir + 'HMF topologies all data.nc'
nc = xr.open_dataset(filepath)
df = nc.to_dataframe()

# compute the datetime and mjd of each time bin
dtarray = df.index.to_pydatetime()
df['datetime'] = dtarray
df['mjd'] = htime.datetime2mjd(dtarray)
df = df.reset_index()

#find teh CR range
start_mjd = df['mjd'][0] # 1625
stop_mjd = df['mjd'][len(df)-1]
window_length = 27.27
dmjd = 1




#flag ICMEs
df['isicme'] = False
df['issheath'] = False
icmes = ICMElist.ICMElist(datadir + 'ICME_list\\List of Richardson_Cane ICMEs Since January1996_2022.csv') 
for i in range(0,len(icmes)):
    
    mask = ((df['datetime'] >= icmes['ICME_start'][i])
            & (df['datetime'] < icmes['ICME_end'][i]) )
    df.loc[mask,'isicme'] = True
    
    mask = ((df['datetime'] >= icmes['Shock_time'][i])
            & (df['datetime'] < icmes['ICME_start'][i]) )
    df.loc[mask,'issheath'] = True
   

        
# <codecell> Loop through each CR and calculate helioflux contributions      

window_starts = np.arange(start_mjd, stop_mjd - window_length, dmjd)
nwindows = len(window_starts)


flux_obs = np.ones((nwindows))
flux_gaps =  np.ones((nwindows))
fluxSS_obs = np.ones((nwindows))
fluxnonSS_obs = np.ones((nwindows))
fluxSS_gaps = np.ones((nwindows))
fluxnonSS_gaps = np.ones((nwindows))

flux_obs_sw = np.ones((nwindows))
flux_gaps_sw =  np.ones((nwindows))
fluxSS_obs_sw = np.ones((nwindows))
fluxnonSS_obs_sw = np.ones((nwindows))
fluxSS_gaps_sw = np.ones((nwindows))
fluxnonSS_gaps_sw = np.ones((nwindows))

flux_obs_icme = np.ones((nwindows))
flux_gaps_icme =  np.ones((nwindows))
fluxSS_obs_icme = np.ones((nwindows))
fluxnonSS_obs_icme = np.ones((nwindows))
fluxSS_gaps_icme = np.ones((nwindows))
fluxnonSS_gaps_icme = np.ones((nwindows))

flux_obs_sh = np.ones((nwindows))
flux_gaps_sh =  np.ones((nwindows))
fluxSS_obs_sh = np.ones((nwindows))
fluxnonSS_obs_sh = np.ones((nwindows))
fluxSS_gaps_sh = np.ones((nwindows))
fluxnonSS_gaps_sh = np.ones((nwindows))

N_tot = np.ones((nwindows))
N_tot_icme = np.ones((nwindows))
N_tot_sw = np.ones((nwindows))
N_tot_sh = np.ones((nwindows))

N_gaps = np.ones((nwindows))
N_gaps_sw = np.ones((nwindows))
N_gaps_icme = np.ones((nwindows))
N_gaps_sh = np.ones((nwindows))




df_cr = pd.DataFrame()
df_cr['mjd'] = window_starts + dmjd/2
df_cr['datetime'] = htime.mjd2datetime(df_cr['mjd'].to_numpy())

df_cr['N_tot'] = np.ones((nwindows))*0
df_cr['N_gaps'] = np.ones((nwindows))*0
df_cr['flux_obs'] = np.ones((nwindows))*0
df_cr['flux_gaps'] =  np.ones((nwindows))*0
df_cr['fluxSS_obs'] = np.ones((nwindows))*0
df_cr['fluxnonSS_obs'] = np.ones((nwindows))*0
df_cr['fluxSS_gaps'] = np.ones((nwindows))*0
df_cr['fluxnonSS_gaps'] = np.ones((nwindows))*0

df_cr['N_tot_am'] = np.ones((nwindows))*0
df_cr['N_gaps_am'] = np.ones((nwindows))*0
df_cr['flux_obs_am'] = np.ones((nwindows))*0
df_cr['flux_gaps_am'] =  np.ones((nwindows))*0
df_cr['fluxSS_obs_am'] = np.ones((nwindows))*0
df_cr['fluxnonSS_obs_am'] = np.ones((nwindows))*0
df_cr['fluxSS_gaps_am'] = np.ones((nwindows))*0
df_cr['fluxnonSS_gaps_am'] = np.ones((nwindows))*0

df_cr['N_tot_sh'] = np.ones((nwindows))*0
df_cr['N_gaps_sh'] = np.ones((nwindows))*0
df_cr['flux_obs_sh'] = np.ones((nwindows))*0
df_cr['flux_gaps_sh'] =  np.ones((nwindows))*0
df_cr['fluxSS_obs_sh'] = np.ones((nwindows))*0
df_cr['fluxnonSS_obs_sh'] = np.ones((nwindows))*0
df_cr['fluxSS_gaps_sh'] = np.ones((nwindows))*0
df_cr['fluxnonSS_gaps_sh'] = np.ones((nwindows))*0

df_cr['N_tot_icme'] = np.ones((nwindows))*0
df_cr['N_gaps_icme'] = np.ones((nwindows))*0
df_cr['flux_obs_icme'] = np.ones((nwindows))*0
df_cr['flux_gaps_icme'] =  np.ones((nwindows))*0
df_cr['fluxSS_obs_icme'] = np.ones((nwindows))*0
df_cr['fluxnonSS_obs_icme'] = np.ones((nwindows))*0
df_cr['fluxSS_gaps_icme'] = np.ones((nwindows))*0
df_cr['fluxnonSS_gaps_icme'] = np.ones((nwindows))*0

#convert from effective Br to OSF
au = 1.495978707e11
tfrac = 128 /(27.27*25*60*60)
#osf = 4 * np.pi * au*au *meanBrSS * 1e-9
conv =  4 * np.pi * au*au * 1e-9 *tfrac / 1e14

i = 0
for this_window_start in window_starts:

    #find the two CRs to be compared
    smjd = this_window_start
    fmjd = smjd + window_length
    
    #all data in this range
    mask = (df['mjd'] >= smjd) & (df['mjd'] < fmjd)
    
    datachunk = df[mask]
    
    # find the number of intervals for which data is present
    df_cr['N_tot'].iloc[i] = len(datachunk['Br'])
    df_cr['N_gaps'].iloc[i] = df_cr['N_tot'][i] - len(datachunk['Br'].dropna())
    
    #sum up the Br observations    
    df_cr['flux_obs'].iloc[i] = conv * np.nansum(abs(datachunk['Br']))
    #assume the flux in gaps is at the same rate as the non gaps (Ngaps x Br_avg)
    df_cr['flux_gaps'].iloc[i] = conv * df_cr['N_gaps'][i] * np.nanmean(abs(datachunk['Br']))

    #the SS flux is the closed plus anti sunward, minus twice the sunsward strahl    
    df_cr['fluxSS_obs'].iloc[i] = conv * ( np.nansum(datachunk['N_AS'] * abs(datachunk['Br'])) \
        + np.nansum(datachunk['N_CL'] * abs(datachunk['Br'])) \
            - np.nansum(datachunk['N_SS'] * abs(datachunk['Br'])))
        
    #the non-SS flux is twice the sunward strahl plus the unclassifed (disconnected)
    df_cr['fluxnonSS_obs'].iloc[i] = conv * ( 2* np.nansum(datachunk['N_SS'] * abs(datachunk['Br'])) + \
        np.nansum(datachunk['N_U'] * abs(datachunk['Br'])) )
    
    #assume the flux ratios are the same in the datagaps
    frac_SS = df_cr['fluxSS_obs'][i] /df_cr['flux_obs'][i]
    df_cr['fluxSS_gaps'].iloc[i] = frac_SS * df_cr['flux_gaps'][i]
    frac_nonSS = df_cr['fluxnonSS_obs'][i] /df_cr['flux_obs'][i]
    df_cr['fluxnonSS_gaps'].iloc[i] = frac_nonSS * df_cr['flux_gaps'][i]
    
    
    #Ambient solar wind only
    #==========================================================================
    mask = ((df['mjd'] >= smjd) & (df['mjd'] < fmjd) & ~df['isicme'] & ~df['issheath'])
    datachunk = df[mask]
    
    # find the number of intervals for which data is present
    df_cr['N_tot_am'].iloc[i] = len(datachunk['Br'])
    df_cr['N_gaps_am'].iloc[i] = df_cr['N_tot_am'][i] - len(datachunk['Br'].dropna())
    
    #sum up the Br observations    
    df_cr['flux_obs_am'].iloc[i] = conv * np.nansum(abs(datachunk['Br']))
    #assume the flux in gaps is at the same rate as the non gaps (Ngaps x Br_avg)
    df_cr['flux_gaps_am'].iloc[i] = conv * df_cr['N_gaps_am'][i] * np.nanmean(abs(datachunk['Br']))

    #the SS flux is the closed plus anti sunward, minus twice the sunsward strahl    
    df_cr['fluxSS_obs_am'].iloc[i] = conv * ( np.nansum(datachunk['N_AS'] * abs(datachunk['Br'])) \
        + np.nansum(datachunk['N_CL'] * abs(datachunk['Br'])) \
            - np.nansum(datachunk['N_SS'] * abs(datachunk['Br'])) )
        
    #the non-SS flux is twice the sunward strahl plus the unclassifed (disconnected)
    df_cr['fluxnonSS_obs_am'].iloc[i] = conv * ( 2* np.nansum(datachunk['N_SS'] * abs(datachunk['Br'])) + \
        np.nansum(datachunk['N_U'] * abs(datachunk['Br'])) )
    
    #assume the flux ratios are the same in the datagaps
    frac_SS_sw = df_cr['fluxSS_obs_am'][i] /df_cr['flux_obs_am'][i]
    df_cr['fluxSS_gaps_am'].iloc[i] = frac_SS_sw * df_cr['flux_gaps_am'][i]
    frac_nonSS_sw = df_cr['fluxnonSS_obs_am'][i] /df_cr['flux_obs_am'][i]
    df_cr['fluxnonSS_gaps_am'].iloc[i] = frac_nonSS_sw * df_cr['flux_gaps_am'][i]
    
    #ICME only
    #==========================================================================
    mask = ((df['mjd'] >= smjd) & (df['mjd'] < fmjd) & df['isicme'])
    datachunk = df[mask]
    
    # find the number of intervals for which data is present
    df_cr['N_tot_icme'].iloc[i] = len(datachunk['Br'])
    df_cr['N_gaps_icme'].iloc[i] = df_cr['N_tot_icme'][i] - len(datachunk['Br'].dropna())
    
    if df_cr['N_tot_icme'][i] > 0 :
    
        #sum up the Br observations    
        df_cr['flux_obs_icme'].iloc[i] = conv * np.nansum(abs(datachunk['Br']))
        #assume the flux in gaps is at the same rate as the non gaps (Ngaps x Br_avg)
        df_cr['flux_gaps_icme'].iloc[i] = conv * df_cr['N_gaps_icme'][i] * np.nanmean(abs(datachunk['Br']))
    
        #the SS flux is the closed plus anti sunward, minus twice the sunsward strahl    
        df_cr['fluxSS_obs_icme'].iloc[i] = conv * ( np.nansum(datachunk['N_AS'] * abs(datachunk['Br'])) \
            + np.nansum(datachunk['N_CL'] * abs(datachunk['Br'])) \
                - np.nansum(datachunk['N_SS'] * abs(datachunk['Br'])) )
            
        #the non-SS flux is twice the sunward strahl plus the unclassifed (disconnected)
        df_cr['fluxnonSS_obs_icme'].iloc[i] = conv * ( 2* np.nansum(datachunk['N_SS'] * abs(datachunk['Br'])) + \
            np.nansum(datachunk['N_U'] * abs(datachunk['Br'])) )
        
        #assume the flux ratios are the same in the datagaps
        frac_SS_icme = df_cr['fluxSS_obs_icme'][i] /df_cr['flux_obs_icme'][i]
        df_cr['fluxSS_gaps_icme'].iloc[i] = frac_SS_icme * df_cr['flux_gaps_icme'][i]
        frac_nonSS_icme = df_cr['fluxnonSS_obs_icme'][i] / df_cr['flux_obs_icme'].iloc[i]
        df_cr['fluxnonSS_gaps_icme'].iloc[i] = frac_nonSS_icme * df_cr['flux_gaps_icme'][i]
    
    
    #ICME Sheath only
    #==========================================================================
    mask = ((df['mjd'] >= smjd) & (df['mjd'] < fmjd) & df['issheath'])
    datachunk = df[mask]
    
    # find the number of intervals for which data is present
    df_cr['N_tot_sh'].iloc[i] = len(datachunk['Br'])
    df_cr['N_gaps_sh'].iloc[i] = df_cr['N_tot_sh'][i] - len(datachunk['Br'].dropna())
    
    if df_cr['N_tot_icme'][i] > 0 :
    
        #sum up the Br observations    
        df_cr['flux_obs_sh'].iloc[i] = conv * np.nansum(abs(datachunk['Br']))
        #assume the flux in gaps is at the same rate as the non gaps (Ngaps x Br_avg)
        df_cr['flux_gaps_sh'].iloc[i] = conv * df_cr['N_gaps_sh'][i] * np.nanmean(abs(datachunk['Br']))
    
        #the SS flux is the closed plus anti sunward, minus twice the sunsward strahl    
        df_cr['fluxSS_obs_sh'].iloc[i] = conv * ( np.nansum(datachunk['N_AS'] * abs(datachunk['Br'])) \
            + np.nansum(datachunk['N_CL'] * abs(datachunk['Br'])) \
                - np.nansum(datachunk['N_SS'] * abs(datachunk['Br'])) )
            
        #the non-SS flux is twice the sunward strahl plus the unclassifed (disconnected)
        df_cr['fluxnonSS_obs_sh'].iloc[i] = conv * ( 2* np.nansum(datachunk['N_SS'] * abs(datachunk['Br'])) + \
        
                                                    np.nansum(datachunk['N_U'] * abs(datachunk['Br'])) )
            
        #assume the flux ratios are the same in the datagaps
        frac_SS_sh = df_cr['fluxSS_obs_sh'][i] / df_cr['flux_obs_sh'][i]
        df_cr['fluxSS_gaps_sh'].iloc[i] = frac_SS_sh * df_cr['flux_gaps_sh'][i]
        frac_nonSS_sh = df_cr['fluxnonSS_obs_sh'][i] / df_cr['flux_obs_sh'][i]
        df_cr['fluxnonSS_gaps_sh'].iloc[i] = frac_nonSS_sh * df_cr['flux_gaps_sh'][i]


    i = i + 1
    
    
#save the data
savefile = os.getenv('DBOX') + 'Papers_WIP\\_coauthor\\ManuelaTemmer\\osf_icme_hires.csv'
df_cr.to_csv(savefile)
# <codecell> Plots
    

plt.figure()
           
plt.plot(df_cr['datetime'], df_cr['flux_gaps'] + df_cr['flux_obs'], 
         label = 'Total', color='black', linestyle = 'solid')
plt.plot(df_cr['datetime'], df_cr['flux_obs'],label = 'Total (obs)',
         color='black', linestyle = 'dashed')
plt.plot(df_cr['datetime'], df_cr['flux_gaps'],label = 'Total (data gaps)',
         color='black', linestyle = 'dotted')

plt.plot(df_cr['datetime'], df_cr['fluxSS_gaps'] + df_cr['fluxSS_obs'], 
         label = 'Total SS = OSF', color='red', linestyle = 'solid')
plt.plot(df_cr['datetime'], df_cr['fluxSS_obs'],label = 'SS (obs)',
         color='red', linestyle = 'dashed')
plt.plot(df_cr['datetime'], df_cr['fluxSS_gaps'],label = 'SS (data gaps)',
         color='red', linestyle = 'dotted')

plt.plot(df_cr['datetime'], df_cr['fluxnonSS_gaps'] + df_cr['fluxnonSS_obs'], 
         label = 'Total non-SS flux', color='blue', linestyle = 'solid')
plt.plot(df_cr['datetime'], df_cr['fluxnonSS_obs'],label = 'Non-SS (obs)',
         color='blue', linestyle = 'dashed')
plt.plot(df_cr['datetime'], df_cr['fluxnonSS_gaps'],label = 'Non-SS (data gaps)',
         color='blue', linestyle = 'dotted')
plt.legend()
plt.ylabel('Flux [$x10^{14}$ Wb]')


plt.figure()

plt.plot(df_cr['datetime'], df_cr['fluxSS_obs'] + df_cr['fluxSS_gaps'] ,label = 'Total SS = OSF',
         color='black', linestyle = 'solid')
plt.plot(df_cr['datetime'], df_cr['fluxSS_obs_am'] + df_cr['fluxSS_gaps_am'],
         label = 'Total SS (ambient wind)', color='blue', linestyle = 'solid')
plt.plot(df_cr['datetime'], df_cr['fluxSS_obs_icme'] + df_cr['fluxSS_obs_sh'] +
         df_cr['fluxSS_gaps_icme'] + df_cr['fluxSS_gaps_sh']
         ,label = 'Total SS (ICME+sheath)', color='red', linestyle = 'solid')


plt.plot(df_cr['datetime'], df_cr['fluxnonSS_obs'] + df_cr['fluxnonSS_gaps'],label = 'Total non-SS',
         color='black', linestyle = 'dashed')
plt.plot(df_cr['datetime'], df_cr['fluxnonSS_obs_am'] + df_cr['fluxnonSS_gaps_am']
         ,label = 'Total non-SS (ambient wind)', color='blue', linestyle = 'dashed')
plt.plot(df_cr['datetime'], df_cr['fluxnonSS_obs_icme'] + df_cr['fluxnonSS_obs_sh']
         + df_cr['fluxnonSS_gaps_icme'] + df_cr['fluxnonSS_gaps_sh'],
         label = 'Total non-SS (ICME+sheath)', color='red', linestyle = 'dashed')
#plt.plot(fluxnonSS_gaps,label = 'Non SS Flux in data gaps')
plt.legend()
plt.ylabel('Flux [$x10^{14}$ Wb]')


plt.figure()
plt.plot(df_cr['datetime'], 100 * (df_cr['fluxSS_obs_am'] + df_cr['fluxSS_gaps_am']) \
         / (df_cr['fluxSS_gaps'] + df_cr['fluxSS_obs']),
         label = 'Ambient wind', color='blue', linestyle = 'solid')
plt.plot(df_cr['datetime'], 100 * (df_cr['fluxSS_obs_icme'] + df_cr['fluxSS_gaps_icme']
                                   + df_cr['fluxSS_obs_sh'] + df_cr['fluxSS_gaps_sh']) \
         / (df_cr['fluxSS_gaps'] + df_cr['fluxSS_obs']),
         label = 'ICME+Sheath', color='red', linestyle = 'solid')     
plt.legend()
plt.ylabel('% of OSF')                                                                                 

plt.figure()
plt.plot(df_cr['datetime'], 100 * (df_cr['fluxSS_obs_am'] + df_cr['fluxSS_gaps_am']
                                   + df_cr['fluxSS_obs_sh'] + df_cr['fluxSS_gaps_sh']) \
         / (df_cr['fluxSS_gaps'] + df_cr['fluxSS_obs']),
         label = 'Ambient+Sheath', color='blue', linestyle = 'solid')
plt.plot(df_cr['datetime'], 100 * (df_cr['fluxSS_obs_icme'] + df_cr['fluxSS_gaps_icme']) \
         / (df_cr['fluxSS_gaps'] + df_cr['fluxSS_obs']),
         label = 'ICME', color='red', linestyle = 'solid')     
plt.legend()
plt.ylabel('% of OSF')  


