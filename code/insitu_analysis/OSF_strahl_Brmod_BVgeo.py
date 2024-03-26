# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 17:43:12 2023

@author: mathewjowens

a script to compare OSF estimates from teh strahl method (Frost 2022), the 20hr 
modulus of Br with OMNI data and B and V from geomagnetic reconstructions

"""



import astropy.units as u
import numpy as np
import pandas as pd
import datetime
import h5py
import os as os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import helio_time as htime


import helio_time as htime
#import ReadICMElist_CaneRichardson as ICMElist
import sunspots
import mplot


datadir = os.environ['DBOX'] + 'Data\\'
omnipath = os.environ['DBOX'] + '\\Data_hdf5\omni_1hour.h5'
osfGEOfilepath = os.environ['DBOX'] + 'Data\\Geomag_B_V_2023.txt'

figdir = os.environ['DBOX'] + 'Apps\\Overleaf\\SSN from OSF\\figures\\'

# <codecell> Load and process Anna's topology data

#read in Anna's CR averages of OSF

filepath = datadir + 'FrostAM_SupplementaryData.txt'
column_headings = [
    'year', 'month', 'day', 'hour', 'minute', 'second', 'OSF', 'OSFmin',
    'OSFmax', 'Absolute OSF error (x10^{14} Wb)', 'Open Flux (%)', 'Inverted Flux (%)',
    'Newly-Opened Flux (%)', 'Disconnected Flux (%)', 'Available Data (%)']
osf_strahl =  pd.read_csv(filepath, skiprows=5, delimiter='\t',
                          names=column_headings)

# Create an empty list to store the datetime values
datetime_list = []

# Iterate over the rows and construct datetime objects
for index, row in osf_strahl.iterrows():
    year = int(row['year'])
    month = int(row['month'])
    day = int(row['day'])
    hour = int(row['hour'])
    minute = int(row['minute'])
    second = int(row['second'])

    dt = datetime.datetime(year, month, day, hour, minute, second)
    datetime_list.append(dt)

# Create a new column 'DateTime' from the datetime list
osf_strahl['datetime'] = datetime_list

#create a mjd column
mjd_start = datetime.datetime(1858, 11, 17)
osf_strahl['mjd'] = (osf_strahl['datetime'] - mjd_start).dt.days

#'remove bad data'
osf_strahl['OSF'] = np.where(osf_strahl['OSF'] < -999, np.nan, osf_strahl['OSF'])


# <codecell> load in the omni data and process

AU=149598000
Tsid = 25.38 * 24*60*60
br_window = '20H'

omni_1hour = pd.read_hdf(omnipath)
omni_brwindow_nans = omni_1hour.resample(br_window, on='datetime').mean() 
#omni_brwindow_nans['datetime'] = htime.mjd2datetime(omni_brwindow_nans['mjd'].to_numpy())
#omni_brwindow_nans.reset_index(drop=True, inplace=True)

#compute the constant to convert |Br| at 1 AU to total HMF
Fconst=(1e-3)*4*np.pi*AU*AU/(1e14)

#allow for the distance variation as well as the converstion constant
omni_brwindow_nans['OSF'] = np.abs(omni_brwindow_nans['Bx_gse']) * Fconst * omni_brwindow_nans['r_au'] * omni_brwindow_nans['r_au']

omni_1d = pd.DataFrame(index = omni_brwindow_nans.index)
omni_1d['OSF'] = omni_brwindow_nans['OSF']
omni_1d['mjd'] = omni_brwindow_nans['mjd']
omni_1d['V'] = omni_brwindow_nans['V']
omni_1d['Bmag'] = omni_brwindow_nans['Bmag']
omni_1d['r_au'] = omni_brwindow_nans['r_au']
omni_1d['datetime'] = omni_brwindow_nans.index

#make 27-day means for comparison with strahl estimate
omni_27d = omni_1d.resample('27D', on='datetime').mean() 
omni_27d['datetime'] = htime.mjd2datetime(omni_27d['mjd'].to_numpy())
omni_27d.reset_index(drop=True, inplace=True)

#interpolate these data onto the strahl OSF timestep
osf_strahl['OSF_Br'] = np.interp(osf_strahl['mjd'], omni_27d['mjd'], omni_27d['OSF'], left = np.nan, right=np.nan)

# <codecell> plot the time series

fig = plt.figure()
gs = gridspec.GridSpec(2, 3)


ax = fig.add_subplot(gs[0, 0:2])
ax.plot(omni_27d['datetime'], omni_27d['OSF'], 'k', label = r'OSF from $<|B_R|>_{20}$')
ax.plot(osf_strahl['datetime'], osf_strahl['OSF'], 'r', label = r'OSF from strahl')
ax.set_xlabel(r'Year', fontsize = 14)
ax.set_ylabel(r'OSF [x$10^14$ Wb]', fontsize = 14)
ax.legend(fontsize = 14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

ax = fig.add_subplot(gs[0, 2])

x = osf_strahl['OSF_Br'].to_numpy()
y = osf_strahl['OSF'].to_numpy()

ax.plot(osf_strahl['OSF_Br'], osf_strahl['OSF'],'ko')
xrange = np.array([np.nanmin(x), np.nanmax(x)])
fit_slope, fit_intercept, conf_slope, conf_intercept =  mplot.odr_lin_regress_bootstrap(x, y)
ax.fill_between(xrange, conf_slope[0] * xrange + conf_intercept[0], 
                  conf_slope[1] * xrange + conf_intercept[1], 
                  color='silver', label='Confidence Interval')
ax.plot(x, fit_slope * x + fit_intercept, 'b')
#p, regressparams = mplot.lin_regress(x,y, plotnow = True, ax = ax)

ax.set_xlabel(r'OSF from $<|B_R|>_{20}$ [x$10^{14}$ Wb]', fontsize = 14)
ax.set_ylabel(r'OSF from strahl [x$10^{14}$ Wb]', fontsize = 14)
ax.plot([osf_strahl['OSF_Br'].min(),osf_strahl['OSF_Br'].max() ],  [osf_strahl['OSF_Br'].min(), osf_strahl['OSF_Br'].max() ],'k--')
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


#maek 1yr means for comparison with GEO estimate
omni_1y = omni_1d.resample('1Y', on='datetime').mean() 
omni_1y['datetime'] = htime.mjd2datetime(omni_1y['mjd'].to_numpy())
omni_1y.reset_index(drop=True, inplace=True)

strahl_1y = osf_strahl.resample('1Y', on='datetime').mean() 
strahl_1y['datetime'] = htime.mjd2datetime(strahl_1y['mjd'].to_numpy())
strahl_1y.reset_index(drop=True, inplace=True)

#remove first and least rows, as years are incomplete
strahl_1y = strahl_1y.iloc[1:-1]

ax = fig.add_subplot(gs[1, 0:2])
ax.plot(omni_1y['datetime'], omni_1y['OSF'], 'k', label = r'OSF from $<|B_R|>_{20}$')
ax.plot(strahl_1y['datetime'], strahl_1y['OSF'], 'r', label = r'OSF from strahl')
ax.set_xlabel(r'Year', fontsize = 14)
ax.set_ylabel(r'OSF [x$10^14$ Wb]', fontsize = 14)
ax.legend(fontsize = 14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

ax = fig.add_subplot(gs[1, 2])

x = strahl_1y['OSF_Br'].to_numpy()
y = strahl_1y['OSF'].to_numpy()
ax.plot(osf_strahl['OSF_Br'], osf_strahl['OSF'],'ko')
xrange = np.array([np.nanmin(x), np.nanmax(x)])
fit_slope, fit_intercept, conf_slope, conf_intercept =  mplot.odr_lin_regress_bootstrap(x, y)
ax.fill_between(xrange, conf_slope[0] * xrange + conf_intercept[0], 
                  conf_slope[1] * xrange + conf_intercept[1], 
                  color='silver', label='Confidence Interval')
ax.plot(x, fit_slope * x + fit_intercept, 'b')
#p, regressparams = mplot.lin_regress(x,y, plotnow = True, ax = ax)
ax.set_xlabel(r'OSF from $<|B_R|>_{20}$ [x$10^{14}$ Wb]', fontsize = 14)
ax.set_ylabel(r'OSF from strahl [x$10^{14}$ Wb]', fontsize = 14)
ax.plot([osf_strahl['OSF_Br'].min(),osf_strahl['OSF_Br'].max() ],  [osf_strahl['OSF_Br'].min(), osf_strahl['OSF_Br'].max() ],'k--')
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

rl =  mplot.lin_correl(x, y)
mae = np.nanmean(abs(x - y))

print('r_L = ' + str(rl))
print('MAE = ' + str(mae))

# <codecell> OSF from B and V, using OMNI



#add fractional year
doy, yr = htime.mjd2doyyr(omni_1y['mjd'].to_numpy())
omni_1y['year'] = yr + 0.5

#compute Ideal Parker angle
vrot = 2 * np.pi * omni_1y['r_au'] * AU / Tsid
phi = np.arctan(vrot/omni_1y['V'])
Br = omni_1y['Bmag'] * np.cos(phi)
omni_1y['OSF_BV'] = Br * Fconst *omni_1y['r_au'] *omni_1y['r_au']


# find the best fit
# Specify the degree of the polynomial
degree = 1

# Fit the polynomial to the data
y = omni_1y['OSF'].to_numpy()
x = omni_1y['OSF_BV'].to_numpy()
idx = np.isfinite(x) & np.isfinite(y)

# coefficients = np.polyfit(x[idx], y[idx], degree)
# print(coefficients)
# x_plot = np.linspace(omni_1y['OSF_BV'].min(), omni_1y['OSF_BV'].max(), 100)
# omni_1y['OSF_BV_corr'] = np.polyval(coefficients, omni_1y['OSF_BV'])



fig = plt.figure(figsize = (10,8))
gs = gridspec.GridSpec(2, 3)

#timeseries
ax = fig.add_subplot(gs[0, 0:2])
ax.plot(omni_1y['datetime'],omni_1y['OSF_BV'],'b', label = r'$OSF_{PS}$: OMNI B and V, ideal spiral')
ax.plot(omni_1y['datetime'],omni_1y['OSF'],'k', label = r'$OSF_{BR}$: OMNI $|B_R|$')
ax.plot(strahl_1y['datetime'], strahl_1y['OSF'],'r', label = r'$OSF_S$: $B_R$ + Strahl')
ax.set_xlabel(r'Year', fontsize = 12)
ax.set_ylabel(r'OSF [x$10^14$ Wb]', fontsize = 12)
ax.legend(fontsize = 12, loc = 'lower left')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.text(0.02, 0.95, '(a)', transform=plt.gca().transAxes, fontsize=12)
ax.set_ylim((0,19.5))

#scatterplot
ax = fig.add_subplot(gs[0, 2])
x = omni_1y['OSF_BV'].to_numpy()
y = omni_1y['OSF'].to_numpy()
xrange = np.array([np.nanmin(x), np.nanmax(x)])
bvfit_slope, bvfit_intercept, bvconf_slope, bvconf_intercept =  mplot.lin_regress_bootstrap(x, y)
ax.fill_between(xrange, bvconf_slope[0] * xrange + bvconf_intercept[0], 
                  bvconf_slope[1] * xrange + bvconf_intercept[1], 
                  color='lightskyblue', label='Confidence Interval')
ax.plot(x, y,'ko')
p = [bvfit_slope, bvfit_intercept]
ax.plot(x, bvfit_slope * x + bvfit_intercept, 'b')
# omni_1y['OSF_BV_corr'] = omni_1y['OSF_BV'] *bvfit_slope + bvfit_intercept
#p, regressparams = mplot.lin_regress(x,y, plotnow = True, ax = ax,
#                                     color = 'None', alpha =1, linecolor = 'b')
ax.set_xlim((3,20)); ax.set_ylim((3,20)); ax.plot([3,20 ],  [3,20 ],'k--')
ax.set_xlabel(r'$OSF_{PS}$ [x$10^{14}$ Wb]', fontsize = 12)
ax.set_ylabel(r'$OSF_{BR}$ [x$10^{14}$ Wb]', fontsize = 12)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.text(0.02, 0.95, '(b)', transform=plt.gca().transAxes, fontsize=12)
ax.text(0.02, 0.85, r'$OSF_{BR}$ = ' + "{:.2f}".format(p[0]) + r' $OSF_{PS}$ ' +
         "{:.2f}".format(p[1]), 
        transform=plt.gca().transAxes, fontsize=12, color = 'b')

#generate the corrected OSF estimate from BV
omni_1y['OSF_BV_corr'] = omni_1y['OSF_BV'] *p[0] + p[1]

#time series
ax = fig.add_subplot(gs[1, 0:2])

# confid_int, prob_int = mplot.lin_regress_confid_int(omni_1y['OSF_BV'], regressparams)
# ax.fill_between(omni_1y['datetime'], omni_1y['OSF_BV_corr'] - prob_int, 
#                   omni_1y['OSF_BV_corr'] + prob_int, 
#                   color='lightskyblue')
ax.fill_between(omni_1y['datetime'], bvconf_slope[0] * omni_1y['OSF_BV'] + bvconf_intercept[0], 
                  bvconf_slope[1] * omni_1y['OSF_BV'] + bvconf_intercept[1], 
                  color='lightskyblue', alpha = 1)
ax.plot(omni_1y['datetime'], omni_1y['OSF_BV_corr'] ,'b', 
        label = r'$OSF_{PS}*$: OMNI B and V, corrected ')
ax.plot(omni_1y['datetime'], omni_1y['OSF'],'k', label = r'$OSF_{BR}$: OMNI $|B_R|$ ')
ax.plot(strahl_1y['datetime'], strahl_1y['OSF'],'r', label = r'$OSF_{S}$: $B_R$ + Strahl')
ax.set_xlabel(r'Year', fontsize = 12)
ax.set_ylabel(r'OSF [x$10^14$ Wb]', fontsize = 12)
ax.legend(fontsize = 12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.text(0.02, 0.95, '(c)', transform=plt.gca().transAxes, fontsize=12)

#scatterplot
ax = fig.add_subplot(gs[1, 2])
x = omni_1y['OSF_BV_corr'].to_numpy()
y = omni_1y['OSF'].to_numpy()
xrange = np.array([np.nanmin(x), np.nanmax(x)])
fit_slope, fit_intercept, conf_slope, conf_intercept =  mplot.lin_regress_bootstrap(x, y)
ax.fill_between(xrange, conf_slope[0] * xrange + conf_intercept[0], 
                  conf_slope[1] * xrange + conf_intercept[1], 
                  color='lightskyblue', label='Confidence Interval')
ax.plot(x, y,'ko')
ax.plot(x, fit_slope * x + fit_intercept, 'b')
#p_final, regressparams = mplot.lin_regress(x,y, plotnow = True, ax = ax,
#                                           color = 'lightskyblue', alpha =1, linecolor = 'b')
ax.set_xlabel(r'$OSF_{PS}*$ [x$10^{14}$ Wb]', fontsize = 12)
ax.set_ylabel(r'$OSF_{BR}$ [x$10^{14}$ Wb]', fontsize = 12)
ax.set_xlim((3,13)); ax.set_ylim((3,13)); ax.plot([3,13 ],  [3,13 ],'k--')
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.text(0.02, 0.95, '(d)', transform=plt.gca().transAxes, fontsize=12)

plt.tight_layout()

fig.savefig(figdir + 'OSF_OMNI.pdf')


rl =  mplot.lin_correl(x, y)
mae = np.nanmean(abs(x - y))

print('r_L = ' + str(rl))
print('MAE = ' + str(mae))


# <codecell> read and process the GEO data

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

#compute the OSF from ideal Parker angle
vrot_geo = 2 * np.pi * (osfG['mjd']*0 +1) * AU / Tsid
phi_geo = np.arctan(vrot_geo/osfG['V'])
Br_geo = osfG['B'] * np.cos(phi_geo)
osfG['OSF_BV'] = Br_geo * Fconst 

#apply the correction
osfG['OSF_BV_corr'] = np.polyval(p, osfG['OSF_BV'])


#compute upper and lower limits in the same way.
phi_geo = np.arctan(vrot_geo/osfG['Vmin'])
Br_geo = osfG['Bmin'] * np.cos(phi_geo)
osfG['OSF_BVmin'] = Br_geo * Fconst 
osfG['OSF_BV_corr_min'] = np.polyval(p, osfG['OSF_BVmin'])

phi_geo = np.arctan(vrot_geo/osfG['Vmax'])
Br_geo = osfG['Bmax'] * np.cos(phi_geo)
osfG['OSF_BVmax'] = Br_geo * Fconst 
osfG['OSF_BV_corr_max'] = np.polyval(p, osfG['OSF_BVmax'])


#interp GEO onto OMNI timestep
omni_1y['OSF_GEO_BV'] = np.interp(omni_1y['year'], osfG['year'], osfG['OSF_BV'], left = np.nan, right=np.nan)
omni_1y['OSF_GEO_BV_corr'] = np.interp(omni_1y['year'], osfG['year'], osfG['OSF_BV_corr'], left = np.nan, right=np.nan)
omni_1y['OSF_GEO_BV_corr_min'] = np.interp(omni_1y['year'], osfG['year'], osfG['OSF_BV_corr_min'], left = np.nan, right=np.nan)
omni_1y['OSF_GEO_BV_corr_max'] = np.interp(omni_1y['year'], osfG['year'], osfG['OSF_BV_corr_max'], left = np.nan, right=np.nan)
omni_1y['B_GEO'] = np.interp(omni_1y['year'], osfG['year'], osfG['B'], left = np.nan, right=np.nan)
omni_1y['B_GEO_min'] = np.interp(omni_1y['year'], osfG['year'], osfG['Bmin'], left = np.nan, right=np.nan)
omni_1y['B_GEO_max'] = np.interp(omni_1y['year'], osfG['year'], osfG['Bmax'], left = np.nan, right=np.nan)
omni_1y['V_GEO'] = np.interp(omni_1y['year'], osfG['year'], osfG['V'], left = np.nan, right=np.nan)
omni_1y['V_GEO_min'] = np.interp(omni_1y['year'], osfG['year'], osfG['Vmin'], left = np.nan, right=np.nan)
omni_1y['V_GEO_max'] = np.interp(omni_1y['year'], osfG['year'], osfG['Vmax'], left = np.nan, right=np.nan)

# <codecell> plot OMNI and GEO B and V

#compare B and V

fig = plt.figure(figsize = (10,10))
gs = gridspec.GridSpec(3, 3)

ax = fig.add_subplot(gs[0, 0:2])
ax.fill_between(osfG['year'], osfG['Bmin'],osfG['Bmax'], facecolor = 'pink')
ax.plot(osfG['year'], osfG['B'],'r', label = r'GEO')
ax.plot(omni_1y['year'], omni_1y['Bmag'],'k', label = r'OMNI')
#ax.set_xlabel(r'Year', fontsize=12)
ax.set_ylabel(r'B [nT]', fontsize=12)
ax.legend(ncol = 2, fontsize=12, loc = 'upper right')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.text(0.02, 0.93, '(a)', transform=plt.gca().transAxes, fontsize=12)

ax = fig.add_subplot(gs[0, 2])
ax.errorbar( omni_1y['Bmag'],omni_1y['B_GEO'], 
           yerr = [omni_1y['B_GEO'] - omni_1y['B_GEO_min'], omni_1y['B_GEO_max'] - omni_1y['B_GEO']], fmt = 'ko')
ax.set_ylabel(r'B GEO [nT]', fontsize=12)
ax.set_xlabel(r'B OMNI [nT]', fontsize=12)
#ax.plot(x_plot,  np.polyval(coefficients, x_plot),'b')
ax.plot([omni_1y['Bmag'].min(), omni_1y['Bmag'].max() ], 
        [omni_1y['Bmag'].min(), omni_1y['Bmag'].max() ],'k--')

ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.text(0.02, 0.93, '(b)', transform=plt.gca().transAxes, fontsize=12)



ax = fig.add_subplot(gs[1, 0:2])
ax.fill_between(osfG['year'], osfG['Vmin'],osfG['Vmax'], facecolor = 'pink')
ax.plot(osfG['year'], osfG['V'],'r', label = r'GEO')
ax.plot(omni_1y['year'], omni_1y['V'],'k', label = r'OMNI')
#ax.set_xlabel(r'Year', fontsize=12)
ax.set_ylabel(r'V [km/s]', fontsize=12)
ax.legend(ncol = 2, fontsize=12, loc = 'upper right')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.text(0.02, 0.93, '(c)', transform=plt.gca().transAxes, fontsize=12)

ax = fig.add_subplot(gs[1, 2])
ax.errorbar( omni_1y['V'],omni_1y['V_GEO'], 
           yerr = [omni_1y['V_GEO'] - omni_1y['V_GEO_min'], omni_1y['V_GEO_max'] - omni_1y['V_GEO']], fmt = 'ko')
ax.set_ylabel(r'V GEO [km/s]', fontsize=12)
ax.set_xlabel(r'V OMNI [km/s]', fontsize=12)
#ax.plot(x_plot,  np.polyval(coefficients, x_plot),'b')
ax.plot([omni_1y['V'].min(), omni_1y['V'].max() ], 
        [omni_1y['V'].min(), omni_1y['V'].max() ],'k--')

ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.text(0.02, 0.93, '(d)', transform=plt.gca().transAxes, fontsize=12)



ax = fig.add_subplot(gs[2, 0:2])
ax.fill_between(osfG['year'], osfG['OSF_BV_corr_min'],osfG['OSF_BV_corr_max'], facecolor = 'pink')
ax.plot(osfG['year'], osfG['OSF_BV_corr'],'r', label = r'GEO, $OSF_{PS}*$')
ax.plot(omni_1y['year'], omni_1y['OSF'],'k', label = r'OMNI $OSF_{BR}$')
ax.set_xlabel(r'Year', fontsize=12)
ax.set_ylabel(r'OSF [x$10^{14}$ Wb]', fontsize=12)
ax.legend(ncol=2, fontsize=12, loc = 'upper right')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.text(0.02, 0.93, '(e)', transform=plt.gca().transAxes, fontsize=12)

ax = fig.add_subplot(gs[2, 2])
ax.errorbar( omni_1y['OSF'],omni_1y['OSF_GEO_BV_corr'], 
           yerr = [omni_1y['OSF_GEO_BV_corr'] - omni_1y['OSF_GEO_BV_corr_min'], 
                   omni_1y['OSF_GEO_BV_corr_max'] - omni_1y['OSF_GEO_BV_corr']], fmt = 'ko')
ax.set_ylabel(r'GEO, $OSF_{PS}*$ [x$10^{14}$ Wb]', fontsize = 12)
ax.set_xlabel(r'OMNI, $OSF_{BR}$ [x$10^{14}$ Wb]', fontsize = 12)
ax.plot([omni_1y['OSF'].min(), omni_1y['OSF'].max() ],  [omni_1y['OSF'].min(), omni_1y['OSF'].max() ],'k--')
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.text(0.02, 0.93, '(f)', transform=plt.gca().transAxes, fontsize=12)

plt.tight_layout()

fig.savefig(figdir + 'OSF_GEO.pdf')

x = omni_1y['OSF'].to_numpy()
y = omni_1y['OSF_GEO_BV_corr'].to_numpy() 
valid_indices = ~np.isnan(x) & ~np.isnan(y)
rl = np.corrcoef(x[valid_indices], y[valid_indices])
print('r_L, B_GEO, BOMNI: ' + "{:.2f}".format(rl[0,1]))

# <codecell> plot OSF from GEO and OMNI


# fig = plt.figure()
# gs = gridspec.GridSpec(2, 3)

# ax = fig.add_subplot(gs[0, 0:2])
# ax.plot(omni_1y['year'], omni_1y['OSF'],'k', label = r'OSF from $|B_R|$')
# ax.plot(osfG['year'], osfG['OSF_BV'],'b', label = r'OSF from GEO B and V, ideal spiral')
# ax.set_xlabel(r'Year')
# ax.set_ylabel(r'OSF [x$10^14$ Wb]')
# ax.legend()
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

# ax = fig.add_subplot(gs[0, 2])
# ax.plot(omni_1y['OSF_GEO_BV'], omni_1y['OSF'],'ko')
# ax.set_xlabel(r'OSF from GEO B and V, ideal spiral')
# ax.set_ylabel(r'OSF from $|B_R|$')
# ax.plot(x_plot,  np.polyval(coefficients, x_plot),'b')
# ax.plot(x_plot,  x_plot,'k--')
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)


# ax = fig.add_subplot(gs[1, 0:2])
# ax.plot(omni_1y['year'], omni_1y['OSF'],'k', label = r'OSF from $|B_R|$')
# ax.plot(osfG['year'], osfG['OSF_BV_corr'],'b', label = r'OSF from GEO B and V, corrected')
# ax.set_xlabel(r'Year')
# ax.set_ylabel(r'OSF [x$10^14$ Wb]')
# ax.legend()
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

# ax = fig.add_subplot(gs[1, 2])
# ax.plot(omni_1y['OSF_GEO_BV_corr'], omni_1y['OSF'],'ko')
# ax.set_xlabel(r'OSF from GEO B and V, corrected')
# ax.set_ylabel(r'OSF from $|B_R|$')
# ax.plot([omni_1y['OSF'].min(), omni_1y['OSF'].max() ],  [omni_1y['OSF'].min(), omni_1y['OSF'].max() ],'k--')
# ax.yaxis.tick_right()
# ax.yaxis.set_label_position("right")
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)




# <codecell> GEO summary plot
fig = plt.figure()
gs = gridspec.GridSpec(1, 3)

ax = fig.add_subplot(gs[0, 0:2])
ax.plot(osfG['year'], osfG['OSF_BV_corr'],'r', label = r'OSF from GEO')
ax.plot(omni_1y['year'], omni_1y['OSF'],'k', label = r'OSF from OMNI')
ax.set_xlabel(r'Year', fontsize=12)
ax.set_ylabel(r'OSF [x$10^14$ Wb]', fontsize=12)
ax.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

ax = fig.add_subplot(gs[0, 2])
ax.plot(omni_1y['OSF_GEO_BV_corr'], omni_1y['OSF'],'ko')
ax.set_xlabel(r'OSF from GEO', fontsize=12)
ax.set_ylabel(r'OSF from OMNI', fontsize=12)
ax.plot([omni_1y['OSF'].min(), omni_1y['OSF'].max() ],  [omni_1y['OSF'].min(), omni_1y['OSF'].max() ],'k--')
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# <codecell> plot neutron counts and OSF

oulu = sunspots.load_oulu_nm()
#average to 27d and 1y
oulu_1y = oulu.resample('1Y', on='datetime').mean() 
oulu_1y['datetime'] = htime.mjd2datetime(oulu_1y['mjd'].to_numpy())
oulu_1y.reset_index(drop=True, inplace=True)

oulu_27d = oulu.resample('27D', on='datetime').mean() 
oulu_27d['datetime'] = htime.mjd2datetime(oulu_27d['mjd'].to_numpy())
oulu_27d.reset_index(drop=True, inplace=True)


plt.figure()
ax = plt.subplot(2,1,1)
ax.plot(omni_27d['datetime'], omni_27d['OSF'],'r', label = r'OSF, OMNI, 27-day')
ax.plot(omni_1y['datetime'], omni_1y['OSF'],'k', label = r'OSF, OMNI, 1-yr')
ax.set_ylabel(r'OSF [x$10^{14}$ Wb]', fontsize = 12)
xx = ax.get_xlim()
ax.set_xlim(xx)
ax.legend(fontsize = 12)


ax = plt.subplot(2,1,2)
ax.plot(oulu_27d['datetime'],oulu_27d['counts'], 'r', label = 'Oulu, 27-day')
ax.plot(oulu_1y['datetime'],oulu_1y['counts'], 'k', label = 'Oulu, 1-yr')

#plt.plot(osf_yrs, osf_ssn, 'k') 
ax.set_ylabel(r'Neutron counts/min', fontsize = 12)
ax.set_xlim(xx)
ax.legend(fontsize = 12)

ax.set_xlabel('Year', fontsize = 12)