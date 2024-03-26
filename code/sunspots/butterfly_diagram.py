# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 14:57:00 2023

a script to download and process the sunspot area data from Dave Hathaway 
and Lisa Upton

@author: vy902033
"""



import datetime as datetime
import os as os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import urllib.request


import conversions.helio_time as htime 
import sunspots.sunspots as sunspots
import mplot


data_dir = os.environ['DBOX'] + 'Data\\'
download_now = True
n_spe = 50

    
    
#read in the data
def read_bfly_data(filename = None, data_dir = '', download_now = False):
    
    if download_now:
        urllib.request.urlretrieve('http://solarcyclescience.com/AR_Database/bflydata.txt',
                                   data_dir + 'bflydata.txt')
    
    if filename is None:
        filename = os.path.join(os.environ['DBOX'],'Data','bflydata.txt')
    
    # Read the file
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    data = []
    CRs = []
    
    i = 0
    while i < len(lines):
        if lines[i].find(',') == -1:
            CRs.append(int(lines[i]))
            i += 1
        else:
            #read in the block of data
            list_of_vals = []
            insideblock = True
            while insideblock:
                values = lines[i].split(",")
                vals = np.empty(len(values)-1)
                #convert to ints
                for n in range(0, len(values) -1 ):  #ignore line break at end 
                    vals[n] = int(values[n])
                list_of_vals.extend(vals)
                
                #move to the next line
                i += 1
                
                #check it's not end of the file
                if i >= len(lines):
                    break
                else:
                    if lines[i].find(',') == -1:
                        insideblock = False
            data.append(list_of_vals)

    
    # Convert data to NumPy array
    data = np.array(data).T
    CRs = np.array(CRs)
    
    
    #work out the latitudes
    nlats = len(data[:,0])
    d_sinlat = 2/nlats
    sinlats = np.arange(-1 + d_sinlat/2, 1 - d_sinlat/2 + 0.0001, d_sinlat)

    
    return CRs, sinlats, data

# read it in
CRs, sinlats, data = read_bfly_data(download_now = True)

# compute total sunspot area and mean abs lat
mean_abs_lat = []
total_ss_area = []
for t in range(0, len(CRs)):
    total_ss_area.append(np.sum(data[:,t]))
    mean_abs_lat.append(np.sum(data[:,t]*abs(np.arcsin(sinlats)*180/np.pi)) 
                        / np.sum(data[:,t]) )

total_ss_area = np.array(total_ss_area)
mean_abs_lat = np.array(mean_abs_lat)

#convert to fraction of year    
mjd = htime.crnum2mjd(CRs)
dt = htime.mjd2datetime(mjd)
def year_fraction(date):
    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length

yr = []
for i in range(0,len(dt)):
    t = dt[i]
    yr.append(year_fraction(t))
yr = np.array(yr)

# <codecell> #plot time series

# #get the solar minimum times
# def LoadSolarMinTimes(filepath = 'null'):
#     if filepath == 'null':
#         filepath = os.environ['DBOX'] + 'Data\\SolarMinTimes.txt'
#     solarmintimes_df = pd.read_csv(filepath,
#                          delim_whitespace=True,
#                          names=['fracyear','cyclenum'])
#     doy = (solarmintimes_df['fracyear'] - np.floor(solarmintimes_df['fracyear']))*364 + 1
#     doy = doy.to_numpy()
#     yr = np.floor(solarmintimes_df['fracyear']).to_numpy()
#     yr=yr.astype(int)
#     solarmintimes_df['mjd'] = htime.doyyr2mjd(doy,yr)
    
#     #solarmintimes_df['datetime'] = pd.to_datetime(solarmintimes_df['fracyear'], format='%Y', errors='coerce')
#     solarmintimes_df['datetime'] = htime.mjd2datetime(solarmintimes_df['mjd'].to_numpy())
#     return solarmintimes_df

#read in the solar minimum times
solarmintimes_df = sunspots.LoadSolarMinTimes()

fig = plt.figure()
ax = plt.subplot(311)
im = ax.pcolor(yr, sinlats, data, vmin = 0, vmax = 50, linewidth=0, rasterized=True)
ax.set_xlim((yr[0], yr[-1]))
ax.set_ylabel(r'Sin($\theta$)' , fontsize = 14)
ax.set_yticks([-1, -0.5, 0, 0.5, 1])

#offset colourbar
axins = inset_axes(ax,
                    width="100%",  # width = 50% of parent_bbox width
                    height="100%",  # height : 5%
                    loc='upper right',
                    bbox_to_anchor=(1.03, 0.0, 0.02, 1),
                    bbox_transform=ax.transAxes,
                    borderpad=0,)
cb = fig.colorbar(im, cax = axins, orientation = 'vertical',  pad = -0.1)
cb.ax.tick_params(labelsize=10)
axins.text(0.82,1.07,'Sunspot area [mH]' , 
        fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')


ax = plt.subplot(312)
ax.plot(yr, 100*total_ss_area/1000000, 'k')
ax.set_xlim((yr[0], yr[-1]))
yy = ax.get_ylim()
ax.set_ylim(yy)
#add the solar min times
for n in range(16,len(solarmintimes_df)):
    fyr = year_fraction(solarmintimes_df.loc[n,'datetime'])
    ax.plot([fyr, fyr], yy, 'r--')   
ax.set_ylabel('Sunspot area [% H]', fontsize = 14)

ax = plt.subplot(313)
ax.plot(yr, mean_abs_lat, 'k')
ax.set_xlim((yr[0], yr[-1]))
yy = ax.get_ylim()
ax.set_ylim(yy)
#add the solar min times
for n in range(16,len(solarmintimes_df)):
    fyr = year_fraction(solarmintimes_df.loc[n,'datetime'])
    ax.plot([fyr, fyr], yy, 'r--')
ax.set_ylabel(r'$<|\theta|>$ [deg]', fontsize = 14)    


# <codecell> Superpose the sunspot number
nt = len(yr)
#compute the solar cycle phase for each time stepfor n in range(0,nt): 
phases = np.empty(nt)
for i in range(0,len(solarmintimes_df)-1):
    cyclestart = solarmintimes_df['mjd'][i]
    cycleend = solarmintimes_df['mjd'][i+1]
    
    cyclelength = cycleend -cyclestart
    if (mjd[i] >= cyclestart) & (mjd[i] < cycleend):
        phases[i] = 2*np.pi*(mjd[i]- cyclestart)/cyclelength

#compute the solar cycle phase relative to each solar cycle start and length
phases_scs = np.empty((nt,len(solarmintimes_df)-1))
for i in range(0,len(solarmintimes_df)-1):
    cyclestart = solarmintimes_df['mjd'][i]
    cycleend = solarmintimes_df['mjd'][i+1]
    
    cyclelength = cycleend -cyclestart
    phases_scs[:,i] = 2*np.pi*(mjd - cyclestart) / cyclelength   
    
    
#SPE in years from cycle start
SPE_yrs = np.arange(-2,14,11/n_spe)   
SPE_ssn_yrs = np.empty( (len(SPE_yrs), len(solarmintimes_df)-2) ) *np.nan
for n in range(16, len(solarmintimes_df)-2):
    time_rel_start = yr - year_fraction(solarmintimes_df.loc[n,'datetime'])
    SPE_ssn_yrs[:,n] = np.interp(SPE_yrs, time_rel_start,
                                 total_ss_area, 
                                 left = np.nan, right = np.nan)

#SPE in cycle phase
SPE_phase = np.arange(-2,2*np.pi +2,2*np.pi/n_spe)   
SPE_ssn_phase = np.empty( (len(SPE_phase), len(solarmintimes_df)-2) ) *np.nan
for n in range(16, len(solarmintimes_df)-2):
    SPE_ssn_phase[:,n] = np.interp(SPE_phase, phases_scs[:,n],
                                 total_ss_area, 
                                 left = np.nan, right = np.nan)

# <codecell> Superpose the latitude variations
 
#SPE in years from cycle start  
SPE_lat_yrs = np.empty( (len(SPE_yrs), len(solarmintimes_df)-2) ) *np.nan
for n in range(16, len(solarmintimes_df)-2):
    time_rel_start = yr - year_fraction(solarmintimes_df.loc[n,'datetime'])
    SPE_lat_yrs[:,n] = np.interp(SPE_yrs, time_rel_start,
                                 mean_abs_lat, 
                                 left = np.nan, right = np.nan)

#SPE in cycle phase  
SPE_lat_phase = np.empty( (len(SPE_phase), len(solarmintimes_df)-2) ) *np.nan
for n in range(16, len(solarmintimes_df)-2):
    SPE_lat_phase[:,n] = np.interp(SPE_phase, phases_scs[:,n],
                                 mean_abs_lat, 
                                 left = np.nan, right = np.nan)
    
    
# <codecell> plot the SPE


confid_intervals = [5,10,33]


#sunspot latitude
yy= (0,35)

fig = plt.figure()
ax = plt.subplot(221)
for n in range(16,len(solarmintimes_df)):
    time_rel_start = yr - year_fraction(solarmintimes_df.loc[n,'datetime'])
    ax.plot(time_rel_start, mean_abs_lat)
ax.set_xlim((-2, 14))
ax.set_ylim(yy)
xx = ax.get_xlim()
ax.get_xaxis().set_ticklabels([])
ax.plot([0,0],yy,'k--')


ax = plt.subplot(223)
mplot.plotconfidbands(SPE_yrs, SPE_lat_yrs.T, confid_intervals)
ax.set_xlim(xx)
#add the current cycle
time_rel_start = yr - year_fraction(solarmintimes_df.loc[len(solarmintimes_df)-2,'datetime'])
ax.plot(time_rel_start, mean_abs_lat,'k')
ax.set_xlabel('Years from cycle start', fontsize = 14)
ax.set_ylim(yy)
ax.plot([0,0],yy,'k--')

fig.text(0.04, 0.5, 'Mean |latitude| [deg]', va='center', rotation='vertical', fontsize = 14)


ax = plt.subplot(222)
for n in range(16,len(solarmintimes_df)-1):
    ax.plot(phases_scs[:,n], mean_abs_lat)

ax.set_xlim((-1, 2*np.pi +1))
xx = ax.get_xlim()
ax.get_xaxis().set_ticklabels([])
ax.set_ylim(yy)
ax.get_yaxis().set_ticklabels([])
ax.plot([0,0],yy,'k--')
ax.plot([2*np.pi,2*np.pi],yy,'k--')

ax = plt.subplot(224)
mplot.plotconfidbands(SPE_phase, SPE_lat_phase.T, confid_intervals)
ax.set_xlim(xx)
#add the current cycle
ax.plot(phases_scs[:,len(solarmintimes_df)-2], mean_abs_lat,'k')
ax.set_xlabel('Solar cycle phase [rad]', fontsize = 14)
ax.set_ylim(yy)
ax.get_yaxis().set_ticklabels([])
ax.plot([0,0],yy,'k--')
ax.plot([2*np.pi,2*np.pi],yy,'k--')


#fig.tight_layout()






#sunspot number
yy= (0,0.4)

fig = plt.figure()
ax = plt.subplot(221)
for n in range(16,len(solarmintimes_df)):
    time_rel_start = yr - year_fraction(solarmintimes_df.loc[n,'datetime'])
    ax.plot(time_rel_start, 0.0001*total_ss_area)
ax.set_xlim((-2, 14))
ax.set_ylim(yy)
xx = ax.get_xlim()
ax.get_xaxis().set_ticklabels([])
ax.plot([0,0],yy,'k--')



ax = plt.subplot(223)
mplot.plotconfidbands(SPE_yrs, 0.0001*(SPE_ssn_yrs.T), confid_intervals)
ax.set_xlim(xx)
#add the current cycle
time_rel_start = yr - year_fraction(solarmintimes_df.loc[len(solarmintimes_df)-2,'datetime'])
ax.plot(time_rel_start, 0.0001*total_ss_area,'k')
ax.set_xlabel('Years from cycle start', fontsize = 14)
ax.set_ylim(yy)
ax.plot([0,0],yy,'k--')

fig.text(0.04, 0.5, 'Sunspot area [% H]', va='center', rotation='vertical', fontsize = 14)

ax = plt.subplot(222)
for n in range(16,len(solarmintimes_df)-1):
    ax.plot(phases_scs[:,n], 0.0001*total_ss_area)

ax.set_xlim((-1, 2*np.pi +1))
xx = ax.get_xlim()
ax.get_xaxis().set_ticklabels([])
ax.set_ylim(yy)
ax.get_yaxis().set_ticklabels([])
ax.plot([0,0],yy,'k--')
ax.plot([2*np.pi,2*np.pi],yy,'k--')


ax = plt.subplot(224)
mplot.plotconfidbands(SPE_phase, 0.0001*(SPE_ssn_phase.T), confid_intervals)
ax.set_xlim(xx)
#add the current cycle
ax.plot(phases_scs[:,len(solarmintimes_df)-2], 0.0001*total_ss_area,'k')
ax.set_xlabel('Solar cycle phase [rad]', fontsize = 14)
ax.set_ylim(yy)
ax.get_yaxis().set_ticklabels([])
ax.plot([0,0],yy,'k--')
ax.plot([2*np.pi,2*np.pi],yy,'k--')

#fig.tight_layout()


###############################################################################
# <codecell> Load the Mandip dataset
###############################################################################


download_now = True
data_dir = os.environ['DBOX'] + 'Data\\'
filename = None
if download_now:
    urllib.request.urlretrieve('http://www2.mps.mpg.de/projects/sun-climate/data/indivi_group_area_1874_2023_Mandal.txt',
                               data_dir + 'indivi_group_area_1874_2023_Mandal.txt')

if filename is None:
    filename = os.environ['DBOX'] + 'Data\\indivi_group_area_1874_2023_Mandal.txt'


# Read the ASCII file into a Pandas DataFrame
df = pd.read_csv(filename, skiprows=38, sep='\s+', 
                 names=["year", "month", "day",  "time",
                        "area_proj", "area_corr",  "lat", "lon",
                        "dist_centre", "obs_flag"])   

#create a useful time stamp
df['mjd'] = htime.date2mjd(df['year'], df['month'], df['day']) 
df['datetime'] = htime.mjd2datetime(df['mjd'].to_numpy() )
df['fracyear'] = htime.mjd2fracyear(df['mjd'].to_numpy() )

#remove bad data
mask = df == 999999
df[mask] = np.nan

#now bin this data in time and lat
CRstart = np.ceil(htime.mjd2crnum(df.loc[0,'mjd']))
CRstop = np.ceil(htime.mjd2crnum(df.loc[len(df)-1, 'mjd']))
nt = int(CRstop - CRstart)

sinlat_edges = np.arange(-1, 1.0001, 0.04) 
sinlat_centres = (sinlat_edges[1:]+sinlat_edges[0:-1])/2
nlat = len(sinlat_centres)

ss_area = np.zeros((nlat,nt))
ss_num = np.zeros((nlat,nt))
total_area = np.zeros((nt))
total_num = np.zeros((nt))
mean_lat = np.zeros((nt))
yr = np.zeros((nt))
for t in range(0,nt):
    #find all the spots in this time period
    smjd = htime.crnum2mjd(CRstart + t)
    fmjd = htime.crnum2mjd(CRstart + t + 1)
    
    #total area for this whole time period
    mask = (df['mjd'] >= smjd) & (df['mjd'] < fmjd) 
    time_slice = df[mask].copy()
    
    #total sunspot area
    total_area[t] = np.nansum(time_slice['area_corr'])
    total_num[t] = len(time_slice['area_corr'])
    
    #record teh fractional year
    yr[t] = htime.mjd2fracyear(np.nanmean([fmjd,smjd]))
    
    #computethe area-weighted mean abs lat
    mean_lat[t] = np.nansum(time_slice['area_corr'] * abs(time_slice['lat'])) / total_area[t]
        
    #now loop though the lats and add area
    for i in range(0,nlat):
        minlat = (180 / np.pi) * np.arcsin(sinlat_edges[i])
        maxlat = (180 / np.pi) * np.arcsin(sinlat_edges[i+1]-0.00000000000001)
        
        
        mask = (time_slice['lat'] >= minlat) & (time_slice['lat'] < maxlat)
        
        ss_area[i,t] = np.nansum(time_slice.loc[mask, 'area_corr'])
        ss_num[i,t] = len(time_slice.loc[mask, 'area_corr'])
        
        

# <codecell> Mandal  plot

fig = plt.figure(figsize = (10,8))
ax = plt.subplot(311)
im = ax.pcolor(yr, sinlat_centres, ss_area, vmin = 0, vmax = 1000, linewidth=0, rasterized=True)
ax.set_xlim((yr[0], yr[-1]))
ax.set_ylabel(r'Sin($\theta$)' , fontsize = 14)
ax.set_yticks([-1, -0.5, 0, 0.5, 1])

#offset colourbar
axins = inset_axes(ax,
                    width="100%",  # width = 50% of parent_bbox width
                    height="100%",  # height : 5%
                    loc='upper right',
                    bbox_to_anchor=(1.03, 0.0, 0.02, 1),
                    bbox_transform=ax.transAxes,
                    borderpad=0,)
cb = fig.colorbar(im, cax = axins, orientation = 'vertical',  pad = -0.1)
cb.ax.tick_params(labelsize=10)
axins.text(0.88,1.09,r'Sunspot area [$\mu$H]' , 
        fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')


ax = plt.subplot(312)
ax.plot(yr, 100*total_area/1000000, 'k')
ax.set_xlim((yr[0], yr[-1]))
yy = ax.get_ylim()
ax.set_ylim(yy)
#add the solar min times
for n in range(16,len(solarmintimes_df)):
    fyr = year_fraction(solarmintimes_df.loc[n,'datetime'])
    ax.plot([fyr, fyr], yy, 'r--')   
ax.set_ylabel('Total sunspot\n area [% H]', fontsize = 14)

ax = plt.subplot(313)
ax.plot(yr, mean_lat, 'k')
ax.set_xlim((yr[0], yr[-1]))
yy = ax.get_ylim()
ax.set_ylim(yy)
#add the solar min times
for n in range(16,len(solarmintimes_df)):
    fyr = year_fraction(solarmintimes_df.loc[n,'datetime'])
    ax.plot([fyr, fyr], yy, 'r--')
ax.set_ylabel(r'$<|\theta|>$ [deg]', fontsize = 14)                
        
    
