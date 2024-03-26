# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 14:10:53 2022

@author: mathewjowens
"""

#a script to produce a movie of the WSA-DA runs



import astropy.units as u
import numpy as np
import datetime
import h5py
import os as os
import matplotlib.pyplot as plt
from astropy.io import fits
import shutil
from scipy import interpolate
import glob
import matplotlib as mpl
import pandas as pd

import huxt_inputs as Hin
import huxt as H
import huxt_analysis as HA

import huxt_ensembles as Hens


workingdir = os.environ['DBOX'] + 'Papers_WIP\\_coauthor\\SiegfriedGonzi\\'
bravda_ens_dir = os.environ['DBOX'] + 'python_repos\\BRaVDA\\wsaEns\\'

wsafilename = 'wsa_vel_21.5rs_2010040812_gong.fits' 
newfilename = 'wsa_DA_OMNI_STB.fits' 
starttime = datetime.datetime(2010,4,8, 12)

#ensemble generation parameters
Nens = 500
lat_rot_sigma = 5*np.pi/180 *u.rad
lat_dev_sigma = 2*np.pi/180 *u.rad
long_dev_sigma = 2*np.pi/180 *u.rad

r_min = 21.5*u.solRad
r_max = 240*u.solRad

sigma = 10 #[10] width of the Gaussian for the Kalman filter function [degrees]

def plot_huxt_multi(ax, time, model):
    """
    Plot the HUXt solution at a specified time, and (optionally) overlay the modelled flank location and field of view
    of a specified observer.
    :param ax: Axes handle to plot in.
    :param time: The time to plot. The closest value in model.time_out is selected.
    :param model: A HUXt instance with the solution in.
    :return:
    """
    id_t = np.argmin(np.abs(model.time_out - time))

    # Get plotting data
    lon_arr, dlon, nlon = H.longitude_grid()
    lon, rad = np.meshgrid(lon_arr.value, model.r.value)
    mymap = mpl.cm.viridis
    v_sub = model.v_grid.value[id_t, :, :].copy()
    # Insert into full array
    if lon_arr.size != model.lon.size:
        v = np.zeros((model.nr, nlon)) * np.NaN
        if model.lon.size != 1:
            for i, lo in enumerate(model.lon):
                id_match = np.argwhere(lon_arr == lo)[0][0]
                v[:, id_match] = v_sub[:, i]
        else:
            print('Warning: Trying to contour single radial solution will fail.')
    else:
        v = v_sub

    # Pad out to fill the full 2pi of contouring
    pad = lon[:, 0].reshape((lon.shape[0], 1)) + model.twopi
    lon = np.concatenate((lon, pad), axis=1)
    pad = rad[:, 0].reshape((rad.shape[0], 1))
    rad = np.concatenate((rad, pad), axis=1)
    pad = v[:, 0].reshape((v.shape[0], 1))
    v = np.concatenate((v, pad), axis=1)

    mymap.set_over('lightgrey')
    mymap.set_under([0, 0, 0])
    levels = np.arange(200, 800 + 10, 10)
    cnt = ax.contourf(lon, rad, v, levels=levels, cmap=mymap, extend='both')
    # Remove edgelines that appear in pdfs
    for c in cnt.collections:
        c.set_edgecolor("face")

    cme_colors = ['r', 'c', 'm', 'y', 'deeppink', 'darkorange']
    for j, cme in enumerate(model.cmes):
        cid = np.mod(j, len(cme_colors))
        cme_lons = cme.coords[id_t]['lon']
        cme_r = cme.coords[id_t]['r'].to(u.solRad)
        if np.any(np.isfinite(cme_r)):
            # Pad out to close the profile.
            cme_lons = np.append(cme_lons, cme_lons[0])
            cme_r = np.append(cme_r, cme_r[0])
            ax.plot(cme_lons, cme_r, '-', color=cme_colors[cid], linewidth=3)

    planet_list = ['MERCURY', 'VENUS', 'EARTH', 'STA', 'STB','MARS', 'JUPITER', 'SATURN']
    planet_cols = ['black', 'tan', 'b', 'r', 'm', 'orange']
    # Add other planets if in model domain
    for planet, color in zip(planet_list, planet_cols ):
        obs = model.get_observer(planet)
        deltalon = 0.0*u.rad
        if model.frame == 'sidereal':
            earth_pos = model.get_observer('EARTH')
            deltalon = earth_pos.lon_hae[id_t] - earth_pos.lon_hae[0] 
        obslon = H._zerototwopi_(obs.lon[id_t] + deltalon)
        

        if planet == 'EARTH':
            ax.plot(obslon, obs.r[id_t], 'o', color=color, markersize=12, label=planet)
        elif (obs.r[id_t] > model.r.min()) & (obs.r[id_t] < model.r.max()):
            ax.plot(obslon, obs.r[id_t], 'o', color=color, markersize=12, label=planet)

    ax.set_ylim(0, model.r.value.max())
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.patch.set_facecolor('slategrey')
    return


# <codecell>
wsa_vr_map, vr_longs, vr_lats, br_map, br_longs, br_lats, cr_fits \
    = Hin.get_WSA_maps(workingdir + wsafilename)

#get the huxt params for the start time
cr, cr_lon_init = Hin.datetime2huxtinputs(starttime)


#Use the HUXt ephemeris data to get Earth lat over the CR
#========================================================
dummymodel = H.HUXt(v_boundary=np.ones((128))*400* (u.km/u.s), simtime=27.27*u.day, 
                   cr_num= cr, cr_lon_init = cr_lon_init, 
                   lon_out=0.0*u.deg,
                   r_min=r_min, r_max=r_max)

#retrieve a bodies position at each model timestep:
earth = dummymodel.get_observer('earth')
#get Earth lat as a function of longitude (not time)
E_lat = np.mean(earth.lat_c)
E_r = np.mean(earth.r)

reflats = np.interp(vr_longs,earth.lon_c,earth.lat_c)



# <codecell>
#Now run HUXt with these different solar winds
files = ['wsa_vel_21.5rs_2010040812_gong.fits', 'wsa_DA_OMNI.fits', 'wsa_DA_OMNI_STB.fits']
reflat = np.nanmean(reflats)

#get the HUXt parameters for the start time, allowing for prior CME propagation
cr, cr_lon_init = Hin.datetime2huxtinputs(starttime - datetime.timedelta(days=5))

plt.figure()
for file in files:
    vr_in = Hin.get_WSA_long_profile(workingdir + file, lat = reflat.to(u.deg))
    plt.plot(vr_in, label = file)
plt.legend()

#set up the April 2010 CME.

#copmpute the propagation time to 21.5 rS
vcme = 514*(u.km/u.s)
t_21p5 = (21.5*u.solRad).to(u.km)/vcme + (0*60*60 +12*60)*u.s
cme = H.ConeCME(t_launch=4.5*u.day + t_21p5.to(u.day), longitude=3.0*u.deg, latitude = 0.0*u.deg,
                width=50*u.deg, v=vcme, thickness=5*u.solRad,
                initial_height = 21.5*u.solRad)

for file in files:
    vr_in = Hin.get_WSA_long_profile(workingdir + file, lat = reflat.to(u.deg))
    model = H.HUXt(v_boundary=vr_in, cr_num=cr, cr_lon_init=cr_lon_init, latitude = reflat,
                   simtime=27*u.day, dt_scale=4, r_min = 21.5*u.solRad, frame = 'sidereal')
    model.solve([cme]) 
    #HA.plot_earth_timeseries(model, plot_omni = False)
    
        
    cme_huxt = model.cmes[0]
    stats = cme_huxt.compute_arrival_at_body('EARTH')   
    HA.plot(model, model.time_out[stats['hit_id']])
    #  Print out some arrival stats
    print("**************************************")
    print(file)
    output_text = "Earth arrival:{},  Transit time:{:3.2f} days, Arrival longitude:{:3.2f}, Speed:{:3.2f}" 
    print(output_text.format(stats['t_arrive'].isot, stats['t_transit'], stats['lon'], stats['v']))
 
file = files[0]    
vr_in = Hin.get_WSA_long_profile(workingdir + file, lat = reflat.to(u.deg))
model_noDA = H.HUXt(v_boundary=vr_in, cr_num=cr, cr_lon_init=cr_lon_init, latitude = reflat,
               simtime=27*u.day, dt_scale=4, r_min = 21.5*u.solRad, frame = 'sidereal')
model_noDA.solve([cme]) 
file = files[1]    
vr_in = Hin.get_WSA_long_profile(workingdir + file, lat = reflat.to(u.deg))
model_L1 = H.HUXt(v_boundary=vr_in, cr_num=cr, cr_lon_init=cr_lon_init, latitude = reflat,
               simtime=27*u.day, dt_scale=4, r_min = 21.5*u.solRad, frame = 'sidereal')
model_L1.solve([cme]) 
file = files[2]    
vr_in = Hin.get_WSA_long_profile(workingdir + file, lat = reflat.to(u.deg))
model_L1L5 = H.HUXt(v_boundary=vr_in, cr_num=cr, cr_lon_init=cr_lon_init, latitude = reflat,
               simtime=27*u.day, dt_scale=4, r_min = 21.5*u.solRad, frame = 'sidereal')
model_L1L5.solve([cme])   

models = [model_noDA, model_L1, model_L1L5]
labels = ['No DA', 'L1 DA', 'L1 + L5 DA']

ts_noDA = HA.get_observer_timeseries(model_noDA, observer = 'Earth')
ts_L1 = HA.get_observer_timeseries(model_L1, observer = 'Earth')
ts_L1L5 = HA.get_observer_timeseries(model_L1L5, observer = 'Earth')

tss = [ts_noDA, ts_L1, ts_L1L5]

# <codecell>
#plot the three runs side by side



import matplotlib.gridspec as gridspec
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

from sunpy.net import Fido
from sunpy.net import attrs
from sunpy.timeseries import TimeSeries



time = tss[0]['time']
plotstartdate = time[0] + datetime.timedelta(days =2)
plotstopdate = plotstartdate + datetime.timedelta(days =10)
startdate = time[0]
stopdate = time[len(time)-1]

obs_arrival = datetime.datetime.strptime('2010-04-11T12:28Z', '%Y-%m-%dT%H:%MZ')
obs_v = 431.6


# Download the 1hr OMNI data from CDAweb
trange = attrs.Time(startdate, stopdate)
dataset = attrs.cdaweb.Dataset('OMNI2_H0_MRG1HR')
result = Fido.search(trange, dataset)
downloaded_files = Fido.fetch(result)

# Import the OMNI data
omni = TimeSeries(downloaded_files, concatenate=True)
data = omni.to_dataframe()

# Set invalid data points to NaN
id_bad = data['V'] == 9999.0
data.loc[id_bad, 'V'] = np.NaN

# Create a datetime column
data['datetime'] = data.index.to_pydatetime()

omnidata = pd.DataFrame()
omnidata['datetime'] = time
omnidata['V'] = np.interp(time, data['datetime'], data['V'])

# mask = (data['datetime'] >= startdate) & (data['datetime'] <= stopdate)
# omnidata = data[mask]

def plot_frame(models,labels,tss, omnidata, ylims = [250,750], id_t = 250):
    
    
    fig = plt.figure(figsize = (17,10))
    gs = gridspec.GridSpec(2, 3)
    
    axs=[]
    for n in range(0,3):
        model = models[n]
        ax = fig.add_subplot(gs[0, n], projection='polar')
        plot_huxt_multi(ax, model.time_out[id_t], model)
    
        ax.set_title(labels[n], fontsize = 16)
        #label = "Model time: {:3.1f} hr".format(model.time_out[id_t].to(u.hr).value)
        #ax.text(0.5, -0.05,label, ha='center', fontsize=16, transform=ax.transAxes)
        
        axs.append(ax)
        
    mymap = mpl.cm.viridis
    mymap.set_over('lightgrey')
    mymap.set_under([0, 0, 0])
    norm = mpl.colors.Normalize(vmin=200,vmax=800)
    smp = mpl.cm.ScalarMappable(norm=norm, cmap=mymap)
    
    pos = axs[0].get_position()
    dw = 0.02
    dh = 0.02
    left = pos.x0 - dw
    bottom = pos.y0 + dh
    width = 0.015
    height = pos.height - 2*dh
    cbaxes = fig.add_axes([left, bottom, width, height])
    cbar = fig.colorbar(smp, cax=cbaxes, orientation='vertical', extend='both')
    
    cbaxes.yaxis.set_ticks_position('left')
    cbaxes.yaxis.set_label_position('left')
    cbar.set_label('Solar Wind speed [km/s]')
    
    axs[1].legend(bbox_to_anchor=(0.5, 1.2), loc="center", ncol=5, frameon=False, handletextpad=0.0, columnspacing=0.0)
    
        
    #for the bottom three plots, plot the time series
    for n in range(0,3):
        ax = fig.add_subplot(gs[1, n])
        axs.append(ax)
        model = models[n]
        ts = tss[n]
    
        #get arrival time
        cme_huxt = model.cmes[0]
        stats = cme_huxt.compute_arrival_at_body('EARTH')   
        
        ax.plot(ts['time'],ts['vsw'],'mistyrose')
        ax.plot(ts['time'][0:id_t],ts['vsw'][0:id_t],'r', label = labels[n], linewidth=3)
        
        date_form = DateFormatter("%d")
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(date_form)
        ax.tick_params(axis='x', labelrotation = 0)
        
        ax.plot(omnidata['datetime'], omnidata['V'], 'lightgrey')
        ax.plot(omnidata['datetime'][0:id_t], omnidata['V'][0:id_t],'k', 
                label = 'Obs', linewidth=3)
        
        ax.plot([stats['t_arrive'].to_datetime(), stats['t_arrive'].to_datetime()],
                ylims, 'r--')
        ax.plot([obs_arrival, obs_arrival],
                ylims, 'k--')
        
        ax.set_ylim(ylims)
        ax.set_xlim([plotstartdate, plotstopdate])
        
        ax.legend( loc = 'lower left')
        
        def hours(td):
            days = td.days
            hours = td.seconds//3600
            minutes = (td.seconds//60)%60
            
            tot_hours = days*24 + hours + minutes/60
            return tot_hours
        
        dt = stats['t_arrive'].to_datetime() - obs_arrival
        dv = stats['v'].value - obs_v
        output_text = "CME dt: {:3.1f} hrs,  dV: {:3.1f} km/s" 
        ax.set_title(output_text.format(hours(dt), dv) )
        
    axs[4].set_xlabel('Day of April 2010')
    axs[3].set_ylabel('Solar Wind speed [km/s]')   
    
    return fig


plot_frame(models,labels,tss, omnidata,ylims = [250,750], id_t = 250)


#make a movie
import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage

duration = 10 # video duration, in seconds

start_i = np.argmin(abs(time-plotstartdate))
stop_i = np.argmin(abs(time-plotstopdate))
d_i = stop_i - start_i + 1 

def make_frame(t):
    """
    Produce the frame required by MoviePy.VideoClip.
    Args:
        t: time through the movie
    Returns:
        frame: An image array for rendering to movie clip.
    """
    # Get the time index closest to this fraction of movie duration
    i = np.int32((d_i - 1) * t / duration) + start_i
    fig = plot_frame(models,labels,tss, omnidata,ylims = [250,750], id_t = i)
    frame = mplfig_to_npimage(fig)
    plt.close('all')
    return frame

cr_num = np.int32(model.cr_num.value)
filename = "CME_DA.mp4"
animation = mpy.VideoClip(make_frame, duration=duration)
animation.write_videofile(filename, fps=24, codec='libx264')