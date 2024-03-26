# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 12:57:58 2022


a script to load and process WSA data, run BRAVDA DA and return a modified
WSA file

@author: vy902033
"""


import astropy.units as u
from astropy.time import Time, TimeDelta
import numpy as np
import datetime
import h5py
import os as os
import matplotlib.pyplot as plt
from astropy.io import fits
import shutil
from scipy import interpolate
import glob

#from HUXt
import huxt_inputs as Hin
import huxt as H
import huxt_analysis as HA

#from HUXt_tools
import huxt_ensembles as Hens

#from BRaVDA
import startBravda

#directory for input and output solar wind speed maps
workingdir = os.environ['DBOX'] + 'Data\\BRaVDA\\'
wsafilename = 'wsa_vel_21.5rs_2010040812_gong' 

#bravda ensemble directory
bravda_ens_dir = os.environ['DBOX'] + 'python_repos\\BRaVDA\\masEns\\'

#which spacecraft to assimilate. A=STERA, B=STERB, C=ACE, All are currently assumed to be at Earth lat
runs = ['C', 'B', 'BC', 'ABC']

#forecast tiem. i.e. time of last  in situ observation to be used.
forecasttime = datetime.datetime(2010,4,8, 12)
buffertime_days = 5 # number of days ahead to start the run, to allow CME to propagate

#ensemble generation parameters
Nens = 500
lat_rot_sigma = 5*np.pi/180 *u.rad
lat_dev_sigma = 2*np.pi/180 *u.rad
long_dev_sigma = 2*np.pi/180 *u.rad

#HUXt run parameters
r_min = 21.5*u.solRad
r_max = 240*u.solRad
simtime = 12*u.day

sigma = 10 #[10] width of the Gaussian for the Kalman filter function [degrees]

#set up the April 2010 CME.
vcme = 514*(u.km/u.s)
t_0_cme = datetime.datetime(2010,4,8, 0,0,1) #time at 1 rS
cme_longitude=3.0*u.deg
cme_latitude = 0.0*u.deg
cme_width=50*u.deg
cme_thickness=5*u.solRad

t_1au_obs = datetime.datetime(2010,4,11, 12,28,0)


# <codecell> generate WSA ensemble for the DA

#start the HUXt runs a few days ahead of the forecast time
starttime = forecasttime  - datetime.timedelta(days=buffertime_days)
#BRaVDA rounds to nearest MJD
smjd = int(Time(starttime).mjd)
starttime = Time(smjd, format = 'mjd').datetime
fmjd = int(Time(forecasttime).mjd)
forecasttime = Time(smjd, format = 'mjd').datetime

wsa_vr_map, vr_longs, vr_lats, br_map, br_longs, br_lats, cr_fits \
    = Hin.get_WSA_maps(workingdir + wsafilename + '.fits')
    
x,y = np.meshgrid(vr_longs.value * 180/np.pi,vr_lats.value*180/np.pi)
plt.figure()
plt.pcolor(x, y, wsa_vr_map.value )
plt.title('Original WSA map')

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


    
phi, theta = np.meshgrid(vr_longs, vr_lats, indexing = 'xy')

vr_ensemble = Hens.generate_input_ensemble(phi, theta, wsa_vr_map, 
                            reflats, Nens = Nens, 
                            lat_rot_sigma = lat_rot_sigma, 
                            lat_dev_sigma = lat_dev_sigma,
                            long_dev_sigma = long_dev_sigma)

#resample the ensemble to 128 longitude bins
vr128_ensemble = np.ones((Nens,128))  
dphi = 2*np.pi/128
phi128 = np.linspace(dphi/2, 2*np.pi - dphi/2, 128)
for i in range(0, Nens):
    vr128_ensemble[i,:] = np.interp(phi128,
                  vr_longs.value,vr_ensemble[i,:])

#solar wind speed as a function of longitude         
endata = vr128_ensemble
tdata = phi128*180/np.pi
confid_intervals = [5, 10, 33]

#plt.figure()
#Hens.plotconfidbands(tdata,endata,confid_intervals)
#plt.plot(tdata,vr128_ensemble[0,:],'k',label='Unperturbed')
#plt.legend(facecolor='grey')
#plt.xlabel('Carrington Longitude [deg]')
#plt.ylabel(r'V$_{SW}$ [km/s]')
#plt.title('HUXt input at ' + str(r_in) +' rS')


    
#==============================================================================
#save the ensemble, e,g for use with DA
#==============================================================================

var = 'vr'
if var == 'vr':
    h5f = h5py.File(workingdir + '\\WSA_CR' + str(cr) +'_vin_ensemble.h5', 'w')
    h5f.create_dataset('Vin_ensemble', data=vr128_ensemble)
elif var == 'br':
    h5f = h5py.File(workingdir + '\\WSA_CR' + str(cr) +'_bin_ensemble.h5', 'w')
    h5f.create_dataset('Bin_ensemble', data=vr128_ensemble)            
h5f.attrs['lat_rot_sigma'] = lat_rot_sigma
h5f.attrs['lat_dev_sigma'] = lat_dev_sigma
h5f.attrs['long_dev_sigma'] = long_dev_sigma
filepath = 'get_MAS_vrmap(cr)'  #this is used only to identify the source files. 
h5f.attrs['source_file'] = wsafilename
h5f.attrs['r_in_rS'] = r_min
h5f.attrs['Carrington_rotation'] = cr
h5f.close()    


#also save vr128 to a .dat file for use in BRaVDA
outEnsTxtFile = open(f'{bravda_ens_dir}customensemble.dat', 'w')
np.savetxt(outEnsTxtFile, vr128_ensemble)
outEnsTxtFile.close()


    
    
# <codecell>
    
def _zerototwopi_(angles):
    """
    Function to constrain angles to the 0 - 2pi domain.

    Args:
        angles: a numpy array of angles.
    Returns:
            angles_out: a numpy array of angles constrained to 0 - 2pi domain.
    """
    twopi = 2.0 * np.pi
    angles_out = angles
    a = -np.floor_divide(angles_out, twopi)
    angles_out = angles_out + (a * twopi)
    return angles_out
    
#Run BRaVDA for each run config and create a new FITS file, merging WSA and the DA
for run in runs:
    
    #output filepaths
    outputpath = workingdir + wsafilename + run
    newfilename = wsafilename + run + '.fits' 
    
    #==============================================================================
    #==============================================================================
    #Run startBravda
    #==============================================================================
    #==============================================================================
    startBravda.bravdafunction(forecasttime, obsToUse = run, usecustomens = True,
                               runoutputdir = outputpath, plottimeseries = False)

   
    #read in the posterior solution to blend with the WSA map   
    #BRaVDA files are labelled with teh start time, 27 days previous
    posterior = np.loadtxt(outputpath + '\\posterior\\posterior_MJDstart' + str(fmjd-27) +'.txt')
    
    #the inner boundary value is given as a function of time from the inition time
    post_inner = posterior[0,:]
    #convert to function of longitude
    post_vlong = np.flipud(post_inner)
    #post_vlong = post_inner
    #find the associated carr_longs
    cr, cr_lon_init = Hin.datetime2huxtinputs(forecasttime)
    post_carrlongs = _zerototwopi_(phi128 + cr_lon_init.value)
    
    #interpolate the posterior at the inner boundary to the WSA CarrLong grid
    interp = interpolate.interp1d(post_carrlongs,
                                  post_vlong, kind="nearest",
                                  fill_value="extrapolate")
    post_wsalongs = interp(vr_longs.value)
    
    # Now distort the WSA solution on the basis of the DA at the SS using a Gaussain filter in lat
    sigma_rad = sigma * np.pi/180
    new_map = wsa_vr_map.value *np.nan
    for nlong in range(0,len(vr_longs)):
        
        acelat = reflats[nlong]
        
        #compute the deltas at all latitudes
        delta_lat_ace = abs(vr_lats.value - acelat.value)
        
        #compute the guassian weighting at each lat
        weights_ace = np.exp( -delta_lat_ace*delta_lat_ace/(2*sigma_rad*sigma_rad))
        
        #merge the MAS and assimilated data
        new_map[:,nlong] = ((1-weights_ace)*wsa_vr_map[:,nlong].value +
            weights_ace*post_wsalongs[nlong])
    
    
    
    #rotate the map back around to match the original WSA format
    #===========================================================
    
    #find the required angle of rotation
    hdul = fits.open(workingdir + wsafilename + '.fits')
    cr_num = hdul[0].header['CARROT']
    dgrid = hdul[0].header['GRID'] * np.pi / 180
    carrlong = (hdul[0].header['CARRLONG']) * np.pi / 180
    data = hdul[0].data
    br_map_fits = data[0, :, :]
    vr_map_fits = data[1, :, :]
    hdul.flush() 
    hdul.close()
    
    # compute the Carrington map grids
    vr_long_edges = np.arange(0, 2 * np.pi + 0.00001, dgrid)
    vr_long_centres = (vr_long_edges[1:] + vr_long_edges[:-1]) / 2
    
    vr_lat_edges = np.arange(-np.pi / 2, np.pi / 2 + 0.00001, dgrid)
    vr_lat_centres = (vr_lat_edges[1:] + vr_lat_edges[:-1]) / 2
    
    vr_longs = vr_long_centres * u.rad
    vr_lats = vr_lat_centres * u.rad
    
    
    # rotate the maps so they are in the Carrington frame
    rot_vr_map = np.empty(vr_map_fits.shape)
    for nlat in range(0, len(vr_lat_centres)):
        interp = interpolate.interp1d(_zerototwopi_(vr_long_centres - carrlong),
                                      new_map[nlat, :], kind="nearest",
                                      fill_value="extrapolate")
        rot_vr_map[nlat, :] = interp(vr_long_centres)
    
    new_map = rot_vr_map
    
        
    # plt.figure()
    # plt.pcolor(wsa_vr_map.value)
    
    
    # plt.figure()
    # plt.pcolor(new_map)
    
    #make a copy of the original WSA FITS file
    if not os.path.exists(workingdir + newfilename):
        shutil.copyfile(workingdir + wsafilename + '.fits',
                        workingdir + newfilename)
    
    #modify contents of the FITS file
    
    hdul = fits.open(workingdir + newfilename, mode = 'update')
    data = hdul[0].data
    #paste in the new data
    data[1, :, :] = new_map
    hdul[0].data = data
    hdul.flush() 
    hdul.close()
    
    
    
    #load the old and new files and plot
    # vr_map_fits, vr_longs, vr_lats, br_map, br_longs, br_lats, cr_fits \
    # = Hin.get_WSA_maps(workingdir + wsafilename + '.fits')
    # x,y = np.meshgrid(vr_longs.value * 180/np.pi,vr_lats.value*180/np.pi)
    # plt.figure()
    # plt.pcolor(x,y,vr_map_fits.value)
    
    vr_map_fits, vr_longs, vr_lats, br_map, br_longs, br_lats, cr_fits \
    = Hin.get_WSA_maps(workingdir + newfilename )
    
    plt.figure()
    plt.pcolor(x,y,vr_map_fits.value)
    plt.title('WSA + DA of ' + run)


# <codecell>
#=============================================
#Now run HUXt with these different solar winds
#=============================================

files = []
files.append(wsafilename + '.fits')
for run in runs:
    files.append(wsafilename + run +'.fits')
    
reflat = np.nanmean(reflats)

#read in and plot the HUXt input for each map   
fig, (ax1, ax2) = plt.subplots(2)
for file in files:
    vr_in = Hin.get_WSA_long_profile(workingdir + file, lat = reflat.to(u.deg))
    dlong = 360/len(vr_in)
    longs = np.arange(0, 360, dlong)
    ax1.plot(longs, vr_in, label = file)
ymin, ymax = ax1.get_ylim()
cr_forecast, cr_lon_init_forecast = Hin.datetime2huxtinputs(forecasttime)
ax1.plot([cr_lon_init_forecast.to(u.deg).value, cr_lon_init_forecast.to(u.deg).value], [ymin,ymax],'r')
cr, cr_lon_init = Hin.datetime2huxtinputs(starttime)
ax1.plot([cr_lon_init.to(u.deg).value, cr_lon_init.to(u.deg).value], [ymin,ymax],'r--')
ax1.legend()
ax1.set_xlabel('Carrington longitude [deg]')
ax1.set_ylabel(r'$V_{SW}$ (21.5 rS) [km/s]') 
ax1.set_title('Earth lat = '+str(reflat.to(u.deg)))   

#get the HUXt parameters for the start time, allowing for prior CME propagation
cr, cr_lon_init = Hin.datetime2huxtinputs(starttime)

#run the ambient solution for all runs
for file in files:
    vr_in = Hin.get_WSA_long_profile(workingdir + file, lat = reflat.to(u.deg))
    model = H.HUXt(v_boundary=vr_in, cr_num=cr, cr_lon_init=cr_lon_init, latitude = reflat,
                   simtime=27.27*u.day, dt_scale=4, r_min = r_min, frame = 'sidereal')
    model.solve([])
    earth_series = HA.get_observer_timeseries(model, observer = 'Earth')
    #plot it
    ax2.plot(earth_series['time'],earth_series['vsw'], label = file)
    HA.plot_earth_timeseries(model, plot_omni = True)
ax2.legend()
ax2.set_ylabel(r'$V_{SW}$ (Earth) [km/s]') 



#=============================================
#Now add the CME
#=============================================

#compute the propagation time to 21.5 rS and hence the insertion time relative to run start
t_21p5 = (20.5*u.solRad).to(u.km)/vcme 
t_cme_runstart = ((t_0_cme - starttime).days 
                  + (t_0_cme - starttime).seconds/24/60/60 
                  + t_21p5.to(u.day).value)


cme = H.ConeCME(t_launch=t_cme_runstart*u.day, longitude=cme_longitude, 
                latitude = cme_latitude,
                width=cme_width, v=vcme, thickness=cme_thickness,
                initial_height = r_min)

print('Observed 1 AU arrival time: ')
print(t_1au_obs)
print(' ')
for file in files:
    vr_in = Hin.get_WSA_long_profile(workingdir + file, lat = reflat.to(u.deg))
    model = H.HUXt(v_boundary=vr_in, cr_num=cr, cr_lon_init=cr_lon_init, latitude = reflat,
                   simtime=simtime, dt_scale=4, r_min = r_min, frame = 'sidereal')
    model.solve([cme]) 
    HA.plot_earth_timeseries(model, plot_omni = True)
    
        
    cme_huxt = model.cmes[0]
    stats = cme_huxt.compute_arrival_at_body('EARTH')   
    HA.plot(model, model.time_out[stats['hit_id']])
    #  Print out some arrival stats
    print("**************************************")
    print(file)
    output_text = "Earth arrival:{},  Transit time:{:3.2f} days, Arrival longitude:{:3.2f}, Speed:{:3.2f}" 
    print(output_text.format(stats['t_arrive'].isot, stats['t_transit'], stats['lon'], stats['v']))
    