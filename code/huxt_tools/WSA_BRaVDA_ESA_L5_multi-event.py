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

#directory for input and output solar wind speed maps. where DA maps are stored, if they are not generated at run
rootdir = os.environ['DBOX'] + 'Papers_WIP\\_coauthor\\SiegfriedGonzi\\C2-KIRK-24h-v2\\'

#directory for the cone files. Can be the same or different from rootdir
#conerootdir = os.environ['DBOX'] + 'Papers_WIP\\_coauthor\\SiegfriedGonzi\\C2-KIRK-24h-v2\\'
#conerootdir = os.environ['DBOX'] + 'Papers_WIP\\_coauthor\\SiegfriedGonzi\\C2-MICHAEL-24h-v2\\'
conerootdir = os.environ['DBOX'] + 'Papers_WIP\\_coauthor\\SiegfriedGonzi\\C2-REBECCA-24h-v2\\'

event_number = 11
conefile_number = 0 #zero is unperturbed from forecaster

#bravda ensemble directory
bravda_ens_dir = os.environ['DBOX'] + 'python_repos\\BRaVDA\\masEns\\'

#which spacecraft to assimilate. A=STERA, B=STERB, C=ACE, All are currently assumed to be at Earth lat
runs = ['C', 'ABC']#, 'C', 'B', 'BC', 'ABC']
DAnow = True # run the DA now. Time consuming.
allplots = True

#forecast tiem. i.e. time of last  in situ observation to be used.
buffertime_days = 5 # number of days ahead to start the run, to allow CME to propagate

#ensemble generation parameters
Nens = 500
lat_rot_sigma = 5*np.pi/180 *u.rad
lat_dev_sigma = 2*np.pi/180 *u.rad
long_dev_sigma = 2*np.pi/180 *u.rad

#HUXt run parameters
deacc = True # deaccelerate the WSA to 21.5 rS map prior to making the ensemble
r_min = 21.5*u.solRad
r_max = 240*u.solRad
simtime = 12*u.day

sigma = 10 #[10] width of the Gaussian for the Kalman filter function [degrees]

cme_source = 'CAT' #'GCS' 

# <codecell> CME data
#CAT tool output is in cone2bc.in files

#ICME arrival times and forecast times (i.e. timestep of the WSA input file)
obstimes=['2010-04-11T13:04:00', '2020-11-01T11:54:00', '2021-11-03T19:49:00',
          '2010-05-28T02:58:00', '2009-12-19T10:00:00', '2011-06-17T02:41:00',
          '2010-03-23T22:33:00', '2010-04-05T08:26:00', '2021-10-31T09:13:00',
          '2020-12-10T02:10:00', '2020-10-05T06:52:00', '2021-02-15T18:58:00',
          '2011-05-28T00:14:00', '2011-07-14T12:00:00', '2011-11-01T08:09:00']
foretimes = ['20100409', '20201028', '20211103',
             '20100525', '20091217', '20110615',
             '20100321', '20100404', '20211029',
             '20201208', '20201002', '20210212',
             '20110526', '20110712', '20111030']

#GCS data, provided by Groningen
gcs_v = [386, 192, 1420, 
         666, 477, 736,
         339, 838, 1647,
         1351, np.nan, 193,
         559, 516, 486]
gcs_lon = [19, 358, 358.3,
           19, 356.6, 325.6,
           43, 8.6, 358.5,
           12, np.nan, 29.3,
           127.5, 346.2, 151.6]
gcs_lat = [-7.6, -11, 22,
           -3.4, 8, 0,
           -7.6, -23, -26,
           -29, np.nan, -4.3,
           29, -7.6, -28]
gcs_width = [10.8, 20.4, 38,
             24.2, 34.6, 31.3,
             19.2, 18.3, 55,
             50, np.nan, 15,
             23.3, 17, 9]
gcs_last_obs_height = [15, 15.44, 19,
                       16.11, 15.78, 17,
                       17.44, 14.89, 21.78,
                       18.89, np.nan, 13.33,
                       19.78, 14.78, 18.22]
gcs_last_obs_time = ['2010-04-08T07:54:00', '2020-10-27T01:54:00', '2021-11-02T04:53:00',
                     '2010-05-24T18:24:00', '2009-12-16T08:24:00', '2011-06-14T12:09:00',
                     '2010-03-19T21:24:00', '2010-04-03T12:24:00', '2011-10-28T17:54:00',
                     '2020-12-07T18:30:00', 'nan',                 '2021-02-11T00:24:00',
                     '2011-05-25T03:24:00', '2011-07-11T15:39:00', '2011-10-29T02:24:00']



# <codecell> work out file paths

forecaststr = foretimes[event_number-1]

year = int(forecaststr[0:4])
month = int(forecaststr[4:6])
day = int(forecaststr[6:8])
hour = 0
forecasttime = datetime.datetime(year,month,day, hour)

#start the HUXt runs a few days ahead of the forecast time
starttime = forecasttime  - datetime.timedelta(days=buffertime_days)
#BRaVDA rounds to nearest MJD
smjd = int(Time(starttime).mjd)
starttime = Time(smjd, format = 'mjd').datetime
fmjd = int(Time(forecasttime).mjd)
forecasttime = Time(fmjd, format = 'mjd').datetime

#find the wsa file
eventdir = glob.glob(rootdir + '*' + 'CME-'+str(event_number) )[0]
workingdir = glob.glob(eventdir +'\\swcx\\*')[0] + '\\'
wsafilepath = glob.glob(workingdir + 'wsa_vel_21.5rs*' )[0]
wsafilename = os.path.basename(wsafilepath).replace('.fits','')

#find the cone file
coneeventdir = glob.glob(conerootdir + '*' + 'CME-'+str(event_number))[0]
coneworkingdir = glob.glob(coneeventdir +'\\swcx\\*')[0] + '\\'
conefilepath = glob.glob(coneworkingdir + 'cone2bc*' )[conefile_number]


obs_arrival_datetime = Time(obstimes[event_number-1], format='isot')
# <codecell> generate WSA ensemble for the DA

wsa_vr_map, vr_longs, vr_lats, br_map, br_longs, br_lats, cr_fits \
    = Hin.get_WSA_maps(wsafilepath)
    
if deacc:
    #deaccelerate the WSA map from 1-AU calibrated speeds to expected 21.5 rS values
    for nlat in range (1, len(vr_lats)):
        wsa_vr_map[nlat,:], lon_temp = Hin.map_v_inwards(wsa_vr_map[nlat,:], 215*u.solRad, 
                                                 vr_longs, r_min)
    
x,y = np.meshgrid(vr_longs.value * 180/np.pi,vr_lats.value*180/np.pi)

if allplots:
    plt.figure()
    plt.pcolor(x, y, wsa_vr_map.value, vmin = 250, vmax=650)
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
    
    if DAnow:
        startBravda.bravdafunction(forecasttime, obsToUse = run, usecustomens = True,
                                   runoutputdir = outputpath, plottimeseries = True,
                                   corona = 'WSA')

   
    #read in the posterior solution to blend with the WSA map   
    #BRaVDA files are labelled with teh start time, 27 days previous
    posterior = np.loadtxt(outputpath + '\\posterior\\posterior_MJDstart' + str(fmjd-28) +'.txt')
    
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
    
    if allplots:
        plt.figure()
        plt.pcolor(x,y,vr_map_fits.value , vmin = 250, vmax=650)
        plt.title('WSA + DA of ' + run)
        plt.colorbar()


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
    ax2.plot(earth_series['time'],earth_series['vsw'], label = file)
    
    if allplots:

        fig2, ax = HA.plot_earth_timeseries(model, plot_omni = True)
        ylims = ax[0].get_ylim()
        ax[0].plot([forecasttime, forecasttime], ylims, 'b')
        ax[0].plot([obs_arrival_datetime.to_datetime(), obs_arrival_datetime.to_datetime()], ylims, 'b--')
        
ax2.legend()
ax2.set_ylabel(r'$V_{SW}$ (Earth) [km/s]') 



#=============================================
#Now add the CME
#=============================================

#read in the latest cone file
#conefile = glob.glob(workingdir + 'cone2bc*')[conefile_number]

# #compute the propagation time to 21.5 rS and hence the insertion time relative to run start
# t_21p5 = (20.5*u.solRad).to(u.km)/vcme 
# t_cme_runstart = ((t_0_cme - starttime).days 
#                   + (t_0_cme - starttime).seconds/24/60/60 
#                   + t_21p5.to(u.day).value)


# cme = H.ConeCME(t_launch=t_cme_runstart*u.day, longitude=cme_longitude, 
#                 latitude = cme_latitude,
#                 width=cme_width, v=vcme, thickness=cme_thickness,
#                 initial_height = r_min)

print('Observed 1 AU arrival time: ')
print(obstimes[event_number-1])
print(' ')
dts = []
for file in files:
    vr_in = Hin.get_WSA_long_profile(workingdir + file, lat = reflat.to(u.deg))
    model = H.HUXt(v_boundary=vr_in, cr_num=cr, cr_lon_init=cr_lon_init, latitude = reflat,
                   simtime=simtime, dt_scale=4, r_min = r_min, frame = 'sidereal')
    
    if cme_source == 'CAT':
        cme = Hin.ConeFile_to_ConeCME_list(model, conefilepath)
    #elif cme_source == 'GCS':
    #    cme = Hin.ConeCME_from_params(model, v, lon, lat, width, obstime, obsheight)
        
    model.solve(cme) 
    fig3, ax = HA.plot_earth_timeseries(model, plot_omni = True)
    ylims = ax[0].get_ylim()
    ax[0].plot([forecasttime, forecasttime], ylims, 'b')
    ax[0].plot([obs_arrival_datetime.to_datetime(), obs_arrival_datetime.to_datetime()], ylims, 'b--')
    
        
    cme_huxt = model.cmes[0]
    #  Print out some arrival stats
    print("**************************************")
    print(file)
    print(os.path.basename(conefilepath))
    
    stats = cme_huxt.compute_arrival_at_body('EARTH') 
    if stats['hit_id']:
        output_text = "Earth arrival:{},  Transit time:{:3.2f} days, Arrival longitude:{:3.2f}, Speed:{:3.2f}" 
        print(output_text.format(stats['t_arrive'].isot, stats['t_transit'], stats['lon'], stats['v']))

        #HA.plot(model, model.time_out[stats['hit_id']])
        dt = (obs_arrival_datetime - stats['t_arrive']).jd
    else:
        dt = np.nan
    
    
    print('dt [days] = ' + str(dt))  
   
    #store the results
    dts.append(dt)


print(dts)

    