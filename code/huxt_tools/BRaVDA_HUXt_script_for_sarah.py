# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 14:09:20 2023

@author: vy902033
"""

import astropy.units as u
from astropy.time import Time, TimeDelta
import numpy as np
import datetime
import os as os
from scipy import interpolate
import matplotlib.pyplot as plt

#from HUXt
#sys.path.insert(1, 'C:/Users/invate/OneDrive - University of Reading/PhD Year 1/HUXt-master/HUXt-master/code')
import huxt_inputs as Hin
import huxt as H
import huxt_analysis as HA

#from HUXt_tools
import huxt_ensembles as Hens

#from BRaVDA
import startBravda

#directory for input and output solar wind speed maps
workingdir = os.environ['DBOX'] + 'Data\\BRaVDA\\'
#workingdir = 'C:/Users/invate/OneDrive - University of Reading/PhD Year 1/'


#bravda ensemble directory
bravda_ens_dir = os.environ['DBOX'] + 'python_repos\\BRaVDA\\masEns\\'
#bravda_ens_dir = 'C:/Users/invate/OneDrive - University of Reading/PhD Year 1/' + 'python_repos/BRaVDA/masEns/'

outputpath = workingdir + 'SarahWatson'

#start the HUXt run on this date
huxt_startdate =  datetime.datetime(2021,12,10)
r_min = 30*u.solRad
r_max = 240*u.solRad

#MAS solution for the BRaVDA ensemble and the Br input to HUXt
cr_MAS = 2252 #set to np.nan to use the CR consistent with the startdate

#which spacecraft to assimilate. A=STERA, B=STERB, C=ACE, All are currently assumed to be at Earth lat
obs = ['OMNI', 'STERA']#  ['OMNI'] #

#DA window dates
starttime = datetime.datetime(2021,12,3)
endtime = starttime + datetime.timedelta(days=27.27)


# <codecell> Grab the MAS solution, construct the ensemble needed by BRaVDA
#helper functions
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


#ensemble generation parameters
Nens = 500
lat_rot_sigma = 10*np.pi/180 *u.rad
lat_dev_sigma = 0*np.pi/180 *u.rad
long_dev_sigma = 10*np.pi/180 *u.rad
    
cr_start, cr_lon_init_start = Hin.datetime2huxtinputs(huxt_startdate)

if np.isnan(cr_MAS):
    cr_MAS = cr_start
    print('Using MAS solution for the given HUXt start date')
else:
    print('Using MAS solution for the prescribed CR')
#grab the MAS solution  
Hin.get_MAS_boundary_conditions(cr_MAS)
vr_map, vr_longs, vr_lats = Hin.get_MAS_vr_map(cr_MAS)


#Use the HUXt ephemeris data to get Earth lat over the CR
#========================================================
dummymodel = H.HUXt(v_boundary=np.ones((128))*400* (u.km/u.s), simtime=27.27*u.day, 
                   cr_num= cr_start, cr_lon_init = cr_lon_init_start, 
                   lon_out=0.0*u.deg,
                   r_min=r_min, r_max=r_max)

#retrieve a bodies position at each model timestep:
earth = dummymodel.get_observer('earth')
#get Earth lat as a function of longitude (not time)
E_lat = np.mean(earth.lat_c)
E_r = np.mean(earth.r)

reflats = np.interp(vr_longs,earth.lon_c,earth.lat_c)

#generate WSA ensemble for the DA and put it in the BRaVDA dir
#==============================================================
phi, theta = np.meshgrid(vr_longs, vr_lats, indexing = 'xy')

vr_ensemble = Hens.generate_input_ensemble(phi, theta, vr_map, 
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
    
#also save vr128 to a .dat file for use in BRaVDA
outEnsTxtFile = os.path.join(bravda_ens_dir, 'customensemble.dat')
np.savetxt(outEnsTxtFile, vr128_ensemble)
#outEnsTxtFile.close()



# <codecell> Run BRaVDA

#==============================================================================
#==============================================================================
#Run startBravda
#==============================================================================
#==============================================================================


startBravda.bravdafunction(endtime, obsToAssim = obs , usecustomens = True,
                           runoutputdir = outputpath, plottimeseries = True,
                           precondState = True)

#read in the posterior solution 
smjd = int(Time(starttime).mjd)
actual_startdate = Time(smjd, format = 'mjd').datetime
#post_filepath = glob.glob(outputpath + '\\posterior\\posterior_MJDstart*')
#posterior = np.loadtxt(post_filepath[0]) #THIS CAN FIND THE WRONG FILE. USE MJD START 
posterior = np.loadtxt(outputpath + '\\posterior\\posterior_MJDstart' + str(smjd) +'.txt')

#the inner boundary value is given as a function of time from the inition time
post_inner = posterior[0,:]
#convert from function of time to function of longitude
post_vlong = np.flipud(post_inner)
#post_vlong = post_inner
#find the associated carr_longs
cr, cr_lon_init = Hin.datetime2huxtinputs(actual_startdate)
#resample the ensemble to 128 longitude bins
dphi = 2*np.pi/128
phi128 = np.linspace(dphi/2, 2*np.pi - dphi/2, 128)

post_carrlongs = _zerototwopi_(phi128 + cr_lon_init.value)

#interpolate the posterior at the inner boundary to CarrLong grid
interp = interpolate.interp1d(post_carrlongs,
                              post_vlong, kind="nearest",
                              fill_value="extrapolate")
from scipy.ndimage.filters import uniform_filter1d
v_carrlong = uniform_filter1d(interp(phi128), size = 10) * u.km/u.s


plt.figure()
plt.subplot(2,1,1)
plt.plot(posterior[0,:])

plt.subplot(2,1,2)
plt.plot(v_carrlong)

# # <codecell> Now run HUXt with this time series

# br_in = Hin.get_MAS_br_long_profile(cr_MAS, E_lat.to(u.deg))                                                                            

# model = H.HUXt(v_boundary=v_carrlong, b_boundary=br_in, cr_num=cr, cr_lon_init=cr_lon_init, latitude = E_lat.to(u.deg),
#                simtime=27.27*u.day, dt_scale=4, r_min = r_min, frame = 'sidereal')
# model.solve([])
# earth_series = HA.get_observer_timeseries(model, observer = 'Earth')
# venus_series = HA.get_observer_timeseries(model, observer = 'Venus')
# #plot it
# #fig, ax = plt.subplots(1)
# #ax.plot(earth_series['time'],earth_series['vsw'])
# HA.plot_earth_timeseries(model, plot_omni = True)

