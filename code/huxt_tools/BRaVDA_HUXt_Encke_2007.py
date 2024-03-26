# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 15:27:13 2023

@author: Sarah
"""

import astropy.units as u
from astropy.time import Time, TimeDelta
import numpy as np
import datetime
import os as os
from scipy import interpolate
import matplotlib.pyplot as plt
from sunpy.coordinates.ephemeris import get_horizons_coord

#from HUXt
import sys
sys.path.insert(1, 'C:/Users/invate/OneDrive - University of Reading/PhD Year 1/HUXt/code')
import huxt_inputs as Hin
import huxt as H
import huxt_analysis as HA
import huxt_ensembles as Hens
import huxt_analysis_comet as HAC

#from HUXt_tools
#import huxt_ensembles as Hens

#import sys
#sys.path.insert(1, 'C:/Users/invate/OneDrive - University of Reading/PhD Year 1/BRaVDA-functional_bravda')
#from BRaVDA
import startBravda

#directory for input and output solar wind speed maps
workingdir = os.environ['DBOX'] + 'Data\\BRaVDA\\'
#workingdir = 'C:/Users/invate/OneDrive - University of Reading/PhD Year 1/BRaVDA-functional_bravda/'


#bravda ensemble directory
bravda_ens_dir = os.environ['DBOX'] + 'python_repos\\BRaVDA\\masEns\\'
#bravda_ens_dir = 'C:/Users/invate/OneDrive - University of Reading/PhD Year 1/' + 'python_repos/BRaVDA/masEns/'

outputpath = workingdir + 'SarahWatson'

#start the HUXt run on this date
huxt_startdate =  datetime.datetime(2007, 4, 14)
r_min = 30*u.solRad
r_max = 240*u.solRad
 
#MAS solution for the BRaVDA ensemble and the Br input to HUXt
cr_MAS = 2055 #set to np.nan to use the CR consistent with the startdate
 
#which spacecraft to assimilate. A=STERA, B=STERB, C=ACE, All are currently assumed to be at Earth lat
obs = ['OMNI', 'STERA', 'STERB']
 
#DA window dates
starttime = datetime.datetime(2007, 4, 3)
endtime = starttime + datetime.timedelta(days=27.27)


###############################################################################
cometlat = 30*u.deg
sigma = 10 #[10] width of the Gaussian for the Kalman filter function [degrees]
###############################################################################


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
lat_rot_sigma = 5*np.pi/180 *u.rad
lat_dev_sigma = 2*np.pi/180 *u.rad
long_dev_sigma = 2*np.pi/180 *u.rad
    
cr_start, cr_lon_init_start = Hin.datetime2huxtinputs(huxt_startdate)

if np.isnan(cr_MAS):
    cr_MAS = cr_start
    print('Using MAS solution for the given HUXt start date')
else:
    print('Using MAS solution for the prescribed CR')
#grab the MAS solution  
Hin.get_MAS_boundary_conditions(cr_MAS)
vr_map, vr_longs, vr_lats = Hin.get_MAS_vr_map(cr_MAS)
br_map, br_longs, br_lats = Hin.get_MAS_br_map(cr_MAS)

#Use the HUXt ephemeris data to get Earth lat over the CR
#========================================================
dummymodel = H.HUXt(v_boundary=np.ones((128))*400* (u.km/u.s), simtime=27.27*u.day, 
                   cr_num= cr_start, cr_lon_init = cr_lon_init_start, 
                   lon_out=0.0*u.deg,
                   r_min=r_min, r_max=r_max)

#Use the HUXt ephemeris data to get Comet lat over the CR
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
    
# #also save vr128 to a .dat file for use in BRaVDA
outEnsTxtFile = os.path.join(bravda_ens_dir, 'customensemble.dat')

np.savetxt(outEnsTxtFile, vr128_ensemble)
# #outEnsTxtFile.close()



# <codecell> Run BRaVDA

#==============================================================================
#==============================================================================
#Run startBravda
#==============================================================================
#==============================================================================


startBravda.bravdafunction(endtime, obsToAssim = obs , usecustomens = True,
                           runoutputdir = outputpath, plottimeseries = True )

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



#interpolate the posterior at the inner boundary to the MAS CarrLong grid
interp = interpolate.interp1d(post_carrlongs,
                              post_vlong, kind="nearest",
                              fill_value="extrapolate")
post_wsalongs = interp(vr_longs.value)

# Now distort the WSA solution on the basis of the DA at the SS using a Gaussain filter in lat
sigma_rad = sigma * np.pi/180
new_map = vr_map.value *np.nan
for nlong in range(0,len(vr_longs)):
    
    bravdalat = E_lat
    
    #compute the deltas at all latitudes
    delta_lat_E = abs(vr_lats.value - bravdalat.value)
    
    #compute the guassian weighting at each lat
    weights_ace = np.exp( -delta_lat_E*delta_lat_E/(2*sigma_rad*sigma_rad))
    
    #merge the MAS and assimilated data
    new_map[:,nlong] = ((1-weights_ace)*vr_map[:,nlong].value +
        weights_ace*post_wsalongs[nlong])
new_map= new_map *u.km/u.s

x,y = np.meshgrid(vr_longs.value * 180/np.pi,vr_lats.value*180/np.pi)
#plot the original map
plt.figure()
plt.pcolor(x,y,vr_map.value , vmin = 250, vmax=650)
plt.title('MAS')
plt.colorbar()

#plot the DA map
plt.figure()
plt.pcolor(x,y,new_map.value , vmin = 250, vmax=650)
plt.title('MAS + DA ')
plt.colorbar()

#plot the Br map
xb,yb = np.meshgrid(br_longs.value * 180/np.pi,br_lats.value*180/np.pi)
plt.figure()
plt.pcolor(xb,yb,br_map, vmin = -0.0001, vmax=0.0001)
plt.title('MAS Br ')
plt.colorbar()







# <codecell> Now run HUXt with AT THE COMET LAT

br_in = Hin.get_MAS_br_long_profile(cr_MAS, cometlat)  
                                                                          
#extract the data from the DA map at the comet lat
v_comet_lat = np.ones(len(vr_longs))
for i in range(0, len(vr_longs)):
    v_comet_lat[i] = np.interp(cometlat.to(u.rad).value, vr_lats.value, 
                               new_map[:,i].value)

#interpolate  to HUXt CarrLong grid
interp = interpolate.interp1d(vr_longs.value,
                              v_comet_lat, kind="nearest",
                              fill_value="extrapolate")
from scipy.ndimage.filters import uniform_filter1d
v_carrlong = uniform_filter1d(interp(phi128), size = 10) * u.km/u.s


model_DA = H.HUXt(v_boundary=v_carrlong, b_boundary=br_in, cr_num=cr, cr_lon_init=cr_lon_init, latitude = cometlat,
               simtime=27.27*u.day, dt_scale=4, r_min = r_min)
cme = H.ConeCME(t_launch=model_DA.time_out[934], longitude=-55*u.deg, latitude = -15*u.deg, width=58*u.deg, v=386*(u.km/u.s), thickness=5*u.solRad)
model_DA.solve([cme])
earth_series_DA = HA.get_observer_timeseries(model_DA, observer = 'Earth')

fig3, ax = HA.plot_earth_timeseries(model_DA, plot_omni = True)

#Model results previous to DA for comparison
vr_in = Hin.get_MAS_long_profile(cr_MAS, cometlat)
br_in = Hin.get_MAS_br_long_profile(cr_MAS, cometlat)

model = H.HUXt(v_boundary=vr_in, b_boundary=br_in, cr_num=cr, cr_lon_init=cr_lon_init, latitude = cometlat,
               simtime=27.27*u.day, dt_scale=4, r_min = r_min)
cme = H.ConeCME(t_launch=model.time_out[934], longitude=-55*u.deg, latitude = -15*u.deg, width=58*u.deg, v=386*(u.km/u.s), thickness=5*u.solRad)
model.solve([cme])
earth_series = HA.get_observer_timeseries(model, observer = 'Earth')

fig3, ax = HA.plot_earth_timeseries(model, plot_omni = True)

#plot it
#fig, ax = plt.subplots(1)
#ax.plot(earth_series['time'],earth_series['vsw'])
#HA.plot_earth_timeseries(model, plot_omni = True)

# #Plots Time Series to compare
fig, axs = plt.subplots(2, 1, figsize=(14, 7))
axs[0].plot(earth_series_DA['time'], earth_series_DA['vsw'], 'r', label='HUXt (With DA)')
axs[0].plot(earth_series['time'], earth_series['vsw'], 'k', label='HUXt (No DA)')
axs[0].set_ylim(250, 1000)
axs[0].set_xticklabels([])
axs[0].set_ylabel('Solar Wind Speed (km/s)')
axs[0].grid()
axs[1].plot(earth_series_DA['time'], np.sign(earth_series_DA['bpol'])*0.9, 'r.', markersize=2,  label='HUXt (With DA)')
axs[1].plot(earth_series['time'], np.sign(earth_series['bpol'])*0.95, 'k.', markersize=2, label='HUXt (No DA)')
axs[1].set_ylim(-1.1, 1.1)
axs[1].set_ylabel('B polarity')
axs[1].grid()
axs[1].set_xlabel('Date')
plt.legend(bbox_to_anchor = (1.0, 2), markerscale=6, loc = 'upper right')
fig.subplots_adjust(left=0.07, bottom=0.08, right=0.99, top=0.97, hspace=0.05)

#Get Omni data
import pandas as pd 

# making dataframe 
os.chdir('C:/Users/invate/OneDrive - University of Reading') 
df = pd.read_csv("Omni_APR2007.csv") 

# output the dataframe
print(df)

df['d'] = (pd.to_datetime(df['YYYY'] * 1000 + df['DOY'], format='%Y%j') +
            pd.to_timedelta(df['HR'], unit='h') + pd.to_timedelta(df['MN'], unit='m'))

#For Solar Wind Speed
id_bad = df['2'] == 99999.9
df.loc[id_bad, '2'] = np.NaN

#For B Polarity
id_bad = df['1'] == 9999.99
df.loc[id_bad, '1'] = np.NaN

fig, axs = plt.subplots(2, 1, figsize=(14, 7))
axs[0].plot(df['d'][577:8418], df['2'][577:8418], 'b', markersize=2, label='Observation (OMNI)')
axs[0].plot(earth_series_DA['time'], earth_series_DA['vsw'], 'r', label='HUXt (With DA)')
axs[0].plot(earth_series['time'], earth_series['vsw'], 'y', label='HUXt (No DA)')
axs[0].set_ylim(250, 1000)
axs[0].set_xticklabels([])
axs[0].set_ylabel('Solar Wind Speed (km/s)')
axs[0].grid()
axs[1].plot(df['d'][577:8418], -np.sign(df['1'][577:8418]), 'b.', markersize=2, label='Observation (OMNI)')
axs[1].plot(earth_series_DA['time'], np.sign(earth_series_DA['bpol'])*0.9, 'r.', markersize=2,  label='HUXt (With DA)')
axs[1].plot(earth_series['time'], np.sign(earth_series['bpol'])*0.95, 'y.', markersize=2, label='HUXt (No DA)')
axs[1].set_ylim(-1.1, 1.1)
axs[1].set_ylabel('B polarity')
axs[1].grid()
axs[1].set_xlabel('Date')
plt.legend(bbox_to_anchor = (1.0, 1), markerscale=6, loc = 'upper right')
fig.subplots_adjust(left=0.07, bottom=0.08, right=0.99, top=0.97, hspace=0.05)

#For Comet 1D Time series
Comet_series = HAC.get_observer_timeseries_JPL(model, '90000090')
Comet_series_DA = HAC.get_observer_timeseries_JPL(model_DA, '90000090')

fig, axs = plt.subplots(2, 1, figsize=(14, 7))
axs[0].plot(Comet_series_DA['time'], Comet_series_DA['vsw'], 'r', label='HUXt (With DA)')
axs[0].plot(Comet_series['time'], Comet_series['vsw'], 'k', label='HUXt (No DA)')
axs[0].set_ylim(250, 1000)
axs[0].set_xlim(Comet_series['time'][100], Comet_series['time'][1000])
axs[0].set_xticklabels([])
axs[0].set_ylabel('Solar Wind Speed (km/s)')
axs[0].grid()
axs[0].fill_betweenx(y=np.arange(250, 1000), x1=Comet_series['time'][479], x2=Comet_series['time'][750], color='gray', alpha=0.3)
axs[1].plot(Comet_series_DA['time'], np.sign(Comet_series_DA['bpol'])*0.9, 'r.', markersize=2,  label='HUXt (With DA)')
axs[1].plot(Comet_series['time'], np.sign(Comet_series['bpol'])*0.95, 'k.', markersize=2, label='HUXt (No DA)')
axs[1].set_ylim(-1.1, 1.1)
axs[1].set_xlim(Comet_series['time'][100], Comet_series['time'][1000])
axs[1].set_ylabel('B polarity')
axs[1].grid()
axs[1].set_xlabel('Date')
axs[1].fill_betweenx(y=np.arange(-1.1, 1.1), x1=Comet_series['time'][479], x2=Comet_series['time'][750], color='gray', alpha=0.3)
axs[0].legend(bbox_to_anchor = (1.0, 1), markerscale=6, loc = 'upper right')
fig.subplots_adjust(left=0.07, bottom=0.08, right=0.99, top=0.97, hspace=0.05)


#Plotting Comet on 2D Solar Wind
def Plot2D_HUXt_noDA(i):
    Comet = get_horizons_coord('90000090', model.time_init + model.time_out[i])
    fig, ax = HA.plot(model, model.time_out[i])
    ax.plot(Comet.lon.rad, Comet.radius.solRad, 'w*', markersize=20)
    ax.set_title('No DA', loc='center', fontsize = 20)
    fig, ax = HA.plot_bpol(model, model.time_out[i])
    ax.plot(Comet.lon.rad, Comet.radius.solRad, 'w*', markersize=20)
    ax.set_title('No DA', loc='center', fontsize = 20)


def Plot2D_HUXt_DA(i):
    Comet = get_horizons_coord('90000090', model_DA.time_init + model_DA.time_out[i])
    fig, ax = HA.plot(model_DA, model_DA.time_out[i])
    ax.plot(Comet.lon.rad, Comet.radius.solRad, 'w*', markersize=20)
    ax.set_title('DA', loc='center', fontsize = 20)
    fig, ax = HA.plot_bpol(model_DA, model_DA.time_out[i])
    ax.plot(Comet.lon.rad, Comet.radius.solRad, 'w*', markersize=20)
    ax.set_title('DA', loc='center', fontsize = 20)
    
def Plot_Ecliptic(i):
    Times = []
    Lats = []
    Comet = get_horizons_coord('90000090', model.time_init + model.time_out[i])
    Times.append((Comet.obstime.datetime))
    Lats.append((Comet.lat.deg))
    return Times, Lats

def Plot_Earth_Ecliptic(i):
    Times = []
    Lats = []
    Earth = get_horizons_coord('399', model.time_init + model.time_out[i])
    Times.append((Earth.obstime.datetime))
    Lats.append((Earth.lat.deg))
    return Times, Lats
