# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 16:23:15 2021

@author: mathewjowens
"""
import os
import sys
huxtdir = os.path.abspath(os.environ['DBOX'] + 'python_repos\\HUXt\\code')

#a script to generate and save the HelioMAS 30rS ensembles
import astropy.units as u
import numpy as np

sys.path.append(huxtdir)
import huxt_inputs as Hin
import huxt as H
import huxt_ensembles as Hens

#==============================================================================
N=500 #number of ensemble members
lat_rot_sigma = 5*np.pi/180 #The standard deviation of the Gaussain from which the rotational perturbation is drawn
lat_dev_sigma = 2*np.pi/180 #The standard deviation of the Gaussain from which the linear latitudinal perturbation is drawn                          
long_dev_sigma = 2*np.pi/180 #The standard deviation of the Gaussain from which the linear longitudinal perturbation is drawn
r_in = 30*u.solRad #the radial distance of the speed map
cr_start = 2243
cr_end = 2249
#save location for ensemble h5 files.
savedir = ( os.path.abspath(os.environ['DBOX'] + 
                            'Papers_WIP\\_coauthor\\MattLang\\HelioMASEnsembles_python') )

var = 'vr'
#var = 'br'
#==============================================================================

os.chdir(huxtdir)
for nCR in range(cr_start,cr_end):
    cr = nCR
    
    if var == 'vr':
        vr_map, vr_lats, vr_longs = Hin.get_MAS_vrmap(cr)
    elif  var == 'br':
         vr_map, vr_lats, vr_longs = Hin.get_MAS_brmap(cr)
         
    #vr_map will be a single int if there is no data
    if not isinstance(vr_map, int):
    
        #Use the HUXt ephemeris data to get Earth lat over the CR
        model = H.HUXt(v_boundary=np.ones((128))*400* (u.km/u.s), simtime=27.27*u.day, 
                           dt_scale=4, cr_num= np.floor(cr), lon_out=0.0*u.deg, 
                           r_min=21.5*u.solRad, r_max=215*u.solRad)
        #retrieve a bodies position at each model timestep:
        earth = model.get_observer('earth')
        #get Earth lat as a function of longitude (not time)
        E_lat = np.interp(vr_longs.value,np.flipud(earth.lon_c.value),
                          np.flipud(earth.lat_c.value))*u.rad
        
        #plot the speed map       
        #plt.figure()
        #plt.pcolor(vr_longs.value*180/np.pi, vr_lats.value*180/np.pi, vr_map.value, 
        #           shading='auto',vmin=250, vmax=700)
        #plt.plot(vr_longs*180/np.pi,E_lat*180/np.pi,'r',label = 'Earth')
        #plt.plot(vr_longs*180/np.pi,E_lat*0,'k--')
        #plt.xlabel('Carrington Longitude [deg]')
        #plt.ylabel('Latitude [deg]')
        #plt.title('CR' + str(cr))
        #plt.legend()
        #cbar = plt.colorbar()
        #cbar.set_label(r'V$_{SW}$')
        
        
        #==============================================================================
        #generate the input ensemble
        #==============================================================================
        #generate the meshed grid
        phi, theta = np.meshgrid(vr_longs, vr_lats)
        
        vr_ensemble = Hens.generate_input_ensemble(phi*u.rad, theta*u.rad, vr_map, 
                                              reflats = E_lat, Nens = N,
                                              lat_rot_sigma = lat_rot_sigma, 
                                              lat_dev_sigma = lat_dev_sigma,
                                              long_dev_sigma = long_dev_sigma)
            
        #resample the ensemble to 128 longitude bins
        vr128_ensemble = np.ones((N,128))  
        dphi = 2*np.pi/128
        phi128 = np.linspace(dphi/2, 2*np.pi - dphi/2, 128)
        for i in range(0, N):
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
        import h5py
        
        if var == 'vr':
            h5f = h5py.File(savedir + '\\HelioMAS_CR' + str(cr) +'_vin_ensemble.h5', 'w')
            h5f.create_dataset('Vin_ensemble', data=vr128_ensemble)
        elif var == 'br':
            h5f = h5py.File(savedir + '\\HelioMAS_CR' + str(cr) +'_bin_ensemble.h5', 'w')
            h5f.create_dataset('Bin_ensemble', data=vr128_ensemble)            
        h5f.attrs['lat_rot_sigma'] = lat_rot_sigma
        h5f.attrs['lat_dev_sigma'] = lat_dev_sigma
        h5f.attrs['long_dev_sigma'] = long_dev_sigma
        filepath = 'get_MAS_vrmap(cr)'  #this is used only to identify the source files. 
        h5f.attrs['source_file'] = filepath
        h5f.attrs['r_in_rS'] = r_in
        h5f.attrs['Carrington_rotation'] = cr
        h5f.close()    
