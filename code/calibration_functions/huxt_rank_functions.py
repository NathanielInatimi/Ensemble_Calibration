"""
This script holds the functions which can be used to generate an ensemble HUXt forecast for a specified carrington rotation
Generate rank histograms for ensemble forcasts

"""

import datetime
import numpy as np
import pandas as pd
import xarray as xr
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import matplotlib as mpl
import os

import huxt as H
import huxt_analysis as HA
import huxt_inputs as Hin

import scipy.interpolate

def interpolate_vmap(velocity_map, lats, longs):
    # This Function generates an interpolated solution of velocity map ready for sub-earth path extraction

    # Generate coordinate grid
    lat,long = np.mgrid[:180, :360]

    # This is an array with the shape 2,X --> formatted coordinate grid for interpolation
    X2D = np.array([lat.flatten(), long.flatten()]).T  

    # Run interpolation on velocity map
    Int_2D_solution = scipy.interpolate.LinearNDInterpolator(X2D, velocity_map.flatten())

    return Int_2D_solution

def extract_interpolated_velocity_boundary(interpolated_map_solution, sub_earth_path, longitudes):
    # Function takes in a 2D interpolated MAS map and extracts wind speeds along given sub earth path across given longitudes

    velocity_boundary = interpolated_map_solution(sub_earth_path, longitudes)

    return velocity_boundary[0:-1]

def interpolate_and_resample(omni_data, forecast_series):
    # This function runs an interpolation algorithm on forecast output and outputs a resamples forecast series on the omni data timestep

    Int = scipy.interpolate.CubicSpline(forecast_series['datetime'], forecast_series['vsw'])
    
    data_time_axis = omni_data.index

    interpolated_forecast_output = Int(data_time_axis)

    return interpolated_forecast_output

def perturb_path(E_lat, theta_max, longs, phi_0, n=1):
    # This function generates perturbed latitudes at given longitudes

    phi = longs.value # Longtudes
    theta_e = E_lat.to(u.deg).value + 90 # Earth lattitude

    return theta_e + theta_max*np.sin(n*phi + phi_0)

def save_ens_to_cdf(ensemble_set, cr, variance_dev, no_members):
    # This function saves an ensmeble set for a given carrington rotation 

    # Convert list of panda dataframes into single xarray dataset along new index 'ensemble member'
    #ds_ = xr.concat([df.set_index('datetime').to_xarray() for df in ensemble_set], dim="ensemble_members")
    ds_ = xr.concat([df.to_xarray() for df in ensemble_set], dim="ensemble_members")
    
    # Create/find folder for set deviation and number of ensemble members
    var_dev_folder = f'vardev_{variance_dev}_{no_members}'
    
    var_dev_dir = f'C:\\Users\\ct832900\\Desktop\\Research_Code\\Ensemble_forecasting\\data\\Ensemble_Members\\{var_dev_folder}'

    if not os.path.exists(var_dev_dir):
        os.mkdir(var_dev_dir) 

    # Generate filename for ensemble set (marked by carrington rotation)
    ensemble_set_dir = f'carrot_{cr}_set.nc'
    output_file = f'C:\\Users\\ct832900\\Desktop\\Research_Code\\Ensemble_forecasting\\data\\Ensemble_Members\\{var_dev_folder}\\{ensemble_set_dir}'

    # Save the dataset to a NetCDF file
    ds_.to_netcdf(output_file)

def read_ens_cdf(cr, variance_dev, no_members):
    # This function reads in a netcdf file containing an ensemble set for a given carrington rotation and latudinal variance

    input_file = f'C:\\Users\\ct832900\\Desktop\\Research_Code\\Ensemble_forecasting\\data\\Ensemble_Members\\vardev_{variance_dev}_{no_members}carrot_{cr}_set.nc'
    
    # Read the NetCDF file into an xarray Dataset
    loaded_ds = xr.open_dataset(input_file)

    # Initialize an empty list to store DataFrames
    df_list = []

    # Loop through ensemble members xarray dimension, convert each member to pandas dataframe, append to list
    for i in range(no_members):
        df_list.append(loaded_ds.sel(ensemble_members = i).to_dataframe())

    return df_list

def get_earth_lat(forecast_window, cr, cr_lon_init):

    # HUXt model parameters
    dt_scale = 4
    
    # Use the HUXt ephemeris data to get Earth lat over the CR
    dummymodel = H.HUXt(v_boundary=np.ones(128)*400*(u.km/u.s), simtime=forecast_window, dt_scale=dt_scale, cr_num=cr, cr_lon_init=cr_lon_init, lon_out=0.0*u.deg)

    # Retrieve a bodies position at each model timestep:
    earth = dummymodel.get_observer('earth')

    # Get average Earth lat
    E_lat = np.nanmean(earth.lat_c)

    return E_lat

def generate_ensemble(params):   

    cr = params[0]
    forecast_window = params[1]
    no_members = params[2]
    theta_dev_mean = params[3]
    theta_dev_var = params[4]

    rng = np.random.default_rng() # Initialise random number generator

    E_lat = get_earth_lat(forecast_window, cr=cr, cr_lon_init=360*u.deg) # Get Earth latitude for sub earth paths

    # HUXt model parameters
    dt_scale = 4
    r_min = 30*u.solRad

    # Getting MAS map bits for interpolation and sub earth paths
    MAS_vr_map, MAS_vr_longs, MAS_vr_lats = Hin.get_MAS_vr_map(cr = cr)
    interpolated_MAS_vmap = interpolate_vmap(MAS_vr_map.value, MAS_vr_lats.value, MAS_vr_longs.value)

    # Initialise ensemble member list
    ensemble_members = []

    for i in range(no_members):

        # Generate perturbed latitude path
        perturbed_path = perturb_path(E_lat, rng.uniform(theta_dev_mean, theta_dev_var, 1), MAS_vr_longs, rng.uniform(0,360,1), 1) # Random amplitude and phase, Fixed wave number

        # Generate perturbed velocity boundary
        velocity_boundary = extract_interpolated_velocity_boundary(interpolated_MAS_vmap, perturbed_path, MAS_vr_longs.to(u.deg))

        # Initialise instance of HUXt model
        model = H.HUXt(v_boundary=velocity_boundary*(u.km/u.s), cr_num=cr, simtime=forecast_window, dt_scale=dt_scale, r_min=r_min)
        model.solve([])

        # Extract Earth time series dataFrame
        df_earth_series = HA.get_observer_timeseries(model, observer = 'Earth')
        df_earth_series = df_earth_series.rename(columns = {'time':'datetime'}) # rename time column to match omni data name (for ease later on)
        df_earth_series.set_index('datetime')
        df_temp = df_earth_series.copy()

        ensemble_members.append(df_temp)
    
    save_ens_to_cdf(ensemble_members, cr, theta_dev_var, no_members)

    return
    

def gen_ranked_ensemble(ensemble_members, omni_data): 

    vsw_list = [] # Initialise list for forecast output

    # Prepare data for rank comparison
    omni_chunk = omni_data.loc[ensemble_members[0]['datetime'].iloc[0]:ensemble_members[0]['datetime'].iloc[-1]]
    #omni_chunk = omni_data.loc[ensemble_members[0].index[0]:ensemble_members[0].index[-1]]
    omni_chunk = omni_chunk.dropna(subset = ['V']) # Remove rows with NaN values

    # Interpolate and resample forecast output onto OMNI data time ste[]
    for vsw in ensemble_members:
        vsw_int = interpolate_and_resample(omni_chunk, vsw)
        vsw_list.append(vsw_int)

    # Compare ensemble member output arrays to omni data 
    vsw_arr = np.array(vsw_list)
    ranked_forecast_boolean = np.array([vsw < omni_chunk['V'] for vsw in vsw_arr])
    summed_ranks = np.sum(ranked_forecast_boolean, axis = 0)

    return summed_ranks

def generate_rank_histogram(cr_list, max_pert_mean, no_members):
    
    ranked_forecasts = []
    for cr in cr_list:
        ensemble_set = read_ens_cdf(cr, max_pert_mean, no_members)
        ranked_forecasts.append(gen_ranked_ensemble(ensemble_set, omni_1hour))

    ranked_forecast_arr = np.concatenate(ranked_forecasts) + 1

    return ranked_forecast_arr
