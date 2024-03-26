import datetime
import numpy as np
import pandas as pd
import xarray as xr
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd

import matplotlib as mpl
#from datetime import datetime
import os

import huxt as H
import huxt_analysis as HA
import huxt_inputs as Hin

from scipy import signal
import scipy.interpolate
from scipy import stats
import scipy.optimize as scopt

import random


def generate_catagorical_forecast(forecast, threshold):
    """
    Computes the hit rate and false alarm rate of an input forecast series.

    Parameters:
    - forecast (numpy array): array containing forecast time series values
    - threshold (float): Wind speed threshold value of forecast (above which forecast is catagorised as a 'hit')

    Returns:
    - catagorical_forecast (numpy array): array containing catagorical/binary forecast
    """
    catagorical_forecast = forecast > threshold

    return catagorical_forecast.astype(int)


def gen_probabilistic_forecast(ensemble_members, threshold, ensemble_size): 
    """
    Converts a set of ensemble members into a probabilistic forecast based on a threshold.

    Parameters:
    - ensemble_members (list): List containing separate ensemble member timeseries within each element
    - threshold (float): Wind speed threshold value of forecast (above which forecast is catagorised as a 'hit')
    - ensemble_size (int): Number of individual ensemble members
    
    Returns:
    - probabilistic_forecast (numpy array): probabilistic forecast (between 0 and 1).
    """

    # Compare individual ensemble member forecast timeseries to chosen threshold
    ranked_forecasts = np.array([forecast > threshold for forecast in ensemble_members])

    # Sum along zeroth axis to produce array where each element counts the number of ensemble members above threshold for given timestep
    summed_ranks = np.sum(ranked_forecasts, axis = 0) 

    return summed_ranks/ensemble_size

def gen_catagorical_from_ensemble(ensemble_members, event_threshold, probability_threshold):
    """
    Computes the hit rate and false alarm rate of an input ensembe set based on a given probability threshold.

    Parameters:
    - ensemble_members (list): List containing separate ensemble member timeseries within each element
    - event_threshold (float): Wind speed threshold value of forecast (above which forecast is catagorised as a 'hit')
    - probability_threshold (float): probability threshold for converting probabilistic forecast into a catagorical forecast (above which forecast is catagorised as a 'hit')

    Returns:
    - catagorical_forecast (numpy array): array containing catagorical/binary forecast
    """

    probabilistic_forecast = gen_probabilistic_forecast(ensemble_members=ensemble_members, threshold=event_threshold, ensemble_size=len(ensemble_members))
    catagorical_forecast = probabilistic_forecast > probability_threshold # Convert probabilistic forecast into catagorical forecast using a prob threshold

    return catagorical_forecast.astype(int)

def compute_hit_false_rate(observed_data, forecast, threshold):
    """
    Computes the hit rate and false alarm rate of an input forecast series.

    Parameters:
    - observed_data (numpy array): Array containing observed timeseries values
    - forecast (numpy array): Array containing forecast timeseries values
    - threshold (float): Wind speed threshold value for evaluating forecast (above which a forecast value is catagorised as a 'hit')

    Returns:
    - (true_positive_rate, false_positive_rate) (tuple): lists of hit rate and false alarm rate sorted within a tuple
    """
    # Catagorising/Binarising observed wind speed
    observed_positives = observed_data > threshold
    observed_negatives = observed_data < threshold

    # Create a catagorical/binary forecast
    forecast_positives = forecast > threshold
    forecast_negatives = forecast < threshold

    # Pick out numbers of true positives/false negatives/true negatives/false positives
    true_positives = np.sum(forecast_positives[observed_positives])
    false_negatives = np.sum(forecast_negatives[observed_positives])

    true_negatives = np.sum(forecast_negatives[observed_negatives])
    false_positives = np.sum(forecast_positives[observed_negatives])

    # Compute hit rate and false alarm rate
    true_positive_rate = true_positives / (true_positives + false_negatives)
    false_positive_rate = false_positives / (false_positives + true_negatives)

    return (true_positive_rate, false_positive_rate)

def compute_ensemble_hit_false_rate(observed_data, ensemble_members, event_threshold, probability_threshold):
    """
    Computes the hit rate and false alarm rate of an input ensemble set.
    
    Parameters:
    - ensemble_members (list): List containing separate ensemble member timeseries within each element
    - event_threshold (float): Wind speed threshold value of forecast (above which forecast is catagorised as a 'hit')
    - probability_threshold (float): probability threshold for converting probabilistic forecast into a catagorical forecast (above which forecast is catagorised as a 'hit')

    Returns:
    - (true_positive_rate, false_positive_rate) (tuple): lists of hit rate and false alarm rate sorted within a tuple
    """
    # Catagorising/Binarising observed wind speed
    observed_positives = observed_data > event_threshold
    observed_negatives = observed_data < event_threshold

    # Create a catagorical/binary forecast from ensemble set
    forecast_positives = gen_catagorical_from_ensemble(ensemble_members=ensemble_members, event_threshold=event_threshold, probability_threshold=probability_threshold)
    forecast_negatives =(~forecast_positives.astype(bool)).astype(int)

    # Pick out numbers of true positives/false negatives/true negatives/false positives
    true_positives = np.sum(forecast_positives[observed_positives])
    false_negatives = np.sum(forecast_negatives[observed_positives])

    true_negatives = np.sum(forecast_negatives[observed_negatives])
    false_positives = np.sum(forecast_positives[observed_negatives])

    # Compute TPR and FPR, whilst taking into account edge cases where there are no events
    if (true_positives + false_negatives) == 0:
        true_positive_rate = np.nan
        false_positive_rate = false_positives / (false_positives + true_negatives)

    elif (false_positives + true_negatives) == 0:
        true_positive_rate = true_positives / (true_positives + false_negatives)
        false_positive_rate = np.nan
    
    elif (true_positives + false_negatives) and (false_positives + true_negatives) == 0:
        true_positive_rate = np.nan
        false_positive_rate = np.nan
    else:
        true_positive_rate = true_positives / (true_positives + false_negatives)
        false_positive_rate = false_positives / (false_positives + true_negatives)

 

    return (true_positive_rate, false_positive_rate)

def generate_roc_curve(forecast, observed_data, threshold_range, threshold_num):
    """
    Generates ROC curve elements in form of two lists of true positive rates and false positive rates at varying thresholds.

    Parameters:
    - observed_data (numpy array): Array containing observed timeseries values
    - forecast (numpy array): Array containing forecast timeseries values
    - threshold_range (tuple): (min, max) values of thresholds for computing true/false positive rates
    - threshold_num (int): number of increments between min/max thresholds to compute

    Returns:
    - roc_curve (tuple): two lists of hit rate and false alarm rate at different thresholds
    """
    threshold_list = np.linspace(threshold_range[0], threshold_range[1], threshold_num) # Generate list of thresholds to loop through

    roc_curve = []
    for threshold in threshold_list:
        roc_curve.append(compute_hit_false_rate(observed_data, forecast, threshold))
    
    return roc_curve

def generate_roc_curve_from_ensemble(ensemble_members, observed_data, threshold_range, threshold_num, probability_threshold):
    """
    Generates ROC curve elements in form of two lists of true positive rates and false positive rates at varying thresholds.

    Parameters:
    - ensemble_members (list): List containing separate ensemble member timeseries within each element
    - observed_data (numpy array): Array containing observed timeseries values
    - threshold_range (tuple): (min, max) values of thresholds for computing true/false positive rates
    - threshold_num (int): number of increments between min/max thresholds to compute
    - probability_threshold (float): probability threshold for generating a catagorical forecast (above which forecast is catagorised as a 'hit')

    Returns:
    - roc_curve (tuple): two lists of hit rate and false alarm rate at different thresholds
    """
    threshold_list = np.linspace(threshold_range[0], threshold_range[1], threshold_num) # Generate list of thresholds to loop through

    roc_curve = []
    for threshold in threshold_list:
        roc_curve.append(compute_ensemble_hit_false_rate(observed_data, ensemble_members=ensemble_members, event_threshold=threshold, probability_threshold=probability_threshold))
    
    return roc_curve

def plot_roc_curve(roc_curve):

    rc_hit_rate, rc_false_alarm_rate = zip(*roc_curve)

    plt.figure(dpi=100, figsize=(5,5))
    plt.plot(rc_false_alarm_rate, rc_hit_rate, marker = 'o', markersize = 2, linewidth = 1)
    plt.plot([0,1], [0,1], linestyle = '--', lw = 1, color = 'black')
    plt.title('ROC curve')
    plt.xlabel('false alarm rate')
    plt.ylabel('hit rate')
    plt.show()

    return

def compute_brier_score(forecast, observed_data, threshold):
    """
    Calculates the Brier Score

    Parameters:
    - forecast (numpy array): Array containing forecast timeseries values
    - observed_data (numpy array): Array containing observed timeseries values
    - threshold (float): Wind speed threshold value for evaluating forecast (above which a forecast value is catagorised as a 'hit')

    Returns:
    - brier_score (float): Brier Score.
    """

    y_obs = observed_data > threshold  # catagorising/binarizing observed events using threshold of interest
    y_forecast = forecast > threshold # catagorising/binarizing forecast using threshold of interest

    return np.mean((y_forecast.astype(float) - y_obs.astype(float)) ** 2)

def compute_brier_score_probabilistic(ensemble_members, observed_data, threshold, ensemble_size):
    """
    Calculates the Brier Score for a probabilistic forecast generated by an ensemble

    Parameters:
    - ensemble_members (list): List containing separate ensemble member timeseries within each element
    - observed_data (numpy array): Array containing observed timeseries values
    - threshold (float): Wind speed threshold value for evaluating forecast (above which a forecast value is catagorised as a 'hit')
    - ensemble_size (int): Number of individual ensemble members

    Returns:
    - brier_score (float): Brier Score.
    """

    y_obs = observed_data > threshold  # catagorising/binarizing observed events using threshold of interest
    y_forecast = gen_probabilistic_forecast(ensemble_members, threshold, ensemble_size) 

    return np.mean((y_forecast - y_obs) ** 2)

def plot_brier_scores(forecast, observed_data, threshold_range, threshold_num):

    
    velocity_threshold_arr = np.linspace(threshold_range[0], threshold_range[1], threshold_num)
    brier_scores_arr = np.array([compute_brier_score(forecast=forecast, observed_data=observed_data, threshold=thresh) for thresh in velocity_threshold_arr])

    plt.plot(velocity_threshold_arr, brier_scores_arr, marker = 'o', markersize = 2, linewidth = 1)
    plt.xlabel('Wind Speed Threshold [km/s]')
    plt.ylabel('Brier Score')
    plt.show()

    return

def plot_brier_scores_probabilistic(ensemble_members, observed_data, threshold_range, threshold_num):

    
    velocity_threshold_arr = np.linspace(threshold_range[0], threshold_range[1], threshold_num)
    brier_scores_arr = np.array([compute_brier_score_probabilistic(ensemble_members=ensemble_members, observed_data=observed_data, threshold=thresh, ensemble_size=len(ensemble_members)) for thresh in velocity_threshold_arr])

    plt.plot(velocity_threshold_arr, brier_scores_arr, marker = 'o', markersize = 2, linewidth = 1)
    plt.xlabel('Wind Speed Threshold [km/s]')
    plt.ylabel('Brier Score')
    plt.show()

    return

def read_ens_cdf_var(cr, var_dev, no_members):
    
    input_file = f'C:\\Users\\ct832900\\Desktop\\Research_Code\\Ensemble_forecasting\\data\\Ensemble_Members\\vardev_{var_dev}_{no_members}\\carrot_{cr}_set.nc'

    # Read the NetCDF file into an xarray Dataset
    loaded_ds = xr.open_dataset(input_file)

    # Initialize an empty list to store DataFrames
    df_list = []

    # Loop through ensemble members xarray dimension, convert each member to pandas dataframe, append to list
    for i in range(no_members):
        df_list.append(loaded_ds.sel(ensemble_members = i).to_dataframe())

    return df_list

def interpolate_and_resample(observed_data_index, forecast_index, forecast):
# This function runs an interpolation algorithm on forecast output and outputs a resamples forecast series on the omni data timestep

    Int = scipy.interpolate.CubicSpline(forecast_index, forecast)

    data_time_axis = observed_data_index

    interpolated_forecast_output = Int(data_time_axis)

    return interpolated_forecast_output

def generate_deterministic_forecast(cr, forecast_window, dt_scale, r_min):

    # Use the HUXt ephemeris data to get Earth lat over the CR
    dummymodel = H.HUXt(v_boundary=np.ones(128)*400*(u.km/u.s), simtime=forecast_window, dt_scale=dt_scale, cr_num=cr, lon_out=0.0*u.deg)

    # Retrieve a bodies position at each model timestep:
    earth = dummymodel.get_observer('earth')

    # Get average Earth lat
    E_lat = np.nanmean(earth.lat_c)

    # Get MAS profile for specific carrington rotation and Earth latitude calculated previously
    v_mas = Hin.get_MAS_long_profile(cr, E_lat.to(u.deg))

    model = H.HUXt(v_boundary=v_mas, cr_num=cr, simtime=forecast_window, dt_scale=dt_scale, r_min=r_min)
    model.solve([])

    # Extract Earth time series dataFrame
    df_earth_series = HA.get_observer_timeseries(model, observer = 'Earth')
    df_earth_series = df_earth_series.rename(columns = {'time':'datetime'})
    df_earth_series.set_index('datetime')
    
    return df_earth_series


def read_and_prepare_ensemble_data(file_details, observed_data):

    # Unpack ensemble parameters
    cr = file_details[0]
    ensemble_size = file_details[1]
    lat_var = file_details[2]
    
    # Read in ensemble set for specified latitude variance
    ensemble_members = read_ens_cdf_var(cr=cr, var_dev=lat_var, no_members=ensemble_size) 
    ensemble_members_dti = [df.set_index('datetime') for df in ensemble_members] # reindex each ensemble member by time to make later analysis easier

    # Isolate data during forecast window
    data_chunk = observed_data.loc[pd.to_datetime(ensemble_members_dti[0].index[0]):pd.to_datetime(ensemble_members_dti[0].index[-1])]
    data_chunk = data_chunk.dropna(subset = ['V']) # Remove rows with NaN values

    ensemble_members_reindexed = [interpolate_and_resample(observed_data_index=data_chunk.index, forecast_index=df.index, forecast=df['vsw']) 
                                  for df in ensemble_members_dti]
    
    return ensemble_members_reindexed, data_chunk

