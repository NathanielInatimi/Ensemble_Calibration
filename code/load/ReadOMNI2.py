# -*- coding: utf-8 -*-
"""
A reader for 1-hour OMNI2 data. Produces a dataframe and saves a HDF5 file

Does not download data. Those need to be obtained from:
ftp://cdaweb.gsfc.nasa.gov//pub/data/omni/omni_cdaweb/hourly

Option to add ephemeris data. 
These need to be obtained from https://omniweb.gsfc.nasa.gov/coho/helios/heli.html

The CDF reader is ripped from Heliopy, which is no longer supported

****pytables module must be installed*****

@author: mathewjowens
"""

#Read in OMNI data, save as HDF5
import pandas as pd
import numpy as np
import os
import cdflib
from datetime import datetime
import astropy.units as u
# these are custom functions which need to be in the same directory or the python path
os.chdir(os.path.abspath(os.environ['DBOX'] + '\\python'))
import helio_coords as hcoord
import helio_time as htime 



def zerototwopi(angles):
    """
    Function to constrain angles to the 0 - 2pi domain.

    :param angles: a numpy array of angles
    :return: a numpy array of angles
    """
    twopi = 2.0 * np.pi
    angles_out = angles
    a = -np.floor_divide(angles_out, twopi)
    angles_out = angles_out + (a * twopi)
    return angles_out


#dowload data from ftp://cdaweb.gsfc.nasa.gov//pub/data/omni/omni_cdaweb/hourly
def process_OMNI2(data_dir, starttime,endtime, include_ephemeris=True,
                  ephemeris_dir = os.environ['DBOX'] + 'Data\\ephemeris\\'):
    #set up the OMNI filenames and concatonate into a single dataframe
    #e.g., omnifile = 'D:\\Dropbox\\Data\\OMNI2_CDF\\2016\\omni2_h0_mrg1hr_20160101_v01.cdf'
    dir_name = 'OMNI2_CDF\\'
    data_header = '\\omni2_h0_mrg1hr_'
    data_tail1 = '0101_v01.cdf'
    data_tail2 = '0701_v01.cdf'
    
    years = np.arange(starttime.year,endtime.year+1,1)
    df_list = []
    for year in years:
        filepath = data_dir + dir_name + str(year) + data_header + str(year) + data_tail1
        if os.path.exists(filepath):
            print('Processing: ' + filepath)   
            tempdf = read_OMNI2file(filepath)
            df_list.append(tempdf)
        else:
            print('No file: ' + filepath)   
            
        filepath = data_dir + dir_name + str(year) + data_header + str(year) + data_tail2
        if os.path.exists(filepath):
            print('Processing: ' + filepath)  
            tempdf = read_OMNI2file(filepath)
            df_list.append(tempdf)
        else:
            print('No file: ' + filepath)   
            
            
    #combine the data frames
    data = pd.concat(df_list)
    
    #now reset the index
    data['datetime'] = data.index
    data.reset_index(drop=True, inplace=True)
    
    
    #add in the ephemeris data
    if include_ephemeris:
        data = add_ephermis_to_df(data,
                                  ephemeris_dir = ephemeris_dir)

    return data

def add_ephermis_to_df(df, ephemeris_dir = os.environ['DBOX'] + 'Data\\ephemeris\\'):
    #load the planetary ephemeris data from  https://omniweb.gsfc.nasa.gov/coho/helios/heli.html
    #Choose: Create File, Earth, Heliographic Inertial, 
    #Max time interval allowed, 1-day resolution.
    
    ephemeris_file = ephemeris_dir + 'Earth_HGI.lst'
    
    print('Adding ephemeris from: ' + ephemeris_file)
    
    pos_Earth = pd.read_csv(ephemeris_file,
                         skiprows = 1, delim_whitespace=True,
                         names=['year','doy',
                                'rad_au','HGI_lat','HGI_lon'])
    #convert to mjd
    pos_Earth['mjd'] = htime.doyyr2mjd(pos_Earth['doy'],pos_Earth['year'])
    
    df['r_au'] = np.interp(df['mjd'].to_numpy(),
                                  pos_Earth['mjd'].to_numpy(),pos_Earth['rad_au'].to_numpy(),
                                  left = np.nan, right = np.nan) * u.au
    df['HGI_lat'] = np.interp(df['mjd'].to_numpy(),
                                  pos_Earth['mjd'].to_numpy(),pos_Earth['HGI_lat'].to_numpy(),
                                  left = np.nan, right = np.nan) * u.deg

    #for longitude, unwrap to avoid the 0/2*pi issue
    pos_Earth_unwrap = np.unwrap(pos_Earth['HGI_lon'].to_numpy()*np.pi/180)
    df['HGI_lon'] = np.interp(df['mjd'].to_numpy(),
                                  pos_Earth['mjd'].to_numpy(), pos_Earth_unwrap,
                                  left = np.nan, right = np.nan) 
    df['HGI_lon'] = zerototwopi(df['HGI_lon'])

    #compute the Carrington longitude of Earth
    temp = hcoord.carringtonlatlong_earth(df['mjd'])
    df['Carr_lon'] = temp[:,1]
    
    return df

#read and process a single OMNI CDF file into a dataframe
def read_OMNI2file(filepath):
    cdf = cdflib.CDF(filepath)


    keys = {    'BX_GSE': 'Bx_gse',
                'BY_GSE': 'By_gse',
                'BZ_GSE': 'Bz_gse',
                'V' : 'V',
                'PHI-V' : 'phiV',
                'THETA-V': 'thetaV',
                'N' : 'n_p',
                'T' : 'T_p',
                'ABS_B' : 'Bmag',
                'Ratio': 'ratio_a2p',
                'R': 'SSN',
                'KP' : 'Kp',
                'DST' : 'Dst',
                'AE' : 'AE',
                'PR-FLX_1' : 'pflux_1MeV',
                'PR-FLX_2' : 'pflux_2MeV',
                'PR-FLX_4' : 'pflux_4MeV',
                'PR-FLX_10' : 'pflux_10MeV',
                'PR-FLX_30' : 'pflux_30MeV',
                'PR-FLX_60' : 'pflux_60MeV',
                'MFLX' : 'msphere'}
    
    #extract the whole CDF to a dataframe
    df = cdf2df(cdf)
    
    #retain only the vaariables listed in the keys dictionary, and rename accordingly
    data = pd.DataFrame(index = df.index)
    for key in keys:
    #    print(key)
    #    print(keys[key])
        data[keys[key]] = df[key]
        
    #V transform, sph to cart
    data['Vx_gse'] = - data['V'] * np.cos(data['thetaV']*np.pi/180) * np.cos(data['phiV']*np.pi/180)
    data['Vy_gse'] = data['V'] * np.cos(data['thetaV']*np.pi/180) * np.sin(data['phiV']*np.pi/180)
    data['Vz_gse'] = data['V'] * np.sin(data['thetaV']*np.pi/180) 
    
    #add mjd
    data['mjd'] = htime.datetime2mjd(data.index)
    
    #Add B and V rtn
    data = hcoord.df_gse2rtn(data)

    return data




def cdf2df(cdf):
    """
    Extract all CDF variables to a dataframe, using a datetime index
    
    This is a simplified version of David Stansby's heliopy.data.util routine
    """
    #get the epoch info
    index_info = cdf.varinq('Epoch')
    #check the length of the epoch data
    assert index_info['Last_Rec'] > 0
    
    #get the epoch data
    index = cdf.varget('Epoch')
    #check there aren't mulitple index series
    try:
        index = index[...][:, 0] #this should fail
    except IndexError:
            pass   
        
    index = cdflib.epochs.CDFepoch.breakdown(index, to_np=True)
    index_df = pd.DataFrame({'year': index[:, 0],
                             'month': index[:, 1],
                             'day': index[:, 2],
                             'hour': index[:, 3],
                             'minute': index[:, 4],
                             'second': index[:, 5],
                             'ms': index[:, 6],
                             })
    # Not all CDFs store pass milliseconds
    try:
        index_df['us'] = index[:, 7]
        index_df['ns'] = index[:, 8]
    except IndexError:
        pass
    index = pd.DatetimeIndex(pd.to_datetime(index_df), name='Time')
         
    df = pd.DataFrame(index=index)
    npoints = df.shape[0]
    
    var_list = _get_cdf_vars(cdf)
    keys = {}
    # Get mapping from each attr to sub-variables
    for cdf_key in var_list:
        if cdf_key == 'Epoch':
            keys[cdf_key] = 'Time'
        else:
            keys[cdf_key] = cdf_key
    # Remove index key, as we have already used it to create the index
    keys.pop('Epoch')
    # Remove keys for data that doesn't have the right shape to load in CDF
    # Mapping of keys to variable data
    vars = {cdf_key: cdf.varget(cdf_key) for cdf_key in keys.copy()}
    for cdf_key in keys:
        var = vars[cdf_key]
        if type(var) is np.ndarray:
            key_shape = var.shape
            if len(key_shape) == 0 or key_shape[0] != npoints:
                vars.pop(cdf_key)
        else:
            vars.pop(cdf_key)
    
    # Loop through each key and put data into the dataframe
    for cdf_key in vars:
        df_key = keys[cdf_key]
        # Get fill value for this key
        try:
            fillval = float(cdf.varattsget(cdf_key)['FILLVAL'])
        except KeyError:
            fillval = np.nan
    
        if isinstance(df_key, list):
            for i, subkey in enumerate(df_key):
                data = vars[cdf_key][...][:, i]
                data = _fillval_nan(data, fillval)
                df[subkey] = data
        else:
            # If ndims is 1, we just have a single column of data
            # If ndims is 2, have multiple columns of data under same key
            key_shape = vars[cdf_key].shape
            ndims = len(key_shape)
            if ndims == 1:
                data = vars[cdf_key][...]
                data = _fillval_nan(data, fillval)
                df[df_key] = data
            elif ndims == 2:
                for i in range(key_shape[1]):
                    data = vars[cdf_key][...][:, i]
                    data = _fillval_nan(data, fillval)
                    df[f'{df_key}_{i}'] = data
    return df

def cdfpeek(cdf_loc):
    """
    List all the variables present in a CDF file, along with their size.
    Parameters
    ----------
    cdf_loc : string
        Local location of the cdf file.
    """

    cdf_loc = os.path.expanduser(cdf_loc)
    cdf = cdflib.CDF(cdf_loc)
    info = cdf.cdf_info()
    for key in info:
        print('=' * len(key))
        print(key)
        print('=' * len(key))
        print(info[key])
        
def _get_cdf_vars(cdf):
    # Get list of all the variables in an open CDF file
    var_list = []
    cdf_info = cdf.cdf_info()
    for attr in list(cdf_info.keys()):
        if 'variable' in attr.lower() and len(cdf_info[attr]) > 0:
            for var in cdf_info[attr]:
                var_list += [var]

    return var_list


def _fillval_nan(data, fillval):
    try:
        data[data == fillval] = np.nan
    except ValueError:
        # This happens if we try and assign a NaN to an int type
        pass
    return data


# <codecell>  Example use of load functions.
    

starttime = datetime(1963, 1, 1, 0, 0, 0)
#starttime = datetime(1994, 1, 1, 0, 0, 0)
endtime = datetime(2023, 12, 30, 0, 0, 0)
data_dir = os.environ['DBOX'] + 'Data\\'
save_dir = os.environ['DBOX'] + 'Data_hdf5\\'
include_ephemeris = True
ephemeris_dir = os.environ['DBOX'] + 'Data\\ephemeris\\'

omni_1hour = process_OMNI2(data_dir, starttime,endtime, 
                           include_ephemeris=include_ephemeris,
                  ephemeris_dir = ephemeris_dir)

#omni_1hour = add_ephermis_to_df(omni_1hour, ephemeris_dir = ephemeris_dir)

# <codecell> save and load

#save the data
omni_1hour.to_hdf(save_dir + 'omni_1hour.h5', key = 'data', mode = 'w')

#load the data back in
omni_1hour = pd.read_hdf(save_dir + 'omni_1hour.h5')

#plot the data
import matplotlib.pyplot as plt
plt.plot(omni_1hour['datetime'],omni_1hour['V'])
