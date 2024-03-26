"""
Methods for importing data from the OMNI spacecraft. Based on David Stansby's 
Wind routines provided as part of heliopy

Mathew Owens, 17/10/17


All data is publically available at ftp://spdf.gsfc.nasa.gov/pub/data/omni.
See https://wind.nasa.gov/data_sources.php for more information on different
data products.
"""
import os
import pandas as pd
import numpy as np

from heliopy.data import helper
from heliopy import config
#import heliopy.time as spacetime

data_dir = config['download_dir']
use_hdf = config['use_hdf']
omni_dir = os.path.join(data_dir, 'omni')
remote_omni_dir = 'ftp://spdf.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hourly/'


def hourly(starttime, endtime):
    """
    Import OMNI data products 

    Parameters
    ----------
    starttime : datetime
        Interval start time.
    endtime : datetime
        Interval end time.

    Returns
    -------
    data : DataFrame
    """
   
    data = []
    
    keys = {'Epoch': 'Time',
            'BX_GSE': 'Bx_gse',
            'BY_GSE': 'By_gse',
            'BZ_GSE': 'Bz_gse',
            'V' : 'vp_x',
            'PHI-V' : 'vp_y',
            'THETA-V': 'vp_z',
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
    badvalues = {'Bx_gse': 999.9,
                     'By_gse': 999.9,
                     'Bz_gse': 999.9,
                     'vp_x' : 9999.0,
                     'vp_y' : 999.900024,
                     'vp_z': 999.900024,
                     'n_p' : 999.900024,
                     'T_p' : 9999999.0,
                     'Bmag' : 999.9,
                     'ratio_a2p' : 9.999,
                     'pflux_1MeV' : 999999,
                     'pflux_2MeV' : 99999,
                     'pflux_4MeV' : 99999,
                     'pflux_10MeV' : 99999,
                     'pflux_30MeV' : 99999,
                     'pflux_60MeV' : 99999}
    
    
    #load the two data files for each year
    for yr in range(starttime.year,endtime.year+1,1):
        print('Loading OMNI ',str(yr))
        this_relative_dir = str(yr)
        # Absolute path to local directory for this data file
        local_dir = os.path.join(omni_dir, this_relative_dir)
        
        #there are 2 moni files per year. load them both
        filename1 = 'omni2_h0_mrg1hr_' +\
            str(yr) +\
            '0101_v01.cdf'
        hdfname = filename1[:-4] + '.hdf'
        hdfloc = os.path.join(local_dir, hdfname)
        if os.path.isfile(hdfloc):
            df = pd.read_hdf(hdfloc)
            data.append(df)
            continue

        helper.checkdir(local_dir)
        remote_url = remote_omni_dir + str(yr) +'/'
        cdf = helper.load(filename1, local_dir, remote_url, guessversion=True)


        df = helper.cdf2df(cdf,
                           index_key='Epoch',
                           keys=keys,
                           badvalues=badvalues)
        if use_hdf:
            df.to_hdf(hdfloc, 'mag', mode='w', format='f')
        data.append(df)
        
        #second file
        filename2 = 'omni2_h0_mrg1hr_' +\
            str(yr) +\
            '0701_v01.cdf'
        hdfname = filename2[:-4] + '.hdf'
        hdfloc = os.path.join(local_dir, hdfname)
        if os.path.isfile(hdfloc):
            df = pd.read_hdf(hdfloc)
            data.append(df)
            continue

        helper.checkdir(local_dir)
        remote_url = remote_omni_dir + str(yr) +'/'
        cdf = helper.load(filename2, local_dir, remote_url, guessversion=True)


        df = helper.cdf2df(cdf,
                           index_key='Epoch',
                           keys=keys,
                           badvalues=badvalues)
        if use_hdf:
            df.to_hdf(hdfloc, 'mag', mode='w', format='f')
        data.append(df)
    
    #compile the list of dataframes into a single dataframe and cut out the time period of interest
    tempdf=helper.timefilter(data, starttime, endtime)
    
    #convert from spherical coords to cartesian for V_p
    Vmag=tempdf['vp_x']
    phiV=tempdf['vp_y']*np.pi/180
    thetV=tempdf['vp_z']*np.pi/180
        
        
    tempdf['vp_x']=-Vmag*np.cos(thetV)*np.cos(phiV)
    tempdf['vp_y']=Vmag*np.cos(thetV)*np.sin(phiV)
    tempdf['vp_z']=Vmag*np.sin(thetV)
        
    return tempdf


