"""
Methods for importing data from the merged magnetic field and plasma data from 
the Ulysses spacecraft. Based on David Stansby's Wind routines provided as part
of heliopy

Mathew Owens, 2/11/17


All data is publically available at ftp://spdf.gsfc.nasa.gov/pub/data/ulysses/.
See https://wind.nasa.gov/data_sources.php for more information on different
data products.
"""
import os
import pandas as pd
import numpy as np

from heliopy.data import helper
from heliopy import config

data_dir = config['download_dir']
use_hdf = config['use_hdf']
ulysses_dir = os.path.join(data_dir, 'ulysses')
remote_ulysses_dir = 'ftp://spdf.gsfc.nasa.gov/pub/data/ulysses/merged/coho1hr_magplasma/'


def hourly(starttime, endtime):
    """
    Import merged plasma and magnetic field data products from Ulysses.

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
            'BR': 'Br_rtn',
            'BT': 'Bt_rtn',
            'BN': 'Bn_rtn',
            'plasmaFlowSpeed' : 'vp_r',
            'azimuthAngle' : 'vp_t',
            'elevAngle': 'vp_n',
            'protonDensity' : 'n_p',
            'protonTempLarge' : 'Tlarge_p',
            'protonTempSmall' : 'Tsmall_p',
            'ABS_B' : 'Bmag',
            'alphaDensity': 'ratio_a2p',
            'protonFlux1_LET' : 'pflux_1LET',
            'protonFlux2_LET' : 'pflux_2LET',
            'protonFlux3_LET' : 'pflux_3LET',
            'protonFlux4_LET' : 'pflux_4LET',
            'protonFlux5_LET' : 'pflux_5LET',
            'protonFlux1_HET' : 'pflux_1HET',
            'protonFlux2_HET' : 'pflux_2HET',
            'heliocentricDistance' : 'SC_R_AU',
            'heliographicLatitude' : 'SC_hlat',
            'heliographicLongitude' : 'SC_hlong'}
    
    badvalues = {'Br_rtn': -9.9999998e+30,
                     'Bt_rtn': -9.9999998e+30,
                     'Bn_rtn': -9.9999998e+30,
                     'vp_r' : -9.9999998e+30,
                     'vp_t' : -9.9999998e+30,
                     'vp_n': -9.9999998e+30,
                     'n_p' : -9.9999998e+30,
                     'Tlarge_p' : -9.9999998e+30,
                     'Tsmall_p' : -9.9999998e+30,
                     'Bmag' : -9.9999998e+30,
                     'ratio_a2p' : -9.9999998e+30,
                     'pflux_1LET' : -9.9999998e+30,
                     'pflux_2LET' : -9.9999998e+30,
                     'pflux_3LET' : -9.9999998e+30,
                     'pflux_4LET' : -9.9999998e+30,
                     'pflux_5LET' : -9.9999998e+30,
                     'pflux_1HET' : -9.9999998e+30,
                     'pflux_2HET' : -9.9999998e+30}

    #load the two data files for each year
    for yr in range(starttime.year,endtime.year+1,1):
        #find the last month required
        mn_end = 13
        if yr == endtime.year :
            mn_end = endtime.month+1
        
        for mn in range(1,mn_end,1):
            #print(str(yr) + ' ' +str(mn) )
            this_relative_dir = str(yr)
            # Absolute path to local directory for this data file
            local_dir = os.path.join(ulysses_dir, this_relative_dir)
            
            #create the month string with leading zeroes
            monstr=str(mn)
            if mn<10:
                monstr='0'+str(mn)
            
            #create the filename
            filename1 = 'uy_coho1hr_merged_mag_plasma_' +\
                str(yr) + monstr + '01_v01.cdf'
            hdfname = filename1[:-4] + '.hdf'
            hdfloc = os.path.join(local_dir, hdfname)
            if os.path.isfile(hdfloc):
                df = pd.read_hdf(hdfloc)
                data.append(df)
                continue
    
            helper.checkdir(local_dir)
            remote_url = remote_ulysses_dir + str(yr) +'/'
            cdf = helper.load(filename1, local_dir, remote_url, guessversion=True)
    
    
            df = helper.cdf2df(cdf,
                               index_key='Epoch',
                               keys=keys,
                               badvalues=badvalues)
            if use_hdf:
                df.to_hdf(hdfloc, 'mag', mode='w', format='f')
            data.append(df)

    
    #compile the list of dataframes into a single dataframe and cut out the time period of interest
    tempdf=helper.timefilter(data, starttime, endtime)
    
    #convert alpha density to alpha-to-proton ratio
    tempdf['ratio_a2p']=tempdf['n_p']/tempdf['ratio_a2p']
    
    #convert from spherical coords to cartesian for V_p
    Vmag=tempdf['vp_r']
    phiV=tempdf['vp_t']*np.pi/180
    thetV=tempdf['vp_n']*np.pi/180
    tempdf['vp_r']=Vmag*np.cos(thetV)*np.cos(phiV)
    tempdf['vp_t']=-Vmag*np.cos(thetV)*np.sin(phiV)
    tempdf['vp_n']=Vmag*np.sin(thetV)
        
    return tempdf


