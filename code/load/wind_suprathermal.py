# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 14:52:37 2017

@author: vy902033
"""
import os
import pandas as pd
import numpy as np

import heliopy.time as spacetime
from heliopy.data import helper
from heliopy import config

data_dir = config['download_dir']
use_hdf = config['use_hdf']
wind_dir = os.path.join(data_dir, 'wind')
remote_wind_dir = 'ftp://spdf.gsfc.nasa.gov/pub/data/wind/'


def threedp_elpd(starttime, endtime):
    """
    Import 'pm' wind data.

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
    # Directory relative to main WIND data directory
    relative_dir = os.path.join('3dp', '3dp_elpd')

    daylist = spacetime.daysplitinterval(starttime, endtime)
    data = []
    for day in daylist:
        date = day[0]
        this_relative_dir = os.path.join(relative_dir, str(day[0].year))
        # Absolute path to local directory for this data file
        local_dir = os.path.join(wind_dir, this_relative_dir)
        filename = 'wi_elpd_3dp_' +\
            str(date.year) +\
            str(date.month).zfill(2) +\
            str(date.day).zfill(2) +\
            '_v02.cdf'
        hdfname = filename[:-4] + 'hdf'
        hdfloc = os.path.join(local_dir, hdfname)
        if os.path.isfile(hdfloc):
            df = pd.read_hdf(hdfloc)
            data.append(df)
            continue

        helper.checkdir(local_dir)
        remote_url = remote_wind_dir + this_relative_dir
        cdf = helper.load(filename, local_dir, remote_url, guessversion=True)

        keys = {'MAGF': ['b_x', 'b_y', 'b_z'],
                'VSW': ['v_x', 'v_y', 'v_z'],
                'PANGLE': ['pangle1','pangle2','pangle3','pangle4','pangle5','pangle6','pangle7','pangle8'],
                'Epoch': 'Time'}
        df = helper.cdf2df(cdf, index_key='Epoch', keys=keys)
        if use_hdf:
            df.to_hdf(hdfloc, 'elpd', mode='w', format='f')
        data.append(df)

    return helper.timefilter(data, starttime, endtime)