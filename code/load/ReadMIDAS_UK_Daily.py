# -*- coding: utf-8 -*-
"""
A script to read in the MIDAS UK Daily Weather observations.

Data from: http://catalogue.ceda.ac.uk/uuid/954d743d1c07d1dd034c131935db54e0
"""

import pandas as pd
import numpy as np
import os as os

MIDAS_dir = os.environ['DBOX'] + 'Data\\MIDAS\\UK_Hourly\\'
year=1922



MIDAS_filehead='midas_wxhrly_'

#load the header names
MIDAS_headers = MIDAS_dir + 'WH_Column_Headers.txt'
df_headers = pd.read_csv(MIDAS_headers)
colnames=list(df_headers)

#create the file path
MIDAS_file = MIDAS_dir + MIDAS_filehead + str(year) +'01-' + str(year) + '12.txt'

df = pd.read_csv(MIDAS_file,index_col=colnames[0],names=colnames)
                 
