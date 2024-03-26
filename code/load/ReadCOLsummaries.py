# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 14:39:28 2017

A function to read and process the COL monthly summaries, obtained from Roger 
Brugge. This is an amateur weather network(?) which records thunder days.
@author: vy902033
"""

import pandas as pd
import os as os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt



path = os.environ['DBOX'] + 'Data\\Climate_Met_data\\RogerBrugge\\COL monthly summaries\\'

startyr=1946
stopyr=2017

col_specification =[(0, 4), (5, 13), (14,19),(20,44),(45,51),(52,57),(58,62),
                        (63,67),(68,73),(74,78),(79,85),(86,89),(90,94),(95,102),
                        (103,106),(107,111),(112,115),(116,119),(120,123),
                        (124,128),(129,134),(135,139),(140,149),(150,154),(155,159),
                        (160,164),(165,169)]
col_headers=['mmyy',	'sitecode',	'altitude',	'sitename', 	
             'T_meanmax','T_meanmin','T_highmax','T_lowmax','T_highmin','Tlowmin',
             'Rain_total', 'Rain_days0p2mm',	'Rain_days1mm','Rain_max',
             'N_airfrost','N_groundfrost','N_snowfall', 'N_snowlie','N_thunder',
             'N_haillt5mm','N_hailgt5mm','N_fog',
             'Sun_total', 'Sun_maxhours','SunlessDays',
             'T_soil30cm','T_soil100cm']

for year in range(startyr,stopyr+1):
    print(year)
    
    filepath=path+str(year)
    #using pandas fix width with a column specification   
    data = pd.read_fwf(filepath, colspecs=col_specification,header=None)
    
    #remove any dates that are nans (usually from blank lines)
    data = data[pd.notnull(data[0])]
    #redo the indices of the dataframe
    data.index = range(len(data))
    
    #remove datagaps, which use 'x' in the COL monthly data
    for i in range(4,27):
        data[i]=data[i].astype(str)
        data[i].replace({'x' : np.nan}, inplace=True)
        #data[i].replace({'vs' : np.nan}, inplace=True)
        #convert any strings to numeric
        data[i]=data[i].astype(float)
        
    #convert to datetime from format DDMM
    mm=np.floor(data[0]/100)
    #yy=data[0] - dd*100    
    dfdt=[]
    for i in range(0,len(data)):
            dfdt.append(datetime(year,int(mm[i]),15))
    #replace the index with the datetime objects
    data.index=dfdt
    

        
    if year==startyr:
        coldata=data.copy()
    else:
        coldata=coldata.append(data)

#sort out the headers
coldata.columns=col_headers
#order the data by time step
coldata.sort_index(inplace=True)

plt.figure(figsize=(8,6), dpi=100)
plt.plot(coldata['N_thunder'].resample('A',label='center').mean()/12)
#plt.plot(coldata['N_thunder'].rolling('26280H',min_periods=1000).mean(),'k')