# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 17:55:39 2021

@author: mathewjowens

A set of functions to load and process ACE 64-second merged mag and swe data. 
The data must first be downloaded in HDF4 format from the CDAweb FTP site:
    cdaweb.gsfc.nasa.gov/pub/data/ace/magswe/level2_hdf/64sec
    
"""


import pandas as pd
import numpy as np
import os
import pyhdf.HDF 
import pyhdf.V   
import pyhdf.VS 
import pyhdf.SD  
# these are custom functions which need to be in the same directory or the python path
import helio_time as htime



 
def read_HDF(filepath):
    """
    Read the contents of a HDF4 Vgroup to a dataframe
    """
    hdf = pyhdf.HDF.HDF(filepath)    
    vs = hdf.vstart()
    v  = hdf.vgstart()
    
    
    vg=v.attach(2)
    members = vg.tagrefs()    
    for tag, ref in members:
        #extract the actual data
        vd = vs.attach(ref)
        
        #get the data info
        nrecs, intmode, fields, size, name = vd.inquire()
        
        #convert from list to array
        temp = np.array(vd[:])
        
        #add each field to a dataframe
        i = 0
        df = pd.DataFrame()
        for field in fields:
            #print('Field "',field,'": ',vd[0][i])      
            df[field] = temp[:,i]     
            i = i + 1
    
        vd.detach() 
        
    vg.detach()    
    
    return df

def read_ACEmerged64sec_HDF(filepath):
    """
    Read and process a single annual ACE merged 64 sec HDF
    """
    
    df = read_HDF(filepath)
    
    keys = {    'B_rtn_r': 'Br',
                'B_rtn_t': 'Bt',
                'B_rtn_n': 'Bn',
                'V_rtn_r': 'Vr',
                'V_rtn_t': 'Vt',
                'V_rtn_n': 'Vn',
                'Np' : 'n_p',
                'Tp' : 'T_p',
                'Bmag' : 'Bmag',
                'Alpha_ratio': 'ratio_a2p',
                'pos_gse_x' : 'pos_gse_x',
                'pos_gse_y' : 'pos_gse_y',
                'pos_gse_z' : 'pos_gse_z',
                'pos_gsm_x' : 'pos_gsm_x',
                'pos_gsm_y' : 'pos_gsm_y',
                'pos_gsm_z' : 'pos_gsm_z',
                'pos_hs_x' : 'pos_hs_x',
                'pos_hs_y' : 'pos_hs_y',
                'pos_hs_z' : 'pos_hs_z',}
    
    
    #retain only the vaariables listed in the keys dictionary, and rename accordingly
    data = pd.DataFrame()
    for key in keys:
        #print(keys[key])
        data[keys[key]] = df[key]
        #remove fill vals
        if not "pos_" in key:
            mask = data[keys[key]] < -9999
            data.loc[mask, keys[key]] = np.nan
    #compute the datetime 
    data['mjd'] = htime.doyyr2mjd(df['fp_doy'].astype(float),df['year'].astype(int))    
    data['datetime'] = htime.mjd2datetime(data['mjd'].to_numpy())    
     
    return data

def process_ACEmerged64sec(data_dir, startyr,endyr):
    #set up the ACE filenames and concatonate into a single dataframe
    
    dir_name = 'ACE_hdf_64sec_merged\\'
    file_header = 'magswe_data_64sec_year'
    file_tail = '.hdf'
    
    
    years = np.arange(startyr,endyr+1,1)
    df_list = []
    for year in years:
        filepath = data_dir + dir_name + file_header +str(year) +file_tail
        if os.path.exists(filepath):
            print('Processing: ' + filepath)   
            tempdf = read_ACEmerged64sec_HDF(filepath)
            df_list.append(tempdf)
        else:
            print('No file: ' + filepath)                           
            
    #combine the data frames
    data = pd.concat(df_list)
    
    
    return data

# <codecell> Example of using the ACE merged 64 sec reader

startyr = 1998
endyr = 2021

data_dir = os.environ['DBOX'] + 'Data\\'
save_dir = os.environ['DBOX'] + 'Data_hdf5\\'

ace64sec =  process_ACEmerged64sec(data_dir, startyr,endyr)

#save the data
ace64sec.to_hdf(save_dir + 'ace64sec.h5', key = 'data', mode = 'w')


#load the data back in
ace64sec = pd.read_hdf(save_dir + 'ace64sec.h5')
   
    
    
    
    
    
   