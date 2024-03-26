# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:06:43 2023

a collection of sunspot functions

@author: vy902033
"""

import numpy as np
import pandas as pd
import datetime as datetime
from astropy.time import Time
import os as os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import urllib.request
import requests
from bs4 import BeautifulSoup
from scipy.interpolate import griddata

import conversions.helio_time as htime
import conversions.helio_coords as hcoords
import plotting.mplot as mplot


# <codecell> Data loaders

def _setup_dirs_():
    cwd = os.path.abspath(os.path.dirname(__file__))
    root = os.path.dirname(cwd)
    datapath = os.path.join(root,'Data')
    
    return datapath

def LoadSSN(filepath='null', download_now = False, sminpath = None):
    #(dowload from http://www.sidc.be/silso/DATA/SN_m_tot_V2.0.csv)
    
        
    if filepath == 'null':
        datapath =  _setup_dirs_()
        filepath = os.path.join(datapath,'SN_m_tot_V2.0.csv')
            
    if download_now:
        urllib.request.urlretrieve('http://www.sidc.be/silso/DATA/SN_m_tot_V2.0.csv', filepath)
        

    
        
    col_specification =[(0, 4), (5, 7), (8,16),(17,23),(24,29),(30,35)]
    ssn_df=pd.read_fwf(filepath, colspecs=col_specification,header=None)
    dfdt=np.empty_like(ssn_df[0], dtype=datetime.datetime)
    dfTime=np.empty_like(ssn_df[0], dtype=Time)
    for i in range(0,len(ssn_df)):
        date_string = str(int(ssn_df[0][i])) + '-' + str(int(ssn_df[1][i])) + '-15' 
        
        dfdt[i] = datetime.datetime(int(ssn_df[0][i]),int(ssn_df[1][i]),15)
        dfTime[i] = Time(date_string, format='iso')
        
        
    #replace the index with the datetime objects
    ssn_df['datetime']=dfdt  
    ssn_df['ssn']=ssn_df[3]
    ssn_df['mjd'] = htime.datetime2mjd(dfdt)


    #delete the unwanted columns
    ssn_df.drop(0,axis=1,inplace=True)
    ssn_df.drop(1,axis=1,inplace=True)
    ssn_df.drop(2,axis=1,inplace=True)
    ssn_df.drop(3,axis=1,inplace=True)
    ssn_df.drop(4,axis=1,inplace=True)
    ssn_df.drop(5,axis=1,inplace=True)
    
    #add the 13-month running smooth
    window = 13*30
    temp = ssn_df.rolling(str(window)+'D', on='datetime').mean()
    ssn_df['smooth'] = np.interp(ssn_df['mjd'],temp['mjd'],temp['ssn'],
                                              left =np.nan, right =np.nan)
    #drop the incomplete window at the start
    mask = ssn_df['mjd'] < ssn_df['mjd'].loc[0] + window/2
    ssn_df.loc[mask, 'smooth'] = np.nan
    
    
    #add in a solar activity index, which normalises the cycle magnitude
    #approx solar cycle length, in months
    nwindow = int(11*365)
    
    #find maximum value in a 1-solar cycle bin centred on current time
    temp_max = ssn_df.rolling(str(nwindow) +'D', on='datetime').max()
    temp_mean = ssn_df.rolling(str(nwindow) +'D', on='datetime').mean()
    ssn_df['rollingmax'] = np.interp(ssn_df['mjd'],temp_mean['mjd'],temp_max['smooth'],
                                              left =np.nan, right =np.nan)
    
    #drop the incomplete window at the start
    mask = ssn_df['mjd'] < ssn_df['mjd'].loc[0] + nwindow/2
    ssn_df.loc[mask, 'rollingmax'] = np.nan
    
    #fill the max value at the end of the series
    fillval_end = ssn_df['rollingmax'].dropna().values[-1]
    fillval_start = ssn_df['rollingmax'].dropna().values[0]
    #drop the incomplete window at the start
    mask = ssn_df['mjd'] < (ssn_df['mjd'].loc[0] + 11*365.25/2)
    ssn_df.loc[mask, 'rollingmax'] = fillval_start
    #fill forward too
    ssn_df['rollingmax'] = ssn_df['rollingmax'].fillna(fillval_end) 
    
    #create a Solar Activity Index, as SSN normalised to the max smoothed value in
    #1-sc window centred on current tim
    ssn_df['sai'] = ssn_df['smooth']/ssn_df['rollingmax']
    
    
    
    # #compute phase
    # if sminpath is None:
    #     datapath =  _setup_dirs_()
    #     sminpath = os.path.join(datapath,'SolarMinTimes.txt')
    # smin_df = LoadSolarMinTimes(filepath = sminpath)
    # solarmin_mjd = smin_df['mjd'].to_numpy()
    # ssn_df['phase'] = solar_cycle_phase(ssn_df['mjd'], solarmin_mjd)
    
    #add fractional year.
    ssn_df['fracyear'] = htime.mjd2fracyear(ssn_df['mjd'].to_numpy())
    
    return ssn_df


def LoadSSNyr(filepath='null', download_now = False, sminpath = None):
    #downloads and process the annual SILSO SSN back to 1700
    
    #(dowload from https://www.sidc.be/SILSO/DATA/SN_y_tot_V2.0.csv)
        
    if filepath == 'null':
        datapath =  _setup_dirs_()
        filepath = os.path.join(datapath,'SN_y_tot_V2.0.csv')
            
    if download_now:
        urllib.request.urlretrieve('https://www.sidc.be/SILSO/DATA/SN_y_tot_V2.0.csv', filepath)
        

    col_specification =[(0, 4), (8, 13)]
    ssn_df=pd.read_fwf(filepath, colspecs=col_specification,header=None)
    dfdt=np.empty_like(ssn_df[0], dtype=datetime.datetime)
    dfTime=np.empty_like(ssn_df[0], dtype=Time)
    for i in range(0,len(ssn_df)):
        date_string = str(int(ssn_df[0][i])) + '-7' + '-1' 
        
        dfdt[i] = datetime.datetime(int(ssn_df[0][i]),7,1)
        dfTime[i] = Time(date_string, format='iso')
        
        
    #replace the index with the datetime objects
    ssn_df['datetime']=dfdt
    ssn_df['ssn']=ssn_df[1]
    ssn_df['mjd'] = htime.datetime2mjd(dfdt)


    #delete the unwanted columns
    ssn_df.drop(0,axis=1,inplace=True)
    ssn_df.drop(1,axis=1,inplace=True)

    
    #add in a solar activity index, which normalises the cycle magnitude
    #approx solar cycle length, in months
    nwindow = int(11*365)
    
    #find maximum value in a 1-solar cycle bin centred on current time
    temp_max = ssn_df.rolling(str(nwindow) +'D', on='datetime').max()
    temp_mean = ssn_df.rolling(str(nwindow) +'D', on='datetime').mean()
    ssn_df['rollingmax'] = np.interp(ssn_df['mjd'],temp_mean['mjd'],temp_max['ssn'],
                                              left =np.nan, right =np.nan)
    
    #drop the incomplete window at the start
    mask = ssn_df['mjd'] < ssn_df['mjd'].loc[0] + nwindow/2
    ssn_df.loc[mask, 'rollingmax'] = np.nan
    
    #fill the max value at the end of the series
    fillval_end = ssn_df['rollingmax'].dropna().values[-1]
    fillval_start = ssn_df['rollingmax'].dropna().values[0]
    #drop the incomplete window at the start
    mask = ssn_df['mjd'] < (ssn_df['mjd'].loc[0] + 11*365.25/2)
    ssn_df.loc[mask, 'rollingmax'] = fillval_start
    #fill forward too
    ssn_df['rollingmax'] = ssn_df['rollingmax'].fillna(fillval_end) 
    
    #create a Solar Activity Index, as SSN normalised to the max smoothed value in
    #1-sc window centred on current tim
    ssn_df['sai'] = ssn_df['ssn']/ssn_df['rollingmax']
    
    
    
    #compute phase
    if sminpath is None:
        datapath =  _setup_dirs_()
        sminpath = os.path.join(datapath,'SolarMinTimes.txt')
    smin_df = LoadSolarMinTimes(filepath = sminpath)
    solarmin_mjd = smin_df['mjd'].to_numpy()
    ssn_df['phase'] = solar_cycle_phase(ssn_df['mjd'], solarmin_mjd)
    
    #add fractional year.
    ssn_df['fracyear'] = htime.mjd2fracyear(ssn_df['mjd'].to_numpy())
    
    
    return ssn_df



def LoadBBgsn(filepath='null', download_now = False):
    #(dowload from https://www.sidc.be/SILSO/DATA/GroupNumber/GNbb2_y.txt)
        
    if filepath == 'null' and download_now:
        datapath =  _setup_dirs_()
        urllib.request.urlretrieve('https://www.sidc.be/SILSO/DATA/GroupNumber/GNbb2_y.txt',
                                   os.path.join(datapath, 'GNbb2_y.txt'))
        
    if filepath == 'null':
        datapath =  _setup_dirs_()
        filepath = os.path.join(datapath,'GNbb2_y.txt')
        
    gsn_df = pd.read_csv(filepath,
                             delim_whitespace=True,
                             names=['fracyear','gsn', 'gsn_std'])
    doy = (gsn_df['fracyear'] - np.floor(gsn_df['fracyear']))*364 + 1
    doy = doy.to_numpy()
    yr = np.floor(gsn_df['fracyear']).to_numpy()
    yr=yr.astype(int)
    gsn_df['mjd'] = htime.doyyr2mjd(doy,yr)
    
    gsn_df['ssn'] = gsn_df['gsn']*12.28
        
        
    
    return gsn_df

def LoadSolarMinTimes(filepath = None):
    #get the solar minimum times
    
    if filepath is None:
        datapath =  _setup_dirs_()
        filepath = os.path.join(datapath,'SolarMinTimes.txt')
        
    solarmintimes_df = pd.read_csv(filepath,
                         delim_whitespace=True,
                         names=['fracyear','cyclenum'])
    doy = (solarmintimes_df['fracyear'] - np.floor(solarmintimes_df['fracyear']))*364 + 1
    doy = doy.to_numpy()
    yr = np.floor(solarmintimes_df['fracyear']).to_numpy()
    yr=yr.astype(int)
    solarmintimes_df['mjd'] = htime.doyyr2mjd(doy,yr)
    
    #solarmintimes_df['datetime'] = pd.to_datetime(solarmintimes_df['fracyear'], format='%Y', errors='coerce')
    solarmintimes_df['datetime'] = htime.mjd2datetime(solarmintimes_df['mjd'].to_numpy())
    
    return solarmintimes_df

def load_usoskin_osf(data_dir = None, download_now = False):

    if data_dir is None:
        data_dir =  _setup_dirs_()
        
    osffilepath = os.path.join(data_dir,'Usoskin2023_osf.dat')
    ssnfilepath = os.path.join(data_dir,'Usoskin2023_ssn.dat')
    
    if download_now:
        urllib.request.urlretrieve('http://cdsarc.u-strasbg.fr/ftp/J/A+A/649/A141/osf.dat',
                                    osffilepath)
        urllib.request.urlretrieve('http://cdsarc.u-strasbg.fr/ftp/J/A+A/649/A141/osn.dat',
                                    ssnfilepath)


    usoskin_osf_df = pd.read_csv(osffilepath,
                         delim_whitespace=True,
                         names=['intyear','osf', 'osf_1-sigma','osf_smooth22'])
    
    usoskin_ssn_df = pd.read_csv(ssnfilepath,
                         delim_whitespace=True,
                         names=['intyear','ssn', 'ssn_1-sigma','ssn_smooth22'])
    
    #copy the SSN data to the OSF df
    usoskin_osf_df['ssn'] = usoskin_ssn_df['ssn']
    usoskin_osf_df['ssn_1-sigma'] = usoskin_ssn_df['ssn_1-sigma']
    usoskin_osf_df['ssn_smooth22'] = usoskin_ssn_df['ssn_smooth22']
    
    
    #create fracyear and MJD columns
    usoskin_osf_df['fracyear'] = usoskin_osf_df['intyear'] + 0.5
    
    doy = (usoskin_osf_df['fracyear'] - np.floor(usoskin_osf_df['fracyear']))*364 + 1
    doy = doy.to_numpy()
    yr = np.floor(usoskin_osf_df['fracyear']).to_numpy()
    yr=yr.astype(int)
    usoskin_osf_df['mjd'] = htime.doyyr2mjd(doy,yr)
    
    return usoskin_osf_df


def load_oulu_nm(filepath = None):
    #data have to be generated at hourly resolution from the online form at:
    #https://cosmicrays.oulu.fi/
    
    
    if filepath is None:
        data_dir =  _setup_dirs_()
        filepath = os.path.join(data_dir, 'OULU.dat')
        
    # Define a custom converter function for the date and time columns
    def datetime_converter(date_str, time_str):
        datetime_str = f"{date_str} {time_str}"
        return datetime.datetime.strptime(datetime_str, "%Y.%m.%d %H:%M:%S")
    
   
    
    # Read the ASCII file into a Pandas DataFrame
    df = pd.read_csv(filepath, skiprows=16, sep=" ", usecols=[0, 1, 2, 3, 4, 5], 
                     names=["Date", "time", "fracyear", "Uncorrected_Count_Rates", 
                            "counts", "pressure"], 
                     skipfooter=3, engine='python')
    
    # Combine date and time columns into a single datetime column
    df["datetime"] = df.apply(lambda row: datetime_converter(row["Date"], row["time"]), axis=1)
    
    # Drop the separate date and time columns
    df.drop(columns=["Date", "time"], inplace=True)

    #add the mjd 
    df["Time"] = Time(df['datetime'])
    df["mjd"]= df.apply(lambda row: row["Time"].mjd, axis=1)
    

    return df


def load_oulu_phi(filepath = None, download_now = True):
    #function to load the heliospheric modulation potential estimated from neutron
    #monitor data, at: https://cosmicrays.oulu.fi/phi/Phi_Table_2017.txt
    
    if download_now:
        datapath =  _setup_dirs_()
        urllib.request.urlretrieve('https://cosmicrays.oulu.fi/phi/Phi_Table_2017.txt',
                                   os.path.join(datapath,'Phi_Table_2017.txt'))
      
    if filepath is None:
        data_dir =  _setup_dirs_()
        filepath = os.path.join(data_dir, 'Phi_Table_2017.txt')
        
    valid_lines=[]
    # Open the file and read it line by line
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            # Remove non-breaking spaces and other non-alphanumeric characters
            cleaned_line = ''.join(char for char in line if char.isalnum() or char.isspace())
            
            # Use the built-in split function to split by whitespace
            fields = cleaned_line.strip().split()
            
            # Check if the line has the expected number of columns
            if len(fields) == 14:
                # Replace 'N/A' with NaN
                fields = [np.nan if val == 'N/A' else val for val in fields]
                valid_lines.append(fields)

    # Create a DataFrame from the valid lines
    df = pd.DataFrame(valid_lines, columns=['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual'])
    
    # Convert numeric columns to numeric data types (Jan to Annual)
    df[['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual']] = df[['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual']].apply(pd.to_numeric, errors='coerce')
    
    
    # You can set the index if needed
    # df.set_index(0, inplace=True)
    df = df.drop(0, axis=0)
    df = df.reset_index(drop=True)
    
    df = df.drop('Annual', axis=1)
    
    #convert from Year - Month grid to a TIME SERIES
    #===============================================

    # Melt the DataFrame to reshape it
    melted_df = pd.melt(df, id_vars=['Year'], value_name='phi_NM', var_name='Month')
    
    
    # Dictionary to map month abbreviations to their numeric values
    month_abbr_to_num = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    
    # Extract the year and month separately
    melted_df['Year'] = melted_df['Year'].astype(int)
    melted_df['Month'] = melted_df['Month'].map(month_abbr_to_num)
    
    # Create a new column 'Date' by combining Year and Month
    melted_df['Date'] = pd.to_datetime(melted_df[['Year', 'Month']].assign(day=15), format='%Y-%m-%d')
    
    # Sort the DataFrame by 'Date'
    melted_df.sort_values(by='Date', inplace=True)
    

    #convert to mjd and datetime
    for i in range(0,len(melted_df)):
        melted_df.loc[i,'mjd'] = htime.datetime2mjd(melted_df.loc[i,'Date'].to_pydatetime())
    melted_df['datetime'] = htime.mjd2datetime(melted_df['mjd'].to_numpy())
    melted_df['fracyear'] = htime.mjd2fracyear(melted_df['mjd'].to_numpy())   
    
    
    #drop the unneeded columns
    melted_df = melted_df.drop('Year', axis=1)
    melted_df = melted_df.drop('Month', axis=1)
    melted_df = melted_df.drop('Date', axis=1)

    return melted_df  


def load_oulu_phi_extended(filepath = None, download_now = True):
    #function to load the heliospheric modulation potential estimated from neutron
    #monitor + ionisation chamber(?) data, at: https://cosmicrays.oulu.fi/phi/Phi_mon.txt
    
    if download_now:
        datapath =  _setup_dirs_()
        urllib.request.urlretrieve('https://cosmicrays.oulu.fi/phi/Phi_mon.txt',
                                   os.path.join(datapath,'Phi_mon.txt'))
      
    if filepath is None:
        data_dir = _setup_dirs_()
        filepath = os.path.join(data_dir, 'Phi_mon.txt')
        
    valid_lines=[]
    # Open the file and read it line by line
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            # Remove non-breaking spaces and other non-alphanumeric characters
            cleaned_line = ''.join(char for char in line if char.isalnum() or char.isspace())
            
            # Use the built-in split function to split by whitespace
            fields = cleaned_line.strip().split()
            
            # Check if the line has the expected number of columns
            if len(fields) == 14:
                # Replace 'N/A' with NaN
                fields = [np.nan if val == 'N/A' else val for val in fields]
                valid_lines.append(fields)
                
       

    # Create a DataFrame from the valid lines
    df = pd.DataFrame(valid_lines[2:], columns=['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual'])
    
    # Convert numeric columns to numeric data types (Jan to Annual)
    df[['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual']] = df[['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual']].apply(pd.to_numeric, errors='coerce')
    
    
    # You can set the index if needed
    # df.set_index(0, inplace=True)
    df = df.drop(0, axis=0)
    df = df.reset_index(drop=True)
    
    df = df.drop('Annual', axis=1)
    
    #convert from Year - Month grid to a TIME SERIES
    #===============================================

    # Melt the DataFrame to reshape it
    melted_df = pd.melt(df, id_vars=['Year'], value_name='phi_NM', var_name='Month')
    
    
    # Dictionary to map month abbreviations to their numeric values
    month_abbr_to_num = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    
    # Extract the year and month separately
    melted_df['Year'] = melted_df['Year'].astype(int)
    melted_df['Month'] = melted_df['Month'].map(month_abbr_to_num)
    
    # Create a new column 'Date' by combining Year and Month
    melted_df['Date'] = pd.to_datetime(melted_df[['Year', 'Month']].assign(day=15), format='%Y-%m-%d')
    
    # Sort the DataFrame by 'Date'
    melted_df.sort_values(by='Date', inplace=True)
    

    #convert to mjd and datetime
    for i in range(0,len(melted_df)):
        melted_df.loc[i,'mjd'] = htime.datetime2mjd(melted_df.loc[i,'Date'].to_pydatetime())
    melted_df['datetime'] = htime.mjd2datetime(melted_df['mjd'].to_numpy())
    melted_df['fracyear'] = htime.mjd2fracyear(melted_df['mjd'].to_numpy())   
    
    
    #drop the unneeded columns
    melted_df = melted_df.drop('Year', axis=1)
    melted_df = melted_df.drop('Month', axis=1)
    melted_df = melted_df.drop('Date', axis=1)

    return melted_df     


def load_14C_phi(filepath = None):
    #function to load the heliospheric modulation potential estimated from 14C
    #supplied by Ilya from Brehm, NatGeo, 2021
     
    if filepath is None:
        data_dir = _setup_dirs_()
        filepath = os.path.join(data_dir, 'HMP_14C_Brehm21_FI_all.res')

    
    df = pd.read_csv(filepath, sep=" ", 
                         names=["Year", "phi_14C", "phi_14C_sigma", "phi_14C_SSA1"])
    #make the year the mid point.
    df['Year'] = df['Year'] +0.5
    #add mjd and datetime
    df['mjd'] = htime.fracyear2mjd(df['Year'].to_numpy())
    df['datetime'] = htime.mjd2datetime(df['mjd'].to_numpy())
    
    return df

def load_wsa_hcstilt(filepath = None, download_now = True):    
    
    if download_now:
        url = 'http://wso.stanford.edu/Tilts.html'
        response = requests.get(url)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Use BeautifulSoup to navigate and extract data from the HTML
        else:
            print("Failed to retrieve the webpage.")

    if filepath is None:
        data_dir = _setup_dirs_()
        filepath = os.path.join(data_dir, 'WSO - Computed _Tilt_ Angle of the HCS.html')
        
    with open(filepath,"r") as f:
        text = f.read()
    soup = BeautifulSoup(text, 'html.parser')
    
    # Extract the text
    data_text = soup.get_text()
    
    # Split the text into rows
    rows = data_text.strip().split('\n')
    
    # Process each row and see if it contains data
    actual_data = []
    for row in rows:
        columns = row.split()  # Split by whitespace
        
        if len(columns) > 1:
            if columns[0] == 'CR':
                actual_data.append(columns)
    
    #now go through the data and add it to an array
    hcstilt = pd.DataFrame()
    for i, row in enumerate(actual_data):
        hcstilt.loc[i,'CR'] = row[1]
        hcstilt.loc[i,'R_av'] = row[4]
        hcstilt.loc[i,'R_n'] = row[5]
        hcstilt.loc[i,'R_s'] = row[6]
        hcstilt.loc[i,'L_av'] = row[7]
        hcstilt.loc[i,'L_n'] = row[8]
        hcstilt.loc[i,'L_s'] = row[9]
        
    for column in hcstilt.columns:
        hcstilt[column] = pd.to_numeric(hcstilt[column], errors='coerce')
        
    #add MJD and datetime
    hcstilt['mjd'] = htime.crnum2mjd(hcstilt['CR'].to_numpy() +0.5)
    hcstilt['datetime'] = htime.mjd2datetime(hcstilt['mjd'].to_numpy())
    #hcstilt['fracyear'] = htime.mjd2fracyear(hcstilt['mjd'].to_numpy())
    
    return hcstilt


def PlotAlternateCycles(solarmintimes_df = None):
    
    if solarmintimes_df is None:
        solarmintimes_df = LoadSolarMinTimes()
        
    for i in range(6, len(solarmintimes_df['mjd'])-1):
        yy=plt.gca().get_ylim()
        if (solarmintimes_df['cyclenum'][i] % 2) == 0:
            rect = patches.Rectangle((solarmintimes_df['datetime'][i],yy[0]),
                                     solarmintimes_df['datetime'][i+1]-solarmintimes_df['datetime'][i],
                                     yy[1]-yy[0],edgecolor='none',facecolor='lightgrey',zorder=0)
            plt.gca().add_patch(rect)       
# <codecell> General solar cycle functions
def compute_phase_SPE(mjd, param, solarmin_mjd = None,
                      nphase = 11, plotnow = True):
    
    #computes the SPE of an input parameter over the solar cycle
    
    #get solar min times if they're not provided
    if solarmin_mjd is None:
        df = LoadSolarMinTimes()
        solarmin_mjd = df['mjd']
    
    
    #first compute the phase relative to each cycle start
    Nt = len(mjd)
    phases_scs = np.empty((Nt,len(solarmin_mjd)-1))
    for i in range(0,len(solarmin_mjd)-1):
        cyclestart = solarmin_mjd[i]
        cycleend = solarmin_mjd[i+1]
        
        cyclelength = cycleend -cyclestart
        phases_scs[:,i] = 2*np.pi*(mjd - cyclestart) / cyclelength   
    
        
    # compute the phase SPEs
    #==========================
    dphase = 2*np.pi/nphase
    SPE_phase = np.arange(-dphase, 2*np.pi + dphase +0.0001, dphase)
    Nphase = len(SPE_phase)
    
    #compute loss for each cycle
    SPE = np.empty((Nphase, len(solarmin_mjd)-1 ))*np.nan
    for i in range(0, len(solarmin_mjd)-1):
        SPE[:,i] = np.interp(SPE_phase, phases_scs[:,i], 
                                 param, left = np.nan, right = np.nan)  
    
    #compute average and std across all cycles
    SPE_AVG = np.empty((Nphase))
    SPE_STD = np.empty((Nphase)) 
    
    for i in range(0,Nphase):
        SPE_AVG[i] = np.nanmean(SPE[i,:])
        SPE_STD[i] = np.nanstd(SPE[i,:])
        
    
    if plotnow:
        
        
        plt.figure(figsize = (6,6))
        
        ax = plt.subplot(1,2,1)
        ax.plot(SPE_phase/(2*np.pi), SPE)
        #ax.plot(SPE_phase, SPE_LOSS_AVG ,'k')
        ax.set_ylabel('Param', fontsize = 12)
        yy = ax.get_ylim()
        ax.text(0.04, 0.93, '(a)', transform=plt.gca().transAxes, fontsize=12)
     
        ax = plt.subplot(1,2,2)
        ax.plot(SPE_phase/(2*np.pi), SPE_AVG)
        #ax.plot(SPE_phase, SPE_LOSS_AVG ,'k')
        ax.set_ylabel('Param', fontsize = 12)
        yy = ax.get_ylim()
        ax.text(0.04, 0.93, '(a)', transform=plt.gca().transAxes, fontsize=12)
        
    return SPE_phase, SPE_AVG, SPE_STD, SPE 


def solar_cycle_phase(mjd, solarmin_mjd = None):
    #computes the solar cycle phase, between 0 and 2 pi
    #solar min times can be specified or the defaults loaded.
    
    if solarmin_mjd is None:
        smin_df = LoadSolarMinTimes()
        solarmin_mjd = smin_df['mjd']
    
    #make a scalar input into an array
    if not hasattr(mjd, "__len__"):
        mjd = [mjd]
    
    Nt = len(mjd)
    phase = np.empty((Nt))
    #compute the solar cycle phase for the OSF times
    for n in range(0,Nt): 
        for i in range(0,len(solarmin_mjd)-1):
            cyclestart = solarmin_mjd[i]
            cycleend = solarmin_mjd[i+1]
            
            cyclelength = cycleend -cyclestart
            if (mjd[n] >= cyclestart) & (mjd[n]  < cycleend):
                phase[n]  = 2*np.pi*(mjd[n]  - cyclestart)/cyclelength
    return phase

# <codecell> OSF model functions



def ssn_analogue_forecast(ssn_df, solarmin_mjd = None, 
                          nphase = 11*12, peak_ssn = 140, plotnow = True,
                          exportdata = False, savepath = None):
    
    #computes teh normalised sunspot number profile with phase. Produces analogue
    #forecast plot 
    
    if solarmin_mjd is None:
        smin_df = LoadSolarMinTimes()
        solarmin_mjd = smin_df['mjd']
        
    Nt = len(ssn_df['mjd'])
    phases_scs = np.empty((Nt,len(solarmin_mjd)-1))
    for i in range(0,len(solarmin_mjd)-1):
        cyclestart = solarmin_mjd[i]
        cycleend = solarmin_mjd[i+1]
        
        cyclelength = cycleend -cyclestart
        phases_scs[:,i] = 2*np.pi*(ssn_df['mjd'] - cyclestart) / cyclelength   

       
    # compute the phase SPEs
    #==========================
    dphase = 2*np.pi/(nphase)
    SPE_phase = np.arange(-dphase, 2*np.pi + dphase +0.0001, dphase)
    Nphase = len(SPE_phase)

    #compute the SPE of the smoothed ssn 
    SPE_SSN = np.empty((Nphase, len(solarmin_mjd)-2 )) * np.nan
    SPE_SSN_NORM = np.empty((Nphase, len(solarmin_mjd)-2 )) * np.nan
    for i in range(0, len(solarmin_mjd)-2):
        SPE_SSN[:,i] = np.interp(SPE_phase, phases_scs[:,i], 
                                 ssn_df['smooth'], left = np.nan, right = np.nan)  
        
        #find the max 13-month smooth value in the cycle
        mask = (ssn_df['mjd'] >= solarmin_mjd[i]) & (ssn_df['mjd'] < solarmin_mjd[i+1]) 
        if mask.any():
            ssn_max = np.nanmax(ssn_df['smooth'].loc[mask])
            SPE_SSN_NORM[:,i] = np.interp(SPE_phase, phases_scs[:,i], 
                                     ssn_df['smooth']/ssn_max, left = np.nan, right = np.nan)  

    #compute average and std across all cycles
    SPE_SSN_AVG = np.empty((Nphase))
    SPE_SSN_STD = np.empty((Nphase))   

    SPE_SSN_NORM_AVG = np.empty((Nphase))
    SPE_SSN_NORM_STD = np.empty((Nphase)) 

    for i in range(0,Nphase):
        SPE_SSN_AVG[i] = np.nanmean(SPE_SSN[i,:])
        SPE_SSN_STD[i] = np.nanstd(SPE_SSN[i,:])
        SPE_SSN_NORM_AVG[i] = np.nanmean(SPE_SSN_NORM[i,:])
        SPE_SSN_NORM_STD[i] = np.nanstd(SPE_SSN_NORM[i,:])
        
    #find the most recent data
    mask = mask = (ssn_df['mjd'] >= solarmin_mjd[len(solarmin_mjd)-2])

    

    if plotnow:
        plt.figure()
        plt.plot(SPE_phase/(2*np.pi), SPE_SSN)
        plt.plot(SPE_phase/(2*np.pi), SPE_SSN_AVG ,'k')
        plt.xlabel('Solar cycle phase', fontsize = 12)
        plt.ylabel('Sunspot number')

    
        plt.figure()
        plt.plot(SPE_phase/(2*np.pi), SPE_SSN_NORM)
        plt.plot(SPE_phase/(2*np.pi), SPE_SSN_NORM_AVG ,'k')
        plt.xlabel('Solar cycle phase')
        plt.ylabel('Normalised sunspot number')
    
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 3)
        ax = fig.add_subplot(gs[0, 0:2])
        ax.plot(ssn_df['datetime'], ssn_df['ssn'], 'r', label = 'Monthly')
        ax.plot(ssn_df['datetime'], ssn_df['smooth'], 'k', label = '13-m smooth')
        ax.set_ylabel('Sunspot number', fontsize = 14)
        ax.set_xlabel('Date', fontsize = 14)
        yy=[0, 350]
        ax.set_ylim(yy)
        ax.legend()
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
    
        ax = fig.add_subplot(gs[0, 2])
        mplot.plotconfidbands(11*SPE_phase/(2*np.pi),SPE_SSN_NORM.T * peak_ssn)
        #add the current cycle
        ax.plot(11*ssn_df['phase'].loc[mask]/(2*np.pi), ssn_df['ssn'].loc[mask],'r', label = 'SC25: Monthly')
        ax.plot(11*ssn_df['phase'].loc[mask]/(2*np.pi), ssn_df['smooth'].loc[mask],'k', label = 'SC25: 13-m smooth')
        ax.set_xlabel('Year through cycle\n' +'(all cycles normalised to 11 yr)', fontsize = 14)
        #plt.ylabel('13-month smoothed sunspot number', fontsize = 14)
        ax.legend(facecolor='silver')
        ax.set_ylim(yy)
        # ax.plot([0,0],yy,'k--')
    
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
    
        fig.tight_layout()
    
    #saev the data
    if exportdata:
        if savepath is None:
            savepath = os.path.join(os.environ['DBOX'] ,'Data','SC25_analogue_forecast.dat')
            
        df = pd.DataFrame()
        sc25_startyr = smin_df.loc[len(smin_df)-2,'fracyear']
        df['fracyear'] = 11*SPE_phase/(2*np.pi) + sc25_startyr
        
        confid_intervals = [5,10,32]
        confid_ts = mplot.getconfidintervals(SPE_SSN_NORM.T * peak_ssn, confid_intervals)  
        df['ssn_median'] = confid_ts[:,0]
        df['ssn_5percent'] = confid_ts[:,1]
        df['ssn_10percent'] = confid_ts[:,2]
        df['ssn_32percent'] = confid_ts[:,3]
        df['ssn_68percent'] = confid_ts[:,4]
        df['ssn_90percent'] = confid_ts[:,5]
        df['ssn_95percent'] = confid_ts[:,6]
        
        df.to_csv(savepath, index=False)
        
        
    return SPE_phase, SPE_SSN_NORM_AVG, SPE_SSN_NORM_STD

def compute_osf_source(ssn, source_ref = 1):
    if source_ref == 1:
        SSNphi = 0.006*12
        source = SSNphi * 50 * (0.234*pow(2.67 + ssn, 0.540) - 0.00153) # OEA2016a
    elif source_ref == 2:
        source = 10*0.94 * ( 0.06 * 30 * np.tanh (0.00675 * ssn *0.6) + 0.683) + 0.01# LEA2013
    #source = SSNphi * (ssn +10) 
    return source

def compute_osfloss_SPE(mjd, osf, ssn, solarmin_mjd = None,
                      nphase = 11, plotnow = True, source_ref = 1):
    
    #computes the fractional loss term required to match the given source and OSF series
    #these must be on the same time step
    
    #get solar min times if they're not provided
    if solarmin_mjd is None:
        df = LoadSolarMinTimes()
        solarmin_mjd = df['mjd']
    
    source = compute_osf_source(ssn, source_ref = source_ref)
    
       
    #first compute the phase relative to each cycle start
    Nt = len(mjd)
    phases_scs = np.empty((Nt,len(solarmin_mjd)-1))
    for i in range(0,len(solarmin_mjd)-1):
        cyclestart = solarmin_mjd[i]
        cycleend = solarmin_mjd[i+1]
        
        cyclelength = cycleend -cyclestart
        phases_scs[:,i] = 2*np.pi*(mjd - cyclestart) / cyclelength   

    #compute the required loss rate comnpared with OSF_geo
    dOSF = np.empty((Nt)) * np.nan
    fracloss = np.empty((Nt)) * np.nan
    for t in range(0, Nt - 1):
        dOSF[t] = osf[t+1] - osf[t]
        fracloss[t] = (source[t] - dOSF[t]) / osf[t]
        
    # compute the phase SPEs
    #==========================
    dphase = 2*np.pi/nphase
    SPE_phase = np.arange(-dphase, 2*np.pi + dphase +0.0001, dphase)
    Nphase = len(SPE_phase)
    
    #compute loss for each cycle
    SPE_LOSS = np.empty((Nphase, len(solarmin_mjd)-1 ))*np.nan
    SPE_SSN = np.empty((Nphase, len(solarmin_mjd)-1 ))*np.nan
    SPE_SSN_NORM = np.empty((Nphase, len(solarmin_mjd)-1 ))*np.nan
    for i in range(0, len(solarmin_mjd)-1):
        SPE_LOSS[:,i] = np.interp(SPE_phase, phases_scs[:,i], 
                                 fracloss, left = np.nan, right = np.nan)  
        SPE_SSN[:,i] = np.interp(SPE_phase, phases_scs[:,i], 
                                 ssn, left = np.nan, right = np.nan)  
        
        #find the max 13-month smooth value in the cycle
        mask = (mjd >= solarmin_mjd[i]) & (mjd < solarmin_mjd[i+1]) 
        if mask.any():
            ssn_max = np.nanmax(ssn[mask])
            SPE_SSN_NORM[:,i] = np.interp(SPE_phase, phases_scs[:,i], 
                                     ssn/ssn_max, left = np.nan, right = np.nan)  

    
    #compute average and std across all cycles
    SPE_LOSS_AVG = np.empty((Nphase))
    SPE_LOSS_STD = np.empty((Nphase)) 
    
    for i in range(0,Nphase):
        SPE_LOSS_AVG[i] = np.nanmean(SPE_LOSS[i,:])
        SPE_LOSS_STD[i] = np.nanstd(SPE_LOSS[i,:])
        
    
    if plotnow:
        
        
        
        
        plt.figure(figsize = (10,10))
        
        ax = plt.subplot(3,2,1)
        ax.plot(SPE_phase/(2*np.pi), SPE_SSN)
        #ax.plot(SPE_phase, SPE_LOSS_AVG ,'k')
        ax.set_ylabel('SSN', fontsize = 12)
        yy = ax.get_ylim()
        ax.text(0.04, 0.93, '(a)', transform=plt.gca().transAxes, fontsize=12)
 
        ax = plt.subplot(3,2,2)
        mplot.plotconfidbands(SPE_phase/(2*np.pi), SPE_SSN.T , confid_intervals = [1, 10, 33], plot_legend=False)
        #ax.plot(SPE_phase, SPE_LOSS_AVG ,'k')
        ax.set_ylim(yy)
        ax.text(0.04, 0.93, '(b)', transform=plt.gca().transAxes, fontsize=12)
        
        
        
        ax = plt.subplot(3,2,3)
        ax.plot(SPE_phase/(2*np.pi), SPE_SSN_NORM)
        #ax.plot(SPE_phase, SPE_LOSS_AVG ,'k')
        ax.set_ylabel('SSN (normalised)', fontsize = 12)
        yy = ax.get_ylim()
        ax.text(0.04, 0.93, '(c)', transform=plt.gca().transAxes, fontsize=12)

        ax = plt.subplot(3,2,4)
        mplot.plotconfidbands(SPE_phase/(2*np.pi), SPE_SSN_NORM.T , confid_intervals = [1, 10, 33], plot_legend=False)
        #ax.plot(SPE_phase, SPE_LOSS_AVG ,'k')
        ax.set_ylim(yy)
        ax.text(0.04, 0.93, '(d)', transform=plt.gca().transAxes, fontsize=12)
        
        ax = plt.subplot(3,2,5)
        ax.plot(SPE_phase/(2*np.pi), SPE_LOSS*100)
        #ax.plot(SPE_phase, SPE_LOSS_AVG ,'k')
        ax.set_ylabel(r'$\chi$, OSF loss [%/yr]', fontsize = 12)
        ax.set_xlabel('Solar cycle phase', fontsize = 12)
        yy = ax.get_ylim()
        ax.text(0.04, 0.93, '(e)', transform=plt.gca().transAxes, fontsize=12)

        ax = plt.subplot(3,2,6)
        mplot.plotconfidbands(SPE_phase/(2*np.pi), SPE_LOSS.T *100, confid_intervals = [1, 10, 33], plot_legend=False)
        #ax.plot(SPE_phase, SPE_LOSS_AVG ,'k')
        ax.set_xlabel('Solar cycle phase', fontsize = 12)
        ax.set_ylim(yy)
        ax.text(0.04, 0.93, '(f)', transform=plt.gca().transAxes, fontsize=12)
        
        
        print('LOSS std: ' + str(np.nanmean(SPE_LOSS_STD)/np.nanmean(SPE_LOSS_AVG)))
        
        
        #plot the SSN and loss timeseries
        #======================================================================
        xx = (htime.mjd2datetime(mjd)[0],htime.mjd2datetime(mjd)[-1])
        
        plt.figure(figsize = (10,10))
        
        ax = plt.subplot(3,1,1)
        ax.plot(htime.mjd2datetime(mjd), osf, 'k')
        ax.set_ylabel(r'OSF [x$10^{14}$ Wb]', fontsize = 14)
        ax.set_xlim(xx)
        yy = ax.get_ylim()
        ax.set_ylim(yy)
        for n in range(0,len(solarmin_mjd)):
            dt = htime.mjd2datetime(solarmin_mjd[n]).item()
            ax.plot([dt, dt],  [0, yy[1]],'r--')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax.text(0.02, 0.91, '(a)', transform=plt.gca().transAxes, fontsize=12)


        
        ax1 = plt.subplot(3,1,2)
        
        ax1.plot(htime.mjd2datetime(mjd), ssn, 'k')
        ax1.set_ylabel(r'SSN', fontsize = 14)
        yy = ax1.get_ylim()
        ax1.set_xlim(xx)
        ax1.set_ylim((0,yy[1]))
        for n in range(0,len(solarmin_mjd)):
            dt = htime.mjd2datetime(solarmin_mjd[n]).item()
            ax1.plot([dt, dt],  [0, yy[1]],'r--')        
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax1.text(0.02, 0.91, '(b)', transform=plt.gca().transAxes, fontsize=12)
        
        ax2 = ax1.twinx()
        ax2.plot(htime.mjd2datetime(mjd), source, 'b')
        ax2.set_ylabel(r'OSF source [x$10^{14}$ Wb yr$^{-1}$]', fontsize = 14, color = 'b')
        yy = ax2.get_ylim()
        ax2.set_xlim(xx)
        ax2.set_ylim((0,yy[1]))
        ax2.tick_params(axis ='y', labelcolor = 'b')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        
        ax = plt.subplot(3,1,3)
        ax.plot(htime.mjd2datetime(mjd), fracloss*100, 'k')
        ax.set_ylabel(r'$\chi$, OSF loss [% yr$^{-1}$]', fontsize = 14)
        ax.set_xlim(xx)
        yy = ax.get_ylim()
        ax.set_ylim((0,yy[1]))
        for n in range(0,len(solarmin_mjd)):
            dt = htime.mjd2datetime(solarmin_mjd[n]).item()
            ax.plot([dt, dt], [0, yy[1]],'r--')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax.set_xlabel('Year', fontsize = 12)
        ax.text(0.02, 0.91, '(c)', transform=plt.gca().transAxes, fontsize=12)

        
    return SPE_phase, SPE_LOSS_AVG, SPE_LOSS_STD, SPE_LOSS 




# <codecell> Vsw reconstruction functions




def osf_and_streamer_belt_width(yrs, ssn, min_yrs, SPE_phase, SPE_LOSS_AVG, 
                                startOSF = 8, startFch = 2, startFsb = 0,
                                S_CH_weights = [0.1, 0.5, 0.3, 0.075, 0.025],
                                chi_CH = 0.22, plotnow = False, source_ref = 1):
    
    
    L= len(yrs)
    mjd = htime.fracyear2mjd(yrs)
    solarmin_mjd = htime.fracyear2mjd(min_yrs)
    
    #compute the OSF source term
    source = compute_osf_source(ssn, source_ref = source_ref)
    
    #compute the SC phase from the year and min years
    phase = solar_cycle_phase(mjd, solarmin_mjd)
    
    #compute the loss term
    loss_frac = np.interp(phase, SPE_phase, SPE_LOSS_AVG, period = 2*np.pi)
    
    
    #set up the arrays
    OSF = np.empty(L)
    F_SB = np.empty(L)
    F_CH = np.empty(L)
    S_CH = np.empty(L)
    chi_SB = np.empty(L)
    loss_abs = np.empty(L)
    SBwidth = np.empty(L)
    
    #initial conditions
    F_SB[0] = startFsb
    OSF[0] = startOSF
    F_CH[0] = startFch
    S_CH[0] = 0
    chi_SB[0] = 0
    
    

    #compute OSF using average loss profile
    for n in range(0, L-1):
        #compute the absolute loss rate
        loss_abs[n] = OSF[n] * loss_frac[n]
        
        #compute the OSF for the next time step
        OSF[n+1] = OSF[n] + source[n] - loss_abs[n] 
        
        #osf can't be negative
        if OSF[n+1] < 0.0:
            OSF[n+1] = 0.0
        
        #the fractional loss of SB flux + CH must equal the total
        chi_SB[n] = loss_frac[n] - chi_CH
        
        #some of the SB flux evolves into CH flux, over a range of time scales given by S_CH_weights
        S_CH[n] = 0 
        if n>1:
            S_CH[n] = S_CH[n] + (source[n-1] - OSF[n-1] * chi_SB[n-1]) * S_CH_weights[0]
        if n>2:
            S_CH[n] = S_CH[n] + (source[n-2] - OSF[n-2] * chi_SB[n-2]) * S_CH_weights[1]
        if n>3:
            S_CH[n] = S_CH[n] + (source[n-3] - OSF[n-3] * chi_SB[n-3]) * S_CH_weights[2]
        if n>4:
            S_CH[n] = S_CH[n] + (source[n-4] - OSF[n-4] * chi_SB[n-4]) * S_CH_weights[3]
        if n>5:
            S_CH[n] = S_CH[n] + (source[n-5] - OSF[n-5] * chi_SB[n-5]) * S_CH_weights[4]
    


        #new flux emerges into the streamer belt, there disconnection loss and CH loss
        F_SB[n+1] =  F_SB[n] + source[n] - chi_SB[n] * OSF[n] - S_CH[n] 
        if F_SB[n+1] < 0:
            F_SB[n+1] = 0
          
        #CH flux has a source from the SB and decays with a constant time
        F_CH[n+1] = F_CH[n]  - chi_CH * OSF[n] + S_CH[n]
        
        if F_CH[n+1] < 0:
            F_CH[n+1] = 0
        
        SBwidth[n] = np.sin(1.0 - F_CH[n] /OSF[n])
        
    #the last year of SBwidth is likely nonesense
    SBwidth[-1] = np.nan
    #as are teh first 6 years
    SBwidth[0:6] = np.nan
        
    if plotnow:    
        fig = plt.figure(figsize=(9,6))
    
        ax = plt.subplot(4,1,1)
        ax.plot(yrs, OSF, 'b', label = 'SSN-based model')
        ax.fill_between(yrs, ssn*0, ssn/100, label = 'SSN/100', facecolor ='grey')
        ax.set_ylim((0,14.5))
        ax.set_ylabel(r'OSF [x$10^14$ Wb]', fontsize = 12)
        ax.set_xlabel(r'Year', fontsize = 12)
        ax.legend(ncol = 4, fontsize = 12, loc = 'upper left')
    
    
        ax = plt.subplot(4,1,2)
        ax.plot(yrs, OSF, 'k', label = 'OSF')
        ax.plot(yrs, F_SB, 'b', label = 'F_SB')
        ax.plot(yrs, F_CH, 'r', label = 'F_CH')
        ax.legend(ncol = 3, fontsize = 12, loc = 'upper left')
    
        ax = plt.subplot(4,1,3)
        ax.plot(yrs, source, 'k', label = 'source')
        ax.plot(yrs, S_CH, 'r', label = 'S_CH')
        ax.legend(ncol = 3, fontsize = 12, loc = 'upper left')
    
        ax = plt.subplot(4,1,4)
        ax.plot(yrs, SBwidth, 'k', label = 'SBwidth')
        ax.legend(ncol = 3, fontsize = 12, loc = 'upper left')
    
    return OSF, SBwidth


def speed_func_theta(vsw_params, theta):
    #A function to return the solar speed at a latitude theta 
    #(relative to a reference latitude theta=0, which is the 
    #centre of the kinetic flux-rope when used in the context 
    #of that model)
    #
    #vsw_params are [theta_sw0 theta_w V_max V_delta]
    #theta_sw0 is the angle from flux-rope axis to Vsw_min
    #
    #test function: 
    #theta = np.linspace(-np.pi, np.pi, 100)
    #vsw_params = [np.pi/2, np.pi, 800, 400]
    #plt.plot(theta, speed_func_theta(vsw_params,theta))
    #
    #(Mathew Owens, 21/11/05)
    #Ported from Matlab: 15/8/23
    
    #the code for a single value fo theta
    def _speed_func_theta_(theta_sw0, theta_w, V_max, V_delta, theta):

        
        #flip theta if it is greater than pi
        if (theta > np.pi):
            theta=theta-2*np.pi
        vsw = (V_max - 0.5 * V_delta) - 0.5 * V_delta * np.cos(theta * np.pi / theta_w - theta_sw0)
        
        #check it doesn't exceed the max value
        if (abs(theta * np.pi / theta_w -theta_sw0) >= np.pi):
            vsw = V_max
        
        return vsw
    
    #code for multiple theta values
    vec_speed_func_theta = np.vectorize(_speed_func_theta_)
    
    
    theta_sw0 = vsw_params[0]
    theta_w = vsw_params[1]
    V_max = vsw_params[2]
    V_delta = vsw_params[3]
        
    vsw = vec_speed_func_theta(theta_sw0, theta_w, V_max, V_delta, theta)
    
    return vsw

def VswLongAvg_SineCurve(lats, inclination, width, Vfast, dV, plotnow = False):
    #a function to return the longitudinal averaged solar wind speeds for given
    #latitude for a prescribed slow wind band width/depth and inclination.
    #
    #(Mathew Owens, 16/3/16)       
    #Ported to python 15/8/23
    
    #test data
    # lats = np.linspace( -90,90,180) *np.pi/180
    # inclination = 30*np.pi/180
    # width = 60*np.pi/180
    # Vfast=750
    # dV=350
    
    dlong = 2 * np.pi / 180
    
    vlats = speed_func_theta([0, width, Vfast, dV], lats)
    
    
    #now account for the inclination of the streamer belt
    #====================================================
    
    longs = np.arange(-np.pi + dlong/2, np.pi, dlong)
    
    vlist = np.empty((len(longs) * len(lats),))
    vposlist = np.empty((len(longs) * len(lats), 3))
    
    counter = 0
    for i in range(len(lats)):
        for k in range(len(longs)):
            vposlist[counter, 0] = longs[k]
            vposlist[counter, 1] = lats[i]
            vposlist[counter, 2] = 1
            
            vlist[counter] = vlats[i]
            counter += 1
    
    #convert to cartesian coordinates
    X, Y, Z = hcoords.sph2cart(vposlist[:, 0], vposlist[:, 1], vposlist[:, 2])
    X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()
    
    #rotate about X axis by angle inclination
    Xml = X
    Yml = Y * np.cos(inclination) + Z * np.sin(inclination)
    Zml = -Y * np.sin(inclination) + Z * np.cos(inclination)
    
    #convert back to spherical coords
    mlong, mlat, mR = hcoords.cart2sph(Xml, Yml, Zml)
    
    #now average in longitude
    
    v_Xa, v_Xm = np.meshgrid(longs, lats)
    vq = griddata((mlong, mlat), vlist, (v_Xa, v_Xm), method = 'linear')
    
    vlats_longavg = np.nanmean(vq, axis=1)
    
    if plotnow:
        plt.figure()
        plt.plot(np.rad2deg(lats), vlats_longavg)
        plt.xlabel('Latitude (degrees)')
        plt.ylabel('Average Speed')
        plt.title('Average Speed vs. Latitude')
        plt.grid()
        plt.show()

    return vlats_longavg


def HCSinclination(yrs, min_yrs):  
    #ccompute HCS inclination for use with SB width calculation. This is heritage
    #code for eh SB width calculation. Better to use load_wsa_hcstilt() and
    #compute_phase_SPE() to get the latest data.
    #
    #Mathew Owens, 15/8/23   
    
    #convert phase to HCS tilt
    phase = solar_cycle_phase(yrs, min_yrs)
    
    dtilt = np.array([   [0.290444101017667,   0.185703032412197,   0.014049275473509],
       [0.860538160945352,   0.245863772889636,   0.017050905137171],
      [ 1.430632220873037,   0.611783832541170,   0.065108031302459],
      [ 2.000726280800723,   0.544184710365411,   0.039746254726842],
      [ 2.570820340728408,   0.432841654494594,   0.043855889338512],
      [ 3.140914400656093,   0.501222759547090,   0.030439693765475],
      [ 3.711008460583778,   0.401897438567343,   0.035001365105233],
      [ 4.281102520511463,   0.282743338823081,   0.017666796391403],
      [ 4.851196580439149,   0.280998009571087,   0.019759985995491],
      [ 5.421290640366834,   0.214199499108395,   0.017640142498358],
      [ 5.991384700294519,   0.217557320713712,   0.021695761089958]])
        
    inclination = np.interp(phase, dtilt[:,0], dtilt[:,1], period = 2*np.pi)
    
    return inclination

def Vlat_from_SBW(yrs, min_yrs, SBwidth, plotnow = False):
    #convert SBwidth and SC phase into Vsw as a function of lat.
    #based on MAtlab code for Owens et al., SciRep, 2017
    #
    #Mathew Owens, 15/8/23

    #now convert this to a solar wind speed profile
    pwidth = [2.02,   -0.41,    0.46] #[2.4362   -0.4545];%[2.0204   -0.4121    0.4590];
    pvmax = [-256,  973]
    pdv = [ -39,   371] #[0 344];% [-39.3650  370.5516];%[-235.7262  292.9599  263.9678];
    thetminvmax = 0.85 #minimum OSF streamer belt width to consider for vmax correlation
    
    
    #convert phase to HCS tilt
    inclination = HCSinclination(yrs, min_yrs)
    
    lats = np.linspace( -90,90,180) *np.pi/180
    Vband = np.ones((len(yrs), len(lats)))
    beltwidth = np.ones(len(yrs))
    dV = np.ones(len(yrs))
    Vfast =  np.ones(len(yrs))
    for n in range(0, len(yrs)):
        
        beltwidth[n] = np.polyval(pwidth, np.arcsin(SBwidth[n]));
           
        dV[n] = np.polyval(pdv, np.arcsin(SBwidth[n]))
        
        Vfast[n] = np.polyval(pvmax, np.arcsin(SBwidth[n]))
        if np.arcsin(SBwidth[n]) < thetminvmax:
            Vfast[n] = 756
        
        if np.isnan(beltwidth[n]):
            Vband[n,:] =  np.nan
        else:
             Vband[n,:] = VswLongAvg_SineCurve(lats, inclination[n],
                                                        beltwidth[n], Vfast[n], dV[n])
    
    if plotnow:
        startyr = yrs[0]
        stopyr = yrs[-1]
    
        
        fig = plt.figure()
    
        ax = plt.subplot(3,1,1)
        ax.plot(yrs, inclination*180/np.pi , 'k')
        ax.set_ylabel('HCS tilt [degrees]')
        ax.set_xlim([startyr, stopyr])
        
        ax = plt.subplot(3,1,2)
        ax.plot(yrs, np.arcsin(SBwidth) , 'k');
        ax.set_xlim([startyr, stopyr]);
        ax.set_ylabel('sin(SBW)');
        
        ax = plt.subplot(3,1,3)
        im = ax.pcolor(yrs, lats*180/np.pi, np.rot90(Vband),  vmin = 300, vmax = 750)
        ax.set_ylim([-90, 90]); #set(gca,'YTick',[-90 -60 -30 0 30 60 90]);
        #set(gca,'CLim',[250 850]); 
        ax.set_xlim([startyr, stopyr])
        # colorbar; 
        # title('V_{SW} [kms/]');
        ax.set_ylabel('Helio lat [deg]');
        ax.set_xlabel('Year')
        
        #offset colourbar
        axins = inset_axes(ax,
                            width="100%",  # width = 50% of parent_bbox width
                            height="100%",  # height : 5%
                            loc='upper right',
                            bbox_to_anchor=(1.03, 0.0, 0.02, 1),
                            bbox_transform=ax.transAxes,
                            borderpad=0,)
        cb = fig.colorbar(im, cax = axins, orientation = 'vertical',  pad = -0.1)
        cb.ax.tick_params(labelsize=10)
        axins.text(0.99,1.07,r'V$_{SW}$ [km/s]' , 
                fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
        
    return Vband, lats, inclination
