# -*- coding: utf-8 -*-
"""
A reader for Pioneer 10 data

@author: mathewjowens
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

p10dir = 'D:\\Dropbox\\Data\\Pioneer10\\'
save_dir = 'D:\\Dropbox\\Data_hdf5\\'
startyr = 1972
stopyr = 1995

p10_1hour = pd.DataFrame()

#yr = 1983

for yr in range(startyr,stopyr+1):
    print('Loading ' + str(yr))
    
    filepath = p10dir + 'p10_' + str(yr) + '.asc'
    
    
     # WORD  ASCII     MEANING                         UNITS/COMMENTS
                                                                            
     # 1     I4       YEAR                                    1980......... 
     # 2     I4       DECIMAL DAY                              January 1 =Day 1  
     # 3     I3       HOUR                                     (0,1,......23)   
     # 4     F7.2     Spacecraft Heliocentric             astronomical units
     #                   Distance
    
     # 5     F7.1     Heliographic Inertial Latitude        degrees, +/- 90
     #                   of the spacecraft position
                       
     # 6     F7.1     Heliographic Inertial Longitude       degrees, 0-360
     #                   of the spacecraft position
     # 7    F9.4     BR RTN coordinate system                 NANOTESLAS
     # 8    F9.4     BT RTN coordinate system                 NANOTESLAS
     # 9    F9.4     BN RTN coordinate system                 NANOTESLAS
     # 10   F9.4     Scalar B                                 NANOTESLAS 
     # 11   F7.1      Proton Bulk flow speed, RTN                       km/s
     # 12   F7.1      THETA-elevation angle                   Degrees        
     #               of flow velocity vector
     #               (RTN-cordinate system)
     # 13   F7.1      PHI- azimuth angle of                   Degrees
     #               flow velocity vector.
     #               (RTN-coordinate system) 
     # 14   F9.4      Proton density                          [n/cc]
     # 15   F9.0      Proton Temperature                      degrees, K   
                 
     # 16  e13.6     3.45-5.15 MeV H flux, CRT*           (1/(sec-cm**2-ster-MeV)
     # 17  e13.6     30.55-56.47  MeV H flux,CRT*         (1/(sec-cm**2-ster-MeV)
     # 18  e13.6     120.7-227.3 MeV H flux, CRT*         (1/(sec-cm**2-ster-MeV)
                                                         
    temp = pd.read_csv(filepath,
                         skiprows = 0, delim_whitespace=True, index_col=False,
                         names=['year','dec_doy', 'hr', 'R_AU',
                                'HGI_lat_deg', 'HGI_lon_deg', 'BR',
                                'BT','BN','B','V', 'V_theta', 'V_phi',
                                'n_P','T_P','Hflux_low', 'Hflux_medium',
                                'Hflux_high'])
    
    
    
    
    
    temp['datetime'] = pd.to_datetime(temp['year'] * 1000 + np.floor(temp['dec_doy']), format='%Y%j')
    temp['datetime'] = temp['datetime'] + pd.to_timedelta(temp['hr'], unit='h')
    #plasma['datetime'] = htime.mjd2datetime(plasma['mjd'].to_numpy())
    
    
    #remove bad data
    temp.loc[temp['R_AU'] > 999, 'R_AU'] = np.nan
    temp.loc[temp['HGI_lat_deg'] > 9999, 'HGI_lat_deg'] = np.nan
    temp.loc[temp['HGI_lon_deg'] > 9999, 'HGI_lon_deg'] = np.nan
    
    temp.loc[temp['BR'] > 999, 'BR'] = np.nan
    temp.loc[temp['BT'] > 999, 'BT'] = np.nan
    temp.loc[temp['BN'] > 999, 'BN'] = np.nan
    temp.loc[temp['B'] > 999, 'B'] = np.nan
    
    temp.loc[temp['V'] > 9999, 'V'] = np.nan
    temp.loc[temp['V_theta'] > 9999, 'V_theta'] = np.nan
    temp.loc[temp['V_phi'] > 9999, 'V_phi'] = np.nan
    
    temp.loc[temp['n_P'] > 999, 'n_P'] = np.nan
    temp.loc[temp['T_P'] > 9999990, 'T_P'] = np.nan
    
    temp.loc[temp['Hflux_low'] > 9e6, 'Hflux_low'] = np.nan
    temp.loc[temp['Hflux_medium'] > 9e6, 'Hflux_medium'] = np.nan
    temp.loc[temp['Hflux_high'] > 9e6, 'Hflux_high'] = np.nan
    
    #add it to the existing dataframe
    p10_1hour = pd.concat([p10_1hour,temp])




#save the data
p10_1hour.to_hdf(save_dir + 'pioneer10_1hour.h5', key = 'data', mode = 'w')

#load the data back in
p10_1hour = pd.read_hdf(save_dir + 'pioneer10_1hour.h5')

#plot the data
plt.figure()
plt.plot(p10_1hour['datetime'],p10_1hour['V'])

