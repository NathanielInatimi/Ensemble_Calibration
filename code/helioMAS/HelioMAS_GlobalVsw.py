# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 14:30:37 2023

@author: mathewjowens
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:52:10 2022

@author: vy902033
"""
"""
a Script to compute the global Vsw from HelioMAS


Created on Mon Dec 20 11:48:49 2021

@author: mathewjowens
"""

import os
from pyhdf.SD import SD, SDC  
import numpy as np
import astropy.units as u
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import httplib2
import urllib
import requests
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import huxt_inputs as Hin
import helio_time as htime

CRstart = 1625
CRstop = 2269

plt.rcParams.update({'font.size': 16})

downloadnow = False #flag to determine whether HelioMAS data needs to be obtained. Set to False after first use
downloadCRstart = 1625
downloadCRstop = 2269

#datadir = os.environ['DBOX'] + 'python_repos\\SolarWindVariability\data\\'
#heliomasdir = datadir + 'HelioMAS\\'
heliomasdir = os.environ['DBOX'] + 'Data\\HelioMAS\\'

scaleto1au = True
alpha = 0.15 #scaling factor between 30 and 215 rs in acceleration equatoin
rH = 50
pvsw=[1.04,  78.6] #MAS/in situ conversion for Ulysses high-speed wind

#Data reader functions
def LoadSSN(filepath='null'):
    #(dowload from http://www.sidc.be/silso/DATA/SN_m_tot_V2.0.csv)
    if filepath == 'null':
        filepath= os.environ['DBOX'] + 'Data\\SN_m_tot_V2.0.csv'
        
    col_specification =[(0, 4), (5, 7), (8,16),(17,23),(24,29),(30,35)]
    ssn_df=pd.read_fwf(filepath, colspecs=col_specification,header=None)
    dfdt=np.empty_like(ssn_df[0],dtype=datetime)
    for i in range(0,len(ssn_df)):
        dfdt[i] = datetime(int(ssn_df[0][i]),int(ssn_df[1][i]),15)
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
    
    #add in a solar activity index, which normalises the cycle magnitude
    #approx solar cycle length, in months
    nwindow = int(11*12)
    
    #find maximum value in a 1-solar cycle bin centred on current time
    ssn_df['rollingmax'] = ssn_df.rolling(nwindow, center = True).max()['smooth']
    
    #fill the max value at the end of the series
    fillval = ssn_df['rollingmax'].dropna().values[-1]
    ssn_df['rollingmax'] = ssn_df['rollingmax'].fillna(fillval) 
    
    #create a Solar Activity Index, as SSN normalised to the max smoothed value in
    #1-sc window centred on current tim
    ssn_df['sai'] = ssn_df['smooth']/ssn_df['rollingmax']
    
    return ssn_df



def get_helioMAS_output(cr=np.NaN, observatory='', runtype='', runnumber='', masres=''):
    """
    A function to grab the  Vr and Br boundary conditions from MHDweb. An order
    of preference for observatories is given in the function. Checks first if
    the data already exists in the HUXt boundary condition folder

    Parameters
    ----------
    cr : INT
        Carrington rotation number 
    observatory : STRING
        Name of preferred observatory (e.g., 'hmi','mdi','solis',
        'gong','mwo','wso','kpo'). Empty if no preference and automatically selected 
    runtype : STRING
        Name of preferred MAS run type (e.g., 'mas','mast','masp').
        Empty if no preference and automatically selected 
    runnumber : STRING
        Name of preferred MAS run number (e.g., '0101','0201').
        Empty if no preference and automatically selected    

    Returns
    -------
    flag : INT
        1 = successful download. 0 = files exist, -1 = no file found.

    """
    
    assert(np.isnan(cr) == False)
    
    # The order of preference for different MAS run results
    if not masres:
        masres_order = ['high','medium']
    else:
        masres_order = [str(masres)]
           
    if not observatory:
        observatories_order = ['hmi', 'mdi', 'solis', 'gong', 'mwo', 'wso', 'kpo']
    else:
        observatories_order = [str(observatory)]
               
    if not runtype:
        runtype_order = ['masp', 'mas', 'mast']
    else:
        runtype_order = [str(runtype)]
           
    if not runnumber:
        runnumber_order = ['0201', '0101']
    else:
        runnumber_order = [str(runnumber)]
           
    # Get the HUXt boundary condition directory
    #dirs = H._setup_dirs_()
    #_boundary_dir_ = dirs['boundary_conditions'] 
      
    # Example URL: http://www.predsci.com/data/runs/cr2010-medium/mdi_mas_mas_std_0101/helio/br_r0.hdf
    # https://shadow.predsci.com/data/runs/cr2000-medium/mdi_mas_mas_std_0101/helio/vr002.hdf
    heliomas_url_front = 'https://predsci.com/data/runs/cr'
    heliomas_url_end = '_r0.hdf'
    
    brfilename = 'br' + heliomas_url_end
    vrfilename = 'vr' + heliomas_url_end

    # Search MHDweb for a HelioMAS run, in order of preference
    h = httplib2.Http(disable_ssl_certificate_validation=True)
    foundfile = False
    for res in masres_order:
        for masob in observatories_order:
            for masrun in runtype_order:
                for masnum in runnumber_order:
                    urlbase = (heliomas_url_front + str(int(cr)) + '-' + 
                               res + '/' + masob + '_' +
                               masrun + '_mas_std_' + masnum + '/helio/')                                
                    
                    url = urlbase + 'vr' + heliomas_url_end

                    # See if this br file exists
                    resp = h.request(url, 'HEAD')
                    if int(resp[0]['status']) < 400:
                        foundfile = True
                                            
                    # Exit all the loops - clumsy, but works
                    if foundfile: 
                        break
                if foundfile:
                    break
            if foundfile:
                break
        if foundfile:
            break
        
    if foundfile == False:
        print('No data available for given CR and observatory preferences')
        return -1
    
    # Download teh vr and br files
    
    print('Downloading from: ', urlbase)
    #download_file(urlbase + 'br' + heliomas_url_end)
    #download_file(urlbase + 'vr' + heliomas_url_end)
    
    urllib.request.urlretrieve(urlbase + brfilename,
                                 os.path.join(brfilename))
    urllib.request.urlretrieve(urlbase + vrfilename,
                                 os.path.join(vrfilename))
    

        
    return 1




def download_file(url):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                f.write(chunk)
    return local_filename

# <codecell> download HelioMAS data for each observatory
#heliomasdir = 'D:\\Dropbox\\Data\\HelioMAS\\'


#observatories = ['hmi', 'mdi', 'solis', 'gong', 'mwo', 'wso', 'kpo']

observatories = ['solis', 'gong', 'mwo', 'wso', 'kpo']

if downloadnow:
    for obs in observatories:
        #move to the appropriate directory
        MYDIR = heliomasdir + obs
        CHECK_FOLDER = os.path.isdir(MYDIR)
        if not CHECK_FOLDER:
            os.makedirs(MYDIR)
            print("created folder : ", MYDIR)
    
        else:
            print(MYDIR, "folder already exists.")
        os.chdir(MYDIR)
        
        
        for cr in range(downloadCRstart, downloadCRstop):
            #move to the appropriate directory
            CRDIR = MYDIR + '\\CR' + str(cr)
            CHECK_FOLDER = os.path.isdir(CRDIR)
            if not CHECK_FOLDER:
                os.makedirs(CRDIR)
                print("created folder : ", CRDIR)
    
            else:
                print(CRDIR, "folder already exists.")
            os.chdir(CRDIR)
            
            get_helioMAS_output(cr=cr, observatory=obs)
            
# <codecell> Process HelioMAS data

#loop through the HelioMAS solutions and compute Global Vsw 

ssn_df = LoadSSN()

observatories = ['hmi', 'mdi', 'solis', 'gong', 'mwo', 'wso', 'kpo']

lats = np.arange(2.5,180-2.499999,5) * np.pi/180
Nlats = len(lats)-1

df = pd.DataFrame()


for obs in observatories: 
    
    print('Processing ' + obs)
    
    GlobalVsw = np.ones((CRstop-CRstart))*np.nan
    vr_lat = np.ones((CRstop-CRstart,Nlats))*np.nan
    vr_norm_lat = np.ones((CRstop-CRstart,Nlats))*np.nan
    ssn = np.ones((CRstop-CRstart))*np.nan
    sai = np.ones((CRstop-CRstart))*np.nan
    nCR = np.ones((CRstop-CRstart))*np.nan
    
    counter = 0 
    for CR in range(CRstart,CRstop):
        vrfilepath = heliomasdir + obs + '\\CR' + str(CR) + '\\vr_r0.hdf'
        
        nCR[counter] = CR 
        
        smjd = htime.crnum2mjd(CR-1)
        fmjd = htime.crnum2mjd(CR+2)
        
        mask = (ssn_df['mjd'] >= smjd) & (ssn_df['mjd'] < fmjd)
        ssn[counter] = np.nanmean(ssn_df.loc[mask,'ssn'])
        sai[counter] = np.nanmean(ssn_df.loc[mask,'sai'])
        
        
        if (os.path.exists(vrfilepath) == True):
            
            
            #load the Br data
            file = SD(vrfilepath, SDC.READ)
 
            sds_obj = file.select('fakeDim0')  # select sds
            MAS_vr_Xa = sds_obj.get()  # get sds data
            sds_obj = file.select('fakeDim1')  # select sds
            MAS_vr_Xm = sds_obj.get()  # get sds data
            sds_obj = file.select('Data-Set-2')  # select sds
            MAS_vr = sds_obj.get()  # get sds data
 
            # Convert from model to physicsal units
            MAS_vr = MAS_vr * 481.0 * u.km / u.s
            MAS_vr_Xa = MAS_vr_Xa * u.rad
            MAS_vr_Xm = MAS_vr_Xm * u.rad
 
            
            if scaleto1au:
                #scale to 1au using acceleration term
                #comppute new speed
                #MAS_vr =  MAS_vr.value * (1 + alpha * (1 - np.exp(-(215 - 30) / rH)))

                #correct to produce Ulysses fast wind
                MAS_vr = np.polyval(pvsw, MAS_vr.value) *u.km/u.s
            
            #now find teh differences at each lat bin
            for ilat in range(0,Nlats):
                mask_b = ((MAS_vr_Xm >= lats[ilat]*u.rad) & (MAS_vr_Xm < lats[ilat+1]*u.rad))
            
                #compute the mean abs Br at the given lat band
                vr_lat[counter,ilat] = np.nanmean(abs(MAS_vr[:, mask_b].value))
                #normalise to the max value
                vr_norm_lat[counter,:] = vr_lat[counter,:] / np.nanmax(vr_lat[counter,:])
            
            #compute the OSF from full lat resolution
            GlobalVsw[counter] = 0.0
            sine_lats = 0.0
            for ilat in range(0,len(MAS_vr_Xm)):
                theta = MAS_vr_Xm[ilat].value
                absvr = np.nanmean(abs(MAS_vr[:, ilat].value))
                
                GlobalVsw[counter] = GlobalVsw[counter] + np.sin(theta) * absvr
                sine_lats = sine_lats +  np.sin(theta)
                
            GlobalVsw[counter] = GlobalVsw[counter] / sine_lats
            
            
           
        

        else:
            GlobalVsw[counter] = np.nan
            vr_lat[counter,:] = np.nan
            vr_norm_lat[counter,:] = np.nan
    
        counter = counter + 1
    
    #add the data to the data frame
    df['CR'] = nCR
    df['SSN'] = ssn
    df['SAI'] = sai
    df[obs + '_GlobalVsw'] = GlobalVsw
    for ilat in range(0,Nlats):
        df[obs + '_vr_lat_' + str(ilat)] = vr_lat[:,ilat]
        df[obs + '_vr_norm_lat_' + str(ilat)] = vr_norm_lat[:,ilat]

#convert to datetime
df['datetime'] = htime.mjd2datetime(htime.crnum2mjd(df['CR'].values))

#take averages across all observatories
#observatories = ['hmi', 'mdi', 'solis', 'gong', 'mwo', 'wso', 'kpo']
df['GlobalVsw'] = df[['hmi_GlobalVsw','mdi_GlobalVsw','solis_GlobalVsw',
               'gong_GlobalVsw','mwo_GlobalVsw','wso_GlobalVsw', 
               'kpo_GlobalVsw']].mean(axis = 1, skipna=True)

for ilat in range(0,Nlats):
    l = str(ilat)
    df['vr_lat_' + str(ilat)] = df[['hmi_vr_lat_' + l,'mdi_vr_lat_' + l,'solis_vr_lat_' + l,
                   'gong_vr_lat_' + l,'mwo_vr_lat_' + l,'wso_vr_lat_' + l, 
                   'kpo_vr_lat_' + l]].mean(axis = 1, skipna=True)
    
    
    df['vr_norm_lat_' + str(ilat)] = df[['hmi_vr_norm_lat_' + l,'mdi_vr_norm_lat_' + l,'solis_vr_norm_lat_' + l,
                   'gong_vr_norm_lat_' + l,'mwo_vr_norm_lat_' + l,'wso_vr_norm_lat_' + l, 
                   'kpo_vr_norm_lat_' + l]].mean(axis = 1, skipna=True)

    
#add a pole average

mask_max = ((df['SAI'] >= 0.5))
mask_min = ((df['SAI'] < 0.5)) 

Nall = len(np.isfinite(df['GlobalVsw']))   
Nmin = len(np.isfinite(df.loc[mask_min,'GlobalVsw']))
Nmax = len(np.isfinite(df.loc[mask_max,'GlobalVsw']))


# <codecell> plot the HelioMAS results

#plt.rcParams.update({'font.size': 14})
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color='bgkymcr')

plot_observatories = ['hmi', 'mdi', 'solis', 'gong', 'mwo', 'wso', 'kpo']


xx = (datetime(1973,1,1), datetime(2024,1,1))  


# #import OMNI data and compute OSF from 20-hour averages of Br
# datadir = os.environ['DBOX'] + 'Data_hdf5\\'
# omni_1hour = pd.read_hdf(datadir + 'omni_1hour.h5')

# br = omni_1hour.copy()
# br_20H = br.resample('20H', on='datetime').mean() 

# br_20H['datetime'] = br_20H.index
# br_20H.reset_index(drop=True, inplace=True)
# br_20H['absBr'] = abs(br_20H['Bx_gse'])

# br_CR = br_20H.resample('27D', on='datetime').mean() 


dtheta = (lats[1]-lats[0])*180/np.pi
lat_centres = 90 - lats[0:len(lats)-1]*180/np.pi -dtheta/2

#extract data as an array for ploitting purposes
v_lat = np.ones((CRstop-CRstart,Nlats))*np.nan
v_norm_lat = np.ones((CRstop-CRstart,Nlats))*np.nan
for ilat in range(0,Nlats):
    v_lat[:,ilat] = df['vr_lat_' + str(ilat)].to_numpy()
    v_norm_lat[:,ilat] = df['vr_norm_lat_' + str(ilat)].to_numpy()
    

#time series plots
fig = plt.figure()

ax1 = plt.subplot(311)
ax1.plot(df['datetime'],df['SSN']/200, 'k', label = 'SSN/200')
ax1.plot(df['datetime'],df['SAI'], 'r', label = 'SAI')
ax1.set_ylabel('SSN')
ax1.legend(fontsize = 16)
ax1.text(0.01,0.9,'(a)', fontsize = 14, transform=ax1.transAxes, backgroundcolor = 'w')
ax1.set_xlim(xx)


ax2 = plt.subplot(312)
im_v = ax2.pcolor(df['datetime'], lat_centres, v_lat.T)#,norm=plt.Normalize(0,2))
ax2.set_yticks([-90, -45, 0, 45, 90])
ax2.text(0.01,0.9,'(b)', fontsize = 14, transform=ax2.transAxes, backgroundcolor = 'w')
#ax.get_xaxis().set_ticklabels([])
# ax.text(0.02,1.05,'(b)' + r'$<|V_R|>$ [km/s]                                                            ',
#         fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax2.set_xlim(xx)
ax2.set_ylim((-90,90))
ax2.set_ylabel('Latitude [deg]')

#ax.plot([0, 360],[7.5, 7.5],'w--'); ax.plot([0, 360],[-7.5, -7.5],'w--');
#cb = plt.colorbar(im_v); cb.ax.tick_params(labelsize=12)
#cb.ax.set_title(r'$<|\Delta V|>_{CR}$ [km/s]', fontsize = 14)

#offset colourbar
axins = inset_axes(ax2,
                    width="100%",  # width = 50% of parent_bbox width
                    height="100%",  # height : 5%
                    loc='upper right',
                    bbox_to_anchor=(1.03, 0.0, 0.02, 1),
                    bbox_transform=ax2.transAxes,
                    borderpad=0,)
cb = fig.colorbar(im_v, cax = axins, orientation = 'vertical',  pad = -0.1)
cb.ax.tick_params(labelsize=10)
axins.text(1,1.07,'V [km/s]' , 
        fontsize = 14, transform=ax2.transAxes, backgroundcolor = 'w')

ax3 = plt.subplot(313)
for obs in plot_observatories: 
    ax3.plot(df['datetime'], df[obs + '_GlobalVsw'], label = obs)
#plt.legend(fontsize = 16)
ax3.set_ylabel(r'Global $V_{SW}$' +'\nHelioMAS [km/s]', fontsize = 16)
#ax.set_ylim((0,230))
ax3.text(0.01,0.9,'(c)', fontsize = 14, transform=ax3.transAxes, backgroundcolor = 'w')
ax3.set_xlim(xx)


handles, labels = ax3.get_legend_handles_labels()
fig.legend(handles, labels, ncol=len(plot_observatories), loc='lower center', 
           frameon=False, handletextpad=0.2, columnspacing=1.0)


#export the data
data = np.empty((len(df),2))
for n in range(0,len(df)):
    data[n,0] = htime.datetime2fracyear(df['datetime'][n])
    data[n,1] = df['GlobalVsw'][n]
    
export_path = os.environ['DBOX'] + 'Papers_WIP\\_coauthor\\Fatemeh_Rahmanifard\\GlobalVSW.dat'
np.savetxt(export_path, data)