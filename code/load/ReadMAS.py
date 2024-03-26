# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 09:17:41 2022

@author: mathewjowens

A set of functions to download, import and process MAS and HelioMAS data

Lots of this is also in huxt_inputs
"""

import httplib2
import urllib
import os
from pyhdf.SD import SD, SDC  
import numpy as np
import astropy.units as u
import ssl


def get_helioMAS_output(cr=np.NaN, observatory='', runtype='', 
                        runnumber='', masres=''):
    """
    A function to grab the  Vr and Br HelioMAS datacubes from MHDweb. An order
    of preference for observatories is given in the function. 
    
    Puts data in current folder

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
    heliomas_url_front = 'https://shadow.predsci.com/data/runs/cr'
    heliomas_url_end = '002.hdf'
    
    vrfilename = 'vr002.hdf'
    brfilename = 'br002.hdf'
    inputfilename = 'br_r0.hdf'


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
                    url = urlbase + 'br' + heliomas_url_end
                    
                    coronal_urlbase = (heliomas_url_front + str(int(cr)) + '-' + 
                               res + '/' + masob + '_' +
                               masrun + '_mas_std_' + masnum + '/corona/')

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
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    print('Downloading from: ', urlbase)
    urllib.request.urlretrieve(urlbase + 'br' + heliomas_url_end,
                               os.path.join(brfilename))
    urllib.request.urlretrieve(urlbase + 'vr' + heliomas_url_end,
                               os.path.join(vrfilename))
    
    #also grab the input Br
    urllib.request.urlretrieve(coronal_urlbase + 'br_r0.hdf',
                               os.path.join(inputfilename))
        
    return 1

def get_MAS_output(cr=np.NaN, observatory='', runtype='', 
                        runnumber='', masres=''):
    """
    A function to grab the  Br MAS datacube from MHDweb. An order
    of preference for observatories is given in the function. 
    
    Puts data in current folder

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
    heliomas_url_end = '002.hdf'
    

    brfilename = 'br002.hdf'
    #inputfilename = 'br_r0.hdf'


    # Search MHDweb for a HelioMAS run, in order of preference
    h = httplib2.Http(disable_ssl_certificate_validation=True)
    foundfile = False
    for res in masres_order:
        for masob in observatories_order:
            for masrun in runtype_order:
                for masnum in runnumber_order:

                    
                    coronal_urlbase = (heliomas_url_front + str(int(cr)) + '-' + 
                               res + '/' + masob + '_' +
                               masrun + '_mas_std_' + masnum + '/corona/')

                    # See if this br file exists
                    resp = h.request(coronal_urlbase, 'HEAD')
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
    
    # Download the br file
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    print('Downloading from: ', coronal_urlbase)
    urllib.request.urlretrieve(coronal_urlbase + 'br' + heliomas_url_end,
                               os.path.join(brfilename))


        
    return 1

def read_HelioMAS(filepath):
    """
    A function to read in the HelioMAS data cubes

    Parameters
    ----------
    directory_path : INT
        Carrington rotation number

    Returns
    -------
    MAS_vr : NP ARRAY (NDIM = 2)
        Solar wind speed at 30rS, in km/s
    MAS_vr_Xa : NP ARRAY (NDIM = 1)
        Carrington longitude of Vr map, in rad
    MAS_vr_Xm : NP ARRAY (NDIM = 1)
        Latitude of Vr as angle down from N pole, in rad
    MAS_vr_Xr : NP ARRAY (NDIM = 1)
        Radial distance of Vr, in solar radii

    """
    
    assert os.path.exists(filepath)
    
    file = SD(filepath, SDC.READ)
        
    sds_obj = file.select('fakeDim0')  # select sds
    MAS_vr_Xa = sds_obj.get()  # get sds data
    sds_obj = file.select('fakeDim1')  # select sds
    MAS_vr_Xm = sds_obj.get()  # get sds data
    sds_obj = file.select('fakeDim2')  # select sds
    MAS_vr_Xr = sds_obj.get()  # get sds data
    sds_obj = file.select('Data-Set-2')  # select sds
    MAS_vr = sds_obj.get()  # get sds data
    
    # # Convert from model to physicsal units
    # MAS_vr = MAS_vr*481.0 * u.km/u.s
    MAS_vr_Xa = MAS_vr_Xa * u.rad
    MAS_vr_Xm = MAS_vr_Xm * u.rad
    MAS_vr_Xr = MAS_vr_Xr * u.solRad
    
    
    return MAS_vr, MAS_vr_Xa, MAS_vr_Xm, MAS_vr_Xr


def get_MAS_boundary_conditions(cr=np.NaN, observatory='', runtype='', runnumber='', 
                                masres='', savedir = ''):
    """
    A function to grab the  solar wind speed (Vr) and radial magnetic field (Br) boundary conditions from MHDweb.
    An order of preference for observatories is given in the function.
    Checks first if the data already exists in the HUXt boundary condition folder.

    Args:
        cr: Integer Carrington rotation number
        observatory: String name of preferred observatory (e.g., 'hmi','mdi','solis',
            'gong','mwo','wso','kpo'). Empty if no preference and automatically selected.
        runtype: String name of preferred MAS run type (e.g., 'mas','mast','masp').
            Empty if no preference and automatically selected
        runnumber: String Name of preferred MAS run number (e.g., '0101','0201').
            Empty if no preference and automatically selected
        masres: String, specify the resolution of the MAS model run through 'high' or 'medium'.

    Returns:
    flag: Integer, 1 = successful download. 0 = files exist, -1 = no file found.
    """

    assert (np.isnan(cr) == False)

    # The order of preference for different MAS run results
    overwrite = False
    if not masres:
        masres_order = ['high', 'medium']
    else:
        masres_order = [str(masres)]
        overwrite = True  # If the user wants a specific observatory, overwrite what's already downloaded

    if not observatory:
        observatories_order = ['hmi', 'mdi', 'solis', 'gong', 'mwo', 'wso', 'kpo']
    else:
        observatories_order = [str(observatory)]
        overwrite = True  # If the user wants a specific observatory, overwrite what's already downloaded

    if not runtype:
        runtype_order = ['masp', 'mas', 'mast']
    else:
        runtype_order = [str(runtype)]
        overwrite = True

    if not runnumber:
        runnumber_order = ['0201', '0101']
    else:
        runnumber_order = [str(runnumber)]
        overwrite = True

    # Get the HUXt boundary condition directory
    _boundary_dir_ = savedir

    # Example URL: http://www.predsci.com/data/runs/cr2010-medium/mdi_mas_mas_std_0101/helio/br_r0.hdf
    # heliomas_url_front = 'https://shadow.predsci.com/data/runs/cr'
    heliomas_url_front = 'http://www.predsci.com/data/runs/cr'
    heliomas_url_end = '_r0.hdf'

    vrfilename = 'HelioMAS_CR' + str(int(cr)) + '_vr' + heliomas_url_end
    brfilename = 'HelioMAS_CR' + str(int(cr)) + '_br' + heliomas_url_end

    if (os.path.exists(os.path.join(_boundary_dir_, brfilename)) == False or
       os.path.exists(os.path.join(_boundary_dir_, vrfilename)) == False or
       overwrite == True):  # Check if the files already exist

        # Search MHDweb for a HelioMAS run, in order of preference
        h = httplib2.Http(disable_ssl_certificate_validation=False)
        foundfile = False
        urlbase = None
        for res in masres_order:
            for masob in observatories_order:
                for masrun in runtype_order:
                    for masnum in runnumber_order:
                        urlbase = (heliomas_url_front + str(int(cr)) + '-' +
                                   res + '/' + masob + '_' +
                                   masrun + '_mas_std_' + masnum + '/helio/')
                        url = urlbase + 'br' + heliomas_url_end

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

        # Download the vr and br files
        ssl._create_default_https_context = ssl._create_unverified_context

        print('Downloading from: ', urlbase)
        urllib.request.urlretrieve(urlbase + 'br' + heliomas_url_end,
                                   os.path.join(_boundary_dir_, brfilename))
        urllib.request.urlretrieve(urlbase + 'vr' + heliomas_url_end,
                                   os.path.join(_boundary_dir_, vrfilename))

        return 1
    else:
        print('Files already exist for CR' + str(int(cr)))
        return 0


def read_MAS_vr_br(cr, savedir = ''):
    """
    A function to read in the MAS boundary conditions for a given CR

    Args:
        cr: Integer Carrington rotation number

    Returns:
        MAS_vr: Solar wind speed at 30rS, np.array in units of km/s.
        MAS_vr_Xa: Carrington longitude of Vr map, np.array in units of rad.
        MAS_vr_Xm: Latitude of Vr as angle down from N pole, np.array in units of rad.
        MAS_br: Radial magnetic field at 30rS, dimensionless np.array.
        MAS_br_Xa: Carrington longitude of Br map, np.array in units of rad.
        MAS_br_Xm: Latitude of Br as angle down from N pole, np.array in units of rad.
    """
    # Get the boundary condition directory
    _boundary_dir_ = savedir
    # Create the filenames
    heliomas_url_end = '_r0.hdf'
    vrfilename = 'HelioMAS_CR' + str(int(cr)) + '_vr' + heliomas_url_end
    brfilename = 'HelioMAS_CR' + str(int(cr)) + '_br' + heliomas_url_end

    filepath = os.path.join(_boundary_dir_, vrfilename)
    assert os.path.exists(filepath)

    file = SD(filepath, SDC.READ)

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

    filepath = os.path.join(_boundary_dir_, brfilename)
    assert os.path.exists(filepath)
    file = SD(filepath, SDC.READ)

    sds_obj = file.select('fakeDim0')  # select sds
    MAS_br_Xa = sds_obj.get()  # get sds data
    sds_obj = file.select('fakeDim1')  # select sds
    MAS_br_Xm = sds_obj.get()  # get sds data
    sds_obj = file.select('Data-Set-2')  # select sds
    MAS_br = sds_obj.get()  # get sds data

    MAS_br_Xa = MAS_br_Xa * u.rad
    MAS_br_Xm = MAS_br_Xm * u.rad

    return MAS_vr, MAS_vr_Xa, MAS_vr_Xm, MAS_br, MAS_br_Xa, MAS_br_Xm


def get_MAS_long_profile(cr, lat=0.0 * u.deg, savedir = ''):
    """
    Function to download, read and process MAS output to provide a longitude profile at a specified latitude
    of the solar wind speed for use as boundary conditions in HUXt.

    Args:
        cr: Integer Carrington rotation number
        lat: Latitude at which to extract the longitudinal profile, measure up from equator. Float with units of deg

    Returns:
        vr_in: Solar wind speed as a function of Carrington longitude at solar equator.
               Interpolated to HUXt longitudinal resolution. np.array (NDIM = 1) in units of km/s
    """
    assert (np.isnan(cr) == False and cr > 0)
    assert (lat >= -90.0 * u.deg)
    assert (lat <= 90.0 * u.deg)

    # Convert angle from equator to angle down from N pole
    ang_from_N_pole = np.pi / 2 - (lat.to(u.rad)).value

    # Check the data exist, if not, download them
    flag = get_MAS_boundary_conditions(cr, savedir = savedir)
    assert (flag > -1)

    # Read the HelioMAS data
    MAS_vr, MAS_vr_Xa, MAS_vr_Xm, MAS_br, MAS_br_Xa, MAS_br_Xm = read_MAS_vr_br(cr)

    # Extract the value at the given latitude
    vr = np.ones(len(MAS_vr_Xa))
    for i in range(0, len(MAS_vr_Xa)):
        vr[i] = np.interp(ang_from_N_pole, MAS_vr_Xm.value, MAS_vr[i][:].value)

    # Now interpolate on to the HUXt longitudinal grid
    #longs, dlon, nlon = H.longitude_grid(lon_start=0.0 * u.rad, lon_stop=2 * np.pi * u.rad)
    #vr_in = np.interp(longs.value, MAS_vr_Xa.value, vr) * u.km / u.s

    return vr * u.km / u.s



def get_MAS_br_long_profile(cr, lat=0.0 * u.deg, savedir = ''):
    """
    Function to download, read and process MAS output to provide a longitude profile at a specified latitude
    of the Br for use as boundary conditions in HUXt.

    Args:
        cr: Integer Carrington rotation number
        lat: Latitude at which to extract the longitudinal profile, measure up from equator. Float with units of deg

    Returns:
        br_in: Br as a function of Carrington longitude at solar equator.
               Interpolated to HUXt longitudinal resolution. np.array (NDIM = 1)
    """
    assert (np.isnan(cr) == False and cr > 0)
    assert (lat >= -90.0 * u.deg)
    assert (lat <= 90.0 * u.deg)

    # Convert angle from equator to angle down from N pole
    ang_from_N_pole = np.pi / 2 - (lat.to(u.rad)).value

    # Check the data exist, if not, download them
    flag = get_MAS_boundary_conditions(cr, savedir = savedir)
    assert (flag > -1)

    # Read the HelioMAS data
    MAS_vr, MAS_vr_Xa, MAS_vr_Xm, MAS_br, MAS_br_Xa, MAS_br_Xm = read_MAS_vr_br(cr)

    # Extract the value at the given latitude
    br = np.ones(len(MAS_br_Xa))
    for i in range(0, len(MAS_br_Xa)):
        br[i] = np.interp(ang_from_N_pole, MAS_br_Xm.value, MAS_br[i][:])

    # Now interpolate on to the HUXt longitudinal grid
    #longs, dlon, nlon = H.longitude_grid(lon_start=0.0 * u.rad, lon_stop=2 * np.pi * u.rad)
    #br_in = np.interp(longs.value, MAS_br_Xa.value, br) 

    return br


def get_MAS_vr_map(cr, savedir = ''):
    """
    A function to download, read and process MAS output to provide HUXt boundary
    conditions as lat-long maps, along with angle from equator for the maps.
    Maps returned in native resolution, not HUXt resolution.

    Args:
        cr: Integer, Carrington rotation number

    Returns:
        vr_map: Solar wind speed as a Carrington longitude-latitude map. np.array with units of km/s
        vr_lats: The latitudes for the Vr map, relative to the equator. np.array with units of radians
        vr_longs: The Carrington longitudes for the Vr map, np.array with units of radians
    """

    assert (np.isnan(cr) == False and cr > 0)

    # Check the data exist, if not, download them
    flag = get_MAS_boundary_conditions(cr, savedir = savedir)
    if flag < 0:
        return -1, -1, -1

    # Read the HelioMAS data
    MAS_vr, MAS_vr_Xa, MAS_vr_Xm, MAS_br, MAS_br_Xa, MAS_br_Xm = read_MAS_vr_br(cr)

    vr_map = MAS_vr

    # Convert the lat angles from N-pole to equator centred
    vr_lats = (np.pi / 2) * u.rad - MAS_vr_Xm

    # Flip lats, so they're increasing in value
    vr_lats = np.flipud(vr_lats)
    vr_map = np.fliplr(vr_map)
    vr_longs = MAS_vr_Xa

    return vr_map.T, vr_longs, vr_lats


#download all the MAS files
def download_all_MAS_br(crstart = 1625, crend = 2260, masdir = os.environ['DBOX'] + 'Data\\MAS\\',
                        observatories = ['hmi', 'mdi', 'solis', 'gong', 'mwo', 'wso', 'kpo']):
    
    
    for obs in observatories:
        #move to the appropriate directory
        MYDIR = masdir + obs
        CHECK_FOLDER = os.path.isdir(MYDIR)
        if not CHECK_FOLDER:
            os.makedirs(MYDIR)
            print("created folder : ", MYDIR)
    
        else:
            print(MYDIR, "folder already exists.")
        os.chdir(MYDIR)
        
        
        for cr in range(crstart, crend):
            #move to the appropriate directory
            CRDIR = MYDIR + '\\CR' + str(cr)
            CHECK_FOLDER = os.path.isdir(CRDIR)
            if not CHECK_FOLDER:
                os.makedirs(CRDIR)
                print("created folder : ", CRDIR)
    
            else:
                print(CRDIR, "folder already exists.")
            os.chdir(CRDIR)
            
            get_MAS_output(cr=cr, observatory=obs)
            
#download_all_MAS_br()