import numpy as np

def GSE2RTN(mjd, xgse, ygse, zgse): #from Mathew Owens!!
    incl = 7.25*np.pi/180 #inclination of eliptic to helioequator
    #compute latitude at given MJD and small increment, to compute lat gradient
    #=======================================================================
    deltamjd = 1 #the incrememnt to use to calculate the gradient in the Earth's latitude.
    mjd2 = mjd+deltamjd

    #the longitude of the ascending node (from Dusan's coord.pdf document)
    omega1 = 73.666666667 + (0.01395833)*(mjd+3243.72)/365.25
    omega1 = omega1*np.pi/180
    omega2 = 73.666666667 + (0.01395833)*(mjd2+3243.72)/365.25
    omega2 = omega2*np.pi/180

    #ecliptic longitude....
    n1 = mjd-51544.5
    n2 = mjd2-51544.5
    g1 = 357.528+0.9856003*n1
    g2 = 357.528+0.9856003*n2
    g1 = g1*np.pi/180
    g2 = g2*np.pi/180
    L1 = 280.460+0.9856474*n1
    L2 = 280.460+0.9856474*n2
    phi1 = L1+1.915*np.sin(g1)+0.020*np.sin(2*g1)
    phi2 = L2+1.915*np.sin(g2)+0.020*np.sin(2*g2)
    phi1 = phi1*np.pi/180
    phi2 = phi2*np.pi/180

    #ecliptic longitude of the Sun's central meridian
    theta1 = np.arctan(np.cos(incl)*np.tan(phi1-omega1))
    theta2 = np.arctan(np.cos(incl)*np.tan(phi2-omega2))
    #theta2 = atan(cos(incl)*tan(phi2-omega2))

    pos1 = (np.unwrap(phi1-omega1) < np.pi) & (np.unwrap(theta1) < np.pi)
    pos2 = (np.unwrap(phi1-omega1) > np.pi) & (np.unwrap(theta1) > np.pi)
    theta1[pos1] = theta1[pos1]+np.pi
    theta1[pos2] = theta1[pos2]+np.pi
    pos1 = 0
    pos2 = 0
    pos1 = (np.unwrap(phi2-omega2) < np.pi) & (np.unwrap(theta2) < np.pi)
    pos2 = (np.unwrap(phi2-omega2) > np.pi) & (np.unwrap(theta2) > np.pi)
    theta2[pos1] = theta2[pos1]+np.pi
    theta2[pos2] = theta2[pos2]+np.pi

    B01 = np.arcsin(np.sin(theta1)*np.sin(incl))
    B02 = np.arcsin(np.sin(theta2)*np.sin(incl))

    Elat = (np.pi/2)+B01
    dElat = (np.pi/2)+B02


    #dlong=2*pi*deltamjd/t_syn;
    dlong = 2*np.pi*deltamjd/365.25

    #use the gradient of the Earth's latitiude to convert the data
    #(good to within ~0.1 degree)
    #%gradlat=-gradient(Elat,mjd);
    gradlat = np.arctan2((Elat-dElat),dlong)

    Hr = -xgse
    Hphi = -zgse*np.cos(gradlat)+ygse*np.sin(gradlat)
    Hth = -zgse*np.sin(gradlat)-ygse*np.cos(gradlat)

    return Hr, -Hth, Hphi
