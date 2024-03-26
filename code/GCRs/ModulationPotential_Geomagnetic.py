# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:50:22 2023

@author: vy902033

a script to compute the modulation potential from geomagnetic OSF estimates
"""


import numpy as np
import pandas as pd
from datetime import datetime
import os as os
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import conversions.helio_time as htime
import sunspots.sunspots as sunspots 
import plotting.mplot as mplot

confid_level = 0.68

updatenow = False
# Update font size
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,   # Font size for axis labels
    'xtick.labelsize': 12,  # Font size for x-axis tick labels
    'ytick.labelsize': 12,   # Font size for y-axis tick labels
    'legend.fontsize': 12
})
plt.rcParams.update({'font.family': 'Tahoma'})

fontsize = 12

phase_flip = 0.35 # teh phase at which the polairty reversal is assumed [0.35]
hcs_flag = 'R_av' #['R_av']

figdir = os.path.join(os.environ['DBOX'], 'Apps','Overleaf',
                      'A geomagnetic estimate of heliospheric modulation potential','figures')

osffilepath = os.path.join(os.environ['DBOX'], 'Data','OSF_OEL2023.csv')

# <codecell> process the data
osf_df = pd.read_csv(osffilepath)
osf_df['datetime'] = htime.mjd2datetime(osf_df['mjd'].to_numpy()) #datetime gets messed up?

#load the NM modulation potential data
nm_hmp = sunspots.load_oulu_phi(download_now = updatenow)

#average the NM data to 1 yr
nm_1y = nm_hmp.resample('1Y', on='datetime').mean() 
nm_1y['datetime'] = htime.mjd2datetime(nm_1y['mjd'].to_numpy())
nm_1y.reset_index(drop=True, inplace=True)

#add it to the osf dataframe 
osf_df['phi_NM'] = np.interp(osf_df['mjd'], nm_1y['mjd'], nm_1y['phi_NM'], left =np.nan, right =np.nan)

#load the hcs tilt
hcstilt =  sunspots.load_wsa_hcstilt(download_now = updatenow)


#average the NM data to 1 yr
hcs_1y = hcstilt.resample('1Y', on='datetime').mean() 
hcs_1y['datetime'] = htime.mjd2datetime(hcs_1y['mjd'].to_numpy())
hcs_1y.reset_index(drop=True, inplace=True)



#add it to the osf dataframe 
osf_df['wso_hcs_R'] = np.interp(osf_df['mjd'], hcs_1y['mjd'], hcs_1y['R_av'], left =np.nan, right =np.nan)
osf_df['wso_hcs_L'] = np.interp(osf_df['mjd'], hcs_1y['mjd'], hcs_1y['L_av'], left =np.nan, right =np.nan)



#estimate polarity, p, from solar cycle phase. Thomas et al., 2013 used a phase of 0.35 for the polarity change

#normalise the solar cycle phase to 1
osf_df['phase_1'] = osf_df['phase'] / (2*np.pi)


current_pol = 1
osf_df['polarity'] = np.nan
for n in range(0, len(osf_df)-2):
    osf_df.loc[n, 'polarity'] = current_pol
    if (osf_df.loc[n, 'phase_1'] <= phase_flip) & (osf_df.loc[n+1, 'phase_1'] > phase_flip):
        current_pol = current_pol * -1
        
        
        
        
#produce an average HCS variation over the cycle
hcs_phase, hcs_spe_avg, hcs_spe_std, hcs_spe = \
    sunspots.compute_phase_SPE(hcs_1y['mjd'], hcs_1y[hcs_flag], solarmin_mjd = None,
                      nphase = 11, plotnow = False)
    
#compute the new HCS term from the phase SPE
osf_df['hcs_new'] = np.interp(osf_df['phase'], 
                                hcs_phase, hcs_spe_avg, period = 2*np.pi)   


#plot the HCS SPE
#==================
fig = plt.figure(figsize = (10,8))

ax = plt.subplot(2,2,1)
for n in range(0, len(hcs_spe[0,:])):
    thishcs = hcs_spe[:,n]
    if not np.isnan(thishcs).all():
        ax.plot(hcs_phase/(2*np.pi), thishcs, label = 'SC' + str(n-11))
ax.legend()
ax.set(ylabel = r'HCS tilt, $\alpha$ [deg]', xlabel = 'Solar cycle phase')
ax.text(0.03, 0.95, '(a)', transform=plt.gca().transAxes, fontsize=fontsize)

ax = plt.subplot(2,2,2)   
ax.fill_between(hcs_phase/(2*np.pi), hcs_spe_avg - hcs_spe_std, hcs_spe_avg + hcs_spe_std,
                 color = 'silver', zorder = 0 )  
ax.plot(hcs_phase/(2*np.pi), hcs_spe_avg, 'k')  
ax.set(ylabel = r'HCS tilt, $\alpha$ [deg]', xlabel = 'Solar cycle phase')  
ax.text(0.03, 0.95, '(b)', transform=plt.gca().transAxes, fontsize=fontsize)


#compute theta prime
theta_prime = (np.pi/2)*(hcs_spe_avg 
                         - np.nanmin(hcs_spe_avg))/(np.nanmax(hcs_spe_avg) 
                                                    - np.nanmin(hcs_spe_avg)) 
ax = plt.subplot(2,2,3)                                                     
ax.plot(hcs_phase/(2*np.pi), theta_prime*180/np.pi, 'k')  
ax.set(ylabel = r' $\alpha *$ [deg]', xlabel = 'Solar cycle phase',
       yticks = [0,10,20,30,40,50,60,70,80,90])  
ax.text(0.03, 0.95, '(c)', transform=plt.gca().transAxes, fontsize=fontsize)

#compute p_eff

mask_late = hcs_phase/(2*np.pi) > 0.26
mask_early = hcs_phase/(2*np.pi) < 0.3

ax = plt.subplot(2,2,4)                                                     
ax.plot(hcs_phase[mask_late]/(2*np.pi),  1* (1- np.sin(theta_prime[mask_late])), 'r', label = ' p = +1')  
ax.plot(hcs_phase[mask_early]/(2*np.pi), -1*(1 - np.sin(theta_prime[mask_early])), 'b', label = ' p = -1') 
ax.set(ylabel = r' $p*$', xlabel = 'Solar cycle phase')  
ax.legend()
ax.plot(hcs_phase[mask_early]/(2*np.pi),  1* (1- np.sin(theta_prime[mask_early])), 'r--', label = ' p = +1')  
ax.plot(hcs_phase[mask_late]/(2*np.pi), -1*(1 - np.sin(theta_prime[mask_late])), 'b--', label = ' p = -1') 


ax.text(0.03, 0.95, '(d)', transform=plt.gca().transAxes, fontsize=fontsize)

plt.tight_layout()
fig.savefig(os.path.join(figdir, 'HCStilt.pdf'))



#output the average HCS tilt, alpha * and p* values
print('SC phase; alpha;  alpha*; p* (p=1); p* (p=-1)')
for n in range(0,len(hcs_phase)):
    string = ("{:.2f}".format(hcs_phase[n]/(2*np.pi)) + ' ' + 
              "{:.1f}".format(hcs_spe_avg[n]) + ' ' + 
              "{:.1f}".format(theta_prime[n]*180/np.pi) + ' ' + 
              "{:.2f}".format(1* (1- np.sin(theta_prime[n]))) + ' ' +
              "{:.2f}".format(-1* (1- np.sin(theta_prime[n])))
          ) 
    print(string)



        
# #create HCS SPEs from the different polarity cycles
# mask_ppos = osf_df['polarity'] > 0
# mask_pneg = osf_df['polarity'] <= 0

# #produce an average HCS variation over the cycle
# hcs_phase, hcs_spe_avg, hcs_spe_std, hcs_spe = \
#     sunspots.compute_phase_SPE(hcs_1y.loc[mask_ppos,'mjd'], hcs_1y.loc[mask_ppos, hcs_flag], solarmin_mjd = None,
#                       nphase = 11, plotnow = False)
# #compute the new HCS term from the phase SPE
# osf_df.loc[mask_ppos,'hcs_new_pol'] = np.interp(osf_df.loc[mask_ppos,'phase'], 
#                                 hcs_phase, hcs_spe_avg, period = 2*np.pi)  
# #produce an average HCS variation over the cycle
# hcs_phase, hcs_spe_avg, hcs_spe_std, hcs_spe = \
#     sunspots.compute_phase_SPE(hcs_1y.loc[mask_pneg,'mjd'], hcs_1y.loc[mask_pneg, hcs_flag], solarmin_mjd = None,
#                       nphase = 11, plotnow = False)
# #compute the new HCS term from the phase SPE
# osf_df.loc[mask_pneg,'hcs_new_pol'] = np.interp(osf_df.loc[mask_pneg,'phase'], 
#                                 hcs_phase, hcs_spe_avg, period = 2*np.pi)  

#Asvestari 2016 HCS tilt



#hcs tilt is given by equation 1.
# osf_df['HCStilt'] = np.nan
# mask = osf_df['phase_1'] <= 0.4
# osf_df.loc[mask, 'HCStilt'] = 1.5 + 909.5 * osf_df.loc[mask, 'phase_1']*osf_df.loc[mask, 'phase_1']
# mask = osf_df['phase_1'] > 0.4
# osf_df.loc[mask, 'HCStilt'] = 11.1 + 118.8 * (1 - osf_df.loc[mask, 'phase_1']) * (1- osf_df.loc[mask, 'phase_1'])
# mask = osf_df['HCStilt'] > 70
# osf_df.loc[mask, 'HCStilt'] = 70

# plt.figure()
# plt.plot(osf_df['HCStilt'])


           
# <codecell> Asvestari 2016 modulation potential
HCStilt = osf_df['hcs_new']
polarity = osf_df['polarity']


theta_eff = (np.pi/2)*(HCStilt - np.nanmin(HCStilt))/(np.nanmax(HCStilt) - np.nanmin(HCStilt)) 
peff =  polarity * (1 - np.sin(theta_eff))
       

# #rerpoduce Fig5 from Asvestari2106

fig = plt.figure(figsize = (8,10))

ax = plt.subplot(5,1,1)
ax.plot(osf_df['datetime'], osf_df['phi_NM'], 'k')
ax.set_ylabel(r'$\phi$ [MV]')
xx = ax.get_xlim()
ax.set_xlim(xx)
sunspots.PlotAlternateCycles()
ax.text(0.01, 0.05, '(a)', transform=plt.gca().transAxes, fontsize=fontsize)
ax.set_xticklabels([])

ax = plt.subplot(5,1,2)
ax.plot(osf_df['datetime'], osf_df['ssn'], 'k')
ax.set_ylabel(r'SN')
ax.set_xlim(xx)
yy = ax.get_ylim()
ax.set_ylim([0, yy[1]])
sunspots.PlotAlternateCycles()
ax.text(0.01, 0.05, '(b)', transform=plt.gca().transAxes, fontsize=fontsize)
ax.set_xticklabels([])

ax = plt.subplot(5,1,3)
ax.plot(osf_df['datetime'], osf_df['OSF_SSN'], 'b', label = 'SN')
ax.plot(osf_df['datetime'], osf_df['OSF_GEO'], 'k', label = 'GEO')
ax.plot(osf_df['datetime'], osf_df['OSF_OMNI'], 'r', label = 'OMNI')
ax.set_ylabel('OSF, $F_S$ ' + '\n' + r'[x$10^{14}$ Wb]')
ax.set_xlim(xx)
ax.set_ylim([3, 15])
ax.legend(loc = 'upper left', ncol = 3)
sunspots.PlotAlternateCycles()
ax.text(0.01, 0.05, '(c)', transform=plt.gca().transAxes, fontsize=fontsize)
ax.set_xticklabels([])
#compute the OSF correlations for teh periods of overlap
non_nan_indices = osf_df[osf_df['OSF_SSN'].notna() & 
                         osf_df['OSF_GEO'].notna() & 
                         osf_df['OSF_OMNI'].notna()].index
rssn, nssn = mplot.lin_correl(osf_df['OSF_SSN'].loc[non_nan_indices], 
                              osf_df['OSF_OMNI'].loc[non_nan_indices])
rgeo, ngeo = mplot.lin_correl(osf_df['OSF_GEO'].loc[non_nan_indices], 
                              osf_df['OSF_OMNI'].loc[non_nan_indices])
print('r (OSF_OMNI OSF_SSN) = ' + str(rssn))
print('r (OSF_OMNI OSF_GEO) = ' + str(rgeo))
mplot.mengZ(rssn, nssn , rgeo, ngeo)
ax.text(0.85, 0.85, r'$r_L$ = ' + "{:.3f}".format(rssn), color = 'b', transform=plt.gca().transAxes, fontsize=fontsize)
ax.text(0.85, 0.70, r'$r_L$ = ' + "{:.3f}".format(rgeo), color = 'k', transform=plt.gca().transAxes, fontsize=fontsize)


ax = plt.subplot(5,1,4)
ax.plot(osf_df['datetime'], HCStilt, 'k', label = 'Phase')
ax.plot(osf_df['datetime'], osf_df['wso_hcs_R'], 'r', label = 'WSO')
ax.set_ylabel(r'HCS tilt,' + '\n' + r'$\alpha$ [deg]')
ax.set_xlim(xx)
ax.set_ylim([0,90])
ax.set_yticks([0,30,60,90])
ax.legend(loc = 'upper left', ncol = 2)
sunspots.PlotAlternateCycles()
ax.text(0.01, 0.05, '(d)', transform=plt.gca().transAxes, fontsize=fontsize)
ax.set_xticklabels([])

# ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
# color = 'tab:red'
# ax2.set_ylabel(r'sin($\theta_{EFF}$)', color=color)  # we already handled the x-label with ax1
# ax2.plot(osf_df['datetime'], np.sin(theta_eff), color=color)
# ax2.tick_params(axis='y', labelcolor=color)

ax = plt.subplot(5,1,5)
ax.plot(osf_df['datetime'], polarity, 'k')
ax.set_ylabel(r'Polarity, $p$')
ax.set_xlim(xx)
sunspots.PlotAlternateCycles()
ax.text(0.01, 0.05, '(e)', transform=plt.gca().transAxes, fontsize=fontsize)


ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.set_ylabel(r"$p* = p (1 - \sin \alpha *)$", color='r')  # we already handled the x-label with ax1
ax2.plot(osf_df['datetime'], peff, color='r')
ax2.tick_params(axis='y', labelcolor='r')


fig.savefig(os.path.join(figdir , 'data_summary.pdf'))





# <codecell> Models
def phi_model_E2016(params, x):
    phi_0, alpha_0, n, beta = params
    
    osf = x[:,0]
    HCStilt = x[:,1]
    polarity = x[:,2]
    
    return phi_0 * ( (osf/10) ** (n - HCStilt/alpha_0) ) * (1 - beta * polarity)

def objective_E2016(params, x, y_observed):
    y_predicted = phi_model_E2016(params, x)
    #mae = abs(y_predicted - y_observed)
    #error = np.nansum(np.exp(-mae))
    error = np.nanmean((y_predicted - y_observed)**2)
    return error




def phi_model_new(params, x):
    
    hcscoeff, peffcoeff, osfpower, scalingfac = params
    
    osf = x[:,0]
    HCStilt = x[:,1]
    polarity = x[:,2]
    
    
    
    #sinHCS =   np.power( (90.0 - HCStilt)/90.0, exponent)
    sinHCS =  np.power( np.sin( HCStilt*np.pi/180),1)
    #peff =  polarity * (1 - sinHCS)
    
    theta_eff = (np.pi/2)*(HCStilt - np.nanmin(HCStilt))/(np.nanmax(HCStilt) - np.nanmin(HCStilt)) 
    peff =  polarity * (1 - np.sin(theta_eff))
    
    phi = scalingfac* (osf/10) ** osfpower * (1 + hcscoeff * sinHCS) * ( 1 + peffcoeff*peff) 
    
    
    return phi

def objective_new(params, x, y_observed):
    y_predicted = phi_model_new(params, x)
    #mae = abs(y_predicted - y_observed)
    #error = np.nansum(np.exp(-mae))
    error = np.nanmean((y_predicted - y_observed)**2)
    #error = np.nanmean(mae)
    
    #error =  1 - mplot.lin_correl(y_predicted, y_observed)[0]**2
    return error


# <codecell> Optimise the fit params to OSF OMNI

xx_sc = np.array([250, 1100])


#input data
osf = osf_df['OSF_OMNI'].to_numpy()
#hcs = osf_df['HCStilt'].to_numpy()
#hcs = osf_df['hcs_new_pol'].to_numpy()
hcs = osf_df['wso_hcs_R'].to_numpy()
p = osf_df['polarity'].to_numpy()
phi = osf_df['phi_NM'].to_numpy()






#input data
x = np.empty((len(osf_df),3))
x[:,0] = osf
x[:,1] = hcs
x[:,2] = p

#data to match
y_observed = phi

# Initial guesses for the parameters
initial_params = [1473.9 * 0.6, 150, 1.03, 0.095]

# Perform the optimization
result = minimize(objective_E2016, initial_params, args=(x, y_observed), method = 'Nelder-Mead')

# Extract the best-fit parameters
best_fit_params = result.x
print("OMNI, WSO")
print("Best-fit parameters, A2016:", best_fit_params)


phi_E2016_omni = phi_model_E2016(best_fit_params, x)
#fit_slope, fit_intercept, conf_slope, conf_intercept =  mplot.lin_regress_bootstrap(phi_E2016_omni, phi)
pE2016, regressparamsE2016 = mplot.lin_regress(phi_E2016_omni, phi, 
                                               plotnow = False, confid_level = confid_level)
confid_int, prob_int = mplot.lin_regress_confid_int(phi_E2016_omni, regressparamsE2016)

fig = plt.figure(figsize=(10,10))


ax = plt.subplot(2,2,1)
ax.plot(osf_df['datetime'], phi, 'k', label = 'Neutron monitor (U2017)')
xx_ts = ax.get_xlim()
ax.set_xlim(xx_ts)
ax.plot(osf_df['datetime'], phi_E2016_omni,'b', label = 'OMNI, WSO (A2016)')
ax.legend(loc = 'lower left')
yy_ts = ax.get_ylim()
ax.set_ylim(yy_ts)
ax.set_ylabel(r'$\phi$ [MV]')
sunspots.PlotAlternateCycles()
ax.text(0.05, 0.95, '(a)', transform=plt.gca().transAxes, fontsize=12)
#add uncertainty range
# ax.fill_between(osf_df['datetime'], conf_slope[0] * phi_E2016_omni + conf_intercept[0], 
#                   conf_slope[1] * phi_E2016_omni + conf_intercept[1], 
#                   color='pink')
ax.fill_between(osf_df['datetime'], phi_E2016_omni - prob_int, 
                  phi_E2016_omni + prob_int, 
                  color='lightblue', alpha = 1)


ax = plt.subplot(2,2,2)
ax.plot(phi_E2016_omni, phi, 'ro')
rE16omni, nE16omni = mplot.lin_correl(phi_E2016_omni, phi)
maeE16omni = np.nanmean(abs(phi_E2016_omni - phi))
ax.set_title(r'$r_L$ = ' + "{:.3f}".format(rE16omni) + r'; MAE = ' + "{:.2f}".format(maeE16omni))
ax.set_ylabel(r'$\phi$ NM [MV]')
ax.set_xlabel(r'$\phi$ OMNI, WSO (A2016) [MV]')
ax.set_ylim(xx_sc); ax.set_xlim(xx_sc)
#plt.title(r'$r_L$ = ' + "{:.2f}".format(rl[0,1]), fontsize=12) 
ax.plot(xx_sc,xx_sc,'k--')
ax.text(0.05, 0.95, '(b)', transform=plt.gca().transAxes, fontsize=fontsize)

# xrange = xx_sc
# ax.fill_between(xrange, conf_slope[0] * xrange + conf_intercept[0], 
#                   conf_slope[1] * xrange + conf_intercept[1], 
#                   color='pink')
p, regressparams = mplot.lin_regress(phi_E2016_omni, phi, confid_level = confid_level,
                                     plotnow = True, ax = ax, 
                                     color = 'lightblue', linecolor = 'b', alpha =1)


#derive a new model
#==============================================================================




# Initial guesses for the parameters
initial_params = [ 0.7, -0.05, 1, 50]

# Perform the optimization
result = minimize(objective_new, initial_params, args=(x, y_observed), method = 'Nelder-Mead')

# Extract the best-fit parameters
best_fit_params = result.x
osf_p_hcs = phi_model_new(best_fit_params, x)
print("Best-fit parameters, O2024:", best_fit_params)



#that's the correlation maximised, then scale to phi

#idx = np.isfinite(osf_p_hcs) & np.isfinite(y_observed)
#coeffs = np.polyfit(osf_p_hcs[idx], y_observed[idx], 1)
#phi_new = np.polyval(coeffs, osf_p_hcs)

#print("Linear scaling coeffs", coeffs)

phi_new_omni = osf_p_hcs
#fit_slope, fit_intercept, conf_slope, conf_intercept =  mplot.lin_regress_bootstrap(phi_new_omni, phi)
pnew, regressparamsnew = mplot.lin_regress(phi_new_omni, phi, confid_level = confid_level,
                                           plotnow = False, ax = ax)
confid_int, prob_int = mplot.lin_regress_confid_int(phi_new_omni, regressparamsnew)


ax = plt.subplot(2,2,3)
#xvals = phi_model_new(best_fit_params, x)

ax.plot(osf_df['datetime'], phi, 'k', label = 'Neutron monitor (U2017)')
xx = ax.get_xlim()
ax.set_xlim(xx_ts)
ax.plot(osf_df['datetime'], phi_new_omni, 'r', label = 'OMNI, WSO (new model)')
ax.legend(loc = 'lower left')
ax.set_ylim(yy_ts)
ax.set_ylabel(r'$\phi$ [MV]')
ax.set_xlabel('Year')
sunspots.PlotAlternateCycles()
ax.text(0.05, 0.95, '(c)', transform=plt.gca().transAxes, fontsize=fontsize)
#add uncertainty range
# ax.fill_between(osf_df['datetime'], conf_slope[0] * phi_new_omni + conf_intercept[0], 
#                   conf_slope[1] * phi_new_omni + conf_intercept[1], 
#                   color='pink')
ax.fill_between(osf_df['datetime'], phi_new_omni - prob_int, 
                  phi_new_omni + prob_int, 
                  color='pink', alpha = 1)

ax = plt.subplot(2,2,4)
ax.plot(phi_new_omni, phi, 'ro')
rnew_omni, nnew_omni = mplot.lin_correl(phi_new_omni, phi)
maenew_omni = np.nanmean(abs(phi_new_omni - phi))
ax.set_title(r'$r_L$ = ' + "{:.3f}".format(rnew_omni) + r'; MAE = ' + "{:.2f}".format(maenew_omni))
ax.set_ylabel(r'$\phi$ NM [MV]')
ax.set_xlabel(r'$\phi$ OMNI, WSO (new model) [MV]')
ax.set_ylim(xx_sc); ax.set_xlim(xx_sc)
ax.plot(xx_sc,xx_sc,'k--')
ax.text(0.05, 0.95, '(d)', transform=plt.gca().transAxes, fontsize=fontsize)

# xrange = xx_sc
# ax.fill_between(xrange, conf_slope[0] * xrange + conf_intercept[0], 
#                   conf_slope[1] * xrange + conf_intercept[1], 
#                   color='pink', label='Confidence Interval')
p, regressparams = mplot.lin_regress(phi_new_omni, phi, confid_level = confid_level,
                                     plotnow = True, ax = ax, color = 'pink', alpha =1)

plt.tight_layout()


fig.savefig(os.path.join(figdir, 'model_OMNI.pdf'))




mplot.mengZ(rE16omni, nE16omni , rnew_omni, nnew_omni)
# <codecell> Optimise the fit params to OSF OMNI, average HCS
#input data
osf = osf_df['OSF_OMNI'].to_numpy()
#hcs = osf_df['HCStilt'].to_numpy()
#hcs = osf_df['hcs_new_pol'].to_numpy()
hcs = osf_df['hcs_new'].to_numpy()
p = osf_df['polarity'].to_numpy()
phi = osf_df['phi_NM'].to_numpy()




#input data
x = np.empty((len(osf_df),3))
x[:,0] = osf
x[:,1] = hcs
x[:,2] = p

#data to match
y_observed = phi

# Initial guesses for the parameters
initial_params = [1473.9 * 0.6, 150, 1.03, 0.095]

# Perform the optimization
result = minimize(objective_E2016, initial_params, args=(x, y_observed), method = 'Nelder-Mead')

# Extract the best-fit parameters
best_fit_params = result.x

print("OMNI")
print("Best-fit parameters, A2016:", best_fit_params)


phi_E2016_omni = phi_model_E2016(best_fit_params, x)
#fit_slope, fit_intercept, conf_slope, conf_intercept =  mplot.lin_regress_bootstrap(phi_E2016_omni, phi)
pE2016, regressparamsE2016 = mplot.lin_regress(phi_E2016_omni, phi, 
                                               plotnow = False, confid_level = confid_level)
confid_int, prob_int = mplot.lin_regress_confid_int(phi_E2016_omni, regressparamsE2016)

fig = plt.figure(figsize=(10,10))
#xx_sc = np.array([250, 1100])

ax = plt.subplot(2,2,1)
ax.plot(osf_df['datetime'], phi, 'k', label = 'Neutron monitor (U2017)')
xx_ts = ax.get_xlim()
ax.set_xlim(xx_ts)
ax.plot(osf_df['datetime'], phi_E2016_omni,'b', label = 'OMNI (A2016)')
ax.legend(loc = 'lower left')
yy_ts = ax.get_ylim()
ax.set_ylim(yy_ts)
ax.set_ylabel(r'$\phi$ [MV]')
sunspots.PlotAlternateCycles()
ax.text(0.05, 0.95, '(a)', transform=plt.gca().transAxes, fontsize=fontsize)
#add uncertainty range
# ax.fill_between(osf_df['datetime'], conf_slope[0] * phi_E2016_omni + conf_intercept[0], 
#                   conf_slope[1] * phi_E2016_omni + conf_intercept[1], 
#                   color='pink')
ax.fill_between(osf_df['datetime'], phi_E2016_omni - prob_int, 
                  phi_E2016_omni + prob_int, 
                  color='lightblue', alpha = 1)


ax = plt.subplot(2,2,2)
ax.plot(phi_E2016_omni, phi, 'ro')
rE16omni, nE16omni = mplot.lin_correl(phi_E2016_omni, phi)
maeE16omni = np.nanmean(abs(phi_E2016_omni - phi))
ax.set_title(r'$r_L$ = ' + "{:.3f}".format(rE16omni) + r'; MAE = ' + "{:.2f}".format(maeE16omni))
ax.set_ylabel(r'$\phi$ NM [MV]')
ax.set_xlabel(r'$\phi$ OMNI (A2016) [MV]')
ax.set_ylim(xx_sc); ax.set_xlim(xx_sc)
#plt.title(r'$r_L$ = ' + "{:.2f}".format(rl[0,1]), fontsize=12) 
ax.plot(xx_sc,xx_sc,'k--')
ax.text(0.05, 0.95, '(b)', transform=plt.gca().transAxes, fontsize=fontsize)

# xrange = xx_sc
# ax.fill_between(xrange, conf_slope[0] * xrange + conf_intercept[0], 
#                   conf_slope[1] * xrange + conf_intercept[1], 
#                   color='pink')
p, regressparams = mplot.lin_regress(phi_E2016_omni, phi, confid_level = confid_level,
                                     plotnow = True, ax = ax, 
                                     color = 'lightblue', linecolor = 'b', alpha =1)


#derive a new model
#==============================================================================



# Initial guesses for the parameters
initial_params = [ 0.7, -0.05, 1, 50]

# Perform the optimization
result = minimize(objective_new, initial_params, args=(x, y_observed), method = 'Nelder-Mead')

# Extract the best-fit parameters
best_fit_params = result.x
osf_p_hcs = phi_model_new(best_fit_params, x)
print("Best-fit parameters, O2024:", best_fit_params)



phi_new_omni = osf_p_hcs
#fit_slope, fit_intercept, conf_slope, conf_intercept =  mplot.lin_regress_bootstrap(phi_new_omni, phi)
pnew, regressparamsnew = mplot.lin_regress(phi_new_omni, phi, confid_level = confid_level,
                                           plotnow = False, ax = ax)
confid_int, prob_int = mplot.lin_regress_confid_int(phi_new_omni, regressparamsnew)


ax = plt.subplot(2,2,3)
#xvals = phi_model_new(best_fit_params, x)

ax.plot(osf_df['datetime'], phi, 'k', label = 'Neutron monitor (U2017)')
xx = ax.get_xlim()
ax.set_xlim(xx_ts)
ax.plot(osf_df['datetime'], phi_new_omni, 'r', label = 'OMNI (new model)')
ax.legend(loc = 'lower left')
ax.set_ylim(yy_ts)
ax.set_ylabel(r'$\phi$ [MV]')
ax.set_xlabel('Year')
sunspots.PlotAlternateCycles()
ax.text(0.05, 0.95, '(c)', transform=plt.gca().transAxes, fontsize=fontsize)
#add uncertainty range
# ax.fill_between(osf_df['datetime'], conf_slope[0] * phi_new_omni + conf_intercept[0], 
#                   conf_slope[1] * phi_new_omni + conf_intercept[1], 
#                   color='pink')
ax.fill_between(osf_df['datetime'], phi_new_omni - prob_int, 
                  phi_new_omni + prob_int, 
                  color='pink', alpha = 1)

ax = plt.subplot(2,2,4)
ax.plot(phi_new_omni, phi, 'ro')
rnew_omni, nnew_omni = mplot.lin_correl(phi_new_omni, phi)
maenew_omni = np.nanmean(abs(phi_new_omni - phi))
ax.set_title(r'$r_L$ = ' + "{:.3f}".format(rnew_omni) + r'; MAE = ' + "{:.2f}".format(maenew_omni))
ax.set_ylabel(r'$\phi$ NM [MV]')
ax.set_xlabel(r'$\phi$ OSF (new model) [MV]')
ax.set_ylim(xx_sc); ax.set_xlim(xx_sc)
ax.plot(xx_sc,xx_sc,'k--')
ax.text(0.05, 0.95, '(d)', transform=plt.gca().transAxes, fontsize=fontsize)

# xrange = xx_sc
# ax.fill_between(xrange, conf_slope[0] * xrange + conf_intercept[0], 
#                   conf_slope[1] * xrange + conf_intercept[1], 
#                   color='pink', label='Confidence Interval')
p, regressparams = mplot.lin_regress(phi_new_omni, phi, confid_level = confid_level,
                                     plotnow = True, ax = ax, color = 'pink', alpha =1)

plt.tight_layout()


fig.savefig(os.path.join(figdir, 'model_OMNI_averageHCS.pdf'))


mplot.mengZ(rE16omni, nE16omni , rnew_omni, nnew_omni)
# <codecell> Optimise the fit params to OSF GEO
#input data
osf = osf_df['OSF_GEO'].to_numpy()
#hcs = osf_df['HCStilt'].to_numpy()
#hcs = osf_df['hcs_new_pol'].to_numpy()
hcs = osf_df['hcs_new'].to_numpy()
p = osf_df['polarity'].to_numpy()
phi = osf_df['phi_NM'].to_numpy()


#input data
x = np.empty((len(osf_df),3))
x[:,0] = osf
x[:,1] = hcs
x[:,2] = p

#data to match
y_observed = phi

# Initial guesses for the parameters
initial_params = [1473.9 * 0.6, 150, 1.03, 0.095]

# Perform the optimization
result = minimize(objective_E2016, initial_params, args=(x, y_observed), method = 'Nelder-Mead')

# Extract the best-fit parameters
best_fit_params = result.x
print("GEO")
print("Best-fit parameters, A2016:", best_fit_params)
phi_E2016_geo = phi_model_E2016(best_fit_params, x)
#fit_slope, fit_intercept, conf_slope, conf_intercept =  mplot.lin_regress_bootstrap(phi_E2016_geo, phi)
pE2016_geo, regressparamsE2016geo = mplot.lin_regress(phi_E2016_geo, phi, 
                                               plotnow = False, confid_level = confid_level)
confid_int_E2016, prob_int_E2016 = mplot.lin_regress_confid_int(phi_E2016_geo, regressparamsE2016geo)


fig = plt.figure(figsize=(10,10))
#xx_sc = np.array([250, 1100])

ax = plt.subplot(2,2,1)
ax.plot(osf_df['datetime'], phi, 'k', label = 'Neutron monitor (U2017)')
xx_ts = ax.get_xlim()
ax.set_xlim(xx_ts)
ax.plot(osf_df['datetime'], phi_E2016_geo,'b', label = 'GEO (A2016)')
ax.legend()
yy_ts = ax.get_ylim()
ax.set_ylim(yy_ts)
ax.set_ylabel(r'$\phi$ [MV]')
sunspots.PlotAlternateCycles()
ax.text(0.05, 0.95, '(a)', transform=plt.gca().transAxes, fontsize=fontsize)
#add uncertainty range
# ax.fill_between(osf_df['datetime'], conf_slope[0] * phi_E2016_geo + conf_intercept[0], 
#                   conf_slope[1] * phi_E2016_geo + conf_intercept[1], 
#                   color='pink')
ax.fill_between(osf_df['datetime'], phi_E2016_geo - prob_int_E2016, 
                  phi_E2016_geo + prob_int_E2016, 
                  color='lightblue', alpha = 1)


ax = plt.subplot(2,2,2)
ax.plot(phi_E2016_geo, phi, 'ro')
rE16_geo, nE16geo = mplot.lin_correl(phi_E2016_geo, phi)
maeE16_geo = np.nanmean(abs(phi_E2016_geo - phi))
ax.set_title(r'$r_L$ = ' + "{:.3f}".format(rE16_geo) + r'; MAE = ' + "{:.2f}".format(maeE16_geo))
ax.set_ylabel(r'$\phi$ NM [MV]')
ax.set_xlabel(r'$\phi$ GEO (A2016) [MV]')
ax.set_ylim(xx_sc); ax.set_xlim(xx_sc)
#plt.title(r'$r_L$ = ' + "{:.2f}".format(rl[0,1]), fontsize=12) 
ax.plot(xx_sc,xx_sc,'k--')
ax.text(0.05, 0.95, '(b)', transform=plt.gca().transAxes, fontsize=fontsize)

# xrange = xx_sc
# ax.fill_between(xrange, conf_slope[0] * xrange + conf_intercept[0], 
#                   conf_slope[1] * xrange + conf_intercept[1], 
#                   color='pink', label='Confidence Interval')
p_E2016_geo, regressparams_E2016_geo = mplot.lin_regress(phi_E2016_geo, phi, confid_level = confid_level,
                                     plotnow = True, ax = ax, color = 'lightblue',
                                     linecolor = 'b', alpha =1)
#derive a new model
#==============================================================================


# Initial guesses for the parameters
initial_params = [ 0.7, -0.05, 1, 50]

# Perform the optimization
result = minimize(objective_new, initial_params, args=(x, y_observed), method = 'Nelder-Mead')

# Extract the best-fit parameters
best_fit_params = result.x
osf_p_hcs = phi_model_new(best_fit_params, x)
print("Best-fit parameters, O2024:", best_fit_params)



#that's the correlation maximised, then scale to phi

#idx = np.isfinite(osf_p_hcs) & np.isfinite(y_observed)
#coeffs = np.polyfit(osf_p_hcs[idx], y_observed[idx], 1)
#phi_new = np.polyval(coeffs, osf_p_hcs)

#print("Linear scaling coeffs", coeffs)

phi_new_geo = osf_p_hcs
#fit_slope, fit_intercept, conf_slope, conf_intercept =  mplot.lin_regress_bootstrap(phi_new_geo, phi)
pnew_geo, regressparamsnewgeo = mplot.lin_regress(phi_new_geo, phi, 
                                               plotnow = False, confid_level = confid_level)
confid_int_new, prob_int_new = mplot.lin_regress_confid_int(phi_new_geo, regressparamsnewgeo)


ax = plt.subplot(2,2,3)
#xvals = phi_model_new(best_fit_params, x)

yvals = phi
ax.plot(osf_df['datetime'], yvals, 'k', label = 'Neutron monitor (U2017)')
xx = ax.get_xlim()
ax.set_xlim(xx_ts)
ax.plot(osf_df['datetime'], phi_new_geo, 'r', label = 'GEO (new model)')
ax.legend()
ax.set_ylim(yy_ts)
ax.set_ylabel(r'$\phi$ [MV]')
ax.set_xlabel('Year')
sunspots.PlotAlternateCycles()
ax.text(0.05, 0.95, '(c)', transform=plt.gca().transAxes, fontsize=fontsize)
#add uncertainty range
# ax.fill_between(osf_df['datetime'], conf_slope[0] * phi_new_geo + conf_intercept[0], 
#                   conf_slope[1] * phi_new_geo + conf_intercept[1], 
#                   color='pink')
ax.fill_between(osf_df['datetime'], phi_new_geo - prob_int_new, 
                  phi_new_geo + prob_int_new, 
                  color='pink', alpha = 1)

ax = plt.subplot(2,2,4)
ax.plot(phi_new_geo, yvals, 'ro')
rnew_geo, nnew_geo = mplot.lin_correl(phi_new_geo, yvals)
maenew_geo = np.nanmean(abs(phi_new_geo - yvals))
ax.set_title(r'$r_L$ = ' + "{:.3f}".format(rnew_geo) + r'; MAE = ' + "{:.2f}".format(maenew_geo))
ax.set_ylabel(r'$\phi$ NM [MV]')
ax.set_xlabel(r'$\phi$ GEO (new model) [MV]')
ax.set_ylim(xx_sc); ax.set_xlim(xx_sc)
ax.plot(xx_sc,xx_sc,'k--')
ax.text(0.05, 0.95, '(d)', transform=plt.gca().transAxes, fontsize=fontsize)

# xrange = xx_sc
# ax.fill_between(xrange, conf_slope[0] * xrange + conf_intercept[0], 
#                   conf_slope[1] * xrange + conf_intercept[1], 
#                   color='pink', label='Confidence Interval')
p, regressparams = mplot.lin_regress(phi_new_geo, phi, confid_level = confid_level,
                                     plotnow = True, ax = ax, color = 'pink', alpha =1)

plt.tight_layout()

fig.savefig(os.path.join(figdir, 'model_geo.pdf'))

mplot.mengZ(rE16_geo, nE16geo , rnew_geo, nnew_geo )

# <codecell> Optimise the fit params to OSF SSN
#input data
osf = osf_df['OSF_SSN'].to_numpy()
#hcs = osf_df['HCStilt'].to_numpy()
#hcs = osf_df['hcs_new_pol'].to_numpy()
hcs = osf_df['hcs_new'].to_numpy()
p = osf_df['polarity'].to_numpy()
phi = osf_df['phi_NM'].to_numpy()


#input data
x = np.empty((len(osf_df),3))
x[:,0] = osf
x[:,1] = hcs
x[:,2] = p

#data to match
y_observed = phi

# Initial guesses for the parameters
initial_params = [1473.9 * 0.6, 150, 1.03, 0.095]

# Perform the optimization
result = minimize(objective_E2016, initial_params, args=(x, y_observed), method = 'Nelder-Mead')

# Extract the best-fit parameters
best_fit_params = result.x
print('SN')
print("Best-fit parameters, A2106:", best_fit_params)
phi_E2016_sn = phi_model_E2016(best_fit_params, x)
#fit_slope, fit_intercept, conf_slope, conf_intercept =  mplot.lin_regress_bootstrap(phi_E2016_geo, phi)
pE2016_sn, regressparamsE2016sn = mplot.lin_regress(phi_E2016_geo, phi, 
                                               plotnow = False, confid_level = confid_level)
confid_int_E2016_sn, prob_int_E2016_sn = mplot.lin_regress_confid_int(phi_E2016_geo, regressparamsE2016geo)


fig = plt.figure(figsize=(10,10))
#xx_sc = np.array([250, 1100])

ax = plt.subplot(2,2,1)
ax.plot(osf_df['datetime'], yvals, 'k', label = 'Neutron monitor (U2017)')
xx_ts = ax.get_xlim()
ax.set_xlim(xx_ts)
ax.plot(osf_df['datetime'], phi_E2016_sn,'b', label = 'SN (A2016)')
ax.legend()
yy_ts = ax.get_ylim()
ax.set_ylim(yy_ts)
ax.set_ylabel(r'$\phi$ [MV]')
sunspots.PlotAlternateCycles()
ax.text(0.05, 0.95, '(a)', transform=plt.gca().transAxes, fontsize=fontsize)
#add uncertainty range
# ax.fill_between(osf_df['datetime'], conf_slope[0] * phi_E2016_geo + conf_intercept[0], 
#                   conf_slope[1] * phi_E2016_geo + conf_intercept[1], 
#                   color='pink')
ax.fill_between(osf_df['datetime'], phi_E2016_sn - prob_int_E2016_sn, 
                  phi_E2016_sn + prob_int_E2016_sn, 
                  color='lightblue', alpha = 1)


ax = plt.subplot(2,2,2)
ax.plot(phi_E2016_sn, yvals, 'ro')
rE16_sn, n = mplot.lin_correl(phi_E2016_sn, yvals)
maeE16_sn = np.nanmean(abs(phi_E2016_sn - yvals))
ax.set_title(r'$r_L$ = ' + "{:.3f}".format(rE16_sn) + r'; MAE = ' + "{:.2f}".format(maeE16_sn))
ax.set_ylabel(r'$\phi$ NM [MV]')
ax.set_xlabel(r'$\phi$ SN (A2016) [MV]')
ax.set_ylim(xx_sc); ax.set_xlim(xx_sc)
#plt.title(r'$r_L$ = ' + "{:.2f}".format(rl[0,1]), fontsize=12) 
ax.plot(xx_sc,xx_sc,'k--')
ax.text(0.05, 0.95, '(b)', transform=plt.gca().transAxes, fontsize=fontsize)

# xrange = xx_sc
# ax.fill_between(xrange, conf_slope[0] * xrange + conf_intercept[0], 
#                   conf_slope[1] * xrange + conf_intercept[1], 
#                   color='pink', label='Confidence Interval')
p_E2016_sn, regressparams_E2016_sn = mplot.lin_regress(phi_E2016_sn, phi, confid_level = confid_level,
                                     plotnow = True, ax = ax, color = 'lightblue',
                                     linecolor = 'b', alpha =1)
#derive a new model
#==============================================================================


# Initial guesses for the parameters
initial_params = [ 0.7, -0.05, 1, 50]

# Perform the optimization
result = minimize(objective_new, initial_params, args=(x, y_observed), method = 'Nelder-Mead')

# Extract the best-fit parameters
best_fit_params = result.x
osf_p_hcs = phi_model_new(best_fit_params, x)
print("Best-fit parameters, O2024:", best_fit_params)



#that's the correlation maximised, then scale to phi

#idx = np.isfinite(osf_p_hcs) & np.isfinite(y_observed)
#coeffs = np.polyfit(osf_p_hcs[idx], y_observed[idx], 1)
#phi_new = np.polyval(coeffs, osf_p_hcs)

#print("Linear scaling coeffs", coeffs)

phi_new_sn = osf_p_hcs
#fit_slope, fit_intercept, conf_slope, conf_intercept =  mplot.lin_regress_bootstrap(phi_new_geo, phi)
pnew_sn, regressparamsnewsn = mplot.lin_regress(phi_new_sn, phi, 
                                               plotnow = False, confid_level = confid_level)
confid_int_new_sn, prob_int_new_sn = mplot.lin_regress_confid_int(phi_new_geo, regressparamsnewgeo)


ax = plt.subplot(2,2,3)
#xvals = phi_model_new(best_fit_params, x)

yvals = phi
ax.plot(osf_df['datetime'], yvals, 'k', label = 'Neutron monitor (U2017)')
xx = ax.get_xlim()
ax.set_xlim(xx_ts)
ax.plot(osf_df['datetime'], phi_new_sn, 'r', label = 'SN (new model)')
ax.legend()
ax.set_ylim(yy_ts)
ax.set_ylabel(r'$\phi$ [MV]')
ax.set_xlabel('Year')
sunspots.PlotAlternateCycles()
ax.text(0.05, 0.95, '(c)', transform=plt.gca().transAxes, fontsize=fontsize)
#add uncertainty range
# ax.fill_between(osf_df['datetime'], conf_slope[0] * phi_new_geo + conf_intercept[0], 
#                   conf_slope[1] * phi_new_geo + conf_intercept[1], 
#                   color='pink')
ax.fill_between(osf_df['datetime'], phi_new_sn - prob_int_new_sn, 
                  phi_new_sn + prob_int_new_sn, 
                  color='pink', alpha = 1)

ax = plt.subplot(2,2,4)
ax.plot(phi_new_sn, phi, 'ro')
rnew_sn, n = mplot.lin_correl(phi_new_sn, yvals)
maenew_sn = np.nanmean(abs(phi_new_sn - yvals))
ax.set_title(r'$r_L$ = ' + "{:.3f}".format(rnew_sn) + r'; MAE = ' + "{:.2f}".format(maenew_sn))
ax.set_ylabel(r'$\phi$ NM [MV]')
ax.set_xlabel(r'$\phi$ SN (new model) [MV]')
ax.set_ylim(xx_sc); ax.set_xlim(xx_sc)
ax.plot(xx_sc,xx_sc,'k--')
ax.text(0.05, 0.95, '(d)', transform=plt.gca().transAxes, fontsize=fontsize)

# xrange = xx_sc
# ax.fill_between(xrange, conf_slope[0] * xrange + conf_intercept[0], 
#                   conf_slope[1] * xrange + conf_intercept[1], 
#                   color='pink', label='Confidence Interval')
p, regressparams = mplot.lin_regress(phi_new_sn, phi, confid_level = confid_level,
                                     plotnow = True, ax = ax, color = 'pink', alpha =1)

plt.tight_layout()

fig.savefig(os.path.join(figdir, 'model_ssn.pdf'))

# <codecell> put the relevant data into a DataFrame and save
phi_df = osf_df[['fracyear','mjd','B_GEO', 'V_GEO', 'OSF_GEO' ]].copy()
phi_df['phi_geo_A2016'] = phi_E2016_geo
phi_df['phi_geo_A2016_min'] = phi_E2016_geo - prob_int_E2016
phi_df['phi_geo_A2016_max'] = phi_E2016_geo + prob_int_E2016
phi_df['phi_geo'] = phi_new_geo
phi_df['phi_geo_min'] = phi_E2016_geo - prob_int_new
phi_df['phi_geo_max'] = phi_E2016_geo + prob_int_new

#remove data before 1845
mask = phi_df['fracyear'] < 1845
phi_df.drop(phi_df.index[mask], inplace=True)

#save the data
outputfilepath = os.path.join( os.environ['DBOX'], 'Data','HMP_GEO_O2023.csv')
phi_df.to_csv(outputfilepath, index=False, na_rep='nan')

# <codecell> Apply model to OSF(GEO)

#load the ionisation chamber/NM phi estimate
phi_IC = sunspots.load_oulu_phi_extended(filepath = None, download_now = updatenow)
#sample to 1y
IC_1y = phi_IC.resample('1Y', on='datetime').mean() 
IC_1y['datetime'] = htime.mjd2datetime(IC_1y['mjd'].to_numpy())
IC_1y.reset_index(drop=True, inplace=True)
IC_1y['U2017'] = np.interp(IC_1y['mjd'], osf_df['mjd'], osf_df['phi_NM'])



#scale the 2011 NM estimates to the 2017 values, as they use a different LIS
plt.figure()
p_nm, nm_regressparamsnewsn = mplot.lin_regress(IC_1y['phi_NM'], IC_1y['U2017'], 
                                               plotnow = True, confid_level = confid_level)
IC_1y['phi_NM_scaled'] = p_nm[0]*IC_1y['phi_NM'] + p_nm[1]


#load the 14C phi
carbon = sunspots.load_14C_phi()




ymax = 1300
alpha = 0.7
fig = plt.figure(figsize=(8,12))

ax = plt.subplot(3,1,1)
#ax.plot(osf_df['datetime'], phi_E2016_geo, 'b', label = 'OSF(GEO) A2016')
ax.plot(osf_df['datetime'], phi_new_geo, 'r', label = 'GEO (O2024)')
ax.plot(osf_df['datetime'], phi, 'k', label = 'NM (U2017)')
plt.legend(loc = 'upper left')
ax.set_ylabel(r'$\phi$ [MV]')
ax.set_ylim([0,ymax])
sunspots.PlotAlternateCycles()
ax.text(0.01, 0.05, '(a)', transform=plt.gca().transAxes, fontsize=fontsize)
xx = ax.get_xlim()
ax.set_xlim(xx)

#add uncertainty range
# ax.fill_between(osf_df['datetime'], conf_slope[0] * phi_new_geo + conf_intercept[0], 
#                   conf_slope[1] * phi_new_geo + conf_intercept[1], 
#                   color='pink',  alpha = alpha)
ax.fill_between(osf_df['datetime'], phi_new_geo - prob_int_new, 
                  phi_new_geo + prob_int_new, 
                  color='pink', alpha = alpha)
# ax.fill_between(osf_df['datetime'], phi_E2016_geo - prob_int_E2016, 
#                   phi_E2016_geo + prob_int_E2016, 
#                   color='lightblue', alpha = alpha)


mask_ic = IC_1y['fracyear'] < 1951
mask_nm = IC_1y['fracyear'] >= 1951
ax = plt.subplot(3,1,2)
#ax.plot(osf_df['datetime'], phi_E2016_geo, 'b', label = 'OSF(GEO) A2016')
ax.plot(osf_df['datetime'], phi_new_geo, 'r', label = 'GEO (O2024)')
ax.plot(IC_1y.loc[mask_nm,'datetime'], IC_1y.loc[mask_nm, 'phi_NM_scaled'], 'k', label = 'NM (U2011)*')
ax.plot(IC_1y.loc[mask_ic,'datetime'], IC_1y.loc[mask_ic, 'phi_NM_scaled'], 'b', label = 'IC (U2011)*')
plt.legend(loc = 'upper left')
ax.set_ylabel(r'$\phi$ [MV]')
ax.set_ylim([0,ymax])
sunspots.PlotAlternateCycles()
ax.text(0.01, 0.05, '(b)', transform=plt.gca().transAxes, fontsize=fontsize)
ax.set_xlim(xx)

#add uncertainty range
# ax.fill_between(osf_df['datetime'], conf_slope[0] * phi_new_geo + conf_intercept[0], 
#                   conf_slope[1] * phi_new_geo + conf_intercept[1], 
#                   color='pink',  alpha = alpha)
ax.fill_between(osf_df['datetime'], phi_new_geo - prob_int_new, 
                  phi_new_geo + prob_int_new, 
                  color='pink', alpha = alpha)
# ax.fill_between(osf_df['datetime'], phi_E2016_geo - prob_int_E2016, 
#                   phi_E2016_geo + prob_int_E2016, 
#                   color='lightblue', alpha = alpha)

ax = plt.subplot(3,1,3)
#ax.plot(osf_df['datetime'], phi_E2016_geo, 'b', label = 'OSF(GEO) A2016')
ax.plot(osf_df['datetime'], phi_new_geo, 'r', label = 'GEO (O2024)')
ax.plot(carbon['datetime'], carbon['phi_14C'], 'k', label = '14C (B2021)')
plt.legend(loc = 'upper left')
ax.set_ylim([0,ymax])
sunspots.PlotAlternateCycles()
ax.fill_between(carbon['datetime'], carbon['phi_14C'] - carbon['phi_14C_sigma'], 
                carbon['phi_14C'] + carbon['phi_14C_sigma'], 
                 color = 'grey', zorder = 0.1, alpha = alpha)  
ax.set_ylabel(r'$\phi$ [MV]')
ax.set_xlabel('Year')

#add uncertainty range
# ax.fill_between(osf_df['datetime'], conf_slope[0] * phi_new_geo + conf_intercept[0], 
#                   conf_slope[1] * phi_new_geo + conf_intercept[1], 
#                   color='pink', alpha = alpha)
ax.fill_between(osf_df['datetime'], phi_new_geo - prob_int_new, 
                  phi_new_geo + prob_int_new, 
                  color='pink', alpha = alpha)
# ax.fill_between(osf_df['datetime'], phi_E2016_geo - prob_int_E2016, 
#                   phi_E2016_geo + prob_int_E2016, 
#                   color='lightblue', alpha = alpha)

ax.text(0.01, 0.05, '(c)', transform=plt.gca().transAxes, fontsize=fontsize)
ax.set_xlim(xx)


fig.savefig(os.path.join(figdir, 'model_1845_2020.pdf'))

# <codecell> Produce a "motivation" plot of the current HMP estimtaes

fig = plt.figure(figsize=(8,10))

startyr = datetime(1800,1,1)
stopyr = datetime(2025,1,1)

ax = plt.subplot(2,1,1)
ax.plot(osf_df['datetime'], phi, 'b', label = 'NM (U2017)')
ax.plot(carbon['datetime'], carbon['phi_14C'], 'k', label = '14C (B2021)')
plt.legend(loc = 'upper left')
ax.set_ylim([0,ymax])
ax.set_xlim([startyr, stopyr])
sunspots.PlotAlternateCycles()
ax.fill_between(carbon['datetime'], carbon['phi_14C'] - carbon['phi_14C_sigma'], 
                carbon['phi_14C'] + carbon['phi_14C_sigma'], 
                 color = 'grey', zorder = 0.1, alpha = alpha)  
ax.text(0.01, 0.05, '(a)', transform=plt.gca().transAxes, fontsize=fontsize)
xx = ax.get_xlim()
ax.set_xlim(xx)
ax.set_ylabel(r'$\phi$ [MV]')


mask_ic = IC_1y['fracyear'] < 1951
mask_nm = IC_1y['fracyear'] >= 1951

ax = plt.subplot(2,1,2)
ax.plot(IC_1y.loc[mask_nm,'datetime'], IC_1y.loc[mask_nm, 'phi_NM_scaled'], 'b', label = 'NM (U2011)*')
ax.plot(IC_1y.loc[mask_ic,'datetime'], IC_1y.loc[mask_ic, 'phi_NM_scaled'], 'r', label = 'IC (U2011)*')
ax.plot(carbon['datetime'], carbon['phi_14C'], 'k', label = '14C (B2021)')
plt.legend(loc = 'upper left')
ax.set_ylim([0,ymax])
ax.set_xlim([startyr, stopyr])
sunspots.PlotAlternateCycles()
ax.fill_between(carbon['datetime'], carbon['phi_14C'] - carbon['phi_14C_sigma'], 
                carbon['phi_14C'] + carbon['phi_14C_sigma'], 
                 color = 'grey', zorder = 0.1, alpha = alpha)  
ax.text(0.01, 0.05, '(b)', transform=plt.gca().transAxes, fontsize=fontsize)
xx = ax.get_xlim()
ax.set_xlim(xx)
ax.set_xlabel('Year')
ax.set_ylabel(r'$\phi$ [MV]')




fig.savefig(os.path.join(figdir, 'HMP_now_summary.pdf'))