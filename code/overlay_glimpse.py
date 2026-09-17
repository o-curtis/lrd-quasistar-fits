import os, sys, pickle, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import importlib.util
NS='/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/newsrc'
spec=importlib.util.spec_from_file_location('fitg', os.path.join(NS,'fit_glimpse17775.py'))
G=importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
F=G.F
obs_list=G.build_obs()
sps=F.build_sps(include_xi10=True)
model=F.build_model_variant('M3','free','free',mh_mode='free',neb_mode='off',zred=G.Z)
from prospect.models import priors as _priors
model.config_dict['sigma_smooth']['prior']=_priors.TopHat(mini=20.0,maxi=400.0)
model.config_dict['sigma_smooth']['init']=100.0
res=pickle.load(open(os.path.join(NS,'fits_out','dynesty_glimpse17775_m3.pkl'),'rb'))
th=np.asarray(res.samples[int(np.argmax(res.logl))],float)
sps_p=F._make_sps_params(th,model)
wave,stl,sgal,preds=F._eval_model(sps_p,sps,obs_list,model)
o=obs_list[0]; ptl,pgal=preds[0]; m=o['mask']; w=o['wave_obs']/1e4; sc=3631e6
import json
r=json.load(open(os.path.join(NS,'glimpse_result.json'))) if os.path.exists(os.path.join(NS,'glimpse_result.json')) else None
fig,ax=plt.subplots(figsize=(7.2,3.6))
ax.step(w,o['flux']*sc,where='mid',color='0.3',lw=0.7,label='data',zorder=2)
ax.fill_between(w,(o['flux']-o['unc'])*sc,(o['flux']+o['unc'])*sc,step='mid',color='0.3',alpha=0.15,lw=0)
ax.plot(w,(ptl+pgal)*sc,color='#AA3377',lw=1.3,label='total (max-lnL)',zorder=5)
ax.plot(w,ptl*sc,color='#CC3311',lw=1.0,ls='--',label='TLUSTY photosphere',zorder=4)
ax.plot(w,pgal*sc,color='#CCAA00',lw=1.0,ls='-.',label='host',zorder=4)
ax.plot(w[~m],o['flux'][~m]*sc,'.',color='0.75',ms=2,zorder=1)
fphot=float(np.sum(ptl[m])/max(np.sum((ptl+pgal)[m]),1e-30))
ax.set_title('GLIMPSE-17775 (G395M, z=3.5022, mu=2.04):  T=6468  logg=-2.07  phi=131 (+980/-113)  f_phot=%.2f'%fphot,fontsize=8.5,loc='left')
ax.set_xlabel(r'observed wavelength [$\mu$m]'); ax.set_ylabel(r'$F_\nu$ [$\mu$Jy]')
ax.legend(fontsize=7.5,framealpha=0.9); ax.tick_params(direction='in',top=True,right=True)
lomax=np.nanpercentile(o['flux'][m]*sc,99.5); ax.set_ylim(-0.05*lomax,lomax*1.35)
fig.tight_layout(); fig.savefig(os.path.join(NS,'fitml_glimpse17775.png'),dpi=160)
print('saved fitml_glimpse17775.png f_phot=%.3f'%fphot,flush=True)
