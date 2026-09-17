import os, sys, glob, json, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import importlib.util
L2 = '/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/lrds2'
spec = importlib.util.spec_from_file_location('drv', os.path.join(L2, 'fit_desi_lrds2_roar.py'))
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
F = D.F
os.chdir(L2)
name = sys.argv[1]
man = {r['name']: r for r in json.load(open('lrds2_manifest.json'))}
z = float(man[name]['z'])
stem = 'lrds2_' + name.split('.')[0].replace('+','p').replace('-','m')
obs = D.build_obs_desi(name, z)
sps = F.build_sps(include_xi10=True)
model = F.build_model_variant('M3','free','free',mh_mode='free',neb_mode='off',zred=z)
res = pickle.load(open(f'fits_out/dynesty_{stem}.pkl','rb'))
th = np.asarray(res.samples[int(np.argmax(res.logl))], float)
labels = model.theta_labels()
sig_sm = float(th[labels.index('sigma_smooth')])
sps_p = F._make_sps_params(th, model)
wave, spec_tl, spec_gal, _ = sps.get_spectra_components(**sps_p)
zred = float(model.params['zred']); ld = float(model.params['lumdist'])
fig, ax = plt.subplots(figsize=(7.2, 3.6))
kes, ss, cc = 0.34, 5.6704e-5, 2.998e10
ch = np.load(f'fits_out/chain_{stem}.npy')
t, g = ch[:,0], ch[:,1]; phi = kes*ss*t**4/(10**g*cc)
q = np.percentile(phi,[16,50,84]); tq=np.percentile(t,50); gq=np.percentile(g,50)
for o in obs:
    p_tl  = F._smooth_and_interp(wave, spec_tl,  o['wave_obs'], sig_sm, o['sigma_inst'], F.SIGMA_LIB, zred, ld)
    p_gal = F._smooth_and_interp(wave, spec_gal, o['wave_obs'], 0.0,   o['sigma_inst'], F.SIGMA_C3K, zred, ld)
    sc = 3631e6
    ax.step(o['wave_obs']/1e4, o['flux']*sc, where='mid', color='0.3', lw=0.5, zorder=2)
    ax.plot(o['wave_obs']/1e4, (p_tl+p_gal)*sc, color='#AA3377', lw=1.0, zorder=5)
    ax.plot(o['wave_obs']/1e4, p_tl*sc, color='#CC3311', lw=0.8, ls='--', zorder=4)
    ax.plot(o['wave_obs']/1e4, p_gal*sc, color='#CCAA00', lw=0.8, ls='-.', zorder=4)
    ax.plot(o['wave_obs'][~o['mask']]/1e4, o['flux'][~o['mask']]*sc, '.', color='0.8', ms=1.5, zorder=1)
allf = np.concatenate([o['flux'][o['mask']] for o in obs])*3631e6
ax.set_ylim(-0.05*np.nanpercentile(allf,99.5), 1.35*np.nanpercentile(allf,99.5))
ax.set_xlabel(r'observed wavelength [$\mu$m]'); ax.set_ylabel(r'$F_\nu$ [$\mu$Jy]')
ax.set_title('%s (z=%.3f): T=%.0f logg=%.2f phi=%.1f (+%.1f/-%.1f) [max-lnL]' %
             (name, z, tq, gq, q[1], q[2]-q[1], q[1]-q[0]), fontsize=9, loc='left')
ax.tick_params(direction='in', top=True, right=True)
fig.tight_layout(); fig.savefig(f'fits_out/fitml_{stem}.png', dpi=160)
print('saved fitml_%s.png' % stem, flush=True)
