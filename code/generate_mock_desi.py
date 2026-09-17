#!/usr/bin/env python3
"""
R5-DESI injection-recovery mock generator (runs on ROAR, lrds2 dir).
Injects known (teff, logg) photospheres on real DESI noise via the Egg
machinery's _make_sps_params/_eval_model, scaling logL to the template's
continuum. Usage:
  python generate_mock_desi.py TAG TEMPLATE_NAME TEFF LOGG
Writes spectra/MOCK_<TAG>.npz (same schema as real sources, z copied from the
template) + mockmeta_<TAG>.json with truths, so fit_desi_lrds2_roar.py fits it
unchanged (add MOCK entries to lrds2_manifest.json first).
"""
import os, sys, json
import numpy as np
import importlib.util

HERE = '/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/lrds2'
spec = importlib.util.spec_from_file_location('drv', os.path.join(HERE, 'fit_desi_lrds2_roar.py'))
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)
F = D.F

TAG, TMPL, TEFF, LOGG = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
os.chdir(HERE)
man = {r['name']: r for r in json.load(open('lrds2_manifest.json'))}
z = float(man[TMPL]['z'])
obs = D.build_obs_desi(TMPL, z)
sps = F.build_sps(include_xi10=True)
model = F.build_model_variant('M3', 'free', 'free', mh_mode='free', neb_mode='off', zred=z)

idx = {k: i for i, k in enumerate(model.theta_labels())}
th = model.prior_transform(0.5*np.ones(model.ndim))
th[idx['teff']] = TEFF; th[idx['logg']] = LOGG
th[idx['sigma_smooth']] = 60.0
th[idx['logL_star']] = 9.0

def total_model(theta):
    """Per-arm model prediction in maggies, replicating _logl's pipeline."""
    sps_p = F._make_sps_params(theta, model)
    wave, spec_tl, spec_gal, _ = sps.get_spectra_components(**sps_p)
    zred = float(model.params['zred']); lumdist = float(model.params['lumdist'])
    sig_sm = float(theta[{k: i for i, k in enumerate(model.theta_labels())}['sigma_smooth']])
    preds = []
    for o in obs:
        p_tl  = F._smooth_and_interp(wave, spec_tl,  o['wave_obs'], sig_sm,
                                     o['sigma_inst'], F.SIGMA_LIB, zred, lumdist)
        p_gal = F._smooth_and_interp(wave, spec_gal, o['wave_obs'], 0.0,
                                     o['sigma_inst'], F.SIGMA_C3K, zred, lumdist)
        preds.append(p_tl + p_gal)
    return preds

# scale photosphere luminosity to match the real line-free continuum level
pred = total_model(th)
num = den = 0.0
for o, p in zip(obs, pred):
    m = o['mask']
    num += np.nansum(o['flux'][m]); den += np.nansum(np.asarray(p)[m])
scale = num/den
th[idx['logL_star']] += np.log10(scale)
th[idx['logmass']]  += np.log10(scale)
pred = total_model(th)

# assemble a full-grid mock npz in the original (wave, flux, err) layout
d = np.load(os.path.join('spectra', f'{TMPL}.npz'))
w0 = d['wave'].astype(float); e0 = d['err'].astype(float)
fl_mock = np.full_like(w0, np.nan)
_CKMS = 2.998e5; _j = 1e-23
rng = np.random.default_rng(abs(hash(TAG)) % 2**31)
for o, p in zip(obs, pred):
    # model flux is in maggies on o['wave_obs']; convert back to 1e-17 f_lambda
    mag = np.asarray(p)
    f_nu = mag * 3631.0 * _j
    f_lam = f_nu * 2.998e18 / o['wave_obs']**2 / 1e-17
    sel = np.searchsorted(w0, o['wave_obs'])
    sel = np.clip(sel, 0, len(w0)-1)
    fl_mock[sel] = f_lam
noise_ok = np.isfinite(fl_mock) & np.isfinite(e0) & (e0 > 0)
fl_noisy = fl_mock + np.where(noise_ok, rng.normal(size=len(w0))*e0, 0.0)
np.savez(os.path.join('spectra', f'MOCK_{TAG}.npz'), wave=w0, flux=fl_noisy, err=e0, z=z)

kes, ss, cc = 0.34, 5.6704e-5, 2.998e10
phi_true = kes*ss*TEFF**4/(10**LOGG*cc)
json.dump(dict(tag=TAG, template=TMPL, z=z, teff_true=TEFF, logg_true=LOGG,
               phi_true=phi_true, logL_true=float(th[idx['logL_star']])),
          open(f'mockmeta_{TAG}.json', 'w'), indent=1)
print('MOCK_%s written | truth teff=%.0f logg=%.2f phi=%.2f' % (TAG, TEFF, LOGG, phi_true), flush=True)
