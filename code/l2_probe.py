import os, sys, time, json
import numpy as np
sys.path.insert(0, '/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/lrds2')
import importlib.util
spec = importlib.util.spec_from_file_location('drv',
    '/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/lrds2/fit_desi_lrds2_roar.py')
D = importlib.util.module_from_spec(spec)
sys.argv = ['x', 'PROBE', '--dry-run']
pass  # module name 'drv' != '__main__', main guard inert
spec.loader.exec_module(D)
F = D.F
os.chdir('/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/lrds2')
man = {r['name']: r for r in json.load(open('lrds2_manifest.json'))}
name = 'J171741.74+380752.47'; z = float(man[name]['z'])
obs = D.build_obs_desi(name, z)
sps = F.build_sps(include_xi10=True)
model = F.build_model_variant('M3', 'free', 'free', mh_mode='free', neb_mode='off', zred=z)
rng = np.random.default_rng(1)
# warm-up call (compilation/caches)
F._logl(model.prior_transform(0.5*np.ones(model.ndim)), sps, obs, model)
ths = [model.prior_transform(rng.uniform(size=model.ndim)) for _ in range(20)]
t0 = time.time(); nok = 0
for th in ths:
    if np.isfinite(F._logl(th, sps, obs, model)): nok += 1
dt = (time.time()-t0)/20
print('PROBE per-logl %.3f s finite %d/20' % (dt, nok), flush=True)
print('PROBE NLIVE=%s NLIVE_INIT=%s' % (getattr(F,'NLIVE','?'), getattr(F,'NLIVE_INIT','?')), flush=True)
for ncall in (1e5, 3e5, 1e6):
    print('PROBE %dk calls -> %.1f h' % (ncall/1e3, dt*ncall/3600), flush=True)
