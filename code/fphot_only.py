import os, sys, numpy as np, pickle, m3port as M
CAMP = "/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/blueshift-phi-campaign"
STEM, DRV = sys.argv[1], sys.argv[2]
drv = __import__(DRV); model = M.build_model(drv.Z, **drv.MODEL_KWARGS)
res = pickle.load(open(os.path.join(CAMP, "dynesty_%s.pkl" % STEM), "rb"))
th = np.asarray(res.samples[int(np.argmax(res.logl))], float)
sps = M.build_sps(dust_law=drv.DUST_LAW, verbose=False); obs = drv.build_obs()
p = M.predict_components(th, sps, obs, model)[0]; ptl, pgal = p
m = obs[0]["mask"]
ok = m & np.isfinite(ptl) & np.isfinite(pgal)          # NaN-safe (the earlier bug)
fphot = float(np.nansum(ptl[ok]) / max(np.nansum((ptl+pgal)[ok]), 1e-30))
np.save(os.path.join(CAMP, "fphot_%s.npy" % STEM), np.array([fphot]))
print("fphot %s %.4f (ok pixels %d/%d)" % (STEM, fphot, int(ok.sum()), int(m.sum())), flush=True)
