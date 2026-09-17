import os, sys, numpy as np, pickle, m3port as M
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
CAMP = "/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/blueshift-phi-campaign"
STEM, DRV = sys.argv[1], sys.argv[2]
drv = __import__(DRV); model = M.build_model(drv.Z, **drv.MODEL_KWARGS); idx = model.theta_index
ch = np.load(os.path.join(CAMP, "chain_%s.npy" % STEM))
# point estimate: MAXIMUM-LIKELIHOOD posterior sample (component-wise median is off
# the degeneracy ridge and produced systematic offsets in the display)
res = pickle.load(open(os.path.join(CAMP, "dynesty_%s.pkl" % STEM), "rb"))
th = np.asarray(res.samples[int(np.argmax(res.logl))], float)
sps = M.build_sps(dust_law=drv.DUST_LAW, verbose=False); obs = drv.build_obs()
col = lambda k: np.asarray(ch[:, idx[k]]).ravel()
kes, ss, cc = 0.34, 5.6704e-5, 2.998e10
t, g = col("teff"), col("logg"); phi = kes*ss*t**4/(10**g*cc)
Q = lambda a: np.percentile(a, [16,50,84]); tq, gq, pq = Q(t), Q(g), Q(phi)
p = M.predict_components(th, sps, obs, model)[0]; ptl, pgal = p
o = obs[0]; w = o["wave_obs"]/1e4; m = o["mask"]; sc = 3631e6
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.step(w, o["flux"]*sc, where="mid", color="0.3", lw=0.7, label="data", zorder=2)
ax.fill_between(w, (o["flux"]-o["unc"])*sc, (o["flux"]+o["unc"])*sc, step="mid", color="0.3", alpha=0.15, lw=0)
ax.plot(w, (ptl+pgal)*sc, color="#AA3377", lw=1.3, label="total (max-lnL)", zorder=5)
ax.plot(w, ptl*sc, color="#CC3311", lw=1.0, ls="--", label="TLUSTY photosphere", zorder=4)
ax.plot(w, pgal*sc, color="#CCAA00", lw=1.0, ls="-.", label="host", zorder=4)
ax.plot(w[~m], o["flux"][~m]*sc, ".", color="0.75", ms=2, zorder=1)
ax.set_xlabel(r"observed wavelength [$\mu$m]"); ax.set_ylabel(r"$F_\nu$ [$\mu$Jy]")
fphot = float(np.nansum(ptl[m])/max(np.nansum((ptl+pgal)[m]), 1e-30))
ax.set_title("%s:  T=%.0f  logg=%.2f  phi=%.0f (+%.0f/-%.0f)  f_phot=%.2f"
             % (STEM, tq[1], gq[1], pq[1], pq[2]-pq[1], pq[1]-pq[0], fphot), fontsize=9, loc="left")
ax.legend(fontsize=7.5, framealpha=0.9); ax.tick_params(direction="in", top=True, right=True)
lomax = np.nanpercentile(o["flux"][m]*sc, 99.5); ax.set_ylim(-0.05*lomax, lomax*1.35)
fig.tight_layout(); fig.savefig(os.path.join(CAMP, "fitml_%s.png" % STEM), dpi=160)
np.save(os.path.join(CAMP, "fphot_%s.npy" % STEM), np.array([fphot]))
print("saved fitml_%s.png  f_phot=%.3f" % (STEM, fphot), flush=True)
