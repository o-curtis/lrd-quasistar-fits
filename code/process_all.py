#!/usr/bin/env python3
"""Batch-process all landed campaign chains: build SPS once, loop sources.
Usage: process_all.py STEM1:DRIVER1 STEM2:DRIVER2 ..."""
import os, sys, numpy as np
import m3port as M
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
CAMP = "/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/blueshift-phi-campaign"
kes, sig_sb, cc = 0.34, 5.6704e-5, 2.998e10
sps = M.build_sps(dust_law="smc", verbose=False)
for arg in sys.argv[1:]:
    STEM, DRIVER = arg.split(":")
    cf = os.path.join(CAMP, "chain_%s.npy" % STEM)
    if not os.path.exists(cf):
        print("SKIP %s (no chain)" % STEM); continue
    drv = __import__(DRIVER)
    chain = np.load(cf)
    model = M.build_model(drv.Z, **drv.MODEL_KWARGS); idx = model.theta_index
    obs = drv.build_obs()
    col = lambda k: np.asarray(chain[:, idx[k]]).ravel()
    teff, logg = col("teff"), col("logg")
    phi = kes*sig_sb*teff**4/(10**logg*cc)
    q = lambda a: np.percentile(a, [16,50,84])
    tq, gq, pq, Lq, Aq = q(teff), q(logg), q(phi), q(col("logL_star")), q(col("dust2_gal"))
    print("RESULT %s N=%d | Teff=%.0f(+%.0f/-%.0f) logg=%.2f(+%.2f/-%.2f) A_V=%.2f phi=%.1f(+%.1f/-%.1f)"
          % (STEM, len(chain), tq[1], tq[2]-tq[1], tq[1]-tq[0], gq[1], gq[2]-gq[1], gq[1]-gq[0],
             Aq[1], pq[1], pq[2]-pq[1], pq[1]-pq[0]))
    if "MOCK" in STEM and os.path.exists(os.path.join(CAMP, "mock_cliff.npz")):
        z = np.load(os.path.join(CAMP, "mock_cliff.npz"))
        dg = abs(gq[1]-float(z["logg_true"]))/max((gq[2]-gq[0])/2, 1e-6)
        dp = abs(pq[1]-float(z["phi_true"]))/max((pq[2]-pq[0])/2, 1e-6)
        print("   TRUTH logg=%.2f phi=%.1f -> logg %.1f sig, phi %.1f sig"
              % (z["logg_true"], z["phi_true"], dg, dp))
    th = np.array([np.median(chain[:, j]) for j in range(chain.shape[1])])
    o = obs[0]; ptl, pgal = M.predict_components(th, sps, obs, model)[0]
    rng = np.random.default_rng(1); band = []
    for d in chain[rng.choice(len(chain), min(100, len(chain)), replace=False)]:
        try:
            p = M.predict_components(d, sps, obs, model)[0]; band.append(p[0]+p[1])
        except Exception: pass
    lo, hi = np.percentile(np.array(band), [16,84], axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    w = o["wave_obs"]/1e4; m = o["mask"]; sc = 3631e6
    ax.step(w, o["flux"]*sc, where="mid", color="0.3", lw=0.7, label="data", zorder=2)
    ax.fill_between(w, (o["flux"]-o["unc"])*sc, (o["flux"]+o["unc"])*sc, step="mid", color="0.3", alpha=0.15, lw=0)
    ax.fill_between(w, lo*sc, hi*sc, color="#AA3377", alpha=0.25, lw=0, zorder=3)
    ax.plot(w, (ptl+pgal)*sc, color="#AA3377", lw=1.3, label="total (median)", zorder=5)
    ax.plot(w, ptl*sc, color="#CC3311", lw=1.0, ls="--", label="TLUSTY photosphere", zorder=4)
    ax.plot(w, pgal*sc, color="#CCAA00", lw=1.0, ls="-.", label="host", zorder=4)
    ax.plot(w[~m], o["flux"][~m]*sc, ".", color="0.75", ms=2, zorder=1)
    ax.set_xlabel(r"observed wavelength [$\mu$m]"); ax.set_ylabel(r"$F_\nu$ [$\mu$Jy]")
    ax.set_title("%s:  T=%.0f  logg=%.2f  phi=%.0f" % (STEM, tq[1], gq[1], pq[1]), fontsize=9, loc="left")
    ax.legend(fontsize=7.5, framealpha=0.9); ax.tick_params(direction="in", top=True, right=True)
    lomax = np.nanpercentile(o["flux"][m]*sc, 99.5); ax.set_ylim(-0.05*lomax, lomax*1.35)
    fig.tight_layout(); fig.savefig(os.path.join(CAMP, "fit_%s.png" % STEM), dpi=160)
    print("   saved fit_%s.png" % STEM)
