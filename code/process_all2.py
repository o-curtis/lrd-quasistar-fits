#!/usr/bin/env python3
"""Batch-process landed chains: phi + segfault-safe overlay.
Overlay theta = prior_transform(0.5) base (safe host SFH/duste) with the
physical params (teff,logg,logL,dust,logmass,logzsol) swapped to chain medians
-- the pattern proven safe in generate_mock. No posterior band (avoids extreme
draws segfaulting the FSPS C-extension). Usage: STEM:DRIVER ..."""
import os, sys, numpy as np
import m3port as M
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
CAMP = "/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/blueshift-phi-campaign"
kes, sig_sb, cc = 0.34, 5.6704e-5, 2.998e10
sps = M.build_sps(dust_law="smc", verbose=False)
SWAP = ["teff","logg","logL_star","dust2_gal","logmass","logzsol"]
for arg in sys.argv[1:]:
    STEM, DRIVER = arg.split(":")
    cf = os.path.join(CAMP, "chain_%s.npy" % STEM)
    if not os.path.exists(cf): print("SKIP %s" % STEM, flush=True); continue
    drv = __import__(DRIVER); chain = np.load(cf)
    model = M.build_model(drv.Z, **drv.MODEL_KWARGS); idx = model.theta_index
    obs = drv.build_obs()
    col = lambda k: np.asarray(chain[:, idx[k]]).ravel()
    teff, logg = col("teff"), col("logg")
    phi = kes*sig_sb*teff**4/(10**logg*cc)
    Q = lambda a: np.percentile(a, [16,50,84])
    tq, gq, pq, Lq, Aq = Q(teff), Q(logg), Q(phi), Q(col("logL_star")), Q(col("dust2_gal"))
    print("RESULT %s N=%d | Teff=%.0f(+%.0f/-%.0f) logg=%.2f(+%.2f/-%.2f) A_V=%.2f phi=%.1f(+%.1f/-%.1f)"
          % (STEM, len(chain), tq[1],tq[2]-tq[1],tq[1]-tq[0], gq[1],gq[2]-gq[1],gq[1]-gq[0],
             Aq[1], pq[1],pq[2]-pq[1],pq[1]-pq[0]), flush=True)
    if "MOCK" in STEM and os.path.exists(os.path.join(CAMP,"mock_cliff.npz")):
        z = np.load(os.path.join(CAMP,"mock_cliff.npz"))
        dg = abs(gq[1]-float(z["logg_true"]))/max((gq[2]-gq[0])/2,1e-6)
        dp = abs(pq[1]-float(z["phi_true"]))/max((pq[2]-pq[0])/2,1e-6)
        print("   TRUTH logg=%.2f phi=%.1f -> logg recovered %.1fsig, phi %.1fsig"
              % (z["logg_true"], z["phi_true"], dg, dp), flush=True)
    th = model.prior_transform(0.5*np.ones(model.ndim))
    for k in SWAP:
        if k in idx: th[idx[k]] = np.median(col(k))
    try:
        p = M.predict_components(th, sps, obs, model)[0]; ptl, pgal = p
    except Exception as e:
        print("   overlay predict failed: %r" % e, flush=True); continue
    o = obs[0]; w = o["wave_obs"]/1e4; m = o["mask"]; sc = 3631e6
    fig, ax = plt.subplots(figsize=(7.2,3.6))
    ax.step(w, o["flux"]*sc, where="mid", color="0.3", lw=0.7, label="data", zorder=2)
    ax.fill_between(w, (o["flux"]-o["unc"])*sc, (o["flux"]+o["unc"])*sc, step="mid", color="0.3", alpha=0.15, lw=0)
    ax.plot(w, (ptl+pgal)*sc, color="#AA3377", lw=1.3, label="total (median params)", zorder=5)
    ax.plot(w, ptl*sc, color="#CC3311", lw=1.0, ls="--", label="TLUSTY photosphere", zorder=4)
    ax.plot(w, pgal*sc, color="#CCAA00", lw=1.0, ls="-.", label="host", zorder=4)
    ax.plot(w[~m], o["flux"][~m]*sc, ".", color="0.75", ms=2, zorder=1)
    ax.set_xlabel(r"observed wavelength [$\mu$m]"); ax.set_ylabel(r"$F_\nu$ [$\mu$Jy]")
    ax.set_title("%s:  T=%.0f  logg=%.2f  phi=%.0f (+%.0f/-%.0f)"
                 % (STEM, tq[1], gq[1], pq[1], pq[2]-pq[1], pq[1]-pq[0]), fontsize=9, loc="left")
    ax.legend(fontsize=7.5, framealpha=0.9); ax.tick_params(direction="in", top=True, right=True)
    lomax = np.nanpercentile(o["flux"][m]*sc, 99.5); ax.set_ylim(-0.05*lomax, lomax*1.35)
    fig.tight_layout(); fig.savefig(os.path.join(CAMP,"fit_%s.png"%STEM), dpi=160)
    print("   saved fit_%s.png"%STEM, flush=True)
