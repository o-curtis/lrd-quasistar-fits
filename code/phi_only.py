import os, sys, numpy as np, m3port as M
CAMP = "/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/blueshift-phi-campaign"
kes, ss, cc = 0.34, 5.6704e-5, 2.998e10
rows = []
for arg in sys.argv[1:]:
    STEM, DRV = arg.split(":")
    cf = os.path.join(CAMP, "chain_%s.npy" % STEM)
    if not os.path.exists(cf):
        print("SKIP %s (no chain)" % STEM, flush=True); continue
    drv = __import__(DRV); model = M.build_model(drv.Z, **drv.MODEL_KWARGS); idx = model.theta_index
    ch = np.load(cf); col = lambda k: np.asarray(ch[:, idx[k]]).ravel()
    t, g = col("teff"), col("logg"); phi = kes*ss*t**4/(10**g*cc)
    Q = lambda a: np.percentile(a, [16,50,84])
    tq, gq, pq, Aq, Lq = Q(t), Q(g), Q(phi), Q(col("dust2_gal")), Q(col("logL_star"))
    print("%-14s z=%.3f N=%d Teff=%.0f(+%.0f/-%.0f) logg=%.2f(+%.2f/-%.2f) A_V=%.2f logL=%.2f phi=%.1f(+%.1f/-%.1f)"
          % (STEM, drv.Z, len(ch), tq[1],tq[2]-tq[1],tq[1]-tq[0], gq[1],gq[2]-gq[1],gq[1]-gq[0],
             Aq[1], Lq[1], pq[1],pq[2]-pq[1],pq[1]-pq[0]), flush=True)
    rows.append((STEM, drv.Z, pq[1], pq[0], pq[2], gq[1], tq[1]))
    if "MOCK" in STEM and os.path.exists(os.path.join(CAMP, "mock_cliff.npz")):
        z = np.load(os.path.join(CAMP, "mock_cliff.npz"))
        print("   TRUTH logg=%.2f phi=%.1f -> logg %.1fsig, phi %.1fsig"
              % (z["logg_true"], z["phi_true"],
                 abs(gq[1]-float(z["logg_true"]))/max((gq[2]-gq[0])/2,1e-6),
                 abs(pq[1]-float(z["phi_true"]))/max((pq[2]-pq[0])/2,1e-6)), flush=True)
np.save(os.path.join(CAMP, "phi_summary.npy"), np.array(rows, dtype=object))
print("saved phi_summary.npy", flush=True)
