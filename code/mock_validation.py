import os, numpy as np, m3port as M
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
CAMP = "/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/blueshift-phi-campaign"
import fit_mock_cliff as drv
model = M.build_model(drv.Z, **drv.MODEL_KWARGS); idx = model.theta_index
ch = np.load(os.path.join(CAMP, "chain_MOCK_cliff_m3.npy"))
col = lambda k: np.asarray(ch[:, idx[k]]).ravel()
kes, ss, cc = 0.34, 5.6704e-5, 2.998e10
t, g = col("teff"), col("logg"); phi = kes*ss*t**4/(10**g*cc)
z = np.load(os.path.join(CAMP, "mock_cliff.npz"))
fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.9))
for a, dat, truth, lab, xr in [
    (ax[0], g, float(z["logg_true"]), r"$\log g$", None),
    (ax[1], np.log10(phi), np.log10(float(z["phi_true"])), r"$\log_{10}\phi$", None),
    (ax[2], t, float(z["teff_true"]), r"$T_{\rm eff}$ [K]", None)]:
    a.hist(dat, bins=50, color="0.6", edgecolor="none", density=True)
    a.axvline(truth, color="#CC3311", lw=2, label="injected truth")
    a.axvline(np.median(dat), color="#228833", lw=1.5, ls="--", label="recovered median")
    q = np.percentile(dat, [16, 84])
    a.axvspan(q[0], q[1], color="#228833", alpha=0.12)
    a.set_xlabel(lab, fontsize=10); a.set_yticks([])
    a.tick_params(direction="in", labelsize=8)
ax[0].legend(fontsize=7, framealpha=0.9, loc="upper left")
fig.suptitle("R5 injection-recovery: truth recovered within (wide) errors -- unbiased, imprecise",
             fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(CAMP, "mock_validation.png"), dpi=170)
print("saved mock_validation.png")
