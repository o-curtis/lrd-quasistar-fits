import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
z = np.load("mock_cliff.npz")
w = z["wave_obs"]/1e4; m = z["mask"]
fig, ax = plt.subplots(figsize=(7.4,3.4))
ax.errorbar(w, z["flux"]*3631e6, yerr=z["unc"]*3631e6, fmt="o", ms=2, lw=0.4,
            elinewidth=0.4, color="0.5", alpha=0.5, label="mock (truth+noise)", zorder=2)
ax.plot(w, z["truth_flux"]*3631e6, "-", color="#CC3311", lw=1.3,
        label="injected truth (teff=5000, logg=-1.5, phi=12.7)", zorder=4)
ax.plot(w[~m], z["flux"][~m]*3631e6, "x", color="0.7", ms=3, label="masked (lines)", zorder=1)
ax.set_xlabel(r"observed wavelength [$\mu$m]"); ax.set_ylabel(r"$F_\nu$ [$\mu$Jy]")
ax.set_title("R5 injection-recovery mock on the Cliff grid", fontsize=10, loc="left")
ax.legend(fontsize=8, framealpha=0.9); ax.tick_params(direction="in", top=True, right=True)
ax.set_xlim(0.5,5.4)
fig.tight_layout(); fig.savefig("quicklook_mock.png", dpi=160)
print("saved quicklook_mock.png")
