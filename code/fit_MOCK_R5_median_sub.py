#!/usr/bin/env python3
"""R5 mock fit: R5_median_sub (truth injected on real dja_RUBIES_EGS926125.fits noise)."""
import os, argparse
import numpy as np
import m3port as M
CAMP = "/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/blueshift-phi-campaign"
MOCK = os.path.join(CAMP, "mock_R5_median_sub.npz")
Z = 5.285800
STEM = "MOCK_R5_median_sub_m3"
DUST_LAW = "smc"
MODEL_KWARGS = dict(sigma_fix=0.0, teff_lo=2000.0, mh_fix=1.0, av_cap=8.0, dust1_fix=0.0)
WEIGHTS = []
def build_obs():
    z = np.load(MOCK)
    print("  MOCK truth: teff=%.0f logg=%.2f phi=%.2f | fit pixels=%d"
          % (z["teff_true"], z["logg_true"], z["phi_true"], int(z["mask"].sum())), flush=True)
    return [dict(name="PRISM", wave_obs=z["wave_obs"], wave_rest=z["wave_rest"],
                 flux=z["flux"], unc=z["unc"], mask=z["mask"], sigma_inst=float(z["sigma_inst"]))]
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--pool", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    M.fit(STEM, Z, MODEL_KWARGS, build_obs, WEIGHTS, out_dir=CAMP,
          dust_law=DUST_LAW, pool_n=a.pool, dry_run=a.dry_run, source_label="R5_median_sub")
