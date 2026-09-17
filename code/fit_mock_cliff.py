#!/usr/bin/env python3
"""R5 refit: blind fit of the injection-recovery mock (mock_cliff.npz)."""
import os, argparse, numpy as np
import m3port as M

CAMP  = "/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/blueshift-phi-campaign"
MOCK  = os.path.join(CAMP, "mock_cliff.npz")
Z     = 3.55
LABEL = "MOCK (Cliff grid, truth teff=5000 logg=-1.5)"
STEM  = "MOCK_cliff_m3"
DUST_LAW = "smc"
MODEL_KWARGS = dict(sigma_fix=0.0, teff_lo=2000.0, mh_fix=1.0, av_cap=8.0, dust1_fix=0.0)

def build_obs():
    z = np.load(MOCK)
    print("  MOCK truth: teff=%.0f logg=%.2f phi=%.1f | fit pixels=%d"
          % (z["teff_true"], z["logg_true"], z["phi_true"], int(z["mask"].sum())), flush=True)
    return [dict(name="PRISM", wave_obs=z["wave_obs"], wave_rest=z["wave_rest"],
                 flux=z["flux"], unc=z["unc"], mask=z["mask"],
                 sigma_inst=float(z["sigma_inst"]))]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    M.fit(STEM, Z, MODEL_KWARGS, build_obs, [], out_dir=CAMP,
          dust_law=DUST_LAW, pool_n=a.pool, dry_run=a.dry_run, source_label=LABEL)
