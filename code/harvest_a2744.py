import os, sys, json, numpy as np
RFF='/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/water-dot-tests/stack/real-final-fits'
CAMP='/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/blueshift-phi-campaign'
sys.path.insert(0, RFF)
import m3port as M
kes, ss, cc, G = 0.34, 5.6704e-5, 2.998e10, 6.674e-8
LSUN=3.828e33
out={}
for stem, drv_name in [('A2744QSO1_m3','fit_A2744QSO1'), ('A2744QSO1B_m3','fit_A2744QSO1B')]:
    drv=__import__(drv_name)
    model=M.build_model(drv.Z, **drv.MODEL_KWARGS); idx=model.theta_index
    ch=np.load(os.path.join(CAMP,'chain_%s.npy'%stem))
    col=lambda k: np.asarray(ch[:, idx[k]]).ravel()
    t,g,lL=col('teff'),col('logg'),col('logL_star')
    phi=kes*ss*t**4/(10**g*cc)
    R=np.sqrt(10**lL*LSUN/(4*np.pi*ss*t**4))
    vesc=np.sqrt(2*10**g*R)/1e5
    Q=lambda a: list(np.percentile(a,[16,50,84]))
    out[stem]=dict(z=drv.Z, N=len(ch), teff=Q(t), logg=Q(g), logL=Q(lL), phi=Q(phi), vesc=Q(vesc),
                   av=Q(col('dust2_gal')) if 'dust2_gal' in idx else None)
    print(stem, 'T=%.0f logg=%.2f phi=%.1f[%.1f,%.1f]'%(out[stem]['teff'][1],out[stem]['logg'][1],out[stem]['phi'][1],out[stem]['phi'][0],out[stem]['phi'][2]), flush=True)
json.dump(out, open('/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/newsrc/a2744_result.json','w'), indent=1)
print('SAVED')
