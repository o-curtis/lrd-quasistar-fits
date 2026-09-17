#!/usr/bin/env python3
"""R7: two-component vs single-component wind model for the Egg absorption.
Joint fit of the continuum-normalized absorption in Halpha, Hbeta, He I 10830,
Pa-gamma, Pa-delta with a Sobolev-style optical-depth model:
  tau_i(v) = A_i,slow * G(v; v_s, sig_s) + A_i,fast * G(v; v_f, sig_f)
v_s, v_f, sig_s, sig_f are SHARED across all lines (the two kinematic systems);
A_i are per-line optical depths (the level populations). Emission is modeled as
narrow+broad Gaussians outside the absorption window (as in e6). Compares the
two-component model (Delta chi2, per-line amplitude segregation) against a
single-velocity model to decide accelerating-column vs eruptive-shell.
"""
import numpy as np
from astropy.io import fits
from scipy.optimize import least_squares
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

Z, C = 0.1007, 2.998e5
def load(f):
    with fits.open(f) as h:
        s = h["SPECTRUM"].data
        w = s["wave"].astype(float)/(1+Z); fl = s["flux"].astype(float)
        iv = s["ivar"].astype(float); m = s["mask"].astype(int)
    g = (m==1)&(iv>0)&np.isfinite(fl)
    return w[g], fl[g], 1/np.sqrt(iv[g])
mr = load("J1025+1402_MODSR_coadd1d_tellcorr_slitcorr_dered.fits")
fi = load("J1025+1402_FIRE_coadd1d_tellcorr_calib_dered.fits")

# lines: (label, rest wl, dataset, absorption window, emission fit halfwidth)
LINES = [("Ha",6564.61,mr,(-950,-40),2200),
         ("HeI",10833.2,fi,(-1100,-40),2600),("Pag",10941.1,fi,(-900,-40),1600),
         ("Pad",10052.6,fi,(-900,-40),1600)]

def emis(v,c0,an,sn,ab,vb,sb): 
    return c0+an*np.exp(-0.5*(v/sn)**2)+ab*np.exp(-0.5*((v-vb)/sb)**2)

# build normalized absorption arrays per line
DATA=[]
for lab,l0,(w,f,s),win,ehw in LINES:
    v=C*(w/l0-1); sel=np.abs(v)<ehw; vv,ff,ss=v[sel],f[sel],s[sel]
    cont=np.abs(vv)>0.75*ehw
    p=np.polyfit(vv[cont],ff[cont],1); norm=ff/np.polyval(p,vv); ner=ss/np.polyval(p,vv)
    fitsel=~((vv>win[0])&(vv<win[1]))
    try:
        popt,_=__import__("scipy.optimize",fromlist=["curve_fit"]).curve_fit(
            emis,vv[fitsel],norm[fitsel],p0=[1,max(norm.max()-1,.3),150,.5,0,700],
            sigma=np.maximum(ner[fitsel],1e-3),
            bounds=([.5,0,50,0,-300,200],[1.5,np.inf,500,np.inf,300,3000]),maxfev=20000)
        base=emis(vv,*popt)
    except Exception: base=np.ones_like(vv)
    ins=(vv>win[0])&(vv<win[1])
    DATA.append((lab,vv[ins],norm[ins]/base[ins],ner[ins]/base[ins]))

def G(v,v0,sig): return np.exp(-0.5*((v-v0)/sig)**2)
def resid(par,two=True):
    r=[]
    vs,ss=par[0],par[1]
    if two: vf,sf=par[2],par[3]; amps=par[4:]
    else: amps=par[2:]
    for i,(lab,v,y,e) in enumerate(DATA):
        if two: tau=amps[2*i]*G(v,vs,ss)+amps[2*i+1]*G(v,vf,sf)
        else:  tau=amps[i]*G(v,vs,ss)
        r.append((np.exp(-tau)-y)/np.maximum(e,1e-3))
    return np.concatenate(r)

# two-component: [vs,ss,vf,sf, (A_slow,A_fast)x5]
p2=[-190,120,-420,260]+[0.5,0.3]*4
lo2=[-350,40,-900,80]+[0,0]*4; hi2=[-40,400,-150,700]+[5,5]*4
r2=least_squares(resid,p2,bounds=(lo2,hi2),args=(True,),max_nfev=20000)
# single: [vs,ss, A x5]
p1=[-300,300]+[0.5]*4; lo1=[-900,40]+[0]*4; hi1=[-40,700]+[5]*4
r1=least_squares(resid,p1,bounds=(lo1,hi1),args=(False,),max_nfev=20000)
chi2=lambda r,n: np.sum(r.fun**2), 
c2=np.sum(r2.fun**2); c1=np.sum(r1.fun**2); ndat=sum(len(d[1]) for d in DATA)
print("2-comp: vs=%.0f sig_s=%.0f  vf=%.0f sig_f=%.0f  chi2=%.1f/%d (dof %d)"
      %(r2.x[0],r2.x[1],r2.x[2],r2.x[3],c2,ndat,ndat-len(p2)))
print("1-comp: vs=%.0f sig_s=%.0f  chi2=%.1f/%d (dof %d)"%(r1.x[0],r1.x[1],c1,ndat,ndat-len(p1)))
print("Delta chi2 (1c-2c) = %.1f for %d extra params"%(c1-c2,len(p2)-len(p1)))
print("per-line optical depths (A_slow, A_fast):")
for i,(lab,_,_,_) in enumerate(DATA):
    print("  %-4s slow=%.2f fast=%.2f  fast/slow=%.2f"%(lab,r2.x[4+2*i],r2.x[5+2*i],
          r2.x[5+2*i]/max(r2.x[4+2*i],1e-3)))

# figure
fig,axes=plt.subplots(4,1,figsize=(3.6,6.9),sharex=True,gridspec_kw={"hspace":0.09})
vs,ss,vf,sf=r2.x[:4]
for i,((lab,v,y,e),ax) in enumerate(zip(DATA,axes)):
    ax.step(v,y,where="mid",color="0.25",lw=0.8)
    ax.fill_between(v,y-e,y+e,step="mid",color="0.25",alpha=0.18,lw=0)
    vg=np.linspace(v.min(),v.max(),400)
    tau2=r2.x[4+2*i]*G(vg,vs,ss)+r2.x[5+2*i]*G(vg,vf,sf)
    ax.plot(vg,np.exp(-tau2),color="#CC3311",lw=1.4,label="2-comp" if i==0 else None)
    ax.plot(vg,np.exp(-(r2.x[4+2*i]*G(vg,vs,ss))),color="#4477AA",lw=0.8,ls=(0,(4,2)),
            label="slow base" if i==0 else None)
    ax.plot(vg,np.exp(-(r2.x[5+2*i]*G(vg,vf,sf))),color="#EE7733",lw=0.8,ls=(0,(1,1.5)),
            label="fast shell" if i==0 else None)
    ax.axhline(1,color="0.6",lw=0.6,ls=(0,(4,3))); ax.axvline(0,color="0.6",lw=0.5)
    ax.text(0.03,0.12,lab,transform=ax.transAxes,fontsize=9,fontweight="bold")
    ax.set_ylim(min(0.0,np.nanpercentile(y,2)-0.1),1.25); ax.set_xlim(-1100,300)
    ax.tick_params(direction="in",top=True,right=True,labelsize=8)
axes[0].legend(fontsize=6.5,loc="lower left",framealpha=0.9)
axes[-1].set_xlabel(r"velocity [km s$^{-1}$]",fontsize=10)
axes[1].set_ylabel("normalized flux",fontsize=10)
fig.savefig("e7_sobolev.png",dpi=170,bbox_inches="tight")
fig.savefig("e7_sobolev.pdf",bbox_inches="tight")
print("saved e7_sobolev.png/.pdf")
