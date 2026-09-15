import os, sys, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize, minimize_scalar
from scipy.interpolate import BSpline
from scipy.stats import cramervonmises, kstest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from hawkes import fit_exp_mle, fit_soe_mle, rescaled_residuals_exp, _soe_R

BETAS7=np.array([1e-2,1e-1,1,10,100,1000,10000.],float)
OUT=ROOT/'tables_csf'; OUT.mkdir(exist_ok=True)
FIG=ROOT/'figures_csf'; FIG.mkdir(exist_ok=True)

def load_bund():
    d=np.load(ROOT/'data'/'bund.npz',allow_pickle=True)
    out={}
    for k in sorted(d.files):
        arr=d[k]
        up,dn,ta,tb=[np.sort(np.asarray(a,float)) for a in arr]
        out[k]={'up':up,'dn':dn,'P':np.sort(np.r_[up,dn]),'Tr':np.sort(np.r_[ta,tb])}
    return out

# ---------- residual effect sizes ----------
def soe_residuals(t,T,mu,alphas,betas):
    t=np.asarray(t,float); betas=np.asarray(betas,float); alphas=np.asarray(alphas,float)
    R=_soe_R(np.ascontiguousarray(t),np.empty(0),betas)
    idx=np.arange(len(t))[:,None]
    Lam=mu*t + np.sum((alphas/betas)[None,:]*(idx-R),axis=1)
    d=np.diff(np.r_[0.0,Lam]); return d[d>0]

def residual_effects():
    bund=load_bund(); rows=[]
    for day,D in bund.items():
        T=max(D['P'][-1],D['Tr'][-1])
        for proc in ['P','Tr']:
            t=D[proc][D[proc]>0]
            fe=fit_exp_mle(t,np.empty(0),T)
            fs=fit_soe_mle(t,np.empty(0),T,betas=BETAS7)
            _,_,de=rescaled_residuals_exp(t,np.empty(0),T,fe['mu'],fe['alpha'],fe['beta'])
            ds=soe_residuals(t,T,fs['mu'],fs['alpha'],fs['betas'])
            for model,dlt,n,ll in [('Exp',de,fe['n'],fe['loglik']),('SoE',ds,fs['n'],fs['loglik'])]:
                u=1-np.exp(-dlt)
                ks=kstest(u,'uniform')
                cvm=cramervonmises(u,'uniform')
                ac=[]
                z=dlt-dlt.mean(); den=np.dot(z,z)
                for lag in [1,5,10]:
                    ac.append(float(np.dot(z[:-lag],z[lag:])/den) if den>0 and len(z)>lag else np.nan)
                rows.append(dict(day=day,proc=proc,model=model,N=len(t),n=n,loglik=ll,
                                 ks=float(ks.statistic),ks_p=float(ks.pvalue),cvm=float(cvm.statistic),
                                 resid_acf1=ac[0],resid_acf5=ac[1],resid_acf10=ac[2],mean_resid=float(dlt.mean()),var_resid=float(dlt.var())))
    df=pd.DataFrame(rows); df.to_csv(OUT/'A_residual_effects.csv',index=False)
    return df

# ---------- SoE grid sensitivity ----------
def soe_sensitivity():
    bund=load_bund()
    grids=[]
    for K in [5,7,9,11]: grids.append((f'K{K}_1e-2_1e4',np.geomspace(1e-2,1e4,K)))
    grids += [('K7_1e-3_1e4',np.geomspace(1e-3,1e4,7)),('K7_1e-2_1e5',np.geomspace(1e-2,1e5,7)),('K7_1e-3_1e5',np.geomspace(1e-3,1e5,7))]
    prev=OUT/'A_soe_sensitivity.csv'
    rows=pd.read_csv(prev).to_dict('records') if prev.exists() else []
    done=set(pd.DataFrame(rows).grid.unique()) if rows else set()
    for gi,(gname,betas) in enumerate(grids):
        if gname in done:
            print('grid skip',gname,flush=True); continue
        for day,D in bund.items():
            T=max(D['P'][-1],D['Tr'][-1])
            for proc in ['P','Tr']:
                t=D[proc][D[proc]>0]
                fs=fit_soe_mle(t,np.empty(0),T,betas=betas)
                mass=fs['alpha']/fs['betas']
                tau=1/fs['betas']
                bands={
                    'mass_lt_1ms':mass[tau<1e-3].sum(),
                    'mass_1ms_100ms':mass[(tau>=1e-3)&(tau<0.1)].sum(),
                    'mass_0p1_10s':mass[(tau>=0.1)&(tau<10)].sum(),
                    'mass_gt_10s':mass[tau>=10].sum(),
                }
                rows.append(dict(grid=gname,K=len(betas),beta_min=betas.min(),beta_max=betas.max(),day=day,proc=proc,
                                 n=fs['n'],loglik=fs['loglik'],success=fs['success'],**bands))
        pd.DataFrame(rows).to_csv(OUT/'A_soe_sensitivity.csv',index=False)
        print('grid done',gname,flush=True)
    return pd.DataFrame(rows)

# ---------- flexible positive B-spline baseline ----------
def bspline_basis(t,T,knot_step=1800.,degree=3):
    # open uniform knots; all basis functions non-negative and sum to 1 in domain
    internal=np.arange(knot_step,T,knot_step)
    knots=np.r_[np.repeat(0.0,degree+1),internal,np.repeat(T,degree+1)]
    nbas=len(knots)-degree-1
    B=BSpline.design_matrix(np.clip(t,0,T),knots,degree,extrapolate=False).tocsr()
    # exact integral of each basis by antiderivative
    I=np.empty(nbas)
    for j in range(nbas):
        c=np.zeros(nbas); c[j]=1
        sp=BSpline(knots,c,degree,extrapolate=False)
        ant=sp.antiderivative()
        I[j]=ant(T)-ant(0)
    # second-difference roughness surrogate scaled by knot step
    if nbas>=3:
        D2=np.diff(np.eye(nbas),n=2,axis=0)
    else: D2=np.zeros((0,nbas))
    return B,I,D2

def fit_flex_given_R(t,T,B,I,D2,R,S,penalty=1e-3,x0=None):
    N=len(t); m=B.shape[1]; K=R.shape[1]
    rate=max(N/T,1e-6)
    if x0 is None:
        c0=np.repeat(0.45*rate,m)
        a0=np.repeat(0.55*rate/max(K,1)*0.01,K)
        x0=np.r_[c0,a0]
    x0=np.maximum(x0,1e-10)
    def fun(x):
        c=x[:m]; a=x[m:]
        lam=np.asarray(B@c).ravel() + R@a
        if np.any(lam<=0): return 1e100,np.zeros_like(x)
        Dc=D2@c
        ll=np.log(lam).sum()-I@c-S@a-penalty*np.dot(Dc,Dc)
        inv=1/lam
        gc=np.asarray(B.T@inv).ravel()-I-2*penalty*(D2.T@Dc)
        ga=R.T@inv-S
        return -ll,-np.r_[gc,ga]
    res=minimize(fun,x0,jac=True,method='L-BFGS-B',bounds=[(1e-12,None)]*len(x0),options={'maxiter':1600,'ftol':1e-9,'gtol':1e-6,'maxls':100})
    return res

def flex_background_fit(t,T,model='soe',knot_step=1800.,penalty=1e-3,betas=BETAS7,beta_hint=None,raw_fit=None):
    B,I,D2=bspline_basis(t,T,knot_step)
    if model=='soe':
        betas=np.asarray(betas,float); R=_soe_R(np.ascontiguousarray(t),np.empty(0),betas)
        S=np.array([np.sum(1-np.exp(-b*(T-t)))/b for b in betas])
        x0=None
        if raw_fit is not None:
            x0=np.r_[np.repeat(max(raw_fit['mu'],1e-8),B.shape[1]),np.maximum(raw_fit['alpha'],1e-10)]
        res=fit_flex_given_R(t,T,B,I,D2,R,S,penalty,x0=x0)
        m=B.shape[1]; c=res.x[:m]; a=res.x[m:]
        return dict(n=float(np.sum(a/betas)),coeff=c,alpha=a,betas=betas,objective=res.fun,success=res.success,nbas=m)
    # Exponential: profile over beta; coefficients reoptimised for each candidate.
    if beta_hint is None:
        beta_hint=fit_exp_mle(t,np.empty(0),T)['beta']
    logs=np.linspace(np.log(beta_hint)-0.7,np.log(beta_hint)+0.7,3)
    best=None; warm=None
    for lb in logs:
        b=float(np.exp(lb)); R=_soe_R(np.ascontiguousarray(t),np.empty(0),np.array([b]))
        S=np.array([np.sum(1-np.exp(-b*(T-t)))/b])
        res=fit_flex_given_R(t,T,B,I,D2,R,S,penalty,x0=warm)
        warm=res.x
        rec=(res.fun,b,res)
        if best is None or rec[0]<best[0]: best=rec
    _,b,res=best; m=B.shape[1]; c=res.x[:m]; a=float(res.x[m])
    return dict(n=a/b,coeff=c,alpha=np.array([a]),betas=np.array([b]),beta=b,objective=res.fun,success=res.success,nbas=m)

def flexible_background(knot_step=1800.,penalty=1e-3):
    bund=load_bund(); prev=OUT/'A_flexible_background.csv'
    rows=pd.read_csv(prev).to_dict('records') if prev.exists() else []
    done=set(pd.DataFrame(rows).day.astype(str).unique()) if rows else set()
    for di,(day,D) in enumerate(bund.items()):
        if str(day) in done:
            print('flex skip',day,flush=True); continue
        T=max(D['P'][-1],D['Tr'][-1])
        for proc in ['P','Tr']:
            t=D[proc][D[proc]>0]
            raw_e=fit_exp_mle(t,np.empty(0),T); raw_s=fit_soe_mle(t,np.empty(0),T,betas=BETAS7)
            for model in ['exp','soe']:
                f=flex_background_fit(t,T,model=model,knot_step=knot_step,penalty=penalty,beta_hint=raw_e['beta'],raw_fit=(raw_s if model=='soe' else raw_e))
                rows.append(dict(day=day,proc=proc,model=model,knot_step=knot_step,penalty=penalty,n=f['n'],success=f['success'],nbas=f['nbas'],
                                 beta=(f.get('beta',np.nan)),raw_n=(raw_e['n'] if model=='exp' else raw_s['n'])))
        pd.DataFrame(rows).to_csv(OUT/'A_flexible_background.csv',index=False)
        print('flex day',di+1,'/',len(bund),day,flush=True)
    return pd.DataFrame(rows)

# ---------- Hawkes cluster simulation and multiscale stress ----------
def simulate_soe_cluster(T,mu,betas,masses,rng):
    betas=np.asarray(betas,float); masses=np.asarray(masses,float)
    immigrants=np.sort(rng.uniform(0,T,rng.poisson(mu*T)))
    events=list(immigrants)
    queue=list(immigrants)
    q=0
    while q<len(queue):
        parent=queue[q]; q+=1
        for b,nk in zip(betas,masses):
            m=rng.poisson(nk)
            if m:
                ch=parent+rng.exponential(1/b,size=m)
                ch=ch[ch<T]
                if len(ch):
                    events.extend(ch.tolist()); queue.extend(ch.tolist())
        if len(events)>300000: break
    return np.sort(np.asarray(events,float))

def multiscale_stress(reps=30,T=4000.,target_rate=1.5,seed=20260914):
    rng=np.random.default_rng(seed); betas=BETAS7.copy()
    # profiles over kernel mass n_k (not alpha): fast, balanced, slow
    weights={
        'fast':np.array([0.01,0.02,0.04,0.08,0.15,0.30,0.40]),
        'balanced':np.ones(7),
        'slow':np.array([0.40,0.30,0.15,0.08,0.04,0.02,0.01]),
    }
    weights={k:v/v.sum() for k,v in weights.items()}
    prev=OUT/'A_multiscale_stress.csv'
    rows=pd.read_csv(prev).to_dict('records') if prev.exists() else []
    done=set((float(a),str(b)) for a,b in zip(pd.DataFrame(rows).get('n_true',[]),pd.DataFrame(rows).get('profile',[]))) if rows else set()
    for ntrue in [0.50,0.70,0.85,0.95]:
        mu=target_rate*(1-ntrue)
        for prof,w in weights.items():
            if (float(ntrue),str(prof)) in done:
                print('stress skip',ntrue,prof,flush=True); continue
            masses=ntrue*w
            for r in range(reps):
                t=simulate_soe_cluster(T,mu,betas,masses,rng)
                if len(t)<50: continue
                fe=fit_exp_mle(t,np.empty(0),T)
                fs=fit_soe_mle(t,np.empty(0),T,betas=betas)
                # 1-second randomisation: floor to sec + U(0,1), preserve count and sort
                tr=np.sort(np.floor(t)+rng.uniform(0,1,len(t))); tr=tr[tr<T]
                fr=fit_exp_mle(tr,np.empty(0),T)
                rows.append(dict(n_true=ntrue,profile=prof,rep=r,N=len(t),n_exp=fe['n'],n_soe=fs['n'],n_exp_1s=fr['n'],beta_exp=fe['beta']))
            pd.DataFrame(rows).to_csv(OUT/'A_multiscale_stress.csv',index=False)
            print('stress',ntrue,prof,'done',flush=True)
    return pd.DataFrame(rows)

# ---------- symmetric bivariate Hawkes fit ----------
def bivar_stats(up,dn,beta,T):
    # intensity lambda+ = mu + a_s R+ + a_c R- ; lambda- symmetric.
    # Event-level recursions for common beta, plus compensators for parent types.
    up=np.asarray(up,float); dn=np.asarray(dn,float)
    times=np.r_[up,dn]; typ=np.r_[np.ones(len(up),int),-np.ones(len(dn),int)]
    o=np.argsort(times); times=times[o]; typ=typ[o]
    Rp=np.zeros(len(times)); Rm=np.zeros(len(times)); sp=sm=0.; prev=0.
    for i,(ti,ty) in enumerate(zip(times,typ)):
        decay=np.exp(-beta*(ti-prev)); sp*=decay; sm*=decay
        Rp[i]=sp; Rm[i]=sm
        if ty==1: sp+=1
        else: sm+=1
        prev=ti
    # compensator contribution: every parent affects one same and one cross intensity over remaining horizon
    S_up=np.sum(1-np.exp(-beta*(T-up)))/beta
    S_dn=np.sum(1-np.exp(-beta*(T-dn)))/beta
    return times,typ,Rp,Rm,S_up,S_dn

def fit_bivar_common_beta(up,dn,T,beta_hint=None):
    pooled=np.sort(np.r_[up,dn])
    if beta_hint is None: beta_hint=fit_exp_mle(pooled,np.empty(0),T)['beta']
    def solve_beta(beta,x0=None):
        times,typ,Rp,Rm,Sup,Sdn=bivar_stats(up,dn,beta,T)
        # params mu, a_s, a_c >=0
        if x0 is None:
            rate=len(times)/(2*T); x0=np.array([max(rate*.5,1e-6),beta*.2,beta*.2])
        def f(x):
            mu,as_,ac=x
            lam=np.where(typ==1,mu+as_*Rp+ac*Rm,mu+as_*Rm+ac*Rp)
            if np.any(lam<=0): return 1e100,np.zeros(3)
            ll=np.log(lam).sum()-2*mu*T-as_*(Sup+Sdn)-ac*(Sup+Sdn)
            inv=1/lam
            gs=np.sum(inv*np.where(typ==1,Rp,Rm))-(Sup+Sdn)
            gc=np.sum(inv*np.where(typ==1,Rm,Rp))-(Sup+Sdn)
            gm=inv.sum()-2*T
            return -ll,-np.array([gm,gs,gc])
        res=minimize(f,x0,jac=True,method='L-BFGS-B',bounds=[(1e-12,None)]*3,options={'maxiter':300})
        return res
    best=None; warm=None
    for lb in np.linspace(np.log(beta_hint)-1.0,np.log(beta_hint)+1.0,7):
        b=np.exp(lb); res=solve_beta(b,warm); warm=res.x
        if best is None or res.fun<best[0]: best=(res.fun,b,res)
    _,b,res=best; mu,as_,ac=res.x
    return dict(mu=mu,alpha_s=as_,alpha_c=ac,beta=b,n_s=as_/b,n_c=ac/b,n_plus=(as_+ac)/b,n_minus=(as_-ac)/b,loglik=-res.fun,success=res.success)

def bivariate_modes():
    bund=load_bund(); rows=[]
    for i,(day,D) in enumerate(bund.items()):
        T=max(D['P'][-1],D['Tr'][-1])
        f=fit_bivar_common_beta(D['up'],D['dn'],T)
        rows.append(dict(day=day,**f))
        pd.DataFrame(rows).to_csv(OUT/'A_bivariate_modes.csv',index=False)
        print('bivar',i+1,'/',len(bund),day,flush=True)
    return pd.DataFrame(rows)


def merge_min_spacing(t,delta):
    t=np.asarray(t,float)
    if len(t)==0: return t
    keep=[t[0]]
    last=t[0]
    for x in t[1:]:
        if x-last>=delta:
            keep.append(x); last=x
    return np.asarray(keep)

def timestamp_perturbations(seed=20260914):
    bund=load_bund(); rng=np.random.default_rng(seed); rows=[]
    for day,D in bund.items():
        T=max(D['P'][-1],D['Tr'][-1])
        for proc in ['P','Tr']:
            t=D[proc][D[proc]>0]
            # Full-session spacing perturbations.
            for delta in [0.0,0.001,0.01,0.1]:
                tt=t if delta==0 else merge_min_spacing(t,delta)
                fe=fit_exp_mle(tt,np.empty(0),T); fs=fit_soe_mle(tt,np.empty(0),T,betas=BETAS7)
                rows.append(dict(day=day,proc=proc,protocol=('raw' if delta==0 else f'merge_{delta:g}s'),window='day',N=len(tt),n_exp=fe['n'],n_soe=fs['n']))
            # 1-s randomized timestamps, 10-min non-overlapping windows (matching legacy protocol).
            tr=np.sort(np.floor(t)+rng.uniform(0,1,len(t))); tr=tr[tr<T]
            W=600.0
            for start in np.arange(0,T-W+1e-9,W):
                x=tr[(tr>=start)&(tr<start+W)]-start
                if len(x)<20: continue
                fe=fit_exp_mle(x,np.empty(0),W)
                rows.append(dict(day=day,proc=proc,protocol='randomise_1s',window='10min',N=len(x),n_exp=fe['n'],n_soe=np.nan))
    out=pd.DataFrame(rows); out.to_csv(OUT/'A_timestamp_perturbations.csv',index=False); return out


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('task', choices=['residual','soe','flex','stress','bivar','timestamp','all'])
    ap.add_argument('--reps', type=int, default=30)
    a=ap.parse_args()
    tasks=['residual','soe','flex','stress','bivar','timestamp'] if a.task=='all' else [a.task]
    for task in tasks:
        st=time.time(); print('START',task,flush=True)
        if task=='residual': residual_effects()
        elif task=='soe': soe_sensitivity()
        elif task=='flex': flexible_background()
        elif task=='stress': multiscale_stress(reps=a.reps)
        elif task=='bivar': bivariate_modes()
        elif task=='timestamp': timestamp_perturbations()
        print('DONE',task,'sec',time.time()-st,flush=True)
