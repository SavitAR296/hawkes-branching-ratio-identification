"""Archive-based summaries for the revision: bootstrap CIs of medians, paired differences, decomposition table,
aligned 10-min trajectories (R2-8), H3' critical-slowing test incl. exponential-implied and deseasonalised counts (R2-0)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'experiments'))
from hawkes import exp_hawkes_bin_autocorr
import A_csf_additions as A
OUT=ROOT/'tables_rev'; OUT.mkdir(exist_ok=True); rng=np.random.default_rng(20260914)
def bci(x,B=4000,stat=np.median):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; m=stat(x); bs=np.array([stat(rng.choice(x,len(x))) for _ in range(B)])
    return m,np.percentile(bs,2.5),np.percentile(bs,97.5)
fmt=lambda r:'%.3f [%.3f, %.3f]'%r
ds=pd.read_csv(ROOT/'tables/E1_daily_summary.csv'); fb=pd.read_csv(ROOT/'tables_csf/A_flexible_background.csv')
ts=pd.read_csv(ROOT/'tables_csf/A_timestamp_perturbations.csv'); w=pd.read_csv(ROOT/'tables/E1_window_fits.csv')
lines=[]
P=lambda s: lines.append(s)
P('== Table 1 with session-bootstrap 95% CI of the median (20 sessions) ==')
for proc,lab in [('P','Mid-price'),('Tr','Trades')]:
    s=ds[ds.proc==proc].set_index('day'); f=fb[fb.proc==proc]
    fe=f[f.model=='exp'].set_index('day').n; fs=f[f.model=='soe'].set_index('day').n
    P(f'{lab}: Exp const {fmt(bci(s.n_mle))} | SoE const {fmt(bci(s.n_soe))} | Exp flex {fmt(bci(fe))} | SoE flex {fmt(bci(fs))}')
    P(f'   paired dn SoE-Exp const {fmt(bci(s.n_soe-s.n_mle))} ; SoE-Exp flex {fmt(bci(fs.values-fe.values))} ; SoE const-flex {fmt(bci(s.n_soe.values-fs.values))} ; Exp const-flex {fmt(bci(s.n_mle.values-fe.values))}')
    P(f'   Laplace SD median {s.n_sd.median():.4f}; exp beta median {s.beta_mle.median():.0f}; slowest SoE relaxation time median {(1/s.gamma_min_soe).median():.0f} s; mean kernel time {s.m_soe.median():.1f} s')
P('== Decomposition (full session, one dimension changed at a time; paired per session, median [95% CI]) ==')
for proc,lab in [('P','Mid-price'),('Tr','Trades')]:
    s=ds[ds.proc==proc].set_index('day'); f=fb[fb.proc==proc]; fe=f[f.model=='exp'].set_index('day').n; fs=f[f.model=='soe'].set_index('day').n
    t=ts[(ts.proc==proc)&(ts.window=='day')].pivot(index='day',columns='protocol',values=['n_exp','n_soe'])
    wm=w[w.proc==proc].groupby(['day','window'])[['n_mle','n_soe']].median().unstack()
    P(f'{lab}:')
    P(f'   kernel Exp->SoE (const bg)        {fmt(bci(s.n_soe-s.n_mle))}')
    P(f'   background const->spline  Exp     {fmt(bci(fe.values-s.n_mle.values))}   SoE {fmt(bci(fs.values-s.n_soe.values))}')
    for d in ['merge_0.001s','merge_0.01s','merge_0.1s']:
        P(f'   timestamp raw->{d:13s} Exp {fmt(bci(t["n_exp"][d]-t["n_exp"]["raw"]))}   SoE {fmt(bci(t["n_soe"][d]-t["n_soe"]["raw"]))}')
    for win in ['1h','30min','10min']:
        P(f'   window full->{win:6s} (median of windows) Exp {fmt(bci(wm["n_mle"][win].values-s.n_mle.values))}   SoE {fmt(bci(wm["n_soe"][win].values-s.n_soe.values))}')
P('== Aligned 10-min trajectories (R2-8): per-session correlation Exp vs SoE across windows; sign agreement of consecutive changes ==')
for proc,lab in [('P','Mid-price'),('Tr','Trades')]:
    cors=[];rhos=[];agree=[]
    for d,g in w[(w.proc==proc)&(w.window=='10min')].groupby('day'):
        g=g.sort_values('start'); cors.append(np.corrcoef(g.n_mle,g.n_soe)[0,1]); rhos.append(g[['n_mle','n_soe']].corr('spearman').iloc[0,1])
        a=np.sign(np.diff(g.n_mle)); b=np.sign(np.diff(g.n_soe)); agree.append(np.mean(a==b))
    P(f'{lab}: Pearson {fmt(bci(cors))}, Spearman {fmt(bci(rhos))}, sign agreement of changes {fmt(bci(agree))} (0.5 = chance); SoE windows with n>=1: {(w[(w.proc==proc)].n_soe>=1).sum()} of {len(w[w.proc==proc])}')
P('== H3prime critical-slowing test: 1-h windows, 10-s bins, lags 1-6 (means over 260 windows per stream) ==')
h=pd.read_csv(ROOT/'tables/E1_h3prime.csv')
for proc,lab in [('P','Mid-price'),('Tr','Trades')]:
    s=h[h.proc==proc]
    expth=np.array([exp_hawkes_bin_autocorr(1.0,r.n_exp*r.beta_exp,r.beta_exp,10.0,kmax=6)[2] for r in s.itertuples()])
    P(f'{lab}: empirical {np.round(s[[f"rho{j}_emp" for j in range(1,7)]].mean().values,3).tolist()}')
    P(f'   SoE-implied {np.round(s[[f"rho{j}_th" for j in range(1,7)]].mean().values,3).tolist()}  Exp-implied {np.round(np.nanmean(expth,0),4).tolist()}')
    P(f'   slowest SoE relaxation time (1-h fits) median {(1/s.gamma_min).median():.0f} s; implied exp(-gamma*10s) median {s.ar1_th.median():.3f}; lag-1 paired diff SoE-emp {fmt(bci(s.rho1_th-s.rho1_emp))}')
    if proc=='P': P(f'   signed (up-down) counts empirical {np.round(s[[f"rho{j}_ret" for j in range(1,7)]].mean().values,3).tolist()}')
# deseasonalised counts: divide 10-s counts by the cross-session mean intraday profile (30-min bins, 20 sessions)
bund=A.load_bund(); T=50400.0; prof={}
for proc in ['P','Tr']:
    cnt=np.array([np.histogram(D[proc],bins=np.arange(0,T+1,1800))[0] for D in bund.values()]).mean(0); prof[proc]=cnt/cnt.mean()
P('== Deseasonalised 10-s count autocorrelation (counts divided by the 20-session mean 30-min intraday profile) ==')
for proc,lab in [('P','Mid-price'),('Tr','Trades')]:
    raw=[];des=[]
    for D in bund.values():
        for s in np.arange(0,T-3600+1,3600):
            c=np.histogram(D[proc][(D[proc]>=s)&(D[proc]<s+3600)]-s,bins=np.arange(0,3601,10))[0].astype(float)
            if c.sum()<60: continue
            f=prof[proc][np.minimum(((s+np.arange(0,3600,10)+5)//1800).astype(int),len(prof[proc])-1)]
            def acf(x):
                x=x-x.mean(); v=x@x; return [x[:-k]@x[k:]/v for k in range(1,7)]
            raw.append(acf(c)); des.append(acf(c/f))
    P(f'{lab}: raw {np.round(np.mean(raw,0),3).tolist()}  deseasonalised {np.round(np.mean(des,0),3).tolist()}')
P('== Hardiman-Bouchaud moment estimator on raw vs deseasonalised full-session counts ==')
hb=pd.read_csv(ROOT/'tables/E1_hb_estimator.csv')
for proc,lab in [('P','Mid-price'),('Tr','Trades')]:
    out=[]
    for W_ in [1,5,10,30,60,300,600]:
        vals=[]
        for D in bund.values():
            c=np.histogram(D[proc],bins=np.arange(0,T+1e-9,W_))[0].astype(float); f=prof[proc][np.minimum(((np.arange(0,T,W_)+W_/2)//1800).astype(int),len(prof[proc])-1)]
            cd=c/f; m,v=cd.mean(),cd.var(ddof=1); vals.append(1-np.sqrt(m/v) if v>m else 0.0)
        out.append('W=%d: raw %.3f des %.3f'%(W_,hb[(hb.proc==proc)&(hb.W==W_)].n_hb.median(),np.median(vals)))
    P(f'{lab}: '+' | '.join(out))
open(OUT/'R_summaries.txt','w').write('\n'.join(lines)); print('\n'.join(lines))
