"""Penalty x knot scan with blocked CV (R1-6, R2-2) and corrected bootstrap (R1-3, R2-8): summaries -> JSON + text."""
import json, numpy as np, pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'tables_rev'; rng=np.random.default_rng(20260914)
def bci(x,B=4000):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; bs=np.array([np.median(rng.choice(x,len(x))) for _ in range(B)])
    return (float(np.median(x)),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5)))
out={}; L=[]; P=L.append
pen=pd.read_csv(R/'R_penalty_knot_scan.csv'); cv=pd.read_csv(R/'R_penalty_cv.csv')
nd=pen.day.nunique(); P(f'== PENALTY scan: {nd} sessions ==')
cnt=pen.groupby(['proc','knot_min','lam']).day.nunique(); keep=cnt[cnt==nd].index; pen=pen.set_index(['proc','knot_min','lam']).loc[keep].reset_index()
for proc in ['P','Tr']:
    g=pen[pen.proc==proc]; raw_s=g.groupby('day').raw_soe.first(); raw_e=g.groupby('day').raw_exp.first()
    tab=g.groupby(['knot_min','lam']).agg(n=('n_soe','median'),slow=('slow_mass_soe','median'),succ=('succ_soe','mean'),nexp=('n_exp','median'))
    P(f'{proc}: constant-background SoE {raw_s.median():.3f}, Exp {raw_e.median():.3f}')
    P(tab.round(3).to_string())
    lo=tab.n.idxmin(); hi=tab.n.idxmax()
    dmin=(g.set_index('day').groupby(['knot_min','lam']).apply(lambda x:(x.n_soe-raw_s.loc[x.index]).median()))
    gaps=g.dropna(subset=['n_exp']).groupby(['knot_min','lam']).apply(lambda x:np.median(x.n_soe-x.n_exp))
    P(f'  SoE n range over scan: {tab.n.min():.3f} at {lo} to {tab.n.max():.3f} at {hi}; change vs constant: {dmin.min():+.3f} to {dmin.max():+.3f}')
    P(f'  slow mass range: {tab.slow.min():.3f} to {tab.slow.max():.3f}; constant-bg slow mass from optimum check reported separately')
    P(f'  Exp n over its configs: {tab.nexp.dropna().min():.3f} to {tab.nexp.dropna().max():.3f}; SoE-Exp paired gap over configs: {gaps.min():.3f} to {gaps.max():.3f}')
    out[f'pen_{proc}']=dict(n_min=float(tab.n.min()),n_min_at=[float(v) for v in lo],n_max=float(tab.n.max()),n_max_at=[float(v) for v in hi],d_min=float(dmin.min()),d_max=float(dmin.max()),
        slow_min=float(tab.slow.min()),slow_max=float(tab.slow.max()),exp_min=float(tab.nexp.dropna().min()),exp_max=float(tab.nexp.dropna().max()),gap_min=float(gaps.min()),gap_max=float(gaps.max()),
        raw_soe=float(raw_s.median()),raw_exp=float(raw_e.median()),table={f'{k[0]:g}|{k[1]:g}':dict(n=float(v.n),slow=float(v.slow),nexp=(None if pd.isna(v.nexp) else float(v.nexp))) for k,v in tab.iterrows()})
    c=cv[(cv.proc==proc)]; cc=c.groupby(['knot_min','lam']).day.nunique(); c=c.set_index(['knot_min','lam']).loc[cc[cc==nd].index].reset_index()
    piv=c.pivot_table(index='day',columns=['knot_min','lam'],values='cv_ll_per_event'); best=piv.idxmax(axis=1).value_counts()
    ref=(30.0,0.0); rel={k:bci(piv[k]-piv[ref]) for k in piv.columns}
    P(f'  CV: selected configuration counts {dict(best)}'); P('  CV held-out ll/event relative to (30 min, 0): '+'; '.join(f'{k}: {v[0]:+.4f} [{v[1]:+.4f},{v[2]:+.4f}]' for k,v in rel.items()))
    sel=best.idxmax(); nsel=g[(g.knot_min==sel[0])&(g.lam==sel[1])].n_soe; P(f'  modal CV choice {sel}: SoE n {bci(nsel)}')
    out[f'cv_{proc}']=dict(counts={f'{k[0]:g}|{k[1]:g}':int(v) for k,v in best.items()},rel={f'{k[0]:g}|{k[1]:g}':v for k,v in rel.items()},sel=[float(v) for v in sel],n_sel=bci(nsel))
# ---------------- corrected bootstrap (SoE truth refitted with the well-scaled optimiser)
b=pd.read_csv(R/'R_bootstrap_diagnostics.csv'); f=pd.read_csv(R/'R_bootstrap_soe_refit.csv')
s=b[b.model=='SoE'].merge(f,on=['day','proc','rep'],suffixes=('_old',''))
P(f'== BOOTSTRAP (SoE truth refitted) == units {f.groupby(["day","proc"]).ngroups}; same simulated N: {(s.N_old==s.N).mean():.3f}; refits improved by >1 nat: {(s.dll>1).mean():.3f}; |n change| max {np.abs(s.n_hat-s.n_hat_base).max():.4f}')
for proc in ['P','Tr']:
    for model in ['Exp','SoE']:
        g=(b[(b.proc==proc)&(b.model=='Exp')] if model=='Exp' else s[s.proc==proc])
        per=g.groupby('day').agg(obs_ks=('obs_ks','first'),obs_cvm=('obs_cvm','first'),obs_a1=('obs_acf1','first'),obs_a10=('obs_acf10','first'),
             ks975=('ks',lambda x:x.quantile(.975)),cvm975=('cvm',lambda x:x.quantile(.975)),a1lo=('acf1',lambda x:x.quantile(.025)),a1hi=('acf1',lambda x:x.quantile(.975)),
             se=('n_hat','std'),nfit=('n_fit_old' if model=='SoE' else 'n_fit','first'),ncross=('n_cross','median'),xks=('x_ks','median'),xcvm=('x_cvm','median'),xa1=('x_acf1','median'),xa10=('x_acf10','median'))
        outside=((per.obs_ks>per.ks975)|(per.obs_cvm>per.cvm975)|(per.obs_a1<per.a1lo)|(per.obs_a1>per.a1hi)).mean()
        d=dict(obs_ks=per.obs_ks.median(),ks975=per.ks975.median(),obs_cvm=per.obs_cvm.median(),cvm975=per.cvm975.median(),obs_a1=per.obs_a1.median(),a1lo=per.a1lo.median(),a1hi=per.a1hi.median(),
               out_ks=(per.obs_ks>per.ks975).mean(),out_cvm=(per.obs_cvm>per.cvm975).mean(),out_a1=((per.obs_a1<per.a1lo)|(per.obs_a1>per.a1hi)).mean(),outside=outside,
               se=per.se.median(),se_max=per.se.max(),ncross=per.ncross.median(),xks=per.xks.median(),xcvm=per.xcvm.median(),xa1=per.xa1.median(),xa10=per.xa10.median())
        out[f'boot_{proc}_{model}']={k:float(v) for k,v in d.items()}
        P(f"{proc} {model}: obs KS {d['obs_ks']:.3f} | null97.5 {d['ks975']:.4f} | CvM {d['obs_cvm']:.1f} vs {d['cvm975']:.3f} | ACF1 {d['obs_a1']:.3f} vs [{d['a1lo']:.4f},{d['a1hi']:.4f}] | outside any: {outside:.2f} (KS {d['out_ks']:.2f}, CvM {d['out_cvm']:.2f}, ACF1 {d['out_a1']:.2f}) | SE(n) {d['se']:.4f} (max {d['se_max']:.4f}) | other kernel: n {d['ncross']:.3f} KS {d['xks']:.3f} CvM {d['xcvm']:.1f} ACF1 {d['xa1']:.3f} ACF10 {d['xa10']:.3f}")
ds=pd.read_csv(ROOT/'tables/E1_daily_summary.csv'); t=ds[ds.proc=='Tr']
P('Trades exponential rate by session (s^-1): '+str(np.round(np.sort(t.beta_mle.values),1).tolist())); P('   and n: '+str(np.round(t.sort_values('beta_mle').n_mle.values,3).tolist()))
oc=pd.read_csv(R/'R_soe_optimum_check.csv'); B=np.array([1e-2,1e-1,1,10,1e2,1e3,1e4])
for proc in ['P','Tr']:
    m=oc[oc.proc==proc][[f'm{k}' for k in range(7)]].values
    fast=m[:,4:].sum(1); slow=m[:,:3].sum(1); mid=m[:,3]
    P(f'{proc} tight-fit masses: tau<0.1 s {np.median(fast):.3f} [{np.percentile(fast,25):.3f},{np.percentile(fast,75):.3f}], tau=0.1 s {np.median(mid):.3f}, tau>=1 s {np.median(slow):.3f}; per-rate medians {np.round(np.median(m,0),3).tolist()}')
    out[f'mass_{proc}']=dict(fast=float(np.median(fast)),mid=float(np.median(mid)),slow=float(np.median(slow)))
json.dump(out,open(R/'R_summaries4.json','w'),indent=1,default=float); open(R/'R_summaries4.txt','w').write('\n'.join(L)); print('\n'.join(L))
