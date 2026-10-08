"""Bootstrap summaries: null bands, within-model SE, known-misspecification benchmark (R1-3, R2-8)."""
import json, numpy as np, pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'tables_rev'
b=pd.read_csv(R/'R_bootstrap_diagnostics.csv'); out={}; L=[]
for proc in ['P','Tr']:
    for model in ['Exp','SoE']:
        g=b[(b.proc==proc)&(b.model==model)]
        per=g.groupby('day').agg(obs_ks=('obs_ks','first'),obs_cvm=('obs_cvm','first'),obs_a1=('obs_acf1','first'),obs_a10=('obs_acf10','first'),
            ks975=('ks',lambda x:x.quantile(.975)),cvm975=('cvm',lambda x:x.quantile(.975)),a1lo=('acf1',lambda x:x.quantile(.025)),a1hi=('acf1',lambda x:x.quantile(.975)),
            a10lo=('acf10',lambda x:x.quantile(.025)),a10hi=('acf10',lambda x:x.quantile(.975)),se=('n_hat','std'),bias=('n_hat',lambda x:x.mean()),nfit=('n_fit','first'),
            ncross=('n_cross','median'),xks=('x_ks','median'),xcvm=('x_cvm','median'),xa1=('x_acf1','median'),xa10=('x_acf10','median'),reps=('rep','size'))
        per['bias']=per.bias-per.nfit
        out_ks=(per.obs_ks>per.ks975).mean(); out_cvm=(per.obs_cvm>per.cvm975).mean(); out_a1=((per.obs_a1<per.a1lo)|(per.obs_a1>per.a1hi)).mean()
        allin=((per.obs_ks<=per.ks975)&(per.obs_cvm<=per.cvm975)&(per.obs_a1>=per.a1lo)&(per.obs_a1<=per.a1hi)).mean()
        d=dict(reps=int(per.reps.median()),obs_ks=per.obs_ks.median(),ks975=per.ks975.median(),obs_cvm=per.obs_cvm.median(),cvm975=per.cvm975.median(),
               obs_a1=per.obs_a1.median(),a1lo=per.a1lo.median(),a1hi=per.a1hi.median(),a10lo=per.a10lo.median(),a10hi=per.a10hi.median(),
               out_ks=out_ks,out_cvm=out_cvm,out_a1=out_a1,allin=allin,se=per.se.median(),se_max=per.se.max(),bias=per.bias.median(),
               ncross=per.ncross.median(),nfit=per.nfit.median(),xks=per.xks.median(),xcvm=per.xcvm.median(),xa1=per.xa1.median(),xa10=per.xa10.median())
        out[f'{proc}_{model}']=d
        L.append(f"{proc} {model}: reps/session {d['reps']}; obs KS {d['obs_ks']:.3f} vs null97.5 {d['ks975']:.4f} (outside {out_ks:.2f}); obs CvM {d['obs_cvm']:.1f} vs {d['cvm975']:.3f} (outside {out_cvm:.2f}); obs ACF1 {d['obs_a1']:.3f} vs [{d['a1lo']:.4f},{d['a1hi']:.4f}] (outside {out_a1:.2f}); ACF10 band [{d['a10lo']:.4f},{d['a10hi']:.4f}]; all-in fraction {allin:.2f}")
        L.append(f"     within-model SE(n) median {d['se']:.4f} (max {d['se_max']:.4f}); bias of refitted n {d['bias']:+.4f}; n_fit {d['nfit']:.3f}; other kernel on this truth: n {d['ncross']:.3f}, KS {d['xks']:.3f}, CvM {d['xcvm']:.1f}, ACF1 {d['xa1']:.3f}, ACF10 {d['xa10']:.3f}")
# paired: Exp fitted to SoE-truth vs observed Exp n, per session
for proc in ['P','Tr']:
    s=b[(b.proc==proc)&(b.model=='SoE')].groupby('day').n_cross.median(); e=b[(b.proc==proc)&(b.model=='Exp')].groupby('day').n_fit.first()
    L.append(f"{proc}: exponential fitted to SoE-simulated data (pseudo-true at empirical SoE) median {s.median():.3f} vs observed exponential {e.median():.3f}; per-session corr {np.corrcoef(s,e)[0,1]:.2f}; median paired diff {np.median(s.values-e.values):+.3f}")
    out[f'{proc}_pseudo']=dict(sim=s.median(),obs=e.median(),diff=float(np.median(s.values-e.values)))
json.dump(out,open(R/'R_bootstrap_summary.json','w'),indent=1,default=float); open(R/'R_summaries3.txt','w').write('\n'.join(L)); print('\n'.join(L))
