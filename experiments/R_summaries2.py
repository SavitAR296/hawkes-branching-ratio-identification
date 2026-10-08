"""Summaries of the revision tables (timestamp, history, nested, simulations, bivariate, adequacy, sub-samples)."""
import json, numpy as np, pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'tables_rev'; rng=np.random.default_rng(20260914)
def bci(x,B=4000,stat=np.median):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)==0: return (np.nan,np.nan,np.nan)
    bs=np.array([stat(rng.choice(x,len(x))) for _ in range(B)]); return (float(stat(x)),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5)))
f3=lambda r:'%.3f [%.3f, %.3f]'%r
out={}; L=[]; P=L.append
# ---------------- timestamp
fs=pd.read_csv(R/'R_timestamp_fullsession.csv'); w10=pd.read_csv(R/'R_timestamp_10min.csv')
P('== TIMESTAMP full session (median over sessions) ==')
for proc in ['P','Tr']:
    for pr in ['raw','merge_1ms','merge_10ms','merge_0.1s']:
        g=fs[(fs.proc==proc)&(fs.protocol==pr)]
        P(f'{proc} {pr:11s} retained {g.retained.median():.3f}  exp {g.n_exp.median():.3f}  soe {g.n_soe.median():.3f}  beta_exp {g.beta_exp.median():.1f}')
    g=fs[(fs.proc==proc)&(fs.protocol=='randomise_1s')]
    per=g.groupby('day').agg(e=('n_exp','median'),s=('n_soe','median'),elo=('n_exp',lambda x:x.quantile(.025)),ehi=('n_exp',lambda x:x.quantile(.975)),slo=('n_soe',lambda x:x.quantile(.025)),shi=('n_soe',lambda x:x.quantile(.975)),b=('beta_exp','median'),nd=('draw','size'))
    raw=fs[(fs.proc==proc)&(fs.protocol=='raw')].set_index('day')
    P(f'{proc} randomise_1s draws/session {per.nd.median():.0f}: exp median {per.e.median():.3f} (draw range med {per.elo.median():.3f}-{per.ehi.median():.3f}, width {(per.ehi-per.elo).median():.3f}) soe {per.s.median():.3f} ({per.slo.median():.3f}-{per.shi.median():.3f}) beta_exp {per.b.median():.1f}; paired exp-raw {f3(bci(per.e.values-raw.loc[per.index].n_exp.values))} soe-raw {f3(bci(per.s.values-raw.loc[per.index].n_soe.values))}')
    out[f'ts_rand_{proc}']=dict(exp=per.e.median(),exp_lo=per.elo.median(),exp_hi=per.ehi.median(),soe=per.s.median(),soe_lo=per.slo.median(),soe_hi=per.shi.median())
P('== TIMESTAMP 10-min windows, 30-min buffer (median over windows; paired diff vs raw on the same windows) ==')
for proc in ['P','Tr']:
    g=w10[w10.proc==proc]; base=g[g.protocol=='raw'].set_index(['day','start'])
    for pr in ['raw','merge_1ms','merge_10ms','merge_0.1s','randomise_1s']:
        h=g[g.protocol==pr].set_index(['day','start']); j=h.join(base,rsuffix='_raw',how='inner')
        P(f'{proc} {pr:12s} nwin {len(h):5d} exp {h.n_exp.median():.3f} soe {h.n_soe.median():.3f} | dexp {np.median(j.n_exp-j.n_exp_raw):+.3f} dsoe {np.median(j.n_soe-j.n_soe_raw):+.3f} | soe>=1 {(h.n_soe>=1).sum()}')
        out[f'ts10_{proc}_{pr}']=dict(exp=h.n_exp.median(),soe=h.n_soe.median())
# ---------------- history
h=pd.read_csv(R/'R_history_buffer.csv')
P('== HISTORY buffer / boundary exclusion (median over windows) ==')
for proc in ['P','Tr']:
    for wn in ['10min','30min','1h']:
        g=h[(h.proc==proc)&(h.window==wn)]; s=''
        for (H,c),gg in g.groupby(['buffer','cut']):
            s+=f' | b{H/60:.0f}c{c/60:.0f}: {gg.n_exp.median():.3f}/{gg.n_soe.median():.3f}'
            out[f'hist_{proc}_{wn}_b{H/60:.0f}_c{c/60:.0f}']=dict(exp=gg.n_exp.median(),soe=gg.n_soe.median(),soe_ge1=int(gg.soe_ge1.sum()),n=len(gg))
        P(f'{proc} {wn:5s}{s}')
    g=h[(h.proc==proc)]; P(f'{proc} exp bound hits {g.exp_bound.sum()} of {len(g)}; soe>=1 by buffer: '+str(g.groupby(["window","buffer","cut"]).soe_ge1.sum().to_dict()))
# ---------------- nested
nb=pd.read_csv(R/'R_nested_oos.csv')
P('== NESTED free-rate K=1,2,3 and K=7 grid ==')
for proc in ['P','Tr']:
    g=nb[nb.proc==proc]; piv=lambda c: g.pivot(index='day',columns='model',values=c)
    n=piv('n'); aic=piv('aic'); bic=piv('bic'); oos=piv('oos_ll_per_event'); ll=piv('loglik')
    for m in ['K1_exp','K2_free','K3_free','K7_grid']:
        d_oos=oos[m]-oos['K1_exp']
        P(f'{proc} {m:8s} n {f3(bci(n[m]))} dAIC {np.median(aic[m]-aic["K1_exp"]):.0f} dBIC {np.median(bic[m]-bic["K1_exp"]):.0f} oos/ev {oos[m].median():.4f} d_oos {f3(bci(d_oos))} win {np.mean(d_oos>0):.2f}')
        out[f'nest_{proc}_{m}']=dict(n=bci(n[m]),dAIC=float(np.median(aic[m]-aic['K1_exp'])),dBIC=float(np.median(bic[m]-bic['K1_exp'])),oos=float(oos[m].median()),doos=bci(d_oos),win=float(np.mean(d_oos>0)))
    d73=oos['K7_grid']-oos['K3_free']; P(f'{proc} K7-K3 oos {f3(bci(d73))} win {np.mean(d73>0):.2f}; K7 vs K3 dAIC {np.median(aic["K7_grid"]-aic["K3_free"]):.0f} dBIC {np.median(bic["K7_grid"]-bic["K3_free"]):.0f}')
    out[f'nest_{proc}_K7mK3']=dict(doos=bci(d73),win=float(np.mean(d73>0)),dAIC=float(np.median(aic["K7_grid"]-aic["K3_free"])),dBIC=float(np.median(bic["K7_grid"]-bic["K3_free"])))
    for m in ['K2_free','K3_free']:
        bet=g[g.model==m].betas.str.split(';',expand=True).astype(float); ms=g[g.model==m].masses.str.split(';',expand=True).astype(float)
        P(f'{proc} {m} median rates {np.round(bet.median().values,3).tolist()} masses {np.round(ms.median().values,3).tolist()}')
# ---------------- simulations
sc=pd.read_csv(R/'R_sim_controls.csv')
P('== SIM grid_burnin ==')
g=sc[sc.design=='grid_burnin']
for (T,pf,nt),gg in g.groupby(['T','profile','n_true']):
    P(f'T={T:.0f} {pf:8s} n={nt:.2f} reps {len(gg):2d} Nmed {gg.N.median():.0f} exp {gg.n_exp.mean():.3f}±{gg.n_exp.std():.3f} soe {gg.n_soe.mean():.3f}±{gg.n_soe.std():.3f} | nohist exp {gg.n_exp_nohist.mean():.3f} soe {gg.n_soe_nohist.mean():.3f}')
    out[f'sim_burn_{T:.0f}_{pf}_{nt:.2f}']=dict(N=gg.N.median(),exp=gg.n_exp.mean(),exp_sd=gg.n_exp.std(),soe=gg.n_soe.mean(),soe_sd=gg.n_soe.std(),exp_nh=gg.n_exp_nohist.mean(),soe_nh=gg.n_soe_nohist.mean(),reps=len(gg))
P('== SIM controls ==')
for d in ['exp_dgp','offgrid6','powerlaw25']:
    for (pf,nt),gg in sc[sc.design==d].groupby(['profile','n_true']):
        P(f'{d:10s} {pf:10s} n={nt:.2f} reps {len(gg)} Nmed {gg.N.median():.0f} exp {gg.n_exp.mean():.3f}±{gg.n_exp.std():.3f} soe {gg.n_soe.mean():.3f}±{gg.n_soe.std():.3f} beta_exp {gg.beta_exp.median():.2f}')
        out[f'sim_{d}_{pf}_{nt:.2f}']=dict(exp=gg.n_exp.mean(),exp_sd=gg.n_exp.std(),soe=gg.n_soe.mean(),soe_sd=gg.n_soe.std())
for d in ['cox_logOU','regime2']:
    gg=sc[sc.design==d]
    P(f'{d}: reps {len(gg)} Nmed {gg.N.median():.0f} exp {gg.n_exp.mean():.3f}±{gg.n_exp.std():.3f} soe {gg.n_soe.mean():.3f}±{gg.n_soe.std():.3f} exp_flex {gg.n_exp_flex.mean():.3f}±{gg.n_exp_flex.std():.3f} soe_flex {gg.n_soe_flex.mean():.3f}±{gg.n_soe_flex.std():.3f} beta_exp med {gg.beta_exp.median():.4f}')
    out[f'sim_{d}']=dict(exp=gg.n_exp.mean(),exp_sd=gg.n_exp.std(),soe=gg.n_soe.mean(),soe_sd=gg.n_soe.std(),exp_flex=gg.n_exp_flex.mean(),soe_flex=gg.n_soe_flex.mean(),soe_flex_sd=gg.n_soe_flex.std(),exp_flex_sd=gg.n_exp_flex.std())
gg=sc[sc.design=='slow95_multistart']
P(f'slow95 multistart: reps {len(gg)} two-optima frac {np.mean(gg.n_modes==2):.2f}; ll_gap median {gg.ll_gap_2nd_mode.median():.2f} (range {gg.ll_gap_2nd_mode.min():.2f}-{gg.ll_gap_2nd_mode.max():.2f}); all_success {gg.all_success.mean():.2f}; best n_exp dist {np.round(np.percentile(gg.n_exp,[10,25,50,75,90]),3).tolist()}; best beta median {gg.beta_exp.median():.2f}; alt n median {gg.n_alt.median():.3f}; N med {gg.N.median():.0f}')
best_slow=gg.beta_exp<10; P(f'   best optimum slow-rate (beta<10) in {best_slow.mean():.2f} of reps; n_exp when slow {gg.n_exp[best_slow].median():.3f}, when fast {gg.n_exp[~best_slow].median():.3f}')
out['slow95']=dict(two=float(np.mean(gg.n_modes==2)),gap=float(gg.ll_gap_2nd_mode.median()),succ=float(gg.all_success.mean()),slowfrac=float(best_slow.mean()),n_slow=float(gg.n_exp[best_slow].median()) if best_slow.any() else None,n_fast=float(gg.n_exp[~best_slow].median()) if (~best_slow).any() else None)
# ---------------- bivariate
bv=pd.read_csv(R/'R_bivariate.csv'); ma=pd.read_csv(R/'R_bivariate_modes_acf.csv')
P('== BIVARIATE ==')
d1=bv.ll_sep-bv.ll_common; d2=bv.ll_soe-bv.ll_common
P(f'll gain sep vs common: median {d1.median():.1f} (min {d1.min():.1f}); LR=2dl median {2*d1.median():.1f}, df=1; all sessions significant at 1%: {(2*d1>6.63).all()}')
P(f'll gain biv-SoE vs common: median {d2.median():.1f} (min {d2.min():.1f}); dAIC median {-2*d2.median()+2*12:.0f}')
for c in ['beta_common','nplus_common','nminus_common','beta_s','beta_c','nplus_sep','nminus_sep','n_s_soe','n_c_soe','nplus_soe','nminus_soe','tau_act_soe']:
    P(f'  {c:14s} {f3(bci(bv[c]))}'); out[f'bv_{c}']=bci(bv[c])
P(f'  sessions with nminus_soe<0: {(bv.nminus_soe<0).sum()} of {len(bv)}; nminus_sep<0: {(bv.nminus_sep<0).sum()}')
P('  self masses per rate (1e-2..1e4): '+str(np.round(bv[[f"ms{k}" for k in range(7)]].median().values,3).tolist()))
P('  cross masses per rate (1e-2..1e4): '+str(np.round(bv[[f"mc{k}" for k in range(7)]].median().values,3).tolist()))
sl=[0,1,2]; P(f'  slow (beta<=1) self {bv[[f"ms{k}" for k in sl]].sum(1).median():.3f} cross {bv[[f"mc{k}" for k in sl]].sum(1).median():.3f}; mid (10,100) self {bv[["ms3","ms4"]].sum(1).median():.3f} cross {bv[["mc3","mc4"]].sum(1).median():.3f}; fast (1e3,1e4) self {bv[["ms5","ms6"]].sum(1).median():.3f} cross {bv[["mc5","mc6"]].sum(1).median():.3f}')
for c in ['act_emp','act_th','sig_emp','sig_th']:
    P(f'  {c} mean lags1-6 {np.round(ma[[f"{c}{k}" for k in range(1,7)]].mean().values,3).tolist()}')
P(f'  window-level nplus median {ma.nplus.median():.3f} nminus median {ma.nminus.median():.3f}; nminus<0 in {np.mean(ma.nminus<0):.2f} of {len(ma)} windows; sig_th1<0 in {np.mean(ma.sig_th1<0):.2f}')
P(f'  paired act lag1 th-emp {f3(bci(ma.act_th1-ma.act_emp1))}; sig lag1 th {f3(bci(ma.sig_th1,stat=np.mean))} emp {f3(bci(ma.sig_emp1,stat=np.mean))}')
# ---------------- adequacy extra
ad=pd.read_csv(R/'R_adequacy_extra.csv')
P('== ADEQUACY spline / bivariate (medians) ==')
for (proc,m),gg in ad.groupby(['proc','model']):
    P(f'{proc:3s} {m:13s} n {gg.n.median():.3f} KS {gg.ks.median():.3f} CvM {gg.cvm.median():.1f} ACF1 {gg.acf1.median():.3f} ACF10 {gg.acf10.median():.3f}'+(f' se(n+) {gg.se_nplus.median():.4f} se(n-) {gg.se_nminus.median():.4f} n- {gg.n_minus.median():.3f}' if m=='bivar_common' else ''))
    out[f'ad_{proc}_{m}']=dict(n=gg.n.median(),ks=gg.ks.median(),cvm=gg.cvm.median(),acf1=gg.acf1.median(),acf10=gg.acf10.median())
    if m=='bivar_common': out['ad_bivar_se']=dict(sep=gg.se_nplus.median(),sem=gg.se_nminus.median())
# ---------------- sub-samples: activity and announcement days
ds=pd.read_csv(ROOT/'tables/E1_daily_summary.csv'); tot=ds.groupby('day').N.sum(); hi=tot[tot>tot.median()].index; ann=[20140403,20140404,20140430]
P('== SUB-SAMPLES (full-session medians, constant background) ==')
for proc in ['P','Tr']:
    g=ds[ds.proc==proc].set_index('day')
    for lab,idx in [('high-activity',hi),('low-activity',tot.index.difference(hi)),('announcement',pd.Index(ann)),('non-announcement',tot.index.difference(ann))]:
        gg=g.loc[idx]; P(f'{proc} {lab:16s} k={len(gg):2d} exp {gg.n_mle.median():.3f} soe {gg.n_soe.median():.3f} gap {np.median(gg.n_soe-gg.n_mle):.3f}')
        out[f'sub_{proc}_{lab}']=dict(k=len(gg),exp=gg.n_mle.median(),soe=gg.n_soe.median(),gap=float(np.median(gg.n_soe-gg.n_mle)))
json.dump(out,open(R/'R_numbers.json','w'),indent=1,default=float); open(R/'R_summaries2.txt','w').write('\n'.join(L)); print('\n'.join(L))
