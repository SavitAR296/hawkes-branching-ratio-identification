"""Figures added in revision: critical-slowing test, bivariate SoE modes, stress map with controls (+ penalty and robustness when available)."""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'experiments'))
from hawkes import exp_hawkes_bin_autocorr
import A_csf_additions as A
FIG=ROOT/'figures_rev'; FIG.mkdir(exist_ok=True); R=ROOT/'tables_rev'; rng=np.random.default_rng(20260914)
plt.rcParams.update({'font.size':9,'axes.titlesize':9,'legend.fontsize':7.5,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
CE,CS,CR,CK='#1f77b4','#ff7f0e','#2ca02c','black'
def mband(X,B=2000):
    X=np.asarray(X,float); m=np.nanmean(X,0); idx=rng.integers(0,len(X),(B,len(X))); bs=np.nanmean(X[idx],1)
    return m,np.percentile(bs,2.5,0),np.percentile(bs,97.5,0)
lags=np.arange(1,7)
def fig_csd():
    h=pd.read_csv(ROOT/'tables/E1_h3prime.csv'); bund=A.load_bund(); T=50400.0
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.9))
    for ax,proc,lab in [(axs[0],'P','(a) mid-price changes'),(axs[1],'Tr','(b) trades')]:
        s=h[h.proc==proc]
        cnt=np.array([np.histogram(D[proc],bins=np.arange(0,T+1,1800))[0] for D in bund.values()]).mean(0); prof=cnt/cnt.mean()
        des=[]
        for r in s.itertuples():
            D=bund[str(r.day)] if str(r.day) in bund else bund[r.day]
            c=np.histogram(D[proc][(D[proc]>=r.start)&(D[proc]<r.start+3600)]-r.start,bins=np.arange(0,3601,10))[0].astype(float)
            f=prof[np.minimum(((r.start+np.arange(0,3600,10)+5)//1800).astype(int),len(prof)-1)]; x=c/f; x=x-x.mean(); v=x@x
            des.append([x[:-k]@x[k:]/v for k in lags])
        emp=s[[f'rho{k}_emp' for k in lags]].values; soe=s[[f'rho{k}_th' for k in lags]].values
        ex=np.array([exp_hawkes_bin_autocorr(1.0,r.n_exp*r.beta_exp,r.beta_exp,10.0,kmax=6)[2] for r in s.itertuples()])
        for X,c,ls,lb in [(emp,CK,'-','empirical'),(np.array(des),'grey','--','empirical, deseasonalised'),(soe,CS,'-','SoE-implied'),(ex,CE,'-','exponential-implied')]:
            m,lo,hi=mband(X); ax.plot(lags,m,ls,color=c,marker='o',ms=3,lw=1.3,label=lb); ax.fill_between(lags,lo,hi,color=c,alpha=0.18,lw=0)
        ax.axhline(0,color='k',lw=0.5); ax.set_xlabel('lag (10-s bins)'); ax.set_title(lab,loc='left'); ax.set_ylim(-0.02,0.32)
    axs[0].set_ylabel('autocorrelation of 10-s counts'); axs[0].legend(frameon=False,loc='upper right')
    fig.tight_layout(); fig.savefig(FIG/'R_fig_csd_test.pdf'); plt.close(fig)
def fig_bivariate():
    bv=pd.read_csv(R/'R_bivariate.csv'); ma=pd.read_csv(R/'R_bivariate_modes_acf.csv'); betas=np.array([1e-2,1e-1,1,10,1e2,1e3,1e4])
    fig,axs=plt.subplots(1,3,figsize=(7.4,2.6))
    ms=bv[[f'ms{k}' for k in range(7)]].values; mc=bv[[f'mc{k}' for k in range(7)]].values
    for X,mk,c,lb,off in [(ms,'o',CE,'self $n_{s,k}$',0.92),(mc,'s',CS,'cross $n_{c,k}$',1.08)]:
        med=np.median(X,0); q1,q3=np.percentile(X,25,0),np.percentile(X,75,0)
        axs[0].errorbar(betas*off,med,yerr=[med-q1,q3-med],fmt=mk,color=c,ms=4,capsize=2,label=lb,mfc=(c if mk=='o' else 'white'))
    axs[0].set_xscale('log'); axs[0].set_xlabel(r'rate $\beta_k$ (s$^{-1}$)'); axs[0].set_ylabel(r'kernel mass $\alpha_k/\beta_k$'); axs[0].legend(frameon=False); axs[0].set_title('(a) bivariate SoE masses',loc='left')
    for ax,pre,ttl in [(axs[1],'act','(b) activity $N^+ + N^-$'),(axs[2],'sig','(c) signed $N^+ - N^-$')]:
        for c,suf,lb in [(CK,'emp','empirical'),(CS,'th','bivariate-SoE implied')]:
            m,lo,hi=mband(ma[[f'{pre}_{suf}{k}' for k in lags]].values); ax.plot(lags,m,'-o',color=c,ms=3,lw=1.3,label=lb); ax.fill_between(lags,lo,hi,color=c,alpha=0.18,lw=0)
        ax.axhline(0,color='k',lw=0.5); ax.set_xlabel('lag (10-s bins)'); ax.set_title(ttl,loc='left')
    axs[1].set_ylabel('autocorrelation of 10-s counts'); axs[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG/'R_fig_bivariate_soe.pdf'); plt.close(fig)
def fig_stress():
    sc=pd.read_csv(R/'R_sim_controls.csv'); ar=pd.read_csv(ROOT/'tables_csf/A_multiscale_stress.csv')
    fig,axs=plt.subplots(2,3,figsize=(7.4,5.0)); axs=axs.ravel()
    for ax,pf in zip(axs[:3],['fast','balanced','slow']):
        ax.plot([0.45,1],[0.45,1],'k--',lw=0.6)
        g=sc[(sc.design=='grid_burnin')&(sc.profile==pf)&(sc['T']==4000)].groupby('n_true'); a=ar[ar.profile==pf].groupby('n_true')
        for col,c,lb in [('n_exp',CE,'exponential'),('n_soe',CS,'SoE')]:
            m=g[col].mean(); sd=g[col].std(); ax.errorbar(m.index,m.values,yerr=sd.values,fmt='o-',color=c,ms=3.5,capsize=2,lw=1.2,label=lb+' (stationary)')
            am=a[col].mean(); ax.plot(am.index+0.008,am.values,'o',mfc='white',color=c,ms=3.5,label=lb+' (archived)')
        gl=sc[(sc.design=='grid_burnin')&(sc.profile==pf)&(sc['T']==50400)].groupby('n_true')
        if len(gl):
            for col,c in [('n_exp',CE),('n_soe',CS)]:
                ax.plot(gl[col].mean().index-0.008,gl[col].mean().values,'D',color=c,ms=4.5,mec='k',mew=0.5,label=('$T=50{,}400$ s' if col=='n_soe' else None))
        ax.set_title(f'({"abc"[["fast","balanced","slow"].index(pf)]}) {pf} profile',loc='left'); ax.set_ylim(0,1.02); ax.set_xlabel('true $n$')
    axs[0].set_ylabel('estimated $n$'); axs[0].legend(frameon=False,fontsize=5.6,loc='lower right')
    ax=axs[3]; ax.plot([0.45,1],[0.45,1],'k--',lw=0.6)
    for d,mk,lb in [('offgrid6','s','off-grid 6 rates'),('powerlaw25','^','power-law-like 25 rates'),('exp_dgp','o','exponential truth')]:
        g=sc[(sc.design==d)&((sc.profile=='beta100')|(sc.design!='exp_dgp'))].groupby('n_true')
        ax.plot(g.n_soe.mean().index,g.n_soe.mean().values,mk+'-',color=CS,ms=4,lw=1,label='SoE, '+lb); ax.plot(g.n_exp.mean().index,g.n_exp.mean().values,mk+':',color=CE,ms=4,lw=1)
    ax.set_title('(d) controls (a)--(b): off-grid truths',loc='left'); ax.set_xlabel('true $n$'); ax.set_ylabel('estimated $n$'); ax.set_ylim(0,1.02); ax.legend(frameon=False,fontsize=5.8,loc='upper left')
    ax=axs[4]; labs=[]; x=0
    for d,lb in [('cox_logOU','log-OU'),('regime2','regime')]:
        g=sc[sc.design==d]
        for j,(col,c,h) in enumerate([('n_exp',CE,None),('n_soe',CS,None),('n_exp_flex',CE,'//'),('n_soe_flex',CS,'//')]):
            ax.bar(x+j*0.2,g[col].mean(),0.18,yerr=g[col].std(),color=c,hatch=h,alpha=(0.55 if h else 1),edgecolor='k',lw=0.4,capsize=2)
        labs.append((x+0.3,lb)); x+=1.1
    ax.set_xticks([p for p,_ in labs]); ax.set_xticklabels([l for _,l in labs]); ax.set_ylim(0,1.02); ax.set_ylabel('spurious $n$ (true $n=0$)')
    ax.set_title('(e) control (c): no self-excitation',loc='left')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=CE,label='Exp, constant'),Patch(color=CS,label='SoE, constant'),Patch(facecolor='white',edgecolor='k',hatch='//',label='30-min spline')],frameon=False,fontsize=5.8,loc='upper center',bbox_to_anchor=(0.5,-0.22),ncol=3,handlelength=1.2,columnspacing=0.8)
    ax=axs[5]; s95=sc[sc.design=='slow95_multistart']; a95=ar[(ar.profile=='slow')&(ar.n_true==0.95)]
    ax.hist(a95.n_exp,bins=np.linspace(0,1,21),color=CE,alpha=0.5,label='archived reps'); ax.hist(s95.n_exp,bins=np.linspace(0,1,21),histtype='step',color='k',lw=1.2,label='multistart best')
    ax.set_xlabel('exponential $\\hat n$'); ax.set_ylabel('replications'); ax.set_title('(f) slow profile, $n=0.95$',loc='left'); ax.legend(frameon=False,fontsize=6.3)
    fig.tight_layout(); fig.savefig(FIG/'R_fig_stress_controls.pdf'); plt.close(fig)
def fig_penalty():
    pen=pd.read_csv(R/'R_penalty_knot_scan.csv'); cv=pd.read_csv(R/'R_penalty_cv.csv'); nd=pen.day.nunique()
    cnt=pen.groupby(['proc','knot_min','lam']).day.nunique(); pen=pen.set_index(['proc','knot_min','lam']).loc[cnt[cnt==nd].index].reset_index()
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.2)); axs=axs.ravel(); cols=dict(zip([5.0,15.0,30.0,60.0,120.0],['#9467bd','#1f77b4','#2ca02c','#ff7f0e','#d62728']))
    X=lambda lam: np.where(np.asarray(lam)==0,1e-4,lam)
    for proc,ls,mk in [('P','-','o'),('Tr','--','s')]:
        g=pen[pen.proc==proc]
        for h,c in cols.items():
            t=g[g.knot_min==h].groupby('lam').agg(n=('n_soe','median'),sl=('slow_mass_soe','median'))
            axs[0].plot(X(t.index),t.n,ls,marker=mk,color=c,ms=3.5,lw=1.1,label=(f'{h:g} min' if proc=='P' else None)); axs[1].plot(X(t.index),t.sl,ls,marker=mk,color=c,ms=3.5,lw=1.1)
        axs[0].axhline(g.groupby('day').raw_soe.first().median(),color='k',ls=ls,lw=0.6); axs[0].axhline(g.groupby('day').raw_exp.first().median(),color='grey',ls=ls,lw=0.6)
        e=g.dropna(subset=['n_exp']).groupby(['knot_min','lam']).n_exp.median(); labs=[f'{k:g} min, $\\lambda$={l:g}' for k,l in e.index]
        axs[3].plot(range(len(e)),e.values,ls,marker=mk,color='#1f77b4' if proc=='P' else '#ff7f0e',label=('mid-price' if proc=='P' else 'trades')); axs[3].axhline(g.groupby('day').raw_exp.first().median(),color='#1f77b4' if proc=='P' else '#ff7f0e',lw=0.6,ls=':')
        ct=pd.read_csv(R/'R_cv_tight.csv'); cc=pd.read_csv(R/'R_cv_constant.csv')
        c=ct[ct.proc==proc].pivot_table(index='day',columns=['knot_min','lam'],values='cv_ll_per_event'); base=cc[cc.proc==proc].set_index('day').cv_ll_per_event
        rel=c.sub(base.loc[c.index],axis=0)
        xs=np.arange(rel.shape[1])+(-0.12 if proc=='P' else 0.12); med=rel.median(); q1=rel.quantile(.25); q3=rel.quantile(.75)
        axs[2].errorbar(xs,med.values,yerr=[med.values-q1.values,q3.values-med.values],fmt=mk,color='#1f77b4' if proc=='P' else '#ff7f0e',ms=4,capsize=2,label=('mid-price' if proc=='P' else 'trades'))
    for a in axs[:2]: a.set_xscale('log'); a.set_xlabel('normalised penalty $\\lambda$ ($\\lambda=0$ at left edge)')
    axs[0].set_ylabel('median SoE $n$'); axs[0].set_title('(a) SoE branching ratio',loc='left'); axs[0].legend(frameon=False,fontsize=6.3,title='knot spacing',title_fontsize=6.5,loc='center right')
    axs[1].set_ylabel(r'median slow mass ($\beta_k\leq1$ s$^{-1}$)'); axs[1].set_title('(b) slow kernel mass',loc='left')
    axs[2].axhline(0,color='k',lw=0.5); axs[2].set_xticks(range(rel.shape[1])); axs[2].set_xticklabels([f'{k:g},{l:g}' for k,l in rel.columns],fontsize=7); axs[2].set_xlabel('(knot spacing min, $\\lambda$)')
    axs[2].set_ylabel('held-out $\\ell$/event vs constant'); axs[2].set_title('(c) blocked cross-validation',loc='left'); axs[2].legend(frameon=False,fontsize=6.5)
    axs[3].set_xticks(range(len(labs))); axs[3].set_xticklabels([l.replace(', ','\n') for l in labs],fontsize=6.5); axs[3].set_ylabel('median exponential $n$'); axs[3].set_ylim(0,1); axs[3].set_title('(d) exponential kernel',loc='left'); axs[3].legend(frameon=False,fontsize=6.5)
    fig.tight_layout(); fig.savefig(FIG/'R_fig_penalty_knot.pdf'); plt.close(fig)


def fig_robustness():
    ts=pd.read_csv(R/'R_timestamp_fullsession.csv'); ds=pd.read_csv(ROOT/'tables/E1_daily_summary.csv'); fb=pd.read_csv(ROOT/'tables_csf/A_flexible_background.csv'); ss=pd.read_csv(ROOT/'tables_csf/A_soe_sensitivity.csv')
    b=pd.read_csv(R/'R_bootstrap_diagnostics.csv'); fx=R/'R_bootstrap_soe_refit.csv'
    if fx.exists():
        f=pd.read_csv(fx); soe=b[b.model=='SoE'].drop(columns=['ks','cvm','acf1','acf5','acf10']).merge(f[['day','proc','rep','ks','cvm','acf1','acf5','acf10']],on=['day','proc','rep']); b=pd.concat([b[b.model=='Exp'],soe])
    def mci(x,B=2000):
        x=np.asarray(x,float); bs=np.array([np.median(rng.choice(x,len(x))) for _ in range(B)]); return np.median(x),np.percentile(bs,2.5),np.percentile(bs,97.5)
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.0)); axs=axs.ravel(); protos=['raw','merge_1ms','merge_10ms','merge_0.1s','randomise_1s']
    for proc,c,off in [('P','#1f77b4',-0.1),('Tr','#ff7f0e',0.1)]:
        for col,mk,o2 in [('n_exp','o',-0.04),('n_soe','s',0.04)]:
            pts=[]
            for pr in protos:
                g=ts[(ts.proc==proc)&(ts.protocol==pr)]; v=g.groupby('day')[col].median() if pr=='randomise_1s' else g.set_index('day')[col]; pts.append(mci(v))
            pts=np.array(pts); xs=np.arange(5)+off+o2
            axs[0].errorbar(xs,pts[:,0],yerr=[pts[:,0]-pts[:,1],pts[:,2]-pts[:,0]],fmt=mk+('-' if col=='n_exp' else '--'),color=c,ms=4,capsize=2,lw=1,mfc=(c if col=='n_exp' else 'white'),label=f'{"mid-price" if proc=="P" else "trades"}, {"Exp" if col=="n_exp" else "SoE"}')
    axs[0].set_xticks(range(5)); axs[0].set_xticklabels(['raw','merge\n1 ms','merge\n10 ms','merge\n0.1 s','randomise\n1 s'],fontsize=7); axs[0].set_ylabel('median full-session $n$'); axs[0].set_title('(a) timestamp protocols',loc='left'); axs[0].legend(frameon=False,fontsize=6,ncol=2,loc='lower right'); axs[0].set_ylim(0.3,1.0)
    cats=[('P','exp'),('P','soe'),('Tr','exp'),('Tr','soe')]
    for j,(proc,m) in enumerate(cats):
        const=ds[ds.proc==proc]['n_mle' if m=='exp' else 'n_soe']; flex=fb[(fb.proc==proc)&(fb.model==m)].n
        for k,(v,mk,lab) in enumerate([(const,'o','constant'),(flex,'s','30-min spline')]):
            q=mci(v); axs[1].errorbar(j+(-0.1 if k==0 else 0.1),q[0],yerr=[[q[0]-q[1]],[q[2]-q[0]]],fmt=mk,color=('k' if k==0 else '#2ca02c'),ms=4,capsize=2,label=(lab if j==0 else None))
    axs[1].set_xticks(range(4)); axs[1].set_xticklabels(['mid-price\nExp','mid-price\nSoE','trades\nExp','trades\nSoE'],fontsize=7); axs[1].set_ylim(0,1); axs[1].set_ylabel('median full-session $n$'); axs[1].set_title('(b) background',loc='left'); axs[1].legend(frameon=False,fontsize=6.5)
    grids=['K5_1e-2_1e4','K7_1e-2_1e4','K9_1e-2_1e4','K11_1e-2_1e4','K7_1e-2_1e5','K7_1e-3_1e4','K7_1e-3_1e5']; gl=['K=5','K=7','K=9','K=11','K=7\nfast+','K=7\nslow+','K=7\nboth+']
    for proc,c in [('P','#1f77b4'),('Tr','#ff7f0e')]:
        v=[ss[(ss.grid==g)&(ss.proc==proc)].n.median() for g in grids]; axs[2].plot(range(len(grids)),v,'-o',color=c,ms=4,label=('mid-price' if proc=='P' else 'trades'))
    axs[2].set_xticks(range(len(grids))); axs[2].set_xticklabels(gl,fontsize=6.5); axs[2].set_ylim(0.9,1.0); axs[2].set_ylabel('median SoE $n$'); axs[2].set_title('(c) SoE basis',loc='left'); axs[2].legend(frameon=False,fontsize=6.5)
    ax=axs[3]; ax2=ax.twinx(); ax2.spines['right'].set_visible(True)
    for j,(proc,m) in enumerate([('P','Exp'),('P','SoE'),('Tr','Exp'),('Tr','SoE')]):
        g=b[(b.proc==proc)&(b.model==m)]; per=g.groupby('day').agg(o=('obs_ks','first'),q=('ks',lambda x:x.quantile(.975)),a=('obs_acf1','first'),lo=('acf1',lambda x:x.quantile(.025)),hi=('acf1',lambda x:x.quantile(.975)))
        ax.bar(j-0.18,per.o.median(),0.34,color='#4c72b0',label=('KS (left)' if j==0 else None)); ax.hlines(per.q.median(),j-0.36,j,color='k',lw=2,label=('KS null 97.5% (left)' if j==0 else None))
        ax2.bar(j+0.18,per.a.median(),0.34,color='#a6c8e0',label=('ACF(1) (right)' if j==0 else None)); ax2.vlines(j+0.18,per.lo.median(),per.hi.median(),color='k',lw=3)
    ax.set_xticks(range(4)); ax.set_xticklabels(['mid-price\nExp','mid-price\nSoE','trades\nExp','trades\nSoE'],fontsize=7); ax.set_ylabel('median KS distance'); ax2.set_ylabel('median residual ACF(1)'); ax2.axhline(0,color='k',lw=0.5)
    ax.set_title('(d) residual effect sizes vs bootstrap null',loc='left'); h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels(); ax.legend(h1+h2,l1+l2,frameon=False,fontsize=6,loc='upper left')
    fig.tight_layout(); fig.savefig(FIG/'R_fig_robustness.pdf'); plt.close(fig)


def fig_moderes():
    import json; M=json.load(open(R/'R_moderes.json')); tk=json.load(open(R/'R_tick_np.json')); bv=pd.read_csv(R/'R_bivariate.csv')
    WS=[0.001,0.01,0.1,1,5,10,30,60,120,300,600,1200]; fig,axs=plt.subplots(1,2,figsize=(7.4,3.0))
    def ser(key,f): a=np.array([M['table'][f'{key}_{W}'][f] for W in WS]); return a[:,0],a[:,1],a[:,2]
    ax=axs[0]
    for key,c,lab in [('mid',CE,'mid-price'),('trd',CS,'trades')]:
        m,lo,hi=ser(key,'nplus'); ax.plot(WS,m,'-o',color=c,ms=3,label=f'{lab}, raw'); ax.fill_between(WS,lo,hi,color=c,alpha=0.15,lw=0)
        m,lo,hi=ser(key,'nplus_des'); ax.plot(WS,m,'--s',color=c,ms=3,mfc='white',label=f'{lab}, deseasonalised')
    for y,c,ls,lab in [(0.577,CE,':','exp. 0.577 / SoE 0.936 (mid-price)'),(0.936,CE,':',None),(0.478,CS,':','exp. 0.478 / SoE 0.950 (trades)'),(0.950,CS,':',None)]:
        ax.axhline(y,color=c,ls=ls,lw=0.8,label=lab)
    ax.plot([100],[np.array(tk['mid_100']['norms'])[0,0]],'^',color=CE,ms=6,mec='k',label='nonparametric, 100-s support'); ax.plot([100],[np.array(tk['trades_100']['norms'])[0,0]],'^',color=CS,ms=6,mec='k')
    ax.set_xscale('log'); ax.set_xlabel('bin width $W$ (s)'); ax.set_ylabel(r'$\hat n_+(W)=1-\sqrt{\mathrm{E}N_W/\mathrm{Var}N_W}$'); ax.set_ylim(0,1); ax.set_title('(a) activity mode',loc='left'); ax.legend(frameon=False,fontsize=5.8,loc='lower right')
    ax=axs[1]
    for key,c,lab in [('mid',CE,'mid-price (up $-$ down)'),('trd',CS,'trades (buy $-$ sell)')]:
        m,lo,hi=ser(key,'nminus'); ax.plot(WS,m,'-o',color=c,ms=3,label=lab); ax.fill_between(WS,lo,hi,color=c,alpha=0.15,lw=0)
    ax.axhline(bv.nminus_soe.median(),color=CE,ls='--',lw=0.9,label='bivariate SoE $n_-$ (mid-price)'); G=np.array(tk['updown_10']['norms']); ax.plot([10],[(G[0,0]+G[1,1])/2-(G[0,1]+G[1,0])/2],'^',color=CE,ms=6,mec='k',label='nonparametric up/down, 10-s support')
    G=np.array(tk['buysell_10']['norms']); ax.plot([10],[(G[0,0]+G[1,1])/2-(G[0,1]+G[1,0])/2],'^',color=CS,ms=6,mec='k',label='nonparametric buy/sell, 10-s support')
    ax.axhline(0,color='k',lw=0.5); ax.set_xscale('log'); ax.set_xlabel('bin width $W$ (s)'); ax.set_ylabel(r'$\hat n_-(W)=1-\sqrt{\mathrm{E}N_W/\mathrm{Var}(N^+_W-N^-_W)}$'); ax.set_ylim(-1.2,1); ax.set_title('(b) signed mode',loc='left'); ax.legend(frameon=False,fontsize=5.8,loc='lower left')
    fig.tight_layout(); fig.savefig(FIG/'R_fig_moderes.pdf'); plt.close(fig)


def fig_scatter():
    lab=pd.read_csv(R/'R_defensible_labels.csv'); fig,axs=plt.subplots(1,2,figsize=(7.4,3.0),sharey=True)
    cols={'kernel':CE,'background':CS,'window':CR}; mk={'kernel':'o','background':'s','window':'D'}
    for ax,proc,ttl in [(axs[0],'P','(a) mid-price changes'),(axs[1],'Tr','(b) trades')]:
        d=lab[(lab.stream==proc)&lab.gap.notna()].copy(); d['gp']=np.maximum(d.gap,d.kernel_gap.fillna(0))
        ax.axvspan(-0.10,0.005,color='grey',alpha=0.15,lw=0)
        for g,c in cols.items():
            dd=d[d.group==g]; ax.scatter(-dd.gp.clip(upper=2.0),dd.n_median,s=22,color=c,marker=mk[g],edgecolor='k',linewidths=0.3,label=g+(' (kernel, forward split)' if g=='kernel' else ' (blocked CV)' if g=='background' else ' (one-window-ahead)'))
        dd=d[(d.group=='background')&(d.gp<=0.1)&d.kernel.isin(['K7_1e-2_1e4','K11_1e-2_1e4'])]; ax.annotate('multiscale backgrounds:\nheld-out flat, $n$ spans %.2f-%.2f'%(dd.n_median.min(),dd.n_median.max()),xy=(-0.01,dd.n_median.min()),xytext=(-1.25,0.5),fontsize=6.5,arrowprops=dict(arrowstyle='->',lw=0.6))
        ax.set_xlim(-2.05,0.05); ax.set_xlabel(r'$\Delta\ell$ to the best of its comparison (nats/event)'); ax.set_title(ttl,loc='left'); ax.set_ylim(0.3,1.0)
    axs[0].set_ylabel('median $n$'); axs[0].legend(frameon=False,fontsize=6,loc='lower left')
    fig.tight_layout(); fig.savefig(FIG/'R_fig_scatter.pdf'); plt.close(fig)


def fig_modes():
    bv=pd.read_csv(ROOT/'tables_csf/A_bivariate_modes.csv').sort_values('day').reset_index(drop=True); h=pd.read_csv(ROOT/'tables/E1_h3prime.csv')
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.8)); x=np.arange(1,len(bv)+1)
    axs[0].plot(x,bv.n_plus,'o-',color=CE,ms=3.5,label='activity $n_+=n_s+n_c$'); axs[0].plot(x,bv.n_minus,'s-',color=CS,ms=3.5,label='signed $n_-=n_s-n_c$')
    axs[0].axhline(bv.n_plus.median(),color=CE,ls='--',lw=0.7); axs[0].axhline(bv.n_minus.median(),color=CS,ls='--',lw=0.7); axs[0].set_xticks(x[::2]); axs[0].set_xlabel('trading session (1 = 1 April 2014)'); axs[0].set_ylabel('mode branching ratio'); axs[0].set_ylim(0,0.7); axs[0].set_title('(a) fitted bivariate eigenmodes',loc='left'); axs[0].legend(frameon=False,fontsize=6.5)
    for proc,c,lab in [('P',CE,'price-change activity'),('Tr',CS,'trade activity')]:
        s=h[h.proc==proc]; axs[1].plot(lags,[s[f'rho{k}_emp'].mean() for k in lags],'-o',color=c,ms=3,label=lab)
    s=h[h.proc=='P']; axs[1].plot(lags,[s[f'rho{k}_ret'].mean() for k in lags],'-^',color=CR,ms=3,label='signed price changes'); axs[1].axhline(0,color='k',lw=0.5)
    axs[1].set_xlabel('lag (10-s bins)'); axs[1].set_ylabel('mean autocorrelation'); axs[1].set_title('(b) activity persists; signed mode reverses',loc='left'); axs[1].legend(frameon=False,fontsize=6.5)
    fig.tight_layout(); fig.savefig(FIG/'R_fig_modes.pdf'); plt.close(fig)


if __name__=='__main__':
    which=sys.argv[1:] or ['csd','bivariate','stress']
    for w in which: {'csd':fig_csd,'bivariate':fig_bivariate,'stress':fig_stress,'penalty':fig_penalty,'robustness':fig_robustness,'moderes':fig_moderes,'scatter':fig_scatter,'modes':fig_modes}[w](); print('fig',w,'ok')
