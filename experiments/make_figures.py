from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures_csf'; OUT.mkdir(exist_ok=True)
T=ROOT/'tables_csf'

def save(fig,name):
    fig.tight_layout(); fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight'); fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight'); plt.close(fig)

def a_robustness():
    ts=pd.read_csv(T/'A_timestamp_perturbations.csv')
    flex=pd.read_csv(T/'A_flexible_background.csv')
    soe=pd.read_csv(T/'A_soe_sensitivity.csv')
    res=pd.read_csv(T/'A_residual_effects.csv')
    fig,ax=plt.subplots(2,2,figsize=(10.5,7.2))
    # timestamp
    order=['raw','merge_0.001s','merge_0.01s','merge_0.1s','randomise_1s']; labels=['raw','merge 1 ms','merge 10 ms','merge 0.1 s','randomise 1 s']
    for j,proc in enumerate(['P','Tr']):
        med=[]; lo=[]; hi=[]
        for p in order:
            v=ts[(ts.proc==proc)&(ts.protocol==p)].n_exp.dropna().to_numpy(); med.append(np.median(v)); lo.append(np.quantile(v,.25)); hi.append(np.quantile(v,.75))
        x=np.arange(len(order))+(-.08 if proc=='P' else .08)
        ax[0,0].errorbar(x,med,yerr=[np.array(med)-lo,np.array(hi)-med],fmt='o-',capsize=3,label='mid-price' if proc=='P' else 'trades')
    ax[0,0].set_xticks(np.arange(len(order)),labels,rotation=20,ha='right'); ax[0,0].set_ylabel('exponential branching ratio'); ax[0,0].set_title('(a) timestamp protocol'); ax[0,0].legend(frameon=False)
    # flexible baseline
    x=np.arange(4); cats=[('P','exp'),('P','soe'),('Tr','exp'),('Tr','soe')]; labs=['price Exp','price SoE','trades Exp','trades SoE']
    raw=[]; fl=[]
    for proc,m in cats:
        g=flex[(flex.proc==proc)&(flex.model==m)]; raw.append(g.raw_n.median()); fl.append(g.n.median())
    ax[0,1].plot(x,raw,'o--',label='constant baseline'); ax[0,1].plot(x,fl,'s-',label='30-min B-spline baseline')
    ax[0,1].set_xticks(x,labs,rotation=20,ha='right'); ax[0,1].set_ylim(0,1); ax[0,1].set_ylabel('median branching ratio'); ax[0,1].set_title('(b) flexible background'); ax[0,1].legend(frameon=False)
    # SoE grid
    grid_order=['K5_1e-2_1e4','K7_1e-2_1e4','K9_1e-2_1e4','K11_1e-2_1e4','K7_1e-2_1e5','K7_1e-3_1e4','K7_1e-3_1e5']
    grid_labels=['K=5','K=7','K=9','K=11','K=7, fast+','K=7, slow+','K=7, both+']
    for proc in ['P','Tr']:
        med=[soe[(soe.grid==g)&(soe.proc==proc)].n.median() for g in grid_order]
        ax[1,0].plot(np.arange(len(grid_order)),med,'o-',label='mid-price' if proc=='P' else 'trades')
    ax[1,0].set_xticks(np.arange(len(grid_order)),grid_labels,rotation=20,ha='right'); ax[1,0].set_ylim(.78,1.0); ax[1,0].set_ylabel('median SoE branching ratio'); ax[1,0].set_title('(c) SoE basis sensitivity'); ax[1,0].legend(frameon=False)
    # residual effects
    names=[]; ks=[]; ac=[]
    for proc,m in [('P','Exp'),('P','SoE'),('Tr','Exp'),('Tr','SoE')]:
        g=res[(res.proc==proc)&(res.model==m)]; names.append(('price ' if proc=='P' else 'trades ')+m); ks.append(g.ks.median()); ac.append(g.resid_acf1.median())
    xx=np.arange(4); ax2=ax[1,1].twinx(); ax[1,1].bar(xx-.16,ks,width=.32,label='KS distance'); ax2.bar(xx+.16,ac,width=.32,alpha=.45,label='residual ACF(1)')
    ax[1,1].set_xticks(xx,names,rotation=20,ha='right'); ax[1,1].set_ylabel('median KS distance'); ax2.set_ylabel('median residual ACF(1)'); ax[1,1].set_title('(d) residual effect sizes')
    handles=[Line2D([0],[0],lw=7,label='KS distance'),Line2D([0],[0],lw=7,alpha=.45,label='residual ACF(1)')]; ax[1,1].legend(handles=handles,frameon=False,loc='upper left')
    save(fig,'A_fig2_robustness')

def a_stress():
    d=pd.read_csv(T/'A_multiscale_stress.csv'); fig,axs=plt.subplots(1,3,figsize=(11,3.5),sharex=True,sharey=True)
    for ax,prof in zip(axs,['fast','balanced','slow']):
        g=d[d.profile==prof].groupby('n_true')[['n_exp','n_soe','n_exp_1s']].agg(['mean','std'])
        x=g.index.to_numpy(float)
        for col,lab,mark in [('n_exp','single Exp','o'),('n_soe','correct SoE','s'),('n_exp_1s','Exp after 1-s randomisation','^')]:
            ax.errorbar(x,g[(col,'mean')],yerr=g[(col,'std')],marker=mark,capsize=3,label=lab)
        ax.plot([.45,1],[.45,1],'k--',lw=1); ax.set_title(prof+' kernel-mass profile'); ax.set_xlabel('true branching ratio')
    axs[0].set_ylabel('estimated branching ratio'); axs[0].legend(frameon=False,fontsize=8); save(fig,'A_fig3_multiscale_stress')

def a_modes():
    b=pd.read_csv(T/'A_bivariate_modes.csv'); h=pd.read_csv(ROOT/'tables'/'E1_h3prime.csv')
    fig,ax=plt.subplots(1,2,figsize=(10.5,3.8))
    xx=np.arange(len(b)); ax[0].plot(xx,b.n_plus,'o-',label=r'activity $n_+=n_s+n_c$'); ax[0].plot(xx,b.n_minus,'s-',label=r'signed $n_-=n_s-n_c$'); ax[0].axhline(b.n_plus.median(),ls='--',lw=1); ax[0].axhline(b.n_minus.median(),ls='--',lw=1); ax[0].set_xlabel('trading session'); ax[0].set_ylabel('mode branching ratio'); ax[0].set_title('(a) fitted bivariate eigenmodes'); ax[0].legend(frameon=False)
    # mean ACF by lag for activity and signed price changes
    hp=h[h.proc=='P']; ht=h[h.proc=='Tr']; lags=np.arange(1,7)
    price=[hp[f'rho{k}_emp'].mean() for k in lags]; trades=[ht[f'rho{k}_emp'].mean() for k in lags]; signed=[hp[f'rho{k}_ret'].mean() for k in lags]
    ax[1].plot(lags,price,'o-',label='price-change activity'); ax[1].plot(lags,trades,'s-',label='trade activity'); ax[1].plot(lags,signed,'^-',label='signed price changes'); ax[1].axhline(0,color='k',lw=.7); ax[1].set_xlabel('lag (10-s bins)'); ax[1].set_ylabel('mean autocorrelation'); ax[1].set_title('(b) activity persists; signed mode reverses'); ax[1].legend(frameon=False)
    save(fig,'A_fig4_modes')

if __name__=='__main__':
    a_robustness(); a_stress(); a_modes(); print('A figures regenerated')
