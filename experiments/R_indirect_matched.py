"""ACF-matched exogenous benchmark: weaken the calibrated background until the 10-s count ACF(1) equals the data's, then compare
slow mass (constant-background SoE); plus the bivariate slow cross-minus-self bound. Seed 20260914."""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'experiments'))
import R_indirect as II, R_revision as RR
R = ROOT / 'tables_rev'; cal = json.load(open(R / 'R_ii_calibration.json')); fits = pd.read_csv(R / 'R_ii_fits.csv'); rng = np.random.default_rng(II.SEED); B = 4000
def run(proc, kind, par, off):
    out = []
    for i in range(20):
        t, T = II.sim_uni(i, proc, kind, par, II.SEED + 4001 * i + off)
        f = RR.fit_soe_masses(t, np.empty(0), T, II.BET, extra_starts=False); a = II.acf_cum(t, T)
        out.append((float((f['alpha'] / II.BET)[II.BET <= 1].sum()), f['n'], a['acf1']))
    return np.array(out)
res = {}; rows = []
for proc in ['P', 'Tr']:
    d = fits[(fits.proc == proc) & (fits.src == 'data')].set_index('day').sort_index(); acf_d = d.acf1.mean()
    for kind in ['logou', 'regime']:
        p0, p1 = cal[f'{kind}_{proc}']['par']
        scales = np.linspace(0.15, 1.0, 8) if kind == 'logou' else np.linspace(0.0, 1.0, 8)
        curve = []
        for k, sc in enumerate(scales):
            par = (p0, max(p1 * sc, 0.02)) if kind == 'logou' else (1.0 + (p0 - 1.0) * sc + 1e-6, p1)
            r = run(proc, kind, par, 17 * k + (0 if kind == 'logou' else 9)); curve.append((float(sc), float(np.median(r[:, 0])), float(np.median(r[:, 1])), float(r[:, 2].mean())))
            rows.append(dict(proc=proc, kind=kind, scale=sc, par0=par[0], par1=par[1], slow=np.median(r[:, 0]), n=np.median(r[:, 1]), acf1=r[:, 2].mean()))
        cv = np.array(curve); sc_star = float(np.interp(acf_d, cv[:, 3], cv[:, 0])) if cv[:, 3].min() <= acf_d <= cv[:, 3].max() else np.nan
        if np.isfinite(sc_star):
            par = (p0, p1 * sc_star) if kind == 'logou' else (1.0 + (p0 - 1.0) * sc_star, p1)
            r = run(proc, kind, par, 777 + (0 if kind == 'logou' else 9)); idx = rng.integers(0, 20, (B, 20))
            dd = np.median(d.slow_soe.values[idx], 1) - np.median(r[:, 0][idx], 1)
            res[f'{proc}_{kind}'] = dict(scale=sc_star, par=[float(par[0]), float(par[1])], acf_sim=float(r[:, 2].mean()), acf_data=float(acf_d), slow_sim=float(np.median(r[:, 0])),
                                         slow_data=float(d.slow_soe.median()), diff=[float(d.slow_soe.median() - np.median(r[:, 0])), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))], n_sim=float(np.median(r[:, 1])))
        else:
            res[f'{proc}_{kind}'] = dict(scale=None, curve=curve)
        print(proc, kind, res[f'{proc}_{kind}'], flush=True)
# bivariate: slow cross-minus-self mass, data vs exogenous benchmark (paired session bootstrap)
bv = pd.read_csv(R / 'R_bivariate.csv').sort_values('day'); dcs = (bv[[f'mc{k}' for k in range(3)]].sum(1) - bv[[f'ms{k}' for k in range(3)]].sum(1)).values
for kind in ['logou', 'regime']:
    s = fits[(fits.proc == 'UD') & (fits.src == kind)].sort_values('day'); scs = (s.slow_cross - s.slow_self).values; idx = rng.integers(0, 20, (B, 20))
    dd = np.median(dcs[idx], 1) - np.median(scs[idx], 1); nm = np.median(bv.nminus_soe.values[idx], 1) - np.median(s.nminus.values[idx], 1)
    res[f'UD_{kind}'] = dict(cs_data=float(np.median(dcs)), cs_sim=float(np.median(scs)), diff=[float(np.median(dcs) - np.median(scs)), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
                             nminus_diff=[float(bv.nminus_soe.median() - s.nminus.median()), float(np.percentile(nm, 2.5)), float(np.percentile(nm, 97.5))])
    print('UD', kind, res[f'UD_{kind}'], flush=True)
pd.DataFrame(rows).to_csv(R / 'R_ii_matched_curve.csv', index=False); json.dump(res, open(R / 'R_ii_matched.json', 'w'), indent=1)
