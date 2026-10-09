"""Residual diagnostics for the bivariate separate-rate and bivariate SoE fits (Table 6): time-rescaling in each dimension,
KS and CvM on the pooled residuals, ACF averaged over the two dimensions; session-bootstrap SE of the median n+/-."""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'experiments'))
import R_revision as RR
OUT = ROOT / 'tables_rev'; bund = RR.load_bund(); bv = pd.read_csv(OUT / 'R_bivariate.csv'); rows = []
for day, D in bund.items():
    T = RR.session_T(D); up = D['up'][D['up'] > 0]; dn = D['dn'][D['dn'] > 0]; b = bv[bv.day == int(day)].iloc[0]
    for model, bs, bc in [('bivar_sep', [b.beta_s], [b.beta_c]), ('bivar_soe', RR.BETAS7, RR.BETAS7)]:
        f = RR.fit_bivar_soe(up, dn, T, bs, bc)
        times = np.r_[up, dn]; typ = np.r_[np.ones(len(up), np.int64), -np.ones(len(dn), np.int64)]; o = np.argsort(times)
        times = np.ascontiguousarray(times[o]); typ = np.ascontiguousarray(typ[o])
        Rp_s, Rm_s = RR._bivar_R(times, typ, f['betas_s']); Rp_c, Rm_c = RR._bivar_R(times, typ, f['betas_c'])
        cu = (np.cumsum(typ == 1) - (typ == 1)).astype(float); cd = (np.cumsum(typ == -1) - (typ == -1)).astype(float)
        ws = f['a'] / f['betas_s']; wc = f['c'] / f['betas_c']
        Lp = f['mu'] * times + ((cu[:, None] - Rp_s) * ws).sum(1) + ((cd[:, None] - Rm_c) * wc).sum(1)
        Lm = f['mu'] * times + ((cd[:, None] - Rm_s) * ws).sum(1) + ((cu[:, None] - Rp_c) * wc).sum(1)
        res = []
        for sgn, L in [(1, Lp), (-1, Lm)]:
            d = np.diff(np.r_[0.0, L[typ == sgn]]); res.append(d[d > 0])
        k, c, _ = RR.diag_stats(np.r_[res[0], res[1]]); a = np.mean([RR.diag_stats(r)[2] for r in res], axis=0)
        rows.append(dict(day=day, model=model, nplus=f['n_s'] + f['n_c'], nminus=f['n_s'] - f['n_c'], ks=k, cvm=c, acf1=a[0], acf5=a[1], acf10=a[2], mean_resid=float(np.mean(np.r_[res[0], res[1]]))))
df = pd.DataFrame(rows); df.to_csv(OUT / 'R_bivar_resid.csv', index=False)
rng = np.random.default_rng(20260914); out = {}
for m, g in df.groupby('model'):
    idx = rng.integers(0, len(g), (4000, len(g))); se = {k: float(np.median(g[k].values[idx], 1).std(ddof=1)) for k in ['nplus', 'nminus']}
    out[m] = dict(**{k: float(g[k].median()) for k in ['ks', 'cvm', 'acf1', 'acf10', 'nplus', 'nminus', 'mean_resid']}, se_nplus=se['nplus'], se_nminus=se['nminus'])
    print(m, {k: round(v, 4) for k, v in out[m].items()})
chk = bv.set_index(bv.day.astype(str)); print('refit check |n+ sep - stored| max %.2e, |n+ soe - stored| max %.2e' % (
    np.max(np.abs(df[df.model == 'bivar_sep'].set_index('day').nplus - chk.nplus_sep)), np.max(np.abs(df[df.model == 'bivar_soe'].set_index('day').nplus - chk.nplus_soe))))
json.dump(out, open(OUT / 'R_bivar_resid.json', 'w'), indent=1)
