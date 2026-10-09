"""Summary of the indirect-inference benchmark: data vs most-conservative-exogenous DGP (log-OU, regime), paired session bootstrap."""
import json, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / 'tables_rev'; rng = np.random.default_rng(20260914); B = 4000
f = pd.read_csv(R / 'R_ii_fits.csv'); out = {}; L = []; P = L.append
def paired(a, b):
    """median(a) - median(b) over sessions, paired session bootstrap."""
    a = np.asarray(a, float); b = np.asarray(b, float); idx = rng.integers(0, len(a), (B, len(a)))
    d = np.median(a[idx], 1) - np.median(b[idx], 1); return (float(np.median(a) - np.median(b)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
est = ['soe'] + [f'sp_{h}_{l}' for h, l in [('5', '0'), ('15', '0'), ('30', '0'), ('60', '0'), ('120', '0'), ('30', '0.1')]]
for proc in ['P', 'Tr']:
    d = f[(f.proc == proc) & (f.src == 'data')].set_index('day').sort_index()
    for kind in ['logou', 'regime']:
        s = f[(f.proc == proc) & (f.src == kind)].set_index('day').loc[d.index]
        P(f'== {proc} vs {kind} (20 sessions) ==')
        for e in est:
            sl, n = f'slow_{e}', f'n_{e}'
            r = paired(d[sl], s[sl]); out[f'{proc}_{kind}_{e}'] = dict(slow_data=float(d[sl].median()), slow_sim=float(s[sl].median()), diff=r, n_data=float(d[n].median()), n_sim=float(s[n].median()),
                                                                       share=float(min(1.0, s[sl].median() / d[sl].median())))
            P(f'  {e:10s} slow mass data {d[sl].median():.3f} sim {s[sl].median():.3f} | data-sim {r[0]:+.3f} [{r[1]:+.3f}, {r[2]:+.3f}] | n data {d[n].median():.3f} sim {s[n].median():.3f}')
        r = paired(d.acf1, s.acf1); out[f'{proc}_{kind}_acf'] = dict(data=[float(d[f'acf{k}'].mean()) for k in range(1, 7)], sim=[float(s[f'acf{k}'].mean()) for k in range(1, 7)], diff1=r)
        P(f'  10-s ACF lags 1-6 data {np.round([d[f"acf{k}"].mean() for k in range(1,7)],3).tolist()} sim {np.round([s[f"acf{k}"].mean() for k in range(1,7)],3).tolist()} | lag-1 data-sim {r[0]:+.3f} [{r[1]:+.3f}, {r[2]:+.3f}]')
    for dl in (1, 10, 60):
        vals = {src: f[(f.proc == proc) & (f.src == src)][f'r3_{dl}'] for src in ['data', 'logou', 'regime']}; vals['soe_boot'] = f[(f.proc == proc) & (f.src == 'data')][f'boot_r3_{dl}']
        out[f'{proc}_r3_{dl}'] = {k: [float(v.median()), float(v.quantile(.25)), float(v.quantile(.75))] for k, v in vals.items()}
        P(f'  third-cumulant ratio k3*m/k2^2, {dl:>2d}-s counts: ' + '; '.join(f'{k} {v.median():.2f} [{v.quantile(.25):.2f},{v.quantile(.75):.2f}]' for k, v in vals.items()))
    bacf = f[(f.proc == proc) & (f.src == 'data')]; P(f'  SoE-bootstrap 10-s ACF lag 1 {bacf.boot_acf1.mean():.3f} (data {bacf.acf1.mean():.3f})')
ud = f[f.proc == 'UD']; bv = pd.read_csv(R / 'R_bivariate.csv'); slow = [f'ms{k}' for k in range(3)]; slowc = [f'mc{k}' for k in range(3)]
P('== bivariate (up/down) ==')
P(f"  data: n+ {bv.nplus_soe.median():.3f} n- {bv.nminus_soe.median():.3f} slow self {bv[slow].sum(1).median():.3f} slow cross {bv[slowc].sum(1).median():.3f} signed ACF1 {ud[ud.src=='data'].signed_acf1.mean():.3f}")
for kind in ['logou', 'regime']:
    s = ud[ud.src == kind]; out[f'UD_{kind}'] = {k: float(s[k].median()) for k in ['nplus', 'nminus', 'n_s', 'n_c', 'slow_self', 'slow_cross']}; out[f'UD_{kind}']['signed_acf1'] = float(s.signed_acf1.mean())
    P(f"  {kind}: n+ {s.nplus.median():.3f} n- {s.nminus.median():.3f} slow self {s.slow_self.median():.3f} slow cross {s.slow_cross.median():.3f} signed ACF1 {s.signed_acf1.mean():.3f}")
out['UD_data'] = dict(nplus=float(bv.nplus_soe.median()), nminus=float(bv.nminus_soe.median()), slow_self=float(bv[slow].sum(1).median()), slow_cross=float(bv[slowc].sum(1).median()), signed_acf1=float(ud[ud.src == 'data'].signed_acf1.mean()))
json.dump(out, open(R / 'R_ii_summary.json', 'w'), indent=1); open(R / 'R_ii_summary.txt', 'w').write('\n'.join(L)); print('\n'.join(L))
