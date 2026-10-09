"""T4: (1) relaxation-time admissibility of windows, W >= c/gamma_min (c=5 main, 10 sensitivity), for the 7- and 11-rate grids;
(2) K7 vs K11 held-out difference decomposed by preceding inter-event interval, and the comparison repeated on 1-ms-merged data."""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'experiments'))
import R_revision as RR
from hawkes import fit_soe_mle, soe_slowest_rate, _soe_R
OUT = ROOT / 'tables_rev'; bund = RR.load_bund(); K7 = RR.BETAS7; K11 = RR.SOE_GRIDS['K11_1e-2_1e4']; out = {}
# ---- (1) admissibility
wf = pd.read_csv(ROOT / 'tables/E1_window_fits.csv'); rows = []
for day, D in (bund.items() if not (OUT / 'R_t4_k11_windows.csv').exists() else []):
    T = RR.session_T(D)
    for proc in ['P', 'Tr']:
        t_all = D[proc]
        for W, wn in [(600.0, '10min'), (1800.0, '30min'), (3600.0, '1h')]:
            for s0 in np.arange(0.0, T - W + 1e-9, W):
                x = t_all[(t_all >= s0) & (t_all < s0 + W)] - s0; h = t_all[(t_all >= s0 - 1800) & (t_all < s0)] - s0
                if len(x) < 30: continue
                f = fit_soe_mle(x, h, W, betas=K11); g = soe_slowest_rate(f['alpha'], K11)
                rows.append(dict(day=day, proc=proc, window=wn, W=W, start=s0, n11=f['n'], gamma11=g))
    print('adm', day, flush=True)
k11w = pd.DataFrame(rows) if rows else pd.read_csv(OUT / 'R_t4_k11_windows.csv')
if rows: k11w.to_csv(OUT / 'R_t4_k11_windows.csv', index=False)
adm = {}
for grid, df, ncol, gcol in [('K7', wf.rename(columns={'n_soe': 'n', 'gamma_min_soe': 'g'}), 'n', 'g'), ('K11', k11w.rename(columns={'n11': 'n', 'gamma11': 'g'}), 'n', 'g')]:
    for proc in ['P', 'Tr']:
        for wn in ['10min', '30min', '1h']:
            d = df[(df.proc == proc) & (df.window == wn)].copy(); d['tau'] = 1.0 / d.g.clip(lower=1e-9)
            for c in (5, 10):
                ok = d.W >= c * d.tau; sess = d[ok].groupby('day').n.median(); frac = ok.mean()
                adm[f'{grid}_{proc}_{wn}_c{c}'] = dict(frac=float(frac), n_adm=(float(sess.median()) if len(sess) else None), n_all=float(d.groupby('day').n.median().median()), tau_med=float(d.tau.median()), sessions=int(len(sess)))
            print(grid, proc, wn, 'median tau %.0f s; admissible fraction c=5 %.2f, c=10 %.2f; n(adm, c=5) %s vs all %.3f' % (d.tau.median(), adm[f'{grid}_{proc}_{wn}_c5']['frac'], adm[f'{grid}_{proc}_{wn}_c10']['frac'], adm[f'{grid}_{proc}_{wn}_c5']['n_adm'], adm[f'{grid}_{proc}_{wn}_c5']['n_all']), flush=True)
out['admissibility'] = adm
# ---- (2) K7 vs K11: decomposition of the held-out difference and merged-data comparison
dec = []
for day, D in bund.items():
    T = RR.session_T(D)
    for proc in ['P', 'Tr']:
        for variant in ['raw', 'merge1ms']:
            t = D[proc][D[proc] > 0]
            if variant == 'merge1ms': t = RR.greedy_merge(t, 1e-3)
            Ttr = 0.7 * T; ttr = t[t < Ttr]; tte = t[t >= Ttr] - Ttr; hte = ttr - Ttr; Tte = T - Ttr
            rec = dict(day=day, proc=proc, variant=variant, N_test=len(tte))
            lam = {}
            for name, be in [('K7', K7), ('K11', K11)]:
                f = RR.fit_soe_masses(ttr, np.empty(0), Ttr, be); R = _soe_R(np.ascontiguousarray(tte), np.ascontiguousarray(hte), be)
                lam[name] = f['mu'] + R @ f['alpha']
                comp = f['mu'] * Tte + np.sum([(f['alpha'][k] / be[k]) * (np.sum(1 - np.exp(-be[k] * (Tte - tte))) + np.sum(np.exp(be[k] * hte) - np.exp(-be[k] * (Tte - hte)))) for k in range(len(be))])
                rec[f'll_{name}'] = (np.log(lam[name]).sum() - comp) / len(tte); rec[f'comp_{name}'] = comp / len(tte); rec[f'n_{name}'] = f['n']
            allt = np.r_[hte, tte]; prev = np.diff(allt)[-len(tte):]        # preceding interval of each test event
            dl = np.log(lam['K11']) - np.log(lam['K7'])
            for g, (a, b) in {'lt1ms': (0, 1e-3), '1ms_100ms': (1e-3, 0.1), 'gt100ms': (0.1, np.inf)}.items():
                m = (prev >= a) & (prev < b); rec[f'dll_{g}'] = dl[m].sum() / len(tte); rec[f'frac_{g}'] = m.mean()
            rec['dcomp'] = -(rec['comp_K11'] - rec['comp_K7'])
            dec.append(rec)
    print('dec', day, flush=True)
dec = pd.DataFrame(dec); dec.to_csv(OUT / 'R_t4_decomposition.csv', index=False)
for proc in ['P', 'Tr']:
    for v in ['raw', 'merge1ms']:
        d = dec[(dec.proc == proc) & (dec.variant == v)]
        out[f'dec_{proc}_{v}'] = dict(gain=float((d.ll_K11 - d.ll_K7).median()), n7=float(d.n_K7.median()), n11=float(d.n_K11.median()), **{k: float(d[k].median()) for k in ['dll_lt1ms', 'dll_1ms_100ms', 'dll_gt100ms', 'dcomp', 'frac_lt1ms', 'frac_1ms_100ms', 'frac_gt100ms']})
        print(proc, v, 'K11-K7 held-out gain %.4f/event = events<1ms %.4f (frac %.2f) + 1ms-100ms %.4f (%.2f) + >100ms %.4f (%.2f) + compensator %.4f | n7 %.3f n11 %.3f' % tuple(out[f'dec_{proc}_{v}'][k] for k in ['gain', 'dll_lt1ms', 'frac_lt1ms', 'dll_1ms_100ms', 'frac_1ms_100ms', 'dll_gt100ms', 'frac_gt100ms', 'dcomp', 'n7', 'n11']), flush=True)
json.dump(out, open(OUT / 'R_t4.json', 'w'), indent=1)
