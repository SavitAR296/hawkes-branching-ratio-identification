"""Operational defensibility rule (Section 3.1): label every specification, compute S_conv and S_def with session-bootstrap
intervals for delta in {0.05, 0.10, 0.20}. defensible := (a) scales estimable at the data resolution AND (c) median held-out
log-likelihood per event >= best - delta within the held-out comparison of each dimension; adequate := defensible AND (b)."""
import json, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / 'tables_rev'; C = ROOT / 'tables_csf'; E = ROOT / 'tables'
DELTAS = (0.05, 0.10, 0.20); rng = np.random.default_rng(20260914); B = 4000
out = {}; lines = []; P = lines.append

# ---------------- held-out comparisons, one per dimension (median over sessions of l/event; gap = best - this)
def family(df, key, val):
    med = df.groupby(key)[val].median(); return (med.max() - med).to_dict(), med.to_dict()
HO = {}
for proc in ['P', 'Tr']:
    nk = pd.read_csv(R / 'R_nested_oos.csv'); nk = nk[(nk.proc == proc) & nk.model.isin(['K1_exp', 'K2_free', 'K3_free'])][['day', 'model', 'oos_ll_per_event']]
    hg = pd.read_csv(R / 'R_heldout_grids.csv'); hg = hg[hg.proc == proc].rename(columns={'grid': 'model'})[['day', 'model', 'oos_ll_per_event']]
    kgap, kmed = family(pd.concat([nk, hg]), 'model', 'oos_ll_per_event')
    cv = pd.read_csv(R / 'R_cv_tight.csv'); cv = cv[cv.proc == proc]; cv['model'] = cv.apply(lambda r: f'spline_{r.knot_min:g}_{r.lam:g}', axis=1)
    cc = pd.read_csv(R / 'R_cv_constant.csv'); cc = cc[cc.proc == proc].assign(model='constant')
    bgap, bmed = family(pd.concat([cv[['day', 'model', 'cv_ll_per_event']], cc[['day', 'model', 'cv_ll_per_event']]]), 'model', 'cv_ll_per_event')
    hw = pd.read_csv(R / 'R_heldout_windows.csv'); hw = hw[hw.proc == proc]
    wl = hw.assign(model=hw.window.map({600: '10min', 1800: '30min', 3600: '1h'}), ll=hw.ll_window_per_event)[['day', 'model', 'll']]
    out[f'window_vs_static_{proc}'] = {k: float(np.median(g.ll_window_per_event - g.ll_static_per_event)) for k, g in hw.groupby('window')}
    wgap, wmed = family(wl, 'model', 'll')
    HO[proc] = dict(kernel=kgap, background=bgap, window=wgap)
    P(f'== {proc}: held-out gaps to the best (nats/event) ==')
    for nm, g, m in [('kernel (forward 70/30, constant bg)', kgap, kmed), ('background (blocked CV, 7-rate SoE)', bgap, bmed), ('window (rolling, 7-rate SoE)', wgap, wmed)]:
        P(f'  {nm}: ' + '; '.join(f'{k} {g[k]:.4f}' for k in sorted(g, key=g.get)))
    out[f'gaps_{proc}'] = HO[proc]

# ---------------- specification set with session-level n (S_conv)
def sess(df, col, by='day'):
    return df.groupby(by)[col].median()
import os
CADM = float(os.environ.get('CADM', '5'))
k11w = pd.read_csv(R / 'R_t4_k11_windows.csv')
CREF = os.environ.get('CREF', 'window')        # 'window': gamma_min of the window's own fit; 'session': full-session fit
G_SESS = pd.read_csv(E / 'E1_daily_summary.csv')[['day', 'proc', 'gamma_min_soe']].rename(columns={'gamma_min_soe': 'g_sess'})
def adm_sess(df, col, gcol):
    if CREF == 'session':
        d = df.merge(G_SESS, on=['day', 'proc'], how='left'); d = d[d.W >= CADM / d.g_sess.clip(lower=1e-9)]
    else:
        d = df[df.W >= CADM / df[gcol].clip(lower=1e-9)]
    return d.groupby('day')[col].median()
TAG = '' if (CADM == 5 and CREF == 'window') else ('_sess' if CREF == 'session' else '') + f'_c{CADM:g}'
specs = []   # (proc, kernel, background, window, timestamp, buffer, series indexed by day)
ds = pd.read_csv(E / 'E1_daily_summary.csv'); nk = pd.read_csv(R / 'R_nested_oos.csv'); ss = pd.read_csv(C / 'A_soe_sensitivity.csv')
fb = pd.read_csv(C / 'A_flexible_background.csv'); pen = pd.read_csv(R / 'R_penalty_knot_scan.csv'); wf = pd.read_csv(E / 'E1_window_fits.csv')
ts = pd.read_csv(R / 'R_timestamp_fullsession.csv'); t10 = pd.read_csv(R / 'R_timestamp_10min.csv'); hb = pd.read_csv(R / 'R_history_buffer.csv')
GRIDK = {'K7_1e-2_1e4': 'K7_1e-2_1e4'}
for proc in ['P', 'Tr']:
    d = ds[ds.proc == proc].set_index('day')
    specs.append((proc, 'K1_exp', 'constant', 'full', 'raw', 'none', d.n_mle))
    for m in ['K2_free', 'K3_free']:
        specs.append((proc, m, 'constant', 'full', 'raw', 'none', nk[(nk.proc == proc) & (nk.model == m)].set_index('day').n))
    for g in ss.grid.unique():
        specs.append((proc, g, 'constant', 'full', 'raw', 'none', ss[(ss.proc == proc) & (ss.grid == g)].set_index('day').n))
    for m, kern in [('exp', 'K1_exp'), ('soe', 'K7_1e-2_1e4')]:
        specs.append((proc, kern, 'spline_30_archived', 'full', 'raw', 'none', fb[(fb.proc == proc) & (fb.model == m)].set_index('day').n))
    g = pen[pen.proc == proc]; ct = pd.read_csv(R / 'R_cv_tight.csv'); ct = ct[ct.proc == proc]; tight = {(r_h, r_l): gg.set_index('day').n_tight for (r_h, r_l), gg in ct.groupby(['knot_min', 'lam'])}
    for (h, lam), gg in g.groupby(['knot_min', 'lam']):
        if gg.day.nunique() < 20: continue
        specs.append((proc, 'K7_1e-2_1e4', f'spline_{h:g}_{lam:g}', 'full', 'raw', 'none', tight.get((h, lam), gg.set_index('day').n_soe)))
        if gg.n_exp.notna().all(): specs.append((proc, 'K1_exp', f'spline_{h:g}_{lam:g}', 'full', 'raw', 'none', gg.set_index('day').n_exp))
    kv = pd.read_csv(R / 'R_k11_variants.csv'); kv = kv[kv.proc == proc]
    for v, gg in kv.groupby('variant'):
        if v.startswith('spline'): specs.append((proc, 'K11_1e-2_1e4', v, 'full', 'raw', 'none', gg.set_index('day').n))
    w = wf[wf.proc == proc]; kw = k11w[k11w.proc == proc]
    for win in ['10min', '30min', '1h']:
        specs.append((proc, 'K1_exp', 'constant', win, 'raw', '30min', sess(w[w.window == win], 'n_mle')))
        specs.append((proc, 'K7_1e-2_1e4', 'constant', win, 'raw', '30min', adm_sess(w[w.window == win], 'n_soe', 'gamma_min_soe')))
        specs.append((proc, 'K11_1e-2_1e4', 'constant', win, 'raw', '30min', adm_sess(kw[kw.window == win], 'n11', 'gamma11')))
    for pr in ['merge_1ms', 'merge_10ms', 'merge_0.1s', 'randomise_1s']:
        x = ts[(ts.proc == proc) & (ts.protocol == pr)]
        specs.append((proc, 'K1_exp', 'constant', 'full', pr, 'none', sess(x, 'n_exp'))); specs.append((proc, 'K7_1e-2_1e4', 'constant', 'full', pr, 'none', sess(x, 'n_soe')))
        x = t10[(t10.proc == proc) & (t10.protocol == pr)]
        specs.append((proc, 'K1_exp', 'constant', '10min', pr, '30min', sess(x, 'n_exp'))); specs.append((proc, 'K7_1e-2_1e4', 'constant', '10min', pr, '30min', sess(x, 'n_soe')))
    h = hb[hb.proc == proc]
    for (win, buf, cut), gg in h.groupby(['window', 'buffer', 'cut']):
        if buf == 1800 and cut == 0: continue
        tag = f'{buf/60:g}min' + (f'_cut{cut/60:g}' if cut else '')
        specs.append((proc, 'K1_exp', 'constant', win, 'raw', tag, sess(gg, 'n_exp'))); specs.append((proc, 'K7_1e-2_1e4', 'constant', win, 'raw', tag, sess(gg, 'n_soe')))

# ---------------- labels
def label(sp, delta, bg_rule='cv'):
    """(a) and (c); (c) within estimand classes: session-level (kernel by forward split, background by blocked CV) and
    time-local (kernel as above, window length by one-window-ahead forecasts)."""
    proc, kern, bg, win, tsp, buf, ser = sp; G = HO[proc]; why = []
    if ser.dropna().shape[0] < 10:
        why.append(f'(a) fails: window shorter than {CADM:g} relaxation times in most windows')
    if tsp != 'raw':
        why.append('timestamp transformed: event set differs, (c) undefined; kernel scales below the coarsened resolution')
    if buf not in ('none', '30min'):
        why.append('boundary approximation (history-buffer or cut variant)')
    kg = G['kernel'].get(kern)
    if kg is None or kg > delta: why.append(f'kernel gap {kg:.3f}' if kg is not None else 'kernel not assessed')
    if win == 'full':
        key = 'spline_30_0' if bg == 'spline_30_archived' else bg
        bgp = G['background'].get(key)
        if bgp is None or bgp > delta: why.append(f'background gap {bgp:.3f}' if bgp is not None else 'background not assessed by (c)')
    else:
        wg = G['window'].get(win)
        if wg is None or wg > delta: why.append(f'window gap {wg:.3f}' if wg is not None else 'window not assessed')
    return (not why), '; '.join(why)

def srange(sel):
    sel = [s for s in sel if s[6].dropna().shape[0] >= 10]
    if not sel: return dict(lo=np.nan, lo_ci=(np.nan, np.nan), hi=np.nan, hi_ci=(np.nan, np.nan), lo_spec=None, hi_spec=None, n_specs=0)
    days = sorted(set.intersection(*[set(s[6].dropna().index) for s in sel])); M = np.array([[s[6].loc[dd] for dd in days] for s in sel])
    med = np.median(M, 1); lo, hi = med.min(), med.max(); ilo, ihi = int(med.argmin()), int(med.argmax())
    idx = rng.integers(0, len(days), (B, len(days))); bm = np.median(M[:, idx], axis=2)   # specs x B
    blo, bhi = bm.min(0), bm.max(0)
    return dict(lo=lo, lo_ci=(np.percentile(blo, 2.5), np.percentile(blo, 97.5)), hi=hi, hi_ci=(np.percentile(bhi, 2.5), np.percentile(bhi, 97.5)),
                lo_spec=sel[ilo][1:6], hi_spec=sel[ihi][1:6], n_specs=len(sel))
fmt = lambda r: f"[{r['lo']:.3f} ({r['lo_ci'][0]:.3f}-{r['lo_ci'][1]:.3f}), {r['hi']:.3f} ({r['hi_ci'][0]:.3f}-{r['hi_ci'][1]:.3f})] n_specs={r['n_specs']} | lo: {r['lo_spec']} | hi: {r['hi_spec']}"
rows = []
for proc in ['P', 'Tr']:
    sp = [s for s in specs if s[0] == proc]
    r = srange(sp); out[f'Sconv_{proc}'] = r; P(f'== {proc} S_conv {fmt(r)}')
    for dl in DELTAS:
        sel = [s for s in sp if label(s, dl)[0]]
        r = srange(sel); out[f'Sdef_{proc}_{dl}'] = r; P(f'   S_def[delta={dl}] {fmt(r)}')
        rs = srange([s for s in sel if s[3] == 'full']); out[f'Sdef_session_{proc}_{dl}'] = rs; P(f'      session-level {fmt(rs)}')
        rl = srange([s for s in sel if s[3] != 'full']); out[f'Sdef_local_{proc}_{dl}'] = rl; P(f'      time-local (c={CADM:g}) {fmt(rl)}')
        for dim, keyf in [('background', lambda s: s[2] != 'constant' and s[3] == 'full'), ('window', lambda s: s[3] != 'full'), ('basis', lambda s: s[2] == 'constant' and s[3] == 'full')]:
            sub = [s for s in sel if keyf(s) and s[1] in ('K7_1e-2_1e4', 'K11_1e-2_1e4')] if dim != 'basis' else [s for s in sel if keyf(s)]
            if sub: out[f'Sdef_{proc}_{dl}_{dim}'] = (float(min(np.median(s[6]) for s in sub)), float(max(np.median(s[6]) for s in sub)))
    for s in sp:
        G = HO[proc]; grp = 'timestamp' if s[4] != 'raw' else ('buffer' if s[5] not in ('none', '30min') else ('window' if s[3] != 'full' else ('background' if s[2] != 'constant' else 'kernel')))
        gap = (G['window'].get(s[3]) if grp == 'window' else G['background'].get('spline_30_0' if s[2] == 'spline_30_archived' else s[2]) if grp == 'background' else G['kernel'].get(s[1]) if grp == 'kernel' else None)
        rows.append(dict(stream=proc, kernel=s[1], background=s[2], window=s[3], timestamp=s[4], buffer=s[5], n_median=float(np.median(s[6])), group=grp, gap=gap, kernel_gap=G['kernel'].get(s[1]),
                         **{f'def_{dl}': label(s, dl)[0] for dl in DELTAS}, reason_010=label(s, 0.10)[1]))
lab = pd.DataFrame(rows); lab.to_csv(R / f'R_defensible_labels{TAG}.csv', index=False)
P(f'labelled specifications: {len(lab)} ({(lab.stream=="P").sum()} mid-price, {(lab.stream=="Tr").sum()} trades)')
for proc in ['P', 'Tr']:
    L = lab[(lab.stream == proc) & lab['def_0.1']]
    P(f'{proc} defensible at 0.10: ' + '; '.join(f"{r.kernel}|{r.background}|{r.window} {r.n_median:.3f}" for r in L.sort_values('n_median').itertuples()))
json.dump(out, open(R / f'R_defensible{TAG}.json', 'w'), indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item())
open(R / 'R_defensible.txt', 'w').write('\n'.join(lines)); print('\n'.join(lines))

# ---------------- LaTeX table for the manuscript (Table: defensibility labels and ranges)
def mark(g, proc):
    return ''.join(('\u2713' if g is not None and g <= dl else '\u00b7') for dl in DELTAS)
NAMES = {'K1_exp': 'single exponential ($K=1$)', 'K2_free': '$K=2$, free rates', 'K3_free': '$K=3$, free rates', 'K5_1e-2_1e4': '5 rates, $10^{-2}$--$10^{4}$',
         'K7_1e-2_1e4': '7 rates, $10^{-2}$--$10^{4}$ (primary)', 'K9_1e-2_1e4': '9 rates, $10^{-2}$--$10^{4}$', 'K11_1e-2_1e4': '11 rates, $10^{-2}$--$10^{4}$',
         'K7_1e-2_1e5': '7 rates, $10^{-2}$--$10^{5}$', 'K7_1e-3_1e4': '7 rates, $10^{-3}$--$10^{4}$', 'K7_1e-3_1e5': '7 rates, $10^{-3}$--$10^{5}$',
         'constant': 'constant', '10min': '10 min', '30min': '30 min', '1h': '1 h'}
def bgname(k):
    if k in NAMES: return NAMES[k]
    _, h, l = k.split('_'); return f'spline, {h} min, $\\lambda={l}$'
T = []
T.append(r'\begin{tabular}{llrrcc}'); T.append(r'\toprule')
T.append(r'Comparison (design) & Specification & Gap, mid-price & Gap, trades & Mid-price & Trades \\'); T.append(r'\midrule')
for fam, title in [('kernel', 'Kernel (forward split)'), ('background', 'Background (blocked CV)'), ('window', 'Window (one-window-ahead)')]:
    keys = sorted(set(HO['P'][fam]) | set(HO['Tr'][fam]), key=lambda k: HO['P'][fam].get(k, 9))
    for j, k in enumerate(keys):
        gp, gt = HO['P'][fam].get(k), HO['Tr'][fam].get(k)
        T.append(f"{title if j == 0 else ''} & {bgname(k)} & {gp:.3f} & {gt:.3f} & {mark(gp,'P')} & {mark(gt,'Tr')} \\\\")
    T.append(r'\midrule')
T.append(r'Timestamp & merged or randomised & \multicolumn{2}{c}{event set differs} & $\cdot\cdot\cdot$ & $\cdot\cdot\cdot$ \\'); T.append(r'\midrule')
def rr(r): return f"{r['lo']:.2f} [{r['lo_ci'][0]:.2f}, {r['lo_ci'][1]:.2f}]--{r['hi']:.2f} [{r['hi_ci'][0]:.2f}, {r['hi_ci'][1]:.2f}]"
T.append(r'\multicolumn{2}{l}{$S_{\rm conv}$, all ' + f"{out['Sconv_P']['n_specs']}" + r' specifications} & \multicolumn{2}{l}{' + rr(out['Sconv_P']) + r'} & \multicolumn{2}{l}{' + rr(out['Sconv_Tr']) + r'} \\')
for dl in DELTAS:
    T.append(r'\multicolumn{2}{l}{$S_{\rm def}$(session), $\delta=' + f'{dl:.2f}' + r'$} & \multicolumn{2}{l}{' + rr(out[f'Sdef_session_P_{dl}']) + r'} & \multicolumn{2}{l}{' + rr(out[f'Sdef_session_Tr_{dl}']) + r'} \\')
    T.append(r'\multicolumn{2}{l}{$S_{\rm def}$(local, $c=' + f'{CADM:g}' + r'$), $\delta=' + f'{dl:.2f}' + r'$} & \multicolumn{2}{l}{' + rr(out[f'Sdef_local_P_{dl}']) + r'} & \multicolumn{2}{l}{' + rr(out[f'Sdef_local_Tr_{dl}']) + r'} \\')
T.append(r'\bottomrule'); T.append(r'\end{tabular}')
open(R / f'R_defensible_table{TAG}.tex', 'w').write('\n'.join(T).replace('\u2713', r'\checkmark').replace('\u00b7', r'$\cdot$'))
print('table written')
