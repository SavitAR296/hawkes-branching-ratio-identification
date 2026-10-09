"""Indirect inference for the exogenous share of slow-scale kernel mass (Section 6.1).
Stage 1  calibrate a log-OU and a two-state regime background to deseasonalised count moments (bins of 1, 5, 15, 60 min);
Stage 2  'most conservative exogenous' DGP: millisecond excitation branch only, background = calibrated process x 20-day
         intraday profile; 20 sessions x 50,400 s per stream, seed 20260914;
Stage 3  fit the paper's estimators (SoE; spline-SoE knot scan 5-120 min; bivariate SoE) -> slow mass, n, 10-s count ACF;
Stage 4  compare with the data (paired session bootstrap);
Stage 5  third-order cumulants of counts: data vs SoE bootstrap (pure excitation) vs calibrated Cox-Hawkes (pure exogenous slow).
Run: python experiments/R_indirect.py [calibrate|simulate|summarise]   (resumable; outputs in tables_rev/R_ii_*)"""
import sys, json, time, numpy as np, pandas as pd
from pathlib import Path
from numba import njit
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'experiments'))
import R_revision as RR
OUT = ROOT / 'tables_rev'; SEED = 20260914; TG = 50400.0; WIDTHS = (60, 300, 900, 3600)
BET = RR.BETAS7; FAST = {'P': [5, 6], 'Tr': [4, 5]}      # mid-price: 1e3, 1e4 s^-1; trades: 1e2, 1e3 s^-1 (1-100 ms band)
SPLINES = [(300.0, 0.0), (900.0, 0.0), (1800.0, 0.0), (3600.0, 0.0), (7200.0, 0.0), (1800.0, 0.1)]
bund = RR.load_bund(); DAYS = list(bund.keys())
OC = pd.read_csv(OUT / 'R_soe_optimum_check.csv'); BV = pd.read_csv(OUT / 'R_bivariate.csv')


# ---------------------------------------------------------------- compiled simulators
@njit(cache=True)
def _logou(nsec, tau, sigma, seed):
    np.random.seed(seed); x = np.empty(nsec); a = np.exp(-1.0 / tau); s = sigma * np.sqrt(1.0 - a * a)
    x[0] = sigma * np.random.normal()
    for k in range(1, nsec):
        x[k] = a * x[k - 1] + s * np.random.normal()
    return np.exp(x - 0.5 * sigma * sigma)


@njit(cache=True)
def _regime(nsec, ratio, dur, seed):
    np.random.seed(seed); lo = 2.0 / (1.0 + ratio); hi = 2.0 * ratio / (1.0 + ratio); out = np.empty(nsec)
    state = 1 if np.random.random() < 0.5 else 0; t_next = np.random.exponential(dur)
    for k in range(nsec):
        while k >= t_next:
            state = 1 - state; t_next += np.random.exponential(dur)
        out[k] = hi if state == 1 else lo
    return out


@njit(cache=True)
def _immigrants(mu_sec, seed):
    np.random.seed(seed); cap = int(mu_sec.sum() * 2.0) + 1000; out = np.empty(cap); n = 0
    for k in range(mu_sec.shape[0]):
        m = np.random.poisson(mu_sec[k])
        for j in range(m):
            if n < cap:
                out[n] = k + np.random.random(); n += 1
    return np.sort(out[:n])


@njit(cache=True)
def _offspring(imm, betas, masses, T, seed, cap):
    np.random.seed(seed); buf = np.empty(cap); n = imm.shape[0]
    for i in range(n): buf[i] = imm[i]
    q = 0
    while q < n:
        p = buf[q]; q += 1
        for k in range(betas.shape[0]):
            m = np.random.poisson(masses[k])
            for j in range(m):
                c = p + np.random.exponential(1.0 / betas[k])
                if c < T and n < cap:
                    buf[n] = c; n += 1
    return np.sort(buf[:n])


@njit(cache=True)
def _offspring2(imm, typ, betas, ms, mc, T, seed, cap):
    np.random.seed(seed); tb = np.empty(cap); yb = np.empty(cap, np.int64); n = imm.shape[0]
    for i in range(n):
        tb[i] = imm[i]; yb[i] = typ[i]
    q = 0
    while q < n:
        p = tb[q]; u = yb[q]; q += 1
        for k in range(betas.shape[0]):
            for side in range(2):
                mass = ms[k] if side == 0 else mc[k]; ct = u if side == 0 else 1 - u
                m = np.random.poisson(mass)
                for j in range(m):
                    c = p + np.random.exponential(1.0 / betas[k])
                    if c < T and n < cap:
                        tb[n] = c; yb[n] = ct; n += 1
    o = np.argsort(tb[:n]); return tb[:n][o], yb[:n][o]


# ---------------------------------------------------------------- helpers
def profile_sec(proc):
    C = np.array([np.histogram(bund[d][proc], bins=np.arange(0, TG + 1e-9, 300))[0] for d in DAYS], float).mean(0)
    return np.repeat(C / C.mean(), 300)


PROF = {p: profile_sec(p) for p in ('P', 'Tr')}
PROF['UD'] = PROF['P']


def session_par(i, proc):
    day = DAYS[i]; D = bund[day]; T = RR.session_T(D); t = D[proc][D[proc] > 0]
    m = OC[(OC.day == int(day)) & (OC.proc == proc)][[f'm{k}' for k in range(7)]].values[0]
    return T, len(t) / T, BET[FAST[proc]], m[FAST[proc]]


def background(kind, par, nsec, seed):
    return _logou(nsec, float(par[0]), float(par[1]), seed) if kind == 'logou' else _regime(nsec, float(par[0]), float(par[1]), seed)


def sim_uni(i, proc, kind, par, seed):
    T, rate, fb, fm = session_par(i, proc); nsec = int(np.ceil(T))
    mu = rate * (1 - fm.sum()) * PROF[proc][:nsec] * background(kind, par, nsec, seed)
    imm = _immigrants(mu, seed + 1); imm = imm[imm < T]
    cap = int(len(imm) * 3 / (1 - fm.sum())) + 20000
    return _offspring(imm, fb, fm, T, seed + 2, cap), T


def sim_bivar(i, kind, par, seed):
    day = DAYS[i]; D = bund[day]; T = RR.session_T(D); up = D['up'][D['up'] > 0]; dn = D['dn'][D['dn'] > 0]
    b = BV[BV.day == int(day)]; keep = BET >= 1e3
    ms = np.where(keep, np.array([b[f'ms{k}'].values[0] for k in range(7)]), 0.0); mc = np.where(keep, np.array([b[f'mc{k}'].values[0] for k in range(7)]), 0.0)
    nsec = int(np.ceil(T)); mu = (len(up) + len(dn)) / T * (1 - ms.sum() - mc.sum()) * PROF['UD'][:nsec] * background(kind, par, nsec, seed)
    imm = _immigrants(mu, seed + 1); imm = imm[imm < T]; typ = (np.random.default_rng(seed + 3).random(len(imm)) < 0.5).astype(np.int64)
    tt, yy = _offspring2(imm, typ, BET, ms, mc, T, seed + 2, int(len(imm) * 3 / (1 - ms.sum() - mc.sum())) + 20000)
    return tt[yy == 1], tt[yy == 0], T


def moments(sessions):
    res = {}
    for W in WIDTHS:
        C = np.array([np.histogram(t, bins=np.arange(0, TG + 1e-9, W))[0] for t in sessions], float)
        Y = np.log((C + 0.5) / (C.mean(0) + 0.5))
        res[W] = (Y.var(1, ddof=1), np.array([np.corrcoef(y[:-1], y[1:])[0, 1] for y in Y]))
    return res


def acf_cum(t, T, signed=None):
    """10-s count ACF lags 1-6 and third-cumulant ratios over 1-h windows (windows with >= 60 events)."""
    ac = []; cum = {1: [], 10: [], 60: []}; sac = []
    for s0 in np.arange(0.0, T - 3600 + 1e-9, 3600):
        x = t[(t >= s0) & (t < s0 + 3600)] - s0
        if len(x) < 60: continue
        c = np.histogram(x, bins=np.arange(0, 3601, 10))[0].astype(float); z = c - c.mean(); v = z @ z
        ac.append([z[:-k] @ z[k:] / v for k in range(1, 7)])
        for dl in cum:
            cc = np.histogram(x, bins=np.arange(0, 3600 + 1e-9, dl))[0].astype(float); m = cc.mean(); k2 = cc.var(); k3 = ((cc - m) ** 3).mean()
            cum[dl].append(k3 * m / k2 ** 2 if k2 > 0 else np.nan)
        if signed is not None:
            u, d = signed; uu = u[(u >= s0) & (u < s0 + 3600)] - s0; dd = d[(d >= s0) & (d < s0 + 3600)] - s0
            sg = np.histogram(uu, bins=np.arange(0, 3601, 10))[0] - np.histogram(dd, bins=np.arange(0, 3601, 10))[0]; zs = sg - sg.mean()
            sac.append(zs[:-1] @ zs[1:] / (zs @ zs))
    out = dict(**{f'acf{k}': float(np.mean([a[k - 1] for a in ac])) for k in range(1, 7)}, **{f'r3_{dl}': float(np.nanmedian(cum[dl])) for dl in cum})
    if signed is not None: out['signed_acf1'] = float(np.mean(sac))
    return out


def fit_all(t, T):
    rec = {}
    f = RR.fit_soe_masses(t, np.empty(0), T, BET, extra_starts=False); rec.update(n_soe=f['n'], slow_soe=float((f['alpha'] / BET)[BET <= 1].sum()))
    for ks, lam in SPLINES:
        g = RR.fit_flex_tight(t, T, ks, lam); rec[f'n_sp_{ks/60:g}_{lam:g}'] = g['n']; rec[f'slow_sp_{ks/60:g}_{lam:g}'] = float(g['masses'][BET <= 1].sum())
    return rec


# ---------------------------------------------------------------- stage 1: calibration (common random numbers across the grid)
def calibrate():
    grids = {'logou': [(tau, sg) for tau in np.geomspace(10, 3600, 10) for sg in np.linspace(0.1, 2.0, 20)],
             'regime': [(r, d) for r in (1.2, 1.4, 1.7, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0) for d in np.geomspace(10, 3600, 10)]}
    res = {}
    for proc in ('P', 'Tr'):
        md = moments([bund[d][proc] for d in DAYS]); res[f'data_{proc}'] = {W: [float(md[W][0].mean()), float(md[W][1].mean()), float(md[W][0].std(ddof=1) / np.sqrt(20)), float(md[W][1].std(ddof=1) / np.sqrt(20))] for W in WIDTHS}
        for kind, grid in grids.items():
            best = None; table = []
            for par in grid:
                ms = moments([sim_uni(i, proc, kind, par, SEED + 7919 * i + 101)[0] for i in range(20)])
                obj = sum(((ms[W][0].mean() - res[f'data_{proc}'][W][0]) / res[f'data_{proc}'][W][2]) ** 2 + ((ms[W][1].mean() - res[f'data_{proc}'][W][1]) / res[f'data_{proc}'][W][3]) ** 2 for W in WIDTHS)
                table.append((float(par[0]), float(par[1]), float(obj)))
                if best is None or obj < best[2]: best = (float(par[0]), float(par[1]), float(obj), {W: [float(ms[W][0].mean()), float(ms[W][1].mean())] for W in WIDTHS})
            res[f'{kind}_{proc}'] = dict(par=best[:2], obj=best[2], moments=best[3]); res[f'{kind}_{proc}_grid'] = table
            print('calibrated', proc, kind, best[:3], flush=True)
    json.dump(res, open(OUT / 'R_ii_calibration.json', 'w'), indent=1)


# ---------------------------------------------------------------- stages 2-3: simulate the conservative DGP and fit
def simulate():
    cal = json.load(open(OUT / 'R_ii_calibration.json')); rows, done = RR.resume('R_ii_fits.csv', key='key')
    def save(): pd.DataFrame(rows).to_csv(OUT / 'R_ii_fits.csv', index=False)
    for i, day in enumerate(DAYS):
        D = bund[day]; T = RR.session_T(D)
        for proc in ('P', 'Tr'):
            key = f'data_{proc}_{day}'
            if key not in done:                      # data: same estimators (adds the 120-min spline) + statistics
                t = D[proc][D[proc] > 0]; rec = dict(key=key, src='data', proc=proc, day=day, **fit_all(t, T), **acf_cum(t, T))
                f = RR.fit_soe_masses(t, np.empty(0), T, BET, extra_starts=False)                   # SoE bootstrap sample (pure excitation)
                g = RR.soe_slowest_rate(f['alpha'], BET); ts, _ = RR.simulate_fast(T, f['mu'], BET, f['alpha'] / BET, SEED + 31 * i + (0 if proc == 'P' else 7), min(10 / max(g, 1e-6), 20000.0))
                rec.update({f'boot_{k}': v for k, v in acf_cum(ts, T).items()}); rows.append(rec); save()
            for kind in ('logou', 'regime'):
                key = f'{kind}_{proc}_{day}'
                if key in done: continue
                t, T = sim_uni(i, proc, kind, cal[f'{kind}_{proc}']['par'], SEED + 1009 * i + (0 if proc == 'P' else 503) + (0 if kind == 'logou' else 251))
                rows.append(dict(key=key, src=kind, proc=proc, day=day, N=len(t), **fit_all(t, T), **acf_cum(t, T))); save()
        for kind in ('logou', 'regime'):
            key = f'{kind}_UD_{day}'
            if key in done: continue
            up, dn, T = sim_bivar(i, kind, cal[f'{kind}_P']['par'], SEED + 2003 * i + (0 if kind == 'logou' else 97))
            fb = RR.fit_bivar_soe(up, dn, T, BET, BET); slow = BET <= 1
            rows.append(dict(key=key, src=kind, proc='UD', day=day, n_s=fb['n_s'], n_c=fb['n_c'], nplus=fb['n_s'] + fb['n_c'], nminus=fb['n_s'] - fb['n_c'],
                             slow_self=float((fb['a'] / BET)[slow].sum()), slow_cross=float((fb['c'] / BET)[slow].sum()),
                             **acf_cum(np.sort(np.r_[up, dn]), T, signed=(up, dn)))); save()
        if f'data_UD_{day}' not in done:
            up, dn = D['up'][D['up'] > 0], D['dn'][D['dn'] > 0]
            rows.append(dict(key=f'data_UD_{day}', src='data', proc='UD', day=day, **acf_cum(np.sort(np.r_[up, dn]), T, signed=(up, dn)))); save()
        print('ii', day, flush=True)


if __name__ == '__main__':
    {'calibrate': calibrate, 'simulate': simulate}[sys.argv[1]]()
