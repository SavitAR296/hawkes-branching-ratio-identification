"""Mode-resolved, shape-free count-variance bounds (Section 3.5 / 5.11 / Appendix D).
n+(W) = 1 - sqrt(E N_W / Var N_W) on pooled counts (activity mode; Hardiman-Bouchaud form)
n-(W) = 1 - sqrt(E[N+ + N-]_W / Var[N+ - N-]_W)   (signed mode; immune to common exogenous variation)
Medians over 20 sessions with 4,000-resample session bootstrap; deseasonalised variant; announcement-day exclusion;
08-15 / 15-22 split; simulation check of the identity (Appendix D). Seed 20260914."""
import json, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / 'tables_rev'; T = 50400.0; rng = np.random.default_rng(20260914)
Z = np.load(ROOT / 'data/bund.npz', allow_pickle=True); DAYS = sorted(Z.files); ANN = {'20140403', '20140404', '20140430'}
WS = [0.001, 0.01, 0.1, 1, 5, 10, 30, 60, 120, 300, 600, 1200]


def n_hat(mean, var):
    return 1.0 - np.sqrt(mean / var) if var > 0 and mean > 0 else np.nan


PROF = {}
for key, (i, j) in {'mid': (0, 1), 'trd': (2, 3)}.items():
    C = np.array([np.histogram(np.concatenate([Z[d][i], Z[d][j]]), bins=np.arange(0, T + 1e-9, 1800))[0] for d in DAYS], float).mean(0); PROF[key] = C / C.mean()


def stats(d, key, W, lo=0.0, hi=T, deseason=False):
    """sparse binning (occupied bins only; empty bins counted in the denominator) as in reviewer_checks.py."""
    i, j = (0, 1) if key == 'mid' else (2, 3); nb = int((hi - lo) // W)
    tp = Z[d][i]; tn = Z[d][j]; tp = tp[(tp >= lo) & (tp < lo + nb * W)]; tn = tn[(tn >= lo) & (tn < lo + nb * W)]
    idx = np.concatenate([((tp - lo) // W).astype(np.int64), ((tn - lo) // W).astype(np.int64)])
    ws = np.concatenate([np.ones(len(tp)), -np.ones(len(tn))]); u, inv = np.unique(idx, return_inverse=True)
    ct = np.bincount(inv).astype(float); cs = np.bincount(inv, weights=ws)
    mean_tot = (len(tp) + len(tn)) / nb
    if deseason and W >= 1:
        ct = ct / PROF[key][np.minimum(((lo + (u + 0.5) * W) // 1800).astype(int), 27)]
    m = ct.sum() / nb; vt = (ct * ct).sum() / nb - m * m; ms = cs.sum() / nb; vs = (cs * cs).sum() / nb - ms * ms
    return n_hat(m, vt), n_hat(mean_tot, vs), (vs / mean_tot if mean_tot > 0 else np.nan)


def bci(x, B=4000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; idx = rng.integers(0, len(x), (B, len(x))); bs = np.median(x[idx], 1)
    return [float(np.median(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


out = {}; rows = []
for key in ['mid', 'trd']:
    for W in WS:
        per = np.array([stats(d, key, W) for d in DAYS]); des = np.array([stats(d, key, W, deseason=True)[0] for d in DAYS])
        noann = np.array([stats(d, key, W) for d in DAYS if d not in ANN]); am = np.array([stats(d, key, W, 0.0, 25200.0) for d in DAYS]); pm = np.array([stats(d, key, W, 25200.0, T) for d in DAYS])
        rec = dict(stream=key, W=W, nplus=bci(per[:, 0]), nminus=bci(per[:, 1]), nplus_des=bci(des), VRminus=bci(per[:, 2]),
                   nplus_noann=bci(noann[:, 0]), nminus_noann=bci(noann[:, 1]), nplus_am=bci(am[:, 0]), nminus_am=bci(am[:, 1]), nplus_pm=bci(pm[:, 0]), nminus_pm=bci(pm[:, 1]))
        rows.append(rec); out[f'{key}_{W}'] = rec
        print(f"{key} W={W:>7g}s  n+ {rec['nplus'][0]:+.3f} [{rec['nplus'][1]:+.3f},{rec['nplus'][2]:+.3f}]  n+des {rec['nplus_des'][0]:+.3f}  n- {rec['nminus'][0]:+.3f} [{rec['nminus'][1]:+.3f},{rec['nminus'][2]:+.3f}]  VR- {rec['VRminus'][0]:.3f} | noann n- {rec['nminus_noann'][0]:+.3f} | am/pm n- {rec['nminus_am'][0]:+.3f}/{rec['nminus_pm'][0]:+.3f}  n+ {rec['nplus_am'][0]:+.3f}/{rec['nplus_pm'][0]:+.3f}", flush=True)
pd.DataFrame(rows).to_csv(OUT / 'R_moderes.csv', index=False)
# ---------------- bounds (mid-price: [max(n+(0.1 s), -n-(W*)), n+des(1200)]; trades: [n+(0.1 s), n+des(1200)])
g = lambda k, W, f: out[f'{k}_{W}'][f]
bounds = {}
for Wstar in (60, 300):
    lo_sess = np.array([max(stats(d, 'mid', 0.1)[0], -stats(d, 'mid', Wstar)[1]) for d in DAYS]); hi_sess = np.array([stats(d, 'mid', 1200, deseason=True)[0] for d in DAYS])
    bounds[f'mid_W{Wstar}'] = dict(lo=bci(lo_sess), hi=bci(hi_sess), lo_components=dict(nplus01=g('mid', 0.1, 'nplus'), minus_nminus=[-v for v in g('mid', Wstar, 'nminus')]))
bounds['trd'] = dict(lo=g('trd', 0.1, 'nplus'), hi=g('trd', 1200, 'nplus_des'))
print('bounds', json.dumps(bounds, indent=0)[:600])
# ---------------- Appendix D: simulation check of VR- -> 1/(1-n-)^2, unaffected by common exogenous intensity


def simulate(ns, nc, beta, mu0, sigma, tau, Tsim, seed):
    g_ = np.random.default_rng(seed); n = int(Tsim); X = np.zeros(n)
    if sigma > 0:
        a = np.exp(-1 / tau); s = sigma * np.sqrt(1 - a * a); e = g_.standard_normal(n)
        for i in range(1, n): X[i] = a * X[i - 1] + s * e[i]
    mu = mu0 * np.exp(X - 0.5 * sigma ** 2); k = g_.poisson(mu)
    t = np.repeat(np.arange(n), k) + g_.random(k.sum()); y = g_.integers(0, 2, len(t)) * 2 - 1
    T_all, Y_all, gt, gy = [t], [y], t, y
    while len(gt):
        cs, cc = g_.poisson(ns, len(gt)), g_.poisson(nc, len(gt))
        ts = np.repeat(gt, cs) + g_.exponential(1 / beta, cs.sum()); tc = np.repeat(gt, cc) + g_.exponential(1 / beta, cc.sum())
        gt = np.concatenate([ts, tc]); gy = np.concatenate([np.repeat(gy, cs), -np.repeat(gy, cc)]); keep = gt < Tsim; gt, gy = gt[keep], gy[keep]
        T_all.append(gt); Y_all.append(gy)
    return np.concatenate(T_all), np.concatenate(Y_all)


sim = {}; WSIM = [1, 10, 60, 300, 1200]
for name, ns, nc, sig, mu0 in [('cross_only', 0.0, 0.85, 0.0, 0.03), ('cross_logOU', 0.0, 0.85, 1.0, 0.03), ('noexc_logOU', 0.0, 0.0, 1.0, 0.2)]:
    vals = []
    for r in range(20):
        t, y = simulate(ns, nc, 1.0, mu0, sig, 300.0, 200000.0, 20260914 + r)
        row = []
        for W in WSIM:
            nb = int(200000.0 // W); idx = (t // W).astype(int); m = idx < nb
            tot = np.bincount(idx[m], minlength=nb).astype(float); sg = np.bincount(idx[m], weights=y[m], minlength=nb)
            row += [n_hat(tot.mean(), tot.var()), n_hat(tot.mean(), sg.var())]
        vals.append(row)
    vals = np.array(vals); sim[name] = dict(truth=[ns + nc, ns - nc], W=WSIM, mean=vals.mean(0).tolist(), sd=vals.std(0, ddof=1).tolist())
    print(name, 'truth n+ %.2f n- %+.2f' % (ns + nc, ns - nc), ' '.join(f'W={W}: n+ {vals[:, 2*k].mean():.3f}±{vals[:, 2*k].std(ddof=1):.3f} n- {vals[:, 2*k+1].mean():+.3f}±{vals[:, 2*k+1].std(ddof=1):.3f}' for k, W in enumerate(WSIM)), flush=True)
json.dump(dict(table=out, bounds=bounds, sim=sim), open(OUT / 'R_moderes.json', 'w'), indent=1)
