"""
R_revision.py -- additional analyses for the CSF revision (CHAOS-D-26-09420).
Sub-tasks (run: python experiments/R_revision.py <task>):
  timestamp  same-session/same-window/same-background comparison of all timestamp protocols,
             retained fractions, 1-s randomisation repeated R times (R2-4, R1-7)
  history    history-buffer (0/5/30/60 min) and boundary-exclusion sensitivity (R2-3)
  bivar      bivariate up/down fits: common beta, separate betas, bivariate SoE eigenmodes,
             mode-implied 10-s count autocorrelations (R1-5, R1-13, R2-0)
  penalty    normalised penalty x knot-spacing scan, blocked CV selection (R1-6, R2-2)
  nested     free-rate SoE K=1,2,3 nested comparison, AIC/BIC, held-out log-likelihood (R1-2, R2-8)
  sim        simulation controls: burn-in, correct exponential, off-grid, Cox/regime, long T,
             slow n=0.95 multimodality (R1-4, R2-7)
  boot       refitted parametric bootstrap of adequacy diagnostics (R1-3, R2-8)
Seed 20260914 throughout.  Outputs -> tables_rev/*.csv
"""
import os, sys, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
from scipy.optimize import minimize
from scipy.stats import kstest, cramervonmises
from scipy.interpolate import BSpline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'experiments'))
from hawkes import (fit_exp_mle, fit_soe_mle, _soe_R, soe_slowest_rate, soe_bin_autocorr,
                    bin_counts_autocorr, exp_hawkes_bin_autocorr, rescaled_residuals_exp)
import A_csf_additions as A
BETAS7 = A.BETAS7
OUT = ROOT / 'tables_rev'; OUT.mkdir(exist_ok=True)
SEED = 20260914
load_bund = A.load_bund


def session_T(D):
    return max(D['P'][-1], D['Tr'][-1])


def resume(fname, key='day'):
    """rows already written (list of dicts) and the set of finished keys."""
    f = OUT / fname
    if not f.exists():
        return [], set()
    df = pd.read_csv(f)
    rows = df.to_dict('records')
    done = set(df[key].astype(str).unique()) if key in df else set()
    return rows, done


# ---------------------------------------------------------------- generic log-likelihood evaluators
def soe_loglik(t, hist, T, mu, alphas, betas):
    """log-likelihood on [0,T] of events t with history hist<0, fixed parameters (SoE or Exp)."""
    t = np.ascontiguousarray(t, float); hist = np.ascontiguousarray(hist, float)
    betas = np.asarray(betas, float); alphas = np.asarray(alphas, float)
    R = _soe_R(t, hist, betas)
    lam = mu + R @ alphas
    if np.any(lam <= 0):
        return -np.inf
    S = np.array([np.sum(1 - np.exp(-b * (T - t))) / b + (np.sum(np.exp(b * hist) - np.exp(-b * (T - hist))) / b if len(hist) else 0.0) for b in betas])
    return float(np.sum(np.log(lam)) - mu * T - alphas @ S)


def greedy_merge(t, delta):
    return A.merge_min_spacing(t, delta)


def randomise_1s(t, T, rng):
    tr = np.sort(np.floor(t) + rng.uniform(0, 1, len(t)))
    return tr[tr < T]


# ================================================================ TASK: timestamp
def task_timestamp(R=100):
    bund = load_bund(); rng = np.random.default_rng(SEED)
    rows, done = resume('R_timestamp_fullsession.csv'); wrows, _ = resume('R_timestamp_10min.csv')
    wrows = [r for r in wrows if str(r['day']) in done]
    for day, D in bund.items():
        if str(day) in done: continue
        rng = np.random.default_rng(SEED + int(day))
        T = session_T(D)
        for proc in ['P', 'Tr']:
            t_all = D[proc]; t = t_all[t_all > 0]
            protos = {'raw': t, 'merge_1ms': greedy_merge(t, 1e-3), 'merge_10ms': greedy_merge(t, 1e-2), 'merge_0.1s': greedy_merge(t, 0.1)}
            for name, tt in protos.items():
                fe = fit_exp_mle(tt, np.empty(0), T); fs = fit_soe_mle(tt, np.empty(0), T, betas=BETAS7)
                rows.append(dict(day=day, proc=proc, protocol=name, draw=0, N=len(tt), retained=len(tt) / len(t), n_exp=fe['n'], beta_exp=fe['beta'], n_soe=fs['n']))
            for r in range(R):
                tr = randomise_1s(t, T, rng)
                fe = fit_exp_mle(tr, np.empty(0), T); fs = fit_soe_mle(tr, np.empty(0), T, betas=BETAS7)
                rows.append(dict(day=day, proc=proc, protocol='randomise_1s', draw=r, N=len(tr), retained=1.0, n_exp=fe['n'], beta_exp=fe['beta'], n_soe=fs['n']))
            # 10-min windows with 30-min history: all protocols on the same windows (window effect isolated)
            W, H = 600.0, 1800.0
            protos['randomise_1s'] = randomise_1s(t_all, T, rng)
            for s in np.arange(0.0, T - W + 1e-9, W):
                for name, tt in protos.items():
                    x = tt[(tt >= s) & (tt < s + W)] - s; h = tt[(tt >= s - H) & (tt < s)] - s
                    if len(x) < 30:
                        continue
                    fe = fit_exp_mle(x, h, W); fs = fit_soe_mle(x, h, W, betas=BETAS7)
                    wrows.append(dict(day=day, proc=proc, protocol=name, start=s, N=len(x), n_exp=fe['n'], n_soe=fs['n']))
        pd.DataFrame(rows).to_csv(OUT / 'R_timestamp_fullsession.csv', index=False)
        pd.DataFrame(wrows).to_csv(OUT / 'R_timestamp_10min.csv', index=False)
        print('timestamp', day, flush=True)


# ================================================================ TASK: history
def task_history():
    bund = load_bund(); rows, done = resume('R_history_buffer.csv')
    settings = [(0.0, 0.0), (300.0, 0.0), (1800.0, 0.0), (3600.0, 0.0), (1800.0, 60.0), (1800.0, 300.0)]  # (buffer, excluded initial segment)
    for day, D in bund.items():
        if str(day) in done: continue
        T = session_T(D)
        for proc in ['P', 'Tr']:
            t_all = D[proc]
            for wname, W in [('10min', 600.0), ('30min', 1800.0), ('1h', 3600.0)]:
                for s in np.arange(0.0, T - W + 1e-9, W):
                    for H, cut in settings:
                        a = s + cut
                        x = t_all[(t_all >= a) & (t_all < s + W)] - a
                        h = t_all[(t_all >= s - H) & (t_all < a)] - a
                        if len(x) < 30:
                            continue
                        fe = fit_exp_mle(x, h, W - cut); fs = fit_soe_mle(x, h, W - cut, betas=BETAS7)
                        rows.append(dict(day=day, proc=proc, window=wname, start=s, buffer=H, cut=cut, N=len(x), Nh=len(h),
                                         n_exp=fe['n'], beta_exp=fe['beta'], n_soe=fs['n'], soe_ge1=int(fs['n'] >= 1), exp_bound=int(abs(np.log(fe['beta'])) > 8.9)))
        pd.DataFrame(rows).to_csv(OUT / 'R_history_buffer.csv', index=False)
        print('history', day, flush=True)


# ================================================================ TASK: bivar
@njit(cache=True)
def _bivar_R(times, typ, betas):
    """R_plus[i,k] = sum_{j<i, up} exp(-beta_k (t_i - t_j)), R_minus likewise for down events."""
    n = times.shape[0]; K = betas.shape[0]
    Rp = np.zeros((n, K)); Rm = np.zeros((n, K)); sp = np.zeros(K); sm = np.zeros(K); prev = 0.0
    for i in range(n):
        d = times[i] - prev
        for k in range(K):
            e = np.exp(-betas[k] * d); sp[k] *= e; sm[k] *= e
            Rp[i, k] = sp[k]; Rm[i, k] = sm[k]
        if typ[i] == 1:
            for k in range(K):
                sp[k] += 1.0
        else:
            for k in range(K):
                sm[k] += 1.0
        prev = times[i]
    return Rp, Rm


def fit_bivar_soe(up, dn, T, betas_s, betas_c):
    """Symmetric bivariate Hawkes: lambda+ = mu + sum_k a_k R+_k(beta_s) + sum_l c_l R-_l(beta_c); lambda- mirror.
    Concave in (mu, a, c) >= 0 for fixed rates."""
    times = np.r_[up, dn]; typ = np.r_[np.ones(len(up), np.int64), -np.ones(len(dn), np.int64)]
    o = np.argsort(times); times = np.ascontiguousarray(times[o]); typ = np.ascontiguousarray(typ[o])
    bs = np.asarray(betas_s, float); bc = np.asarray(betas_c, float)
    Rp_s, Rm_s = _bivar_R(times, typ, bs); Rp_c, Rm_c = _bivar_R(times, typ, bc)
    isup = typ == 1
    Xs = np.where(isup[:, None], Rp_s, Rm_s)   # own-type recursion
    Xc = np.where(isup[:, None], Rm_c, Rp_c)   # cross-type recursion
    S_s = np.array([(np.sum(1 - np.exp(-b * (T - up))) + np.sum(1 - np.exp(-b * (T - dn)))) / b for b in bs])
    S_c = np.array([(np.sum(1 - np.exp(-b * (T - up))) + np.sum(1 - np.exp(-b * (T - dn)))) / b for b in bc])
    Ks, Kc = len(bs), len(bc); rate = len(times) / (2 * T)

    def f(x):
        mu = x[0]; a = x[1:1 + Ks]; c = x[1 + Ks:]
        lam = mu + Xs @ a + Xc @ c
        if np.any(lam <= 0):
            return 1e100, np.zeros_like(x)
        ll = np.log(lam).sum() - 2 * mu * T - a @ S_s - c @ S_c
        inv = 1 / lam
        return -ll, -np.r_[inv.sum() - 2 * T, Xs.T @ inv - S_s, Xc.T @ inv - S_c]
    x0 = np.r_[0.5 * rate, np.full(Ks, 0.05 * rate), np.full(Kc, 0.05 * rate)]
    res = minimize(f, x0, jac=True, method='L-BFGS-B', bounds=[(1e-12, None)] * len(x0), options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-7})
    mu = res.x[0]; a = res.x[1:1 + Ks]; c = res.x[1 + Ks:]
    return dict(mu=mu, a=a, c=c, betas_s=bs, betas_c=bc, n_s=np.sum(a / bs), n_c=np.sum(c / bc), loglik=-res.fun, success=res.success, nparam=len(x0))


def task_bivar():
    bund = load_bund(); rows, done = resume('R_bivariate.csv'); mode_rows, _ = resume('R_bivariate_modes_acf.csv')
    mode_rows = [r for r in mode_rows if str(r['day']) in done]
    for day, D in bund.items():
        if str(day) in done: continue
        T = session_T(D); up = D['up'][D['up'] > 0]; dn = D['dn'][D['dn'] > 0]
        f0 = A.fit_bivar_common_beta(up, dn, T)                                   # archived benchmark: common beta
        # separate decay rates, profiled on a 9x9 log-grid around the common-beta optimum (+-2 in log)
        best = None
        for lbs in np.linspace(np.log(f0['beta']) - 2, np.log(f0['beta']) + 2, 9):
            for lbc in np.linspace(np.log(f0['beta']) - 2, np.log(f0['beta']) + 2, 9):
                r = fit_bivar_soe(up, dn, T, [np.exp(lbs)], [np.exp(lbc)])
                if best is None or r['loglik'] > best['loglik']:
                    best = r
        fsep = best
        # bivariate SoE on the seven-rate grid for self and cross excitation
        fb = fit_bivar_soe(up, dn, T, BETAS7, BETAS7)
        npl, nmi = fb['n_s'] + fb['n_c'], fb['n_s'] - fb['n_c']
        g_act = soe_slowest_rate(fb['a'] + fb['c'], BETAS7)
        rows.append(dict(day=day, N=len(up) + len(dn),
                         beta_common=f0['beta'], ll_common=f0['loglik'], nplus_common=f0['n_plus'], nminus_common=f0['n_minus'],
                         beta_s=fsep['betas_s'][0], beta_c=fsep['betas_c'][0], ll_sep=fsep['loglik'], nplus_sep=fsep['n_s'] + fsep['n_c'], nminus_sep=fsep['n_s'] - fsep['n_c'],
                         ll_soe=fb['loglik'], n_s_soe=fb['n_s'], n_c_soe=fb['n_c'], nplus_soe=npl, nminus_soe=nmi, gamma_act_soe=g_act, tau_act_soe=(1 / g_act if g_act > 0 else np.nan),
                         **{f'ms{k}': fb['a'][k] / BETAS7[k] for k in range(7)}, **{f'mc{k}': fb['c'][k] / BETAS7[k] for k in range(7)}))
        # mode-implied 10-s count autocorrelations (activity N+ + N-, signed N+ - N-) over 1-h windows vs empirical
        Delta, W = 10.0, 3600.0
        for s in np.arange(0.0, T - W + 1e-9, W):
            u = up[(up >= s) & (up < s + W)] - s; d_ = dn[(dn >= s) & (dn < s + W)] - s
            if len(u) + len(d_) < 60:
                continue
            fw = fit_bivar_soe(u, d_, W, BETAS7, BETAS7)
            Lam, rho_act = soe_bin_autocorr(1.0, fw['a'] + fw['c'], BETAS7, Delta, kmax=6)
            _, rho_sig = soe_bin_autocorr(1.0, fw['a'] - fw['c'], BETAS7, Delta, kmax=6)
            edges = np.arange(0, W + 1e-9, Delta)
            cu = np.histogram(u, bins=edges)[0].astype(float); cd = np.histogram(d_, bins=edges)[0].astype(float)
            def acf(x):
                x = x - x.mean(); v = np.sum(x * x)
                return [np.sum(x[:-k] * x[k:]) / v if v > 0 else np.nan for k in range(1, 7)]
            ea, es = acf(cu + cd), acf(cu - cd)
            rec = dict(day=day, start=s, nplus=fw['n_s'] + fw['n_c'], nminus=fw['n_s'] - fw['n_c'])
            rec.update({f'act_emp{k}': ea[k - 1] for k in range(1, 7)}); rec.update({f'act_th{k}': rho_act[k - 1] for k in range(1, 7)})
            rec.update({f'sig_emp{k}': es[k - 1] for k in range(1, 7)}); rec.update({f'sig_th{k}': rho_sig[k - 1] for k in range(1, 7)})
            mode_rows.append(rec)
        pd.DataFrame(rows).to_csv(OUT / 'R_bivariate.csv', index=False)
        pd.DataFrame(mode_rows).to_csv(OUT / 'R_bivariate_modes_acf.csv', index=False)
        print('bivar', day, flush=True)


# ================================================================ TASK: penalty (normalised penalty x knot scan + blocked CV)
def flex_fit_norm(t, T, model, knot_step, lam, raw_e, raw_s):
    """Flexible non-negative linear B-spline background with normalised second-difference penalty
    P(c) = lam * N * sum_j (Delta^2 c_j / rbar)^2,  rbar = N/T  (lam dimensionless)."""
    N = len(t); rbar = N / T
    pen = lam * N / rbar ** 2
    return A.flex_background_fit(t, T, model=model, knot_step=knot_step, penalty=pen, beta_hint=raw_e['beta'], raw_fit=(raw_s if model == 'soe' else raw_e))


def blocked_cv_soe(t, T, knot_step, lam, nfold=5, block=1800.0):
    """Interleaved blocked CV for the SoE + spline model: hold out every nfold-th 30-min block.
    Training likelihood uses only events/compensator inside training blocks (history from all events);
    returns held-out log-likelihood per held-out event (averaged over folds)."""
    B, I_full, D2 = A.bspline_basis(t, T, knot_step)
    betas = BETAS7; R = _soe_R(np.ascontiguousarray(t), np.empty(0), betas)
    nb = int(np.ceil(T / block)); blk = np.minimum((t // block).astype(int), nb - 1)
    m = B.shape[1]; N = len(t); rbar = N / T; pen = lam * N / rbar ** 2
    # per-block integrals of basis functions and of each excitation component
    knots = np.r_[np.repeat(0.0, 4), np.arange(knot_step, T, knot_step), np.repeat(T, 4)]
    ants = [BSpline(knots, np.eye(m)[j], 3, extrapolate=False).antiderivative() for j in range(m)]
    edges = np.minimum(np.arange(0, nb + 1) * block, T)
    I_blk = np.array([[ants[j](edges[b + 1]) - ants[j](edges[b]) for j in range(m)] for b in range(nb)])
    S_blk = np.zeros((nb, len(betas)))
    for b in range(nb):
        a0, a1 = edges[b], edges[b + 1]
        for k, bt in enumerate(betas):
            tj = t[t < a1]
            S_blk[b, k] = np.sum(np.exp(-bt * np.maximum(a0 - tj, 0)) - np.exp(-bt * (a1 - tj))) / bt
    out = []
    for fold in range(nfold):
        test = (np.arange(nb) % nfold) == fold
        tr_ev = ~test[blk]; te_ev = test[blk]
        I_tr = I_blk[~test].sum(0); S_tr = S_blk[~test].sum(0); I_te = I_blk[test].sum(0); S_te = S_blk[test].sum(0)
        Btr = B[tr_ev]; Rtr = R[tr_ev]
        def fun(x):
            c = x[:m]; a = x[m:]
            lam_ = np.asarray(Btr @ c).ravel() + Rtr @ a
            if np.any(lam_ <= 0):
                return 1e100, np.zeros_like(x)
            Dc = D2 @ c
            ll = np.log(lam_).sum() - I_tr @ c - S_tr @ a - pen * Dc @ Dc
            inv = 1 / lam_
            return -ll, -np.r_[np.asarray(Btr.T @ inv).ravel() - I_tr - 2 * pen * (D2.T @ Dc), Rtr.T @ inv - S_tr]
        x0 = np.r_[np.full(m, 0.45 * rbar), np.full(len(betas), 0.55 * rbar / len(betas) * 0.01)]
        res = minimize(fun, x0, jac=True, method='L-BFGS-B', bounds=[(1e-12, None)] * len(x0), options={'maxiter': 1600, 'ftol': 1e-9, 'gtol': 1e-6, 'maxls': 100})
        c = res.x[:m]; a = res.x[m:]
        lam_te = np.asarray(B[te_ev] @ c).ravel() + R[te_ev] @ a
        ll_te = np.log(np.maximum(lam_te, 1e-300)).sum() - I_te @ c - S_te @ a
        out.append((ll_te, te_ev.sum()))
    return sum(o[0] for o in out) / max(sum(o[1] for o in out), 1)


def task_penalty():
    bund = load_bund(); rows, done = resume('R_penalty_knot_scan.csv'); cv_rows, _ = resume('R_penalty_cv.csv')
    cv_rows = [r for r in cv_rows if str(r['day']) in done]
    lams = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    knots = [300.0, 900.0, 1800.0, 3600.0, 7200.0]
    exp_cfg = [(900.0, 0.0), (1800.0, 0.0), (1800.0, 10.0), (3600.0, 0.0)]
    cv_cfg = [(900.0, 0.0), (1800.0, 0.0), (1800.0, 0.1), (1800.0, 10.0), (3600.0, 0.0)]
    for day, D in bund.items():
        if str(day) in done:
            continue
        t0 = time.time(); T = session_T(D)
        for proc in ['P', 'Tr']:
            t = D[proc][D[proc] > 0]
            raw_e = fit_exp_mle(t, np.empty(0), T); raw_s = fit_soe_mle(t, np.empty(0), T, betas=BETAS7)
            for ks in knots:
                for lam in lams:
                    if ks != 1800.0 and lam not in (0.0, 1e-2, 1.0, 100.0):
                        continue
                    fs = flex_fit_norm(t, T, 'soe', ks, lam, raw_e, raw_s)
                    mass = fs['alpha'] / BETAS7
                    rec = dict(day=day, proc=proc, knot_min=ks / 60, lam=lam, nbas=fs['nbas'], n_soe=fs['n'], slow_mass_soe=mass[BETAS7 <= 1].sum(),
                               succ_soe=fs['success'], raw_soe=raw_s['n'], raw_exp=raw_e['n'])
                    if (ks, lam) in exp_cfg:
                        fe = flex_fit_norm(t, T, 'exp', ks, lam, raw_e, raw_s); rec.update(n_exp=fe['n'], beta_exp=fe['beta'])
                    rows.append(rec)
            for ks, lam in cv_cfg:
                try:
                    cv_rows.append(dict(day=day, proc=proc, knot_min=ks / 60, lam=lam, cv_ll_per_event=blocked_cv_soe(t, T, ks, lam)))
                except Exception as e:
                    cv_rows.append(dict(day=day, proc=proc, knot_min=ks / 60, lam=lam, cv_ll_per_event=np.nan, err=str(e)[:60]))
        pd.DataFrame(rows).to_csv(OUT / 'R_penalty_knot_scan.csv', index=False)
        pd.DataFrame(cv_rows).to_csv(OUT / 'R_penalty_cv.csv', index=False)
        print('penalty', day, '%.0fs' % (time.time() - t0), flush=True)


# ================================================================ TASK: nested free-rate SoE + held-out likelihood
def fit_soe_free(t, hist, T, K, starts):
    best = None
    for st in starts:
        def obj(lb):
            r = fit_soe_mle(t, hist, T, betas=np.exp(lb)); return -r['loglik']
        res = minimize(obj, np.log(np.asarray(st, float)), method='Nelder-Mead', options={'xatol': 1e-3, 'fatol': 1e-4, 'maxiter': 400})
        r = fit_soe_mle(t, hist, T, betas=np.exp(res.x)); r['betas'] = np.exp(res.x)
        if best is None or r['loglik'] > best['loglik']:
            best = r
    return best


def task_nested():
    bund = load_bund(); rows, done = resume('R_nested_oos.csv')
    starts = {2: [[1e3, 1e-1], [1e2, 1.0], [1e3, 10.0]], 3: [[1e3, 10.0, 1e-1], [1e2, 1.0, 1e-2], [1e4, 1e2, 1.0]]}
    for day, D in bund.items():
        if str(day) in done: continue
        T = session_T(D)
        for proc in ['P', 'Tr']:
            t = D[proc][D[proc] > 0]; N = len(t)
            Ttr = 0.7 * T; ttr = t[t < Ttr]; tte = t[t >= Ttr] - Ttr; hte = ttr - Ttr; Tte = T - Ttr
            cands = {}
            fe = fit_exp_mle(t, np.empty(0), T); cands['K1_exp'] = (fe['mu'], np.array([fe['alpha']]), np.array([fe['beta']]), fe['loglik'], 3)
            for K in (2, 3):
                r = fit_soe_free(t, np.empty(0), T, K, starts[K]); cands[f'K{K}_free'] = (r['mu'], r['alpha'], r['betas'], r['loglik'], 1 + 2 * K)
            fs = fit_soe_mle(t, np.empty(0), T, betas=BETAS7); cands['K7_grid'] = (fs['mu'], fs['alpha'], BETAS7, fs['loglik'], 8)
            # held-out: refit each candidate on [0,0.7T) (free rates refitted), evaluate on [0.7T,T) with observed history
            fe2 = fit_exp_mle(ttr, np.empty(0), Ttr); oos = {'K1_exp': soe_loglik(tte, hte, Tte, fe2['mu'], [fe2['alpha']], [fe2['beta']])}
            for K in (2, 3):
                r2 = fit_soe_free(ttr, np.empty(0), Ttr, K, starts[K]); oos[f'K{K}_free'] = soe_loglik(tte, hte, Tte, r2['mu'], r2['alpha'], r2['betas'])
            fs2 = fit_soe_mle(ttr, np.empty(0), Ttr, betas=BETAS7); oos['K7_grid'] = soe_loglik(tte, hte, Tte, fs2['mu'], fs2['alpha'], BETAS7)
            for name, (mu, al, be, ll, p) in cands.items():
                rows.append(dict(day=day, proc=proc, model=name, N=N, n=float(np.sum(al / be)), loglik=ll, nparam=p, aic=2 * p - 2 * ll, bic=p * np.log(N) - 2 * ll,
                                 oos_ll_per_event=oos[name] / max(len(tte), 1), N_test=len(tte), betas=';'.join(f'{b:.4g}' for b in np.sort(be)), masses=';'.join(f'{m:.3g}' for m in (al / be)[np.argsort(be)])))
        pd.DataFrame(rows).to_csv(OUT / 'R_nested_oos.csv', index=False)
        print('nested', day, flush=True)


# ================================================================ TASK: simulation controls
def simulate_soe_burnin(T, mu, betas, masses, rng, burn):
    """stationary sample on [0,T]: simulate on [-burn, T], return (events>=0, history<0)."""
    s = A.simulate_soe_cluster(T + burn, mu, betas, masses, rng) - burn
    return s[s >= 0], s[s < 0]


def fit_pair(t, hist, T, betas=BETAS7):
    fe = fit_exp_mle(t, hist, T); fs = fit_soe_mle(t, hist, T, betas=betas)
    return fe, fs


def task_sim(reps=30):
    rng = np.random.default_rng(SEED); rows, _ = resume('R_sim_controls.csv')
    have = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['design', 'profile', 'n_true', 'T'])
    def finished(design, profile, ntrue, T, need):
        m = (have.design == design) & (have.profile.astype(str) == str(profile)) & (np.isclose(have.n_true.astype(float), ntrue)) & (np.isclose(have['T'].astype(float), T))
        return int(m.sum()) >= need
    W = {k: v / v.sum() for k, v in {'fast': np.array([.01, .02, .04, .08, .15, .30, .40]), 'balanced': np.ones(7), 'slow': np.array([.40, .30, .15, .08, .04, .02, .01])}.items()}
    def add(**kw):
        rows.append(kw); pd.DataFrame(rows).to_csv(OUT / 'R_sim_controls.csv', index=False)
    # (a) stress map with burn-in (stationary start) and history conditioning, T=4000 ; (e) long T=50400 for balanced/slow at n=.85,.95 (10 reps)
    for prof, w in W.items():
        for ntrue in [0.50, 0.70, 0.85, 0.95]:
            masses = ntrue * w; mu = 1.5 * (1 - ntrue)
            g = soe_slowest_rate(masses * BETAS7, BETAS7); burn = min(10.0 / max(g, 1e-6), 20000.0)
            for T, R in [(4000.0, reps), (50400.0, 10 if (prof != 'fast' and ntrue >= 0.85) else 0)]:
                if R == 0 or finished('grid_burnin', prof, ntrue, T, R): continue
                rng = np.random.default_rng(SEED + hash((prof, ntrue, T)) % 100000)
                for r in range(R):
                    t, h = simulate_soe_burnin(T, mu, BETAS7, masses, rng, burn)
                    if len(t) < 50: continue
                    fe, fs = fit_pair(t, h, T); fe0, fs0 = fit_pair(t, np.empty(0), T)
                    add(design='grid_burnin', profile=prof, n_true=ntrue, T=T, rep=r, N=len(t), burn=burn, n_exp=fe['n'], beta_exp=fe['beta'], n_soe=fs['n'], n_exp_nohist=fe0['n'], n_soe_nohist=fs0['n'])
            print('sim grid', prof, ntrue, flush=True)
    # (b) correctly specified single exponential DGP
    for beta in [1.0, 100.0]:
        for ntrue in [0.50, 0.70, 0.85, 0.95]:
            if finished('exp_dgp', f'beta{beta:g}', ntrue, 4000.0, reps): continue
            mu = 1.5 * (1 - ntrue); burn = min(10.0 / (beta * (1 - ntrue)), 20000.0)
            for r in range(reps):
                t, h = simulate_soe_burnin(4000.0, mu, np.array([beta]), np.array([ntrue]), rng, burn)
                if len(t) < 50: continue
                fe, fs = fit_pair(t, h, 4000.0)
                add(design='exp_dgp', profile=f'beta{beta:g}', n_true=ntrue, T=4000.0, rep=r, N=len(t), burn=burn, n_exp=fe['n'], beta_exp=fe['beta'], n_soe=fs['n'])
        print('sim exp dgp', beta, flush=True)
    # (c) off-grid multiscale DGPs: geometric midpoints (6 rates) and a dense 25-rate power-law-like kernel (mass ~ beta^-0.3 over 1e-2..1e4)
    offgrid = {'offgrid6': (np.geomspace(10 ** -1.5, 10 ** 3.5, 6), np.ones(6)), 'powerlaw25': (np.geomspace(1e-2, 1e4, 25), np.geomspace(1e-2, 1e4, 25) ** -0.3)}
    for name, (bt, wt) in offgrid.items():
        wt = wt / wt.sum()
        for ntrue in [0.50, 0.70, 0.85, 0.95]:
            if finished(name, name, ntrue, 4000.0, reps): continue
            masses = ntrue * wt; mu = 1.5 * (1 - ntrue); g = soe_slowest_rate(masses * bt, bt); burn = min(10.0 / max(g, 1e-6), 20000.0)
            for r in range(reps):
                t, h = simulate_soe_burnin(4000.0, mu, bt, masses, rng, burn)
                if len(t) < 50: continue
                fe, fs = fit_pair(t, h, 4000.0)
                add(design=name, profile=name, n_true=ntrue, T=4000.0, rep=r, N=len(t), burn=burn, n_exp=fe['n'], beta_exp=fe['beta'], n_soe=fs['n'])
        print('sim offgrid', name, flush=True)
    # (d) non-self-exciting controls with time-varying / stochastic intensity, T = 50400 s (one session), fitted with constant and 30-min spline backgrounds
    def cox_logou(T, rng, mean_rate=0.6, tau=900.0, sig=0.5):
        dt = 1.0; n = int(T / dt); x = np.zeros(n); x[0] = rng.normal(0, sig)
        a = np.exp(-dt / tau); s = sig * np.sqrt(1 - a ** 2)
        for i in range(1, n): x[i] = a * x[i - 1] + s * rng.normal()
        lam = mean_rate * np.exp(x - sig ** 2 / 2); lmax = lam.max()
        cand = np.sort(rng.uniform(0, T, rng.poisson(lmax * T))); keep = rng.uniform(0, 1, len(cand)) < lam[np.minimum((cand / dt).astype(int), n - 1)] / lmax
        return cand[keep]
    def regime(T, rng, rates=(0.3, 0.9), mean_dur=900.0):
        ev = []; s = 0; tcur = 0.0
        while tcur < T:
            d = rng.exponential(mean_dur); a, b = tcur, min(T, tcur + d)
            ev.append(np.sort(rng.uniform(a, b, rng.poisson(rates[s] * (b - a))))); tcur = b; s = 1 - s
        return np.concatenate(ev)
    for name, gen in [('cox_logOU', cox_logou), ('regime2', regime)]:
        if finished(name, name, 0.0, 50400.0, 20): continue
        for r in range(20):
            t = gen(50400.0, rng); T = 50400.0
            fe, fs = fit_pair(t, np.empty(0), T)
            fle = A.flex_background_fit(t, T, model='exp', knot_step=1800., penalty=1e-3, beta_hint=fe['beta'], raw_fit=fe)
            fls = A.flex_background_fit(t, T, model='soe', knot_step=1800., penalty=1e-3, raw_fit=fs)
            add(design=name, profile=name, n_true=0.0, T=T, rep=r, N=len(t), burn=0.0, n_exp=fe['n'], beta_exp=fe['beta'], n_soe=fs['n'], n_exp_flex=fle['n'], n_soe_flex=fls['n'])
        print('sim nonSE', name, flush=True)
    # (f) slow n=0.95 cell: multistart diagnostics on the archived design (no burn-in, T=4000) and with burn-in
    masses = 0.95 * W['slow']; mu = 1.5 * 0.05
    for r in range(0 if finished('slow95_multistart', 'slow', 0.95, 4000.0, reps) else reps):
        t = A.simulate_soe_cluster(4000.0, mu, BETAS7, masses, rng)
        fits = []
        for lb0 in np.linspace(-2, 8, 6):
            x0 = np.log([0.5 * len(t) / 4000.0, 0.5, 1.0]); x0[2] = lb0; x0[1] = lb0 + np.log(0.5)
            res = minimize(lambda th: __import__('hawkes').exp_loglik(th, np.ascontiguousarray(t), np.empty(0), 4000.0), x0, jac=True, method='L-BFGS-B', bounds=[(-20, 10), (-20, 10), (-9, 9)])
            m_, a_, b_ = np.exp(res.x); fits.append((-res.fun, a_ / b_, b_, res.success))
        fits.sort(key=lambda z: -z[0]); ll = np.array([z[0] for z in fits])
        add(design='slow95_multistart', profile='slow', n_true=0.95, T=4000.0, rep=r, N=len(t), burn=0.0, n_exp=fits[0][1], beta_exp=fits[0][2], ll_best=fits[0][0], ll_gap_2nd_mode=float(ll[0] - ll[-1]),
            n_modes=int(np.sum(np.abs(np.log(np.array([z[2] for z in fits])) - np.log(fits[0][2])) > 1.0) > 0) + 1, all_success=int(all(z[3] for z in fits)),
            n_alt=[z[1] for z in fits if abs(np.log(z[2]) - np.log(fits[0][2])) > 1.0][:1][0] if any(abs(np.log(z[2]) - np.log(fits[0][2])) > 1.0 for z in fits) else np.nan)
    print('sim slow95 done', flush=True)


# ================================================================ TASK: refitted parametric bootstrap of diagnostics
def diag_stats(dlt):
    u = 1 - np.exp(-dlt); ks = kstest(u, 'uniform').statistic; cvm = cramervonmises(u, 'uniform').statistic
    z = dlt - dlt.mean(); den = z @ z
    ac = [float(z[:-l] @ z[l:] / den) for l in (1, 5, 10)]
    return ks, cvm, ac


@njit(cache=True)
def _sim_cluster_nb(T0, T1, mu, betas, masses, seed, cap):
    """compiled cluster simulation on [T0,T1] (used only by the bootstrap; numba RNG seeded per replication)."""
    np.random.seed(seed)
    n_imm = np.random.poisson(mu * (T1 - T0))
    if n_imm > cap:
        n_imm = cap
    buf = np.empty(cap)
    for i in range(n_imm):
        buf[i] = T0 + (T1 - T0) * np.random.random()
    n = n_imm; q = 0
    while q < n:
        p = buf[q]; q += 1
        for k in range(betas.shape[0]):
            m = np.random.poisson(masses[k])
            for j in range(m):
                c = p + np.random.exponential(1.0 / betas[k])
                if c < T1 and n < cap:
                    buf[n] = c; n += 1
    return np.sort(buf[:n])


def simulate_fast(T, mu, betas, masses, seed, burn, cap=3000000):
    s = _sim_cluster_nb(-float(burn), float(T), float(mu), np.ascontiguousarray(betas, dtype=np.float64),
                        np.ascontiguousarray(masses, dtype=np.float64), int(seed), int(cap))
    return s[s >= 0], s[s < 0]


def task_boot(B=60):
    """Refitted parametric bootstrap: simulate from each fitted full-session model (stationary burn-in), refit both kernels
    without history (as for the data), record diagnostics of the true-model fit (null band) and of the other kernel
    (x_*: for SoE truth this is the known-misspecification benchmark)."""
    bund = load_bund(); rows, _ = resume('R_bootstrap_diagnostics.csv')
    done = set((str(r['day']), r['proc']) for r in rows)
    for di, (day, D) in enumerate(bund.items()):
        T = session_T(D)
        for pi, proc in enumerate(['P', 'Tr']):
            if (str(day), proc) in done:
                continue
            t0 = time.time(); t = D[proc][D[proc] > 0]
            fe = fit_exp_mle(t, np.empty(0), T); fs = fit_soe_mle(t, np.empty(0), T, betas=BETAS7)
            de = A.soe_residuals(t, T, fe['mu'], np.array([fe['alpha']]), np.array([fe['beta']]))
            ds = A.soe_residuals(t, T, fs['mu'], fs['alpha'], fs['betas'])
            obs = {'Exp': diag_stats(de), 'SoE': diag_stats(ds)}
            specs = [('Exp', fe['mu'], np.array([fe['alpha']]), np.array([fe['beta']]), fe['n']), ('SoE', fs['mu'], fs['alpha'], BETAS7, fs['n'])]
            for mi, (model, mu, al, be, nfit) in enumerate(specs):
                masses = al / be; g = soe_slowest_rate(al, be); burn = min(10.0 / max(g, 1e-6), 20000.0)
                for b in range(B):
                    seed = (SEED + 1000003 * di + 10007 * pi + 1009 * mi + b) % 4294967291
                    ts, _ = simulate_fast(T, mu, be, masses, seed, burn)
                    if len(ts) < 100:
                        continue
                    f_e = fit_exp_mle(ts, np.empty(0), T); f_s = fit_soe_mle(ts, np.empty(0), T, betas=BETAS7)
                    d_e = A.soe_residuals(ts, T, f_e['mu'], np.array([f_e['alpha']]), np.array([f_e['beta']]))
                    d_s = A.soe_residuals(ts, T, f_s['mu'], f_s['alpha'], f_s['betas'])
                    own, other = (d_e, d_s) if model == 'Exp' else (d_s, d_e)
                    k1, c1, a1 = diag_stats(own); k2, c2, a2 = diag_stats(other)
                    rows.append(dict(day=day, proc=proc, model=model, rep=b, N=len(ts), n_fit=nfit,
                                     n_hat=(f_e['n'] if model == 'Exp' else f_s['n']), n_cross=(f_s['n'] if model == 'Exp' else f_e['n']),
                                     ks=k1, cvm=c1, acf1=a1[0], acf5=a1[1], acf10=a1[2], x_ks=k2, x_cvm=c2, x_acf1=a2[0], x_acf10=a2[2],
                                     obs_ks=obs[model][0], obs_cvm=obs[model][1], obs_acf1=obs[model][2][0], obs_acf5=obs[model][2][1], obs_acf10=obs[model][2][2]))
            pd.DataFrame(rows).to_csv(OUT / 'R_bootstrap_diagnostics.csv', index=False)
            print('boot', day, proc, '%.0fs' % (time.time() - t0), flush=True)


# ================================================================ TASK: adequacy of spline-background and bivariate candidates
def task_adeq():
    """Time-rescaling diagnostics for the 30-min spline-background fits (both kernels) and the common-rate bivariate fit;
    observed-information SEs of n+ and n- conditional on the profiled common rate (R2-8)."""
    bund = load_bund(); rows = []
    for day, D in bund.items():
        T = session_T(D)
        for proc in ['P', 'Tr']:
            t = D[proc][D[proc] > 0]
            raw_e = fit_exp_mle(t, np.empty(0), T); raw_s = fit_soe_mle(t, np.empty(0), T, betas=BETAS7)
            knots = np.r_[np.repeat(0.0, 4), np.arange(1800.0, T, 1800.0), np.repeat(T, 4)]
            for model in ['exp', 'soe']:
                f = A.flex_background_fit(t, T, model=model, knot_step=1800., penalty=1e-3, beta_hint=raw_e['beta'], raw_fit=(raw_s if model == 'soe' else raw_e))
                ant = BSpline(knots, f['coeff'], 3, extrapolate=False).antiderivative()
                bg = ant(np.clip(t, 0, T)) - ant(0.0)
                be = np.asarray(f['betas'], float); al = np.asarray(f['alpha'], float)
                R = _soe_R(np.ascontiguousarray(t), np.empty(0), be)
                Lam = bg + np.sum((al / be)[None, :] * (np.arange(len(t))[:, None] - R), axis=1)
                d = np.diff(np.r_[0.0, Lam]); d = d[d > 0]
                k, c, a = diag_stats(d)
                rows.append(dict(day=day, proc=proc, model=model + '_spline', n=f['n'], ks=k, cvm=c, acf1=a[0], acf5=a[1], acf10=a[2]))
        up = D['up'][D['up'] > 0]; dn = D['dn'][D['dn'] > 0]
        fb = A.fit_bivar_common_beta(up, dn, T); mu, a_s, a_c, b = fb['mu'], fb['alpha_s'], fb['alpha_c'], fb['beta']
        times = np.r_[up, dn]; typ = np.r_[np.ones(len(up)), -np.ones(len(dn))]; o = np.argsort(times); times = times[o]; typ = typ[o]
        n_ = len(times); Sp = np.zeros(n_); Sm = np.zeros(n_); cu = np.zeros(n_); cd = np.zeros(n_); sp = sm = 0.0; nu = nd = 0; prev = 0.0
        for i in range(n_):
            e = np.exp(-b * (times[i] - prev)); sp *= e; sm *= e
            Sp[i] = sp; Sm[i] = sm; cu[i] = nu; cd[i] = nd
            if typ[i] > 0: sp += 1.0; nu += 1
            else: sm += 1.0; nd += 1
            prev = times[i]
        Lp = mu * times + (a_s / b) * (cu - Sp) + (a_c / b) * (cd - Sm)
        Lm = mu * times + (a_s / b) * (cd - Sm) + (a_c / b) * (cu - Sp)
        res = []
        for sgn, L in [(1, Lp), (-1, Lm)]:
            d = np.diff(np.r_[0.0, L[typ == sgn]]); res.append(d[d > 0])
        k, c, _ = diag_stats(np.r_[res[0], res[1]]); a = np.mean([diag_stats(r)[2] for r in res], axis=0)
        own = np.where(typ > 0, Sp, Sm); cross = np.where(typ > 0, Sm, Sp); lam = mu + a_s * own + a_c * cross
        X = np.c_[np.ones(n_), own, cross]; H = (X / lam[:, None] ** 2).T @ X
        try:
            cov = np.linalg.inv(H)
            se_p = float(np.sqrt(np.array([0, 1 / b, 1 / b]) @ cov @ np.array([0, 1 / b, 1 / b])))
            se_m = float(np.sqrt(np.array([0, 1 / b, -1 / b]) @ cov @ np.array([0, 1 / b, -1 / b])))
        except np.linalg.LinAlgError:
            se_p = se_m = np.nan
        rows.append(dict(day=day, proc='UD', model='bivar_common', n=fb['n_plus'], n_minus=fb['n_minus'], se_nplus=se_p, se_nminus=se_m,
                         ks=k, cvm=c, acf1=a[0], acf5=a[1], acf10=a[2]))
        pd.DataFrame(rows).to_csv(OUT / 'R_adequacy_extra.csv', index=False)
        print('adeq', day, flush=True)


# ================================================================ well-scaled SoE fit (masses parametrisation) and bootstrap refinement
def fit_soe_masses(t, hist, T, betas, extra_starts=True):
    """MLE over (mu, m_k = alpha_k/beta_k): same concave problem as fit_soe_mle, but well scaled; starts from the
    fit_soe_mle solution and from equal masses, tight tolerances; returns the best."""
    t = np.ascontiguousarray(t, float); hist = np.ascontiguousarray(hist, float); betas = np.asarray(betas, float)
    base = fit_soe_mle(t, hist, T, betas=betas)
    R = _soe_R(t, hist, betas) * betas[None, :]
    S = np.array([np.sum(1 - np.exp(-b * (T - t))) / b + (np.sum(np.exp(b * hist) - np.exp(-b * (T - hist))) / b if len(hist) else 0.0) for b in betas]) * betas
    def f(x):
        mu = x[0]; m = x[1:]; lam = mu + R @ m
        if np.any(lam <= 0): return 1e100, np.zeros_like(x)
        inv = 1 / lam
        return -(np.log(lam).sum() - mu * T - m @ S), -np.r_[inv.sum() - T, R.T @ inv - S]
    K = len(betas); rate = len(t) / T
    starts = [np.r_[base['mu'], base['alpha'] / betas]]
    if extra_starts:
        starts.append(np.r_[0.5 * rate, np.full(K, 0.5 / K)])
    best = None
    for x0 in starts:
        res = minimize(f, np.maximum(x0, 1e-10), jac=True, method='L-BFGS-B', bounds=[(1e-12, None)] * (K + 1),
                       options={'ftol': 1e-15, 'gtol': 1e-9, 'maxiter': 50000, 'maxfun': 50000})
        if best is None or res.fun < best.fun: best = res
    mu = best.x[0]; m = best.x[1:]
    return dict(mu=mu, alpha=m * betas, betas=betas, n=float(m.sum()), loglik=-best.fun, success=best.success,
                base_n=base['n'], base_loglik=base['loglik'])


def task_soecheck():
    """Is every headline full-session SoE fit at the optimum? Compare fit_soe_mle with the well-scaled refit."""
    bund = load_bund(); rows = []
    for day, D in bund.items():
        T = session_T(D)
        for proc in ['P', 'Tr']:
            t = D[proc][D[proc] > 0]; f = fit_soe_masses(t, np.empty(0), T, BETAS7)
            rows.append(dict(day=day, proc=proc, n_base=f['base_n'], n_tight=f['n'], dll=f['loglik'] - f['base_loglik'], **{f'm{k}': f['alpha'][k] / BETAS7[k] for k in range(7)}))
    df = pd.DataFrame(rows); df.to_csv(OUT / 'R_soe_optimum_check.csv', index=False); return df


def task_bootfix():
    """Re-fit the SoE-truth bootstrap replications with the well-scaled optimiser (same seeds, same simulations)."""
    bund = load_bund(); rows, _ = resume('R_bootstrap_soe_refit.csv')
    done = set((str(r['day']), r['proc']) for r in rows)
    for di, (day, D) in enumerate(bund.items()):
        T = session_T(D)
        for pi, proc in enumerate(['P', 'Tr']):
            if (str(day), proc) in done: continue
            t0 = time.time(); t = D[proc][D[proc] > 0]
            fs = fit_soe_mle(t, np.empty(0), T, betas=BETAS7)          # bootstrap truth = archived headline fit (as in task_boot)
            masses = fs['alpha'] / BETAS7; g = soe_slowest_rate(fs['alpha'], BETAS7); burn = min(10.0 / max(g, 1e-6), 20000.0)
            for b in range(60):
                seed = (SEED + 1000003 * di + 10007 * pi + 1009 * 1 + b) % 4294967291
                ts, _ = simulate_fast(T, fs['mu'], BETAS7, masses, seed, burn)
                f = fit_soe_masses(ts, np.empty(0), T, BETAS7, extra_starts=False)
                d = A.soe_residuals(ts, T, f['mu'], f['alpha'], f['betas']); k1, c1, a1 = diag_stats(d)
                rows.append(dict(day=day, proc=proc, rep=b, N=len(ts), n_fit=fs['n'], n_hat=f['n'], n_hat_base=f['base_n'], dll=f['loglik'] - f['base_loglik'],
                                 ks=k1, cvm=c1, acf1=a1[0], acf5=a1[1], acf10=a1[2]))
            pd.DataFrame(rows).to_csv(OUT / 'R_bootstrap_soe_refit.csv', index=False)
            print('bootfix', day, proc, '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    task = sys.argv[1]; t0 = time.time(); print('START', task, flush=True)
    {'soecheck': task_soecheck, 'bootfix': task_bootfix, 'adeq': task_adeq, 'timestamp': task_timestamp, 'history': task_history, 'bivar': task_bivar, 'penalty': task_penalty, 'nested': task_nested, 'sim': task_sim, 'boot': task_boot}[task]()
    print('DONE', task, '%.0f s' % (time.time() - t0), flush=True)
