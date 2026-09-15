"""
E1 (Layer 1, real data): Euro-Bund futures, 20 trading days of April 2014 (tick-datasets, X-DataInitiative),
four event streams: mid-price up, mid-price down, buyer-initiated trades, seller-initiated trades.

Outputs (tables/):
  E1_window_fits.csv     one row per (day, process, window length, window start): MLE, Laplace posterior, SoE kernel
  E1_hb_estimator.csv    Hardiman-Bouchaud moment estimator per (day, process, bin width)
  E1_daily_summary.csv   per-day full-session fits + residual KS
  E1_h3prime.csv         per 1h-window: empirical bin-count autocorrelations vs exact Hawkes prediction
"""
import sys, os, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hawkes import (fit_exp_mle, fit_exp_laplace, fit_soe_mle, hb_estimator, rescaled_residuals_exp,
                    exp_hawkes_bin_autocorr, bin_counts_autocorr, soe_bin_autocorr, soe_slowest_rate, rescaled_residuals_soe)
BETAS = np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])   # s^-1, geometric grid spanning 0.1 ms .. 100 s

ROOT = os.path.join(os.path.dirname(__file__), "..")
d = np.load(os.path.join(ROOT, "data", "bund.npz"), allow_pickle=True)
days = sorted(d.files)

def load_day(k):
    arr = d[k]
    up, dn, ta, tb = [np.sort(np.asarray(a, dtype=float)) for a in arr]
    P = np.sort(np.concatenate([up, dn]))     # mid-price changes
    Tr = np.sort(np.concatenate([ta, tb]))    # trades that do not move the mid
    return dict(up=up, dn=dn, P=P, Tr=Tr)

WINDOWS = {"10min": 600.0, "30min": 1800.0, "1h": 3600.0, "day": None}
HIST_LEN = 1800.0   # history used for the intensity at the window start
rows, hb_rows, daily_rows, h3_rows = [], [], [], []
t0 = time.time()
for k in days:
    D = load_day(k)
    T_day = max(D["P"][-1], D["Tr"][-1])
    for proc in ["P", "Tr"]:
        t_all = D[proc]
        # ---- full-session fits + diagnostics ----
        t_in = t_all[t_all > 0]
        fm = fit_exp_mle(t_in, np.empty(0), T_day)
        fl = fit_exp_laplace(t_in, np.empty(0), T_day)
        fs = fit_soe_mle(t_in, np.empty(0), T_day, betas=BETAS)
        ks_stat, ks_p, _ = rescaled_residuals_exp(t_in, np.empty(0), T_day, fm["mu"], fm["alpha"], fm["beta"])
        ks_soe, ksp_soe = rescaled_residuals_soe(t_in, T_day, fs["mu"], fs["alpha"], fs["betas"])
        gmin = soe_slowest_rate(fs["alpha"], fs["betas"])
        daily_rows.append(dict(day=k, proc=proc, N=len(t_in), T=T_day, n_mle=fm["n"], beta_mle=fm["beta"], mu_mle=fm["mu"],
                               n_bayes=fl["n_mean"], n_sd=fl["n_sd"], n_lo=fl["n_lo90"], n_hi=fl["n_hi90"],
                               n_soe=fs["n"], m_soe=fs["m"], gamma_min_soe=gmin, ll_exp=fm["loglik"], ll_soe=fs["loglik"], ks_stat=ks_stat, ks_p=ks_p,
                               ks_soe=ks_soe, ksp_soe=ksp_soe, **{f"a{j}": fs["alpha"][j] for j in range(len(BETAS))}))
        # ---- HB moment estimator ----
        for W in [1, 5, 10, 30, 60, 300, 600]:
            nh, _ = hb_estimator(t_in, T_day, W)
            hb_rows.append(dict(day=k, proc=proc, W=W, n_hb=nh))
        # ---- rolling windows ----
        for wname, W in WINDOWS.items():
            if W is None:
                continue
            starts = np.arange(0.0, T_day - W + 1e-9, W)
            for s in starts:
                m_in = (t_all >= s) & (t_all < s + W)
                m_h = (t_all >= s - HIST_LEN) & (t_all < s)
                t_in = t_all[m_in] - s
                hist = t_all[m_h] - s
                if len(t_in) < 30:
                    continue
                fm = fit_exp_mle(t_in, hist, W)
                fl = fit_exp_laplace(t_in, hist, W, beta0=max(fm["beta"], 1e-3))
                row = dict(day=k, proc=proc, window=wname, W=W, start=s, N=len(t_in), n_mle=fm["n"], beta_mle=fm["beta"],
                           mu_mle=fm["mu"], n_bayes=fl["n_mean"], n_map=fl["n_map"], n_sd=fl["n_sd"], n_lo=fl["n_lo90"],
                           n_hi=fl["n_hi90"], sd_logbeta=fl["sd_logbeta"], ll_exp=fm["loglik"])
                fs = fit_soe_mle(t_in, hist, W, betas=BETAS)
                row.update(n_soe=fs["n"], m_soe=fs["m"], ll_soe=fs["loglik"], gamma_min_soe=soe_slowest_rate(fs["alpha"], fs["betas"]))
                rows.append(row)
                # ---- H3': critical slowing down in activity (bin-count autocorrelation) ----
                if wname == "1h":
                    Delta = 10.0
                    emp, nb = bin_counts_autocorr(t_in, W, Delta, kmax=6)
                    Lam_s, th = soe_bin_autocorr(fs["mu"], fs["alpha"], fs["betas"], Delta, kmax=6)
                    gmin = soe_slowest_rate(fs["alpha"], fs["betas"])
                    rec = dict(day=k, proc=proc, start=s, Delta=Delta, N=len(t_in), n_exp=fm["n"], beta_exp=fm["beta"],
                               n_soe=fs["n"], m_soe=fs["m"], n_bayes=fl["n_mean"], n_sd=fl["n_sd"], gamma_min=gmin,
                               ar1_th=np.exp(-gmin * Delta) if gmin > 0 else np.nan)
                    rec.update({f"rho{j+1}_emp": emp[j] for j in range(6)})
                    rec.update({f"rho{j+1}_th": th[j] for j in range(6)})
                    edges = np.arange(0, W + 1e-9, Delta)
                    if proc == "P":
                        cu = np.histogram(D["up"][(D["up"] >= s) & (D["up"] < s + W)] - s, bins=edges)[0]
                        cd = np.histogram(D["dn"][(D["dn"] >= s) & (D["dn"] < s + W)] - s, bins=edges)[0]
                        r = (cu - cd).astype(float); r -= r.mean(); v = np.sum(r * r)
                        rec.update({f"rho{j+1}_ret": (np.sum(r[:-(j+1)] * r[(j+1):]) / v if v > 0 else np.nan) for j in range(6)})
                    h3_rows.append(rec)
    print(k, "done  %.0fs" % (time.time() - t0), flush=True)

os.makedirs(os.path.join(ROOT, "tables"), exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "tables", "E1_window_fits.csv"), index=False)
pd.DataFrame(hb_rows).to_csv(os.path.join(ROOT, "tables", "E1_hb_estimator.csv"), index=False)
pd.DataFrame(daily_rows).to_csv(os.path.join(ROOT, "tables", "E1_daily_summary.csv"), index=False)
pd.DataFrame(h3_rows).to_csv(os.path.join(ROOT, "tables", "E1_h3prime.csv"), index=False)
print("saved; total time %.0fs" % (time.time() - t0))
