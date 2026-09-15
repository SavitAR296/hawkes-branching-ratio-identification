"""
Layer 1: Bayesian Hawkes endogeneity.

Exponential kernel  phi(s) = alpha * exp(-beta s),  n = alpha/beta.
Sum-of-exponentials kernel  phi(s) = sum_k alpha_k exp(-beta_k s) with a fixed
geometric grid of beta_k (approximates slowly decaying / power-law kernels).

Everything is O(N) via the Ozaki (1979) recursion.  Events *before* the window
start are used as history (they enter the intensity and the compensator but not
the log-likelihood sum), which removes most of the left-boundary bias of short
windows.
"""
import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.stats import kstest, beta as beta_dist
from scipy.special import expit, logit

# ----------------------------------------------------------------------------
# exponential kernel: log-likelihood and analytic gradient (wrt mu, alpha, beta)
# ----------------------------------------------------------------------------
@njit(cache=True)
def _exp_loglik_grad(t, hist, T, mu, alpha, beta):
    """t: event times in [0,T] (sorted); hist: event times < 0 (history, sorted).
    Returns (loglik, dmu, dalpha, dbeta)."""
    n = t.shape[0]
    # history contribution at time 0:  R0 = sum_j exp(beta*t_j),  dR0 = sum_j t_j exp(beta t_j)  (t_j<0)
    R = 0.0
    dR = 0.0
    for j in range(hist.shape[0]):
        e = np.exp(beta * hist[j])
        R += e
        dR += hist[j] * e
    ll = 0.0
    g_mu = 0.0
    g_a = 0.0
    g_b = 0.0
    prev = 0.0
    # compensator part of history: (alpha/beta) * sum_j [exp(beta t_j) - exp(-beta (T - t_j))]
    comp_h = 0.0
    dcomp_h = 0.0  # derivative wrt beta of sum_j [exp(beta t_j) - exp(-beta(T-t_j))]
    for j in range(hist.shape[0]):
        e0 = np.exp(beta * hist[j])
        e1 = np.exp(-beta * (T - hist[j]))
        comp_h += e0 - e1
        dcomp_h += hist[j] * e0 + (T - hist[j]) * e1
    for i in range(n):
        d = t[i] - prev
        e = np.exp(-beta * d)
        # R_i = e^{-beta d}(R_{i-1}) evaluated just before t_i (excluding t_i itself)
        dR = e * (dR - d * R)
        R = e * R
        lam = mu + alpha * R
        ll += np.log(lam)
        g_mu += 1.0 / lam
        g_a += R / lam
        g_b += alpha * dR / lam
        # add current event to the state (for the next step)
        R += 1.0
        prev = t[i]
    # compensator over [0,T] of the events inside
    S = 0.0
    dS = 0.0
    for i in range(n):
        e1 = np.exp(-beta * (T - t[i]))
        S += 1.0 - e1
        dS += (T - t[i]) * e1
    S_tot = S + comp_h
    dS_tot = dS + dcomp_h
    ll += -mu * T - (alpha / beta) * S_tot
    g_mu += -T
    g_a += -(1.0 / beta) * S_tot
    g_b += (alpha / beta**2) * S_tot - (alpha / beta) * dS_tot
    return ll, g_mu, g_a, g_b


def exp_loglik(theta, t, hist, T):
    """theta = (log mu, log alpha, log beta). Returns -loglik and gradient (for minimize)."""
    mu, alpha, beta = np.exp(theta)
    ll, gm, ga, gb = _exp_loglik_grad(t, hist, T, mu, alpha, beta)
    g = np.array([gm * mu, ga * alpha, gb * beta])
    return -ll, -g


def fit_exp_mle(t, hist, T, x0=None):
    """MLE for the exponential kernel (unconstrained: n may exceed 1)."""
    t = np.ascontiguousarray(t, dtype=np.float64)
    hist = np.ascontiguousarray(hist, dtype=np.float64)
    N = len(t)
    if x0 is None:
        rate = max(N / T, 1e-6)
        x0 = np.log([0.5 * rate, 0.5 * 1.0, 1.0])
    best = None
    for start in [x0, x0 + np.array([0, 0.5, 1.0]), x0 + np.array([0.3, -0.5, -1.0])]:
        res = minimize(exp_loglik, start, args=(t, hist, T), jac=True, method="L-BFGS-B",
                       bounds=[(-20, 10), (-20, 10), (-9, 9)])
        if best is None or res.fun < best.fun:
            best = res
    mu, alpha, beta = np.exp(best.x)
    return dict(mu=mu, alpha=alpha, beta=beta, n=alpha / beta, loglik=-best.fun, N=N, T=T, success=best.success)


# ----------------------------------------------------------------------------
# Bayesian version (Laplace approximation): parameters (logit n, log beta, log mu),
# priors  n ~ Beta(2,2),  log beta ~ N(log beta0, 1),  log mu ~ N(log mu0, 2)
# ----------------------------------------------------------------------------
def _neg_logpost(x, t, hist, T, log_beta0, log_mu0):
    z, lb, lm = x
    n = expit(z)
    beta = np.exp(lb)
    mu = np.exp(lm)
    alpha = n * beta
    ll, gm, ga, gb = _exp_loglik_grad(t, hist, T, mu, alpha, beta)
    # chain rule: alpha = n*beta ; dalpha/dz = beta * n(1-n); d alpha / d lb = alpha
    d_z = ga * beta * n * (1 - n)
    d_lb = ga * alpha + gb * beta
    d_lm = gm * mu
    # log prior
    lp = (2 - 1) * np.log(n) + (2 - 1) * np.log(1 - n) + np.log(n * (1 - n))  # Beta(2,2) on n, Jacobian of logit
    lp += -0.5 * (lb - log_beta0) ** 2
    lp += -0.5 * (lm - log_mu0) ** 2 / 4.0
    g_lp = np.array([(1 - n) - n + (1 - 2 * n), -(lb - log_beta0), -(lm - log_mu0) / 4.0])
    f = -(ll + lp)
    g = -(np.array([d_z, d_lb, d_lm]) + g_lp)
    return f, g


def fit_exp_laplace(t, hist, T, beta0=1.0, mu0=None, x0=None):
    """MAP + Laplace posterior for (n, beta, mu). Returns posterior mean/sd of n in logit space and
    a 90% credible interval for n (Gaussian in logit space)."""
    t = np.ascontiguousarray(t, dtype=np.float64)
    hist = np.ascontiguousarray(hist, dtype=np.float64)
    N = len(t)
    rate = max(N / T, 1e-6)
    if mu0 is None:
        mu0 = 0.5 * rate
    args = (t, hist, T, np.log(beta0), np.log(mu0))
    if x0 is None:
        x0 = np.array([logit(0.5), np.log(beta0), np.log(mu0)])
    best = None
    for start in [x0, x0 + np.array([1.5, 1.0, 0.0]), x0 + np.array([-1.5, -1.0, 0.0])]:
        res = minimize(_neg_logpost, start, args=args, jac=True, method="L-BFGS-B",
                       bounds=[(-12, 12), (-9, 9), (-20, 10)])
        if best is None or res.fun < best.fun:
            best = res
    x = best.x
    # Hessian by central differences of the analytic gradient
    H = np.zeros((3, 3))
    eps = 1e-4
    for k in range(3):
        xp = x.copy(); xp[k] += eps
        xm = x.copy(); xm[k] -= eps
        gp = _neg_logpost(xp, *args)[1]
        gm = _neg_logpost(xm, *args)[1]
        H[:, k] = (gp - gm) / (2 * eps)
    H = 0.5 * (H + H.T)
    try:
        cov = np.linalg.inv(H)
        if np.any(np.diag(cov) <= 0):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        cov = np.full((3, 3), np.nan)
    n_map = expit(x[0]); beta = np.exp(x[1]); mu = np.exp(x[2])
    sd_z = np.sqrt(cov[0, 0]) if np.isfinite(cov[0, 0]) else np.nan
    # posterior moments of n by Gauss-Hermite in logit space
    if np.isfinite(sd_z):
        nodes, weights = np.polynomial.hermite_e.hermegauss(40)
        zs = x[0] + sd_z * nodes
        ns = expit(zs)
        w = weights / weights.sum()
        n_mean = np.sum(w * ns)
        n_sd = np.sqrt(np.sum(w * (ns - n_mean) ** 2))
        lo, hi = expit(x[0] - 1.6449 * sd_z), expit(x[0] + 1.6449 * sd_z)
    else:
        n_mean = n_map; n_sd = np.nan; lo = hi = np.nan
    return dict(n_map=n_map, n_mean=n_mean, n_sd=n_sd, n_lo90=lo, n_hi90=hi, beta=beta, mu=mu,
                sd_logbeta=np.sqrt(cov[1, 1]) if np.isfinite(cov[1, 1]) else np.nan,
                N=N, T=T, logpost=-best.fun, cov=cov, x=x)


# ----------------------------------------------------------------------------
# sum-of-exponentials kernel with fixed beta grid (captures slowly decaying kernels)
# ----------------------------------------------------------------------------
@njit(cache=True)
def _soe_R(t, hist, betas):
    """Returns matrix R[i,k] = sum_{j<i} exp(-beta_k (t_i - t_j)) including history,
    and S[k] = sum over all events (incl. history) of [exp(-beta_k max(0,-t_j)) ... ] compensator pieces."""
    n = t.shape[0]
    K = betas.shape[0]
    R = np.zeros((n, K))
    state = np.zeros(K)
    for k in range(K):
        s = 0.0
        for j in range(hist.shape[0]):
            s += np.exp(betas[k] * hist[j])
        state[k] = s
    prev = 0.0
    for i in range(n):
        d = t[i] - prev
        for k in range(K):
            state[k] = state[k] * np.exp(-betas[k] * d)
            R[i, k] = state[k]
        for k in range(K):
            state[k] += 1.0
        prev = t[i]
    return R


def fit_soe_mle(t, hist, T, betas=None):
    """MLE of mu, alpha_k >= 0 with fixed betas. n = sum alpha_k/beta_k."""
    t = np.ascontiguousarray(t, dtype=np.float64)
    hist = np.ascontiguousarray(hist, dtype=np.float64)
    if betas is None:
        betas = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
    betas = np.asarray(betas, dtype=np.float64)
    R = _soe_R(t, hist, betas)  # (N,K)
    # compensator integrals: for inside events sum_i (1-exp(-b(T-t_i)))/b ; for history sum_j (exp(b t_j)-exp(-b(T-t_j)))/b
    S = np.zeros(len(betas))
    for k, b in enumerate(betas):
        S[k] = np.sum(1 - np.exp(-b * (T - t))) / b
        if len(hist):
            S[k] += np.sum(np.exp(b * hist) - np.exp(-b * (T - hist))) / b
    N = len(t)

    def f(x):
        mu = x[0]; a = x[1:]
        lam = mu + R @ a
        if np.any(lam <= 0):
            return 1e30, np.zeros_like(x)
        ll = np.sum(np.log(lam)) - mu * T - a @ S
        g = np.empty_like(x)
        inv = 1.0 / lam
        g[0] = np.sum(inv) - T
        g[1:] = R.T @ inv - S
        return -ll, -g

    rate = N / T
    x0 = np.concatenate([[0.5 * rate], 0.5 * rate * 0.1 * betas / betas.sum() * len(betas)])
    x0 = np.maximum(x0, 1e-8)
    res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=[(1e-10, None)] * (1 + len(betas)))
    mu = res.x[0]; a = res.x[1:]
    n = np.sum(a / betas)
    # mean kernel time  m = int s phi / n
    m = np.sum(a / betas**2) / n if n > 0 else np.nan
    return dict(mu=mu, alpha=a, betas=betas, n=n, m=m, loglik=-res.fun, N=N, T=T, success=res.success)


# ----------------------------------------------------------------------------
# Hardiman-Bouchaud moment estimator, residuals, exact bin-count autocorrelation
# ----------------------------------------------------------------------------
def hb_estimator(t, T, W):
    """n_hat = 1 - sqrt(E N_W / Var N_W) from counts in bins of width W (stationary Hawkes:
    Var/E -> (1-n)^-2 as W -> infinity)."""
    edges = np.arange(0, T + 1e-9, W)
    counts = np.histogram(t, bins=edges)[0]
    m, v = counts.mean(), counts.var(ddof=1)
    if v <= m:
        return 0.0, counts
    return 1.0 - np.sqrt(m / v), counts


def rescaled_residuals_exp(t, hist, T, mu, alpha, beta):
    """Time-rescaling theorem: Lambda(t_i)-Lambda(t_{i-1}) should be iid Exp(1). Returns KS statistic, p-value."""
    t = np.asarray(t, dtype=float)
    # compensator at each event time: Lambda(t) = mu t + (alpha/beta) sum_{t_j<t} (1 - exp(-beta (t - t_j)))
    allt = np.concatenate([hist, t])
    Lam = np.empty(len(t))
    # O(N) recursion for sum_{t_j<t_i} exp(-beta (t_i - t_j)) over all previous events (incl history)
    R = 0.0; prev = 0.0
    cnt_hist = len(hist)
    # history state at time 0
    R = np.sum(np.exp(beta * hist)) if cnt_hist else 0.0
    for i, ti in enumerate(t):
        R = R * np.exp(-beta * (ti - prev))
        # number of previous events (incl history) = cnt_hist + i
        Lam[i] = mu * ti + (alpha / beta) * ((cnt_hist + i) - R)
        R += 1.0
        prev = ti
    # subtract the part of history compensator that accrued before 0
    # Lambda restricted to [0, t]: mu t + (alpha/beta) sum_j [exp(-beta max(0,-t_j)) - exp(-beta (t - t_j))]
    corr = (alpha / beta) * np.sum(1 - np.exp(beta * hist)) if cnt_hist else 0.0
    Lam = Lam - corr
    d = np.diff(np.concatenate([[0.0], Lam]))
    d = d[d > 0]
    ks = kstest(d, "expon")
    return ks.statistic, ks.pvalue, d


def exp_hawkes_bin_autocorr(mu, alpha, beta, Delta, kmax=3):
    """Exact autocovariance of bin counts of a stationary exponential Hawkes process
    (Bartlett spectrum; Hawkes 1971). Returns (Lambda, gamma, rho_1..rho_kmax).
       c(tau) = Lambda delta(tau) + c1 exp(-gamma |tau|),  gamma = beta - alpha,
       c1 = Lambda alpha (2 beta - alpha) / (2 gamma).
       Var N_Delta = Lambda Delta + 2 c1 [Delta/gamma - (1-e^{-gamma Delta})/gamma^2]
       Cov_k = c1 (1-e^{-gamma Delta})^2 e^{-gamma (k-1) Delta} / gamma^2,  k>=1.
       Hence rho_{k+1}/rho_k = exp(-beta(1-n)Delta)  exactly."""
    gamma = beta - alpha
    if gamma <= 0:
        return np.nan, gamma, [np.nan] * kmax
    Lam = mu * beta / gamma
    c1 = Lam * alpha * (2 * beta - alpha) / (2 * gamma)
    e = np.exp(-gamma * Delta)
    var = Lam * Delta + 2 * c1 * (Delta / gamma - (1 - e) / gamma**2)
    rhos = [c1 * (1 - e) ** 2 * np.exp(-gamma * (k - 1) * Delta) / gamma**2 / var for k in range(1, kmax + 1)]
    return Lam, gamma, rhos


def bin_counts_autocorr(t, T, Delta, kmax=3):
    edges = np.arange(0, T + 1e-9, Delta)
    c = np.histogram(t, bins=edges)[0].astype(float)
    c = c - c.mean()
    v = np.sum(c * c)
    if v == 0:
        return [np.nan] * kmax, len(c)
    return [np.sum(c[:-k] * c[k:]) / v for k in range(1, kmax + 1)], len(c)


# ----------------------------------------------------------------------------
# sum-of-exponentials kernel: Bartlett spectrum, exact bin-count autocorrelation,
# slowest relaxation rate, and time-rescaling residuals
# ----------------------------------------------------------------------------
def soe_spectrum(omega, mu, alphas, betas):
    """Bartlett spectral density f(omega) = Lambda/(2 pi) / |1 - phi_hat(omega)|^2 (stationary Hawkes)."""
    alphas = np.asarray(alphas); betas = np.asarray(betas)
    n = np.sum(alphas / betas)
    Lam = mu / (1 - n)
    phi_hat = np.sum(alphas[None, :] / (betas[None, :] + 1j * omega[:, None]), axis=1)
    return Lam / (2 * np.pi) / np.abs(1 - phi_hat) ** 2, Lam


def soe_bin_autocorr(mu, alphas, betas, Delta, kmax=6, n_omega=400000):
    """Exact autocorrelations rho_k of counts in bins of width Delta for a stationary Hawkes process
    with kernel sum_k alpha_k exp(-beta_k s), computed from the Bartlett spectrum:
        Cov(N_0, N_k) = int f(omega) |w_Delta(omega)|^2 cos(omega k Delta) d omega,
        |w_Delta(omega)|^2 = 4 sin^2(omega Delta/2)/omega^2.
    The atom Lambda*delta(tau) contributes Lambda*Delta to the variance only."""
    alphas = np.asarray(alphas, float); betas = np.asarray(betas, float)
    n = np.sum(alphas / betas)
    if n >= 1:
        return np.nan, [np.nan] * kmax
    om_max = 400.0 / Delta
    om = np.linspace(1e-9, om_max, n_omega)
    f, Lam = soe_spectrum(om, mu, alphas, betas)
    g = f - Lam / (2 * np.pi)                 # continuous part
    w2 = 4 * np.sin(om * Delta / 2) ** 2 / om**2
    base = g * w2
    var = Lam * Delta + 2 * np.trapezoid(base, om)  # factor 2 for negative frequencies
    rhos = []
    for k in range(1, kmax + 1):
        cov = 2 * np.trapezoid(base * np.cos(om * k * Delta), om)
        rhos.append(cov / var)
    return Lam, rhos


def soe_slowest_rate(alphas, betas):
    """Slowest relaxation rate gamma_min: smallest root in (0, min beta) of 1 = sum alpha_k/(beta_k - gamma).
    Near criticality gamma_min ~ (1-n)/(n m), m = mean kernel time."""
    alphas = np.asarray(alphas, float); betas = np.asarray(betas, float)
    n = np.sum(alphas / betas)
    if n >= 1:
        return 0.0
    from scipy.optimize import brentq
    bmin = betas[alphas > 1e-12].min() if np.any(alphas > 1e-12) else betas.min()
    h = lambda g: 1.0 - np.sum(alphas / (betas - g))
    lo, hi = 0.0, bmin * (1 - 1e-9)
    if h(lo) * h(hi) > 0:
        return (1 - n) / (n * np.sum(alphas / betas**2) / n) if n > 0 else np.nan
    return brentq(h, lo, hi)


def rescaled_residuals_soe(t, T, mu, alphas, betas):
    """KS test of time-rescaled inter-arrivals for the SoE kernel (no history)."""
    t = np.ascontiguousarray(t, dtype=np.float64)
    R = _soe_R(t, np.empty(0), np.asarray(betas, float))
    idx = np.arange(len(t))[:, None]
    Lam = mu * t + np.sum((np.asarray(alphas) / np.asarray(betas))[None, :] * (idx - R), axis=1)
    dlt = np.diff(np.concatenate([[0.0], Lam]))
    dlt = dlt[dlt > 0]
    ks = kstest(dlt, "expon")
    return ks.statistic, ks.pvalue
