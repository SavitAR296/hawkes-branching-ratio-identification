"""Independent nonparametric cross-check with tick 0.8.0.2 (HawkesConditionalLaw, Bacry-Muzy Wiener-Hopf estimator).
Settings as in reviewer_checks.py section 5; day-resampling bootstrap; numerical sensitivity (n_quad=20, delta_lag=0.05)."""
import sys, json, time, warnings, numpy as np
warnings.filterwarnings('ignore')
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / 'tables_rev'
import tick.base.base as tb
from tick.hawkes import HawkesConditionalLaw
def _permissive(cls):
    orig = cls.__setattr__
    def _sa(self, k, v):
        try: orig(self, k, v)
        except AttributeError: object.__setattr__(self, k, v)
    cls.__setattr__ = _sa
def _subs(c):
    for s in c.__subclasses__(): yield s; yield from _subs(s)
for c in list(_subs(tb.Base)): _permissive(c)
Z = np.load(ROOT / 'data/bund.npz', allow_pickle=True); DAYS = sorted(Z.files)
STREAMS = {'mid': lambda d: [np.concatenate([Z[d][0], Z[d][1]])], 'trades': lambda d: [np.concatenate([Z[d][2], Z[d][3]])],
           'updown': lambda d: [Z[d][0], Z[d][1]], 'buysell': lambda d: [Z[d][2], Z[d][3]]}
BANDS = [(1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.0), (1.0, 10.0), (10.0, 100.0)]
def fit(name, max_support, days, n_quad=12, delta_lag=0.1):
    L = HawkesConditionalLaw(claw_method='log', delta_lag=delta_lag, min_lag=5e-4, max_lag=max(500, 5 * max_support), quad_method='log', n_quad=n_quad,
                             min_support=1e-4, max_support=max_support, n_threads=1)
    for d in days: L.incremental_fit([np.ascontiguousarray(np.sort(s).astype(float)) for s in STREAMS[name](d)], compute=False)
    L.compute(); return L
def band_integrals(L, max_support):
    x = np.geomspace(1e-4, max_support, 600); res = {}
    for i in range(len(L.kernels_norms)):
        for j in range(len(L.kernels_norms)):
            v = np.asarray(L.get_kernel_values(i, j, x))
            res[f'k{i}{j}'] = [float(np.trapezoid(v[(x >= a) & (x < b)], x[(x >= a) & (x < b)])) if ((x >= a) & (x < b)).sum() > 1 else 0.0 for a, b in BANDS]
    return res


def main():
    rng = np.random.default_rng(20260914); res = {}; t0 = time.time()
    for name in ['mid', 'trades', 'updown', 'buysell']:
        for ms in [1.0, 10.0, 100.0]:
            L = fit(name, ms, DAYS); G = np.array(L.kernels_norms)
            rec = dict(norms=G.tolist(), bands=band_integrals(L, ms))
            for nq, dl in [(20, 0.1), (12, 0.05)]:
                rec[f'sens_q{nq}_d{dl}'] = np.array(fit(name, ms, DAYS, n_quad=nq, delta_lag=dl).kernels_norms).tolist()
            res[f'{name}_{ms:g}'] = rec; print(name, ms, G.round(3).tolist(), '%.0fs' % (time.time() - t0), flush=True)
    json.dump(res, open(OUT / 'R_tick_np.json', 'w'), indent=1)
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 100; boot = {}
    for name in ['mid', 'trades', 'updown', 'buysell']:
        for ms in [1.0, 10.0, 100.0]:
            if name in ('updown', 'buysell') and ms == 100.0: continue
            vals = []
            for b in range(B):
                days = [DAYS[k] for k in rng.integers(0, len(DAYS), len(DAYS))]
                try: vals.append(np.array(fit(name, ms, days).kernels_norms).tolist())
                except Exception as e: pass
            boot[f'{name}_{ms:g}'] = vals; print('boot', name, ms, len(vals), '%.0fs' % (time.time() - t0), flush=True)
            json.dump(boot, open(OUT / 'R_tick_np_boot.json', 'w'))


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'test'
    if mode == 'main': main()
    if mode == 'test':
        t0 = time.time(); L = fit('mid', 1.0, DAYS); print('mid support 1 s: norms', np.array(L.kernels_norms).round(3).tolist(), '%.1fs' % (time.time() - t0))
        print('attrs:', [a for a in dir(L) if 'support' in a.lower() or 'kernel' in a.lower()][:12])
        print(band_integrals(L, 1.0))
