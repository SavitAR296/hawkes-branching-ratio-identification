"""Pooled signed variance ratio VR- at long bin widths (Section 5.11): the estimator of R_pooled_vr.py extended to
1 h, 2 h, 3.5 h, 7 h and one full session (bin widths that divide the 14-hour session), 4,000-resample session bootstrap,
implied finite-W bound n+ >= VR-^(-1/2) - 1 (Corollary 2), and the share of full-session signed variance carried by the
three sessions with the largest net signed count. Seed 20260914. Added in v1.2.2-revision; does not change any other output."""
import json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; Z = np.load(ROOT / 'data/bund.npz', allow_pickle=True); DAYS = sorted(Z.files); T = 50400.0
idx = np.random.default_rng(20260914).integers(0, 20, (4000, 20)); out = {}
for W in [3600.0, 7200.0, 12600.0, 25200.0, 50400.0]:
    m = int(T // W); S_raw, S_dm, Nn = [], [], []
    for d in DAYS:
        tp = Z[d][0][Z[d][0] < m * W]; tn = Z[d][1][Z[d][1] < m * W]
        s = np.bincount((tp // W).astype(int), minlength=m).astype(float) - np.bincount((tn // W).astype(int), minlength=m)
        S_raw.append((s ** 2).sum()); S_dm.append(((s - s.mean()) ** 2).sum()); Nn.append(len(tp) + len(tn))
    S_raw, S_dm, Nn = map(np.array, (S_raw, S_dm, Nn)); out[str(W)] = {'bins_per_session': m}
    versions = [('raw', S_raw)] + ([('demeaned', S_dm), ('dfcorr', S_dm * m / (m - 1))] if m > 1 else [])
    for name, num in versions:
        v = num.sum() / Nn.sum(); b = num[idx].sum(1) / Nn[idx].sum(1)
        out[str(W)][name] = dict(VR=float(v), VR_ci=[float(x) for x in np.percentile(b, [2.5, 97.5])], bound=float(v ** -0.5 - 1),
                                 bound_ci=[float(x) for x in np.percentile(b ** -0.5 - 1, [2.5, 97.5])])
    if m == 1:
        out[str(W)]['share_top3_sessions'] = float(np.sort(S_raw)[-3:].sum() / S_raw.sum())
    print(W, m, {k: round(r['VR'], 3) for k, r in out[str(W)].items() if isinstance(r, dict)})
json.dump(out, open(ROOT / 'tables_rev/R_pooled_vr_long.json', 'w'), indent=1)
