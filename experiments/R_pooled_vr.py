"""Pooled signed variance ratio VR- (Section 5.11): sessions pooled, three versions (demeaned; demeaned with m/(m-1);
not demeaned), 4,000-resample session bootstrap, implied bound n+ >= VR-^(-1/2) - 1. Seed 20260914."""
import json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; Z = np.load(ROOT / 'data/bund.npz', allow_pickle=True); DAYS = sorted(Z.files); T = 50400.0
idx = np.random.default_rng(20260914).integers(0, 20, (4000, 20)); out = {}
for W in [300.0, 600.0, 1200.0, 1800.0, 3600.0]:
    m = int(T // W); S_raw, S_dm, Nn = [], [], []
    for d in DAYS:
        tp = Z[d][0][Z[d][0] < m * W]; tn = Z[d][1][Z[d][1] < m * W]
        s = np.bincount((tp // W).astype(int), minlength=m).astype(float) - np.bincount((tn // W).astype(int), minlength=m)
        S_raw.append((s ** 2).sum()); S_dm.append(((s - s.mean()) ** 2).sum()); Nn.append(len(tp) + len(tn))
    S_raw, S_dm, Nn = map(np.array, (S_raw, S_dm, Nn)); out[str(W)] = {}
    for name, num in [('demeaned', S_dm), ('dfcorr', S_dm * m / (m - 1)), ('raw', S_raw)]:
        v = num.sum() / Nn.sum(); b = num[idx].sum(1) / Nn[idx].sum(1)
        out[str(W)][name] = dict(VR=float(v), VR_ci=[float(x) for x in np.percentile(b, [2.5, 97.5])], bound=float(v ** -0.5 - 1), bound_ci=[float(x) for x in np.percentile(b ** -0.5 - 1, [2.5, 97.5])])
    print(W, {k: round(r['VR'], 3) for k, r in out[str(W)].items()})
json.dump(out, open(ROOT / 'tables_rev/R_pooled_vr.json', 'w'), indent=1)
