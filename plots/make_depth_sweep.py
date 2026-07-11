"""Depth-sweep figure (pilot 90, llama-3.1-8B, single-layer magn, fv4).

Auto-detects which depths have results in csvs/ (tags Lm{32-d}_K1ev_magn_fv4)
and plots V1-union-V6 steering counts per class + total. Run from repo root.
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES = ['moods', 'places', 'personalities']
COLORS = {'moods': '#3a6ea5', 'places': '#c98a3d', 'personalities': '#6a9a58'}


def union_count(cls, tag):
    u = {}
    for vl in ('', '_v6'):
        f = f"csvs/rfm_{cls}_gpt_oss_outputs_500_concepts_llama_3.1_8B_english_only{vl}_{tag}.csv"
        if not os.path.exists(f):
            return None
        for r in list(csv.reader(open(f)))[1:]:
            u[r[0]] = u.get(r[0], False) or float(r[1]) >= 0.5
    return sum(u.values())


depths, counts = [], {c: [] for c in CLASSES}
for d in range(2, 32):
    tag = f"Lm{32 - d}_K1ev_magn_fv4"
    per = [union_count(c, tag) for c in CLASSES]
    if any(v is None for v in per):
        continue
    depths.append(d)
    for c, v in zip(CLASSES, per):
        counts[c].append(v)

totals = [sum(counts[c][i] for c in CLASSES) for i in range(len(depths))]

fig, ax = plt.subplots(figsize=(7.5, 3.6))
for c in CLASSES:
    ax.plot(depths, counts[c], marker='o', ms=4, lw=1.5, color=COLORS[c], label=c)
ax.plot(depths, totals, marker='s', ms=4, lw=1.8, color='#444', ls='--', label='total (of 90)')
ax.set_xlabel('steered layer (depth)')
ax.set_ylabel('concepts steered (V1 $\\cup$ V6)')
ax.set_xticks(depths if len(depths) <= 12 else range(min(depths), max(depths) + 1, 2))
ax.grid(alpha=0.3)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig('plots/depth_sweep_fv4.png', dpi=150)
print('depths:', depths)
for c in CLASSES:
    print(f"{c}: {counts[c]}")
print('total:', totals)
print('saved plots/depth_sweep_fv4.png')
