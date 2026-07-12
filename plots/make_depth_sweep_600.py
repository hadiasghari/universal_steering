"""Confirmatory 600-concept depth sweep: steered count per supersense x depth.

Heatmap per model (llama depths 10-19, gemma 13-25 step 2; auto-detects which
tags exist in csvs/). Rows sorted by all-layer benchmark count (descending);
a TOTAL row on top; the all-layer benchmark shown as a reference column.
Run from repo root.
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DS = pd.read_csv('bbxdata/bbx_wordnet_ds.csv')
MODELS = {
    'llama_3.1_8B': ('Llama-3.1-8B', 32, range(10, 20)),
    'gemma_2_9B': ('Gemma-2-9B', 42, range(13, 26, 2)),
}


def load(model, tag):
    u = {}
    for vl in ('', '_v6'):
        f = f"csvs/rfm_wordnet_gpt_oss_outputs_500_concepts_{model}_english_only{vl}_{tag}.csv"
        if not os.path.exists(f):
            return None
        for r in list(csv.reader(open(f)))[1:]:
            u[r[0]] = u.get(r[0], False) or float(r[1]) >= 0.5
    return u


for model, (name, nl, drange) in MODELS.items():
    cols, depths = [], []
    for d in drange:
        u = load(model, f"Lm{nl - d}_K1ev_magn_fv4")
        if u is None:
            continue
        depths.append(d)
        cols.append(u)
    if not depths:
        print(f"{name}: no sweep csvs yet, skipping")
        continue
    bench = load(model, 'Lall_K1ev_def_fv4')

    df = DS[['concept', 'supersense']].copy()
    for d, u in zip(depths, cols):
        df[f'd{d}'] = df.concept.map(u)
    df['all-layer'] = df.concept.map(bench)
    g = df.groupby('supersense').sum(numeric_only=True)
    g.index = [s.replace('noun.', '') for s in g.index]
    g = g.sort_values('all-layer', ascending=False)
    mat = pd.concat([pd.DataFrame(g.sum()).T.rename(index={0: 'TOTAL'}), g])

    dcols = [f'd{d}' for d in depths]
    plot_cols = dcols + ['all-layer']
    M = mat[plot_cols].values.astype(float)
    Mn = M.copy()
    Mn[0] /= 600.0
    Mn[1:] /= 25.0

    fig, ax = plt.subplots(figsize=(0.62 * len(plot_cols) + 2.4, 0.30 * len(mat) + 1.4))
    im = ax.imshow(Mn, cmap='YlGnBu', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(plot_cols)))
    ax.set_xticklabels([c.replace('d', '') for c in dcols] + ['all'], fontsize=8)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index, fontsize=8)
    ax.axvline(len(dcols) - 0.5, color='white', lw=2)
    ax.axhline(0.5, color='white', lw=2)
    for i in range(len(mat)):
        for j in range(len(plot_cols)):
            v = int(M[i, j])
            ax.text(j, i, v, ha='center', va='center', fontsize=6.5,
                    color='white' if Mn[i, j] > 0.55 else '#333')
    ax.set_xlabel('steered layer (depth); "all" = all-layer benchmark', fontsize=9)
    ax.set_title(f'{name}: concepts steered per supersense (of 25) by single-layer depth',
                 fontsize=10)
    fig.tight_layout()
    out = f"plots/depth_sweep_600_{model.split('_')[0]}_by_supersense.png"
    fig.savefig(out, dpi=150)
    print('saved', out)
    print(mat[plot_cols].to_string())
