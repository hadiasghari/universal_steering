"""Figure 1: steered concepts per supersense, Llama + Gemma (v4 instrument).
Run from repo root:  ../.venv/bin/python plots/make_fig1.py
"""
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TAG = 'Lall_K1ev_def_fv4'

def union(model):
    out = {}
    for v in [1, 6]:
        vl = '' if v == 1 else f'_v{v}'
        f = f"csvs/rfm_wordnet_gpt_oss_outputs_500_concepts_{model}_english_only{vl}_{TAG}.csv"
        for r in list(csv.reader(open(f)))[1:]:
            out[r[0]] = out.get(r[0], False) or float(r[1]) >= 0.5
    return out

df = pd.read_csv('bbxdata/bbx_wordnet_ds.csv')
lu, gu = union('llama_3.1_8B'), union('gemma_2_9B')

rows = []
for ss, g in df.groupby('supersense'):
    rows.append((ss.replace('noun.', ''),
                 sum(lu[c] for c in g.concept), sum(gu[c] for c in g.concept)))
rows.sort(key=lambda r: -(r[1] + r[2]))
labs, lc, gc = zip(*rows)
x = np.arange(len(labs))

fig, ax = plt.subplots(figsize=(10, 3.8))
ax.bar(x - 0.2, lc, 0.4, label=f'Llama-3.1-8B ({sum(lc)}/600)', color='#3a6ea5')
ax.bar(x + 0.2, gc, 0.4, label=f'Gemma-2-9B ({sum(gc)}/600)', color='#c98a3d')
ax.axhline(np.mean(lc), color='#3a6ea5', lw=0.7, ls=':')
ax.axhline(np.mean(gc), color='#c98a3d', lw=0.7, ls=':')
ax.set_xticks(x)
ax.set_xticklabels(labs, rotation=55, ha='right', fontsize=8)
ax.set_ylabel('steered of 25 (V1$\\cup$V6)')
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig('plots/fig1_opt1.png', dpi=150)
print('saved plots/fig1_opt1.png')
