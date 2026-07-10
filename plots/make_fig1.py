"""Figure 1 (option 1), v4-instrument data: dose-response phenomenon + supersense inventory.
Run from repo root:  ../.venv/bin/python plots/make_fig1.py
"""
import csv
import pickle
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TAG = 'Lall_K1ev_def_fv4'
SUCCESS, FAILURE = 'frog', 'fairway'

def load_judge(v):
    vl = '' if v == 1 else f'_v{v}'
    f = f"csvs/rfm_wordnet_gpt_oss_outputs_500_concepts_llama_3.1_8B_english_only{vl}_{TAG}.csv"
    return {r[0]: float(r[1]) >= 0.5 for r in list(csv.reader(open(f)))[1:]}

def load_gens(v):
    vl = '' if v == 1 else f'_v{v}'
    f = f"cached_outputs/rfm_wordnet_steered_500_concepts_llama_3.1_8B_english_only{vl}_{TAG}.pkl"
    return pickle.load(open(f, 'rb'))

def parse(out):
    txt = out.split('<|start_header_id|>assistant<|end_header_id|>', 1)[-1]
    return txt.replace('<|eot_id|>', '').strip()

df = pd.read_csv('bbxdata/bbx_wordnet_ds.csv')
s1, s6 = load_judge(1), load_judge(6)
union = {c: s1[c] or s6[c] for c in df.concept}
gens = load_gens(6)

fig = plt.figure(figsize=(13, 8.2))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1], hspace=0.42)

# ---- Panel A: dose-response quotes -------------------------------------------
axA = fig.add_subplot(gs[0])
axA.set_axis_off()
axA.set_title("A   Same protocol, opposite outcomes  (steered layer set, rising coefficient)",
              loc='left', fontsize=12, fontweight='bold')
for col, (concept, colr) in enumerate([(SUCCESS, '#1a7a3a'), (FAILURE, '#a33')]):
    outcome = 'steered' if union[concept] else 'not steered'
    axA.text(0.02 + col * 0.5, 1.00, f"'{concept}'  ({outcome})", fontsize=11,
             fontweight='bold', color=colr, transform=axA.transAxes, va='top')
    for row, (coef, out) in enumerate(gens[concept]):
        txt = textwrap.shorten(parse(out).replace('\n', ' '), width=110, placeholder=' ...')
        txt = textwrap.fill(txt, width=58)
        axA.text(0.02 + col * 0.5, 0.86 - row * 0.22, f"c={coef}:  {txt}",
                 fontsize=7.8, family='monospace', transform=axA.transAxes, va='top',
                 bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=colr, lw=0.8, alpha=0.9))

# ---- Panel B: supersense inventory -------------------------------------------
axB = fig.add_subplot(gs[1])
rows = []
for ss, g in df.groupby('supersense'):
    rows.append((ss.replace('noun.', ''), sum(union[c] for c in g.concept)))
rows.sort(key=lambda r: -r[1])
labs, counts = zip(*rows)
x = np.arange(len(labs))
axB.bar(x, counts, color='#3a6ea5')
axB.axhline(np.mean(counts), color='gray', lw=0.8, ls=':')
axB.text(len(labs) - 0.5, np.mean(counts) + 0.35, f"mean {np.mean(counts)/25:.0%}",
         fontsize=8, color='gray', ha='right')
axB.set_xticks(x)
axB.set_xticklabels(labs, rotation=55, ha='right', fontsize=8)
axB.set_ylabel('steered of 25 (V1$\\cup$V6)')
axB.set_title("B   Which concepts steer varies systematically by category  (600 WordNet concepts, Llama-3.1-8B)",
              loc='left', fontsize=12, fontweight='bold')

# verified example annotations (checked against the v4 outcome above)
ANNOT = [('exuberance', 'feeling', True), ('orchestration', 'communication', True),
         ('frog', 'animal', True), ('tibia', 'body', False), ('fairway', 'location', False)]
lab2x = {l: i for i, l in enumerate(labs)}
for word, ss, ok in ANNOT:
    assert union[word] == ok, f"annotation stale: {word}"
    xi = lab2x[ss]
    dx = 22 if word == 'exuberance' else 0
    axB.annotate(f"{'✓' if ok else '✗'} {word}", (xi, counts[xi]),
                 textcoords='offset points', xytext=(dx, 8 if word == 'exuberance' else (14 if ok else 22)),
                 ha='left' if word == 'exuberance' else 'center', fontsize=8,
                 color='#1a7a3a' if ok else '#a33',
                 arrowprops=dict(arrowstyle='-', lw=0.6, color='gray'))

fig.savefig('plots/fig1_opt1.png', dpi=150, bbox_inches='tight')
print('saved plots/fig1_opt1.png')
