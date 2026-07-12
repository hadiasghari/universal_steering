"""K=2-vs-K=1 flip analysis (llama d15, n=600): gained/lost/stable boxplots
on zipf and lambda1/trace. Run from repo root."""
import csv
import io
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
from scipy.stats import mannwhitneyu


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)


def load(tag):
    u = {}
    for vl in ('', '_v6'):
        f = f"csvs/rfm_wordnet_gpt_oss_outputs_500_concepts_llama_3.1_8B_english_only{vl}_{tag}.csv"
        for r in list(csv.reader(open(f)))[1:]:
            u[r[0]] = u.get(r[0], False) or float(r[1]) >= 0.5
    return u


k1 = load('Lm17_K1ev_magn_fv4')
k2 = load('Lm17_K2ev_magn_fv4')
DS = pd.read_csv('bbxdata/bbx_wordnet_ds.csv')
stats = {c: CPUUnpickler(open(f"directions_wordnet_fv4/rfm_{c.replace(' ', '_')}_llama_3_8b_it_eng_only_rfmstats.pkl", 'rb')).load()
         for c in DS.concept}
DS['l1_tr'] = [stats[c][-17]['frac1'] for c in DS.concept]
DS['k1'] = DS.concept.map(k1)
DS['k2'] = DS.concept.map(k2)
gained = DS[~DS.k1 & DS.k2]
lost = DS[DS.k1 & ~DS.k2]
stable = DS[DS.k1 & DS.k2]

fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
for ax, var, label in [(axes[0], 'zipf', 'Zipf frequency'),
                       (axes[1], 'l1_tr', '$\\lambda_1$/trace @ d15')]:
    groups = [gained[var], lost[var], stable[var]]
    u, p = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
    bp = ax.boxplot(groups, labels=[f'gained\n(n={len(gained)})', f'lost\n(n={len(lost)})',
                                    f'stable both\n(n={len(stable)})'],
                    patch_artist=True, showmeans=True, widths=0.55)
    for box, col in zip(bp['boxes'], ['#639922', '#a32d2d', '#888']):
        box.set_facecolor(col)
        box.set_alpha(0.4)
    for med in bp['medians']:
        med.set_color('#222')
    ax.set_title(f'{label}\ngained-vs-lost MW p={p:.3f}', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig('plots/k2_flips_boxplots.png', dpi=150)
print('saved plots/k2_flips_boxplots.png')
