"""Concept WordNet hyponymy depth vs steerability (v4 600 runs, all-layer).

Depth = min_depth() of the concept's synset (first noun synset whose
lexicographer file matches the concept's assigned supersense; fallback: first
noun synset). Two panels: Llama-3.1-8B and Gemma-2-9B, each with its own
steerable set (V1 union V6, Lall_K1ev_def_fv4). Run from repo root.
"""
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import wordnet as wn
from scipy.stats import mannwhitneyu

DS = pd.read_csv('bbxdata/bbx_wordnet_ds.csv')


def synset_depth(word, supersense):
    syns = wn.synsets(word.replace(' ', '_'), 'n')
    if not syns:
        return None
    s = next((x for x in syns if x.lexname() == supersense), syns[0])
    return s.min_depth()


DS['hypo_depth'] = [synset_depth(w, ss) for w, ss in zip(DS.concept, DS.supersense)]
print('concepts without synset:', DS.hypo_depth.isna().sum())

MODELS = [
    ('Llama-3.1-8B',
     'csvs/rfm_wordnet_gpt_oss_outputs_500_concepts_llama_3.1_8B_english_only{v}_Lall_K1ev_def_fv4.csv',
     '#3a6ea5'),
    ('Gemma-2-9B',
     'csvs/rfm_wordnet_gpt_oss_outputs_500_concepts_gemma_2_9B_english_only{v}_Lall_K1ev_def_fv4.csv',
     '#c98a3d'),
]

fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
for ax, (name, csv_tpl, color) in zip(axes, MODELS):
    steer = {}
    for v in ('', '_v6'):
        for r in list(csv.reader(open(csv_tpl.format(v=v))))[1:]:
            steer[r[0]] = steer.get(r[0], False) or float(r[1]) >= 0.5
    df = DS[DS.concept.isin(steer)].dropna(subset=['hypo_depth']).copy()
    df['steered'] = df.concept.map(steer)
    yes = df.loc[df.steered, 'hypo_depth']
    no = df.loc[~df.steered, 'hypo_depth']
    u, p = mannwhitneyu(yes, no, alternative='two-sided')
    bp = ax.boxplot([yes, no], labels=[f'steerable\n(n={len(yes)})', f'not\n(n={len(no)})'],
                    patch_artist=True, showmeans=True, widths=0.55)
    for box in bp['boxes']:
        box.set_facecolor(color)
        box.set_alpha(0.45)
    for med in bp['medians']:
        med.set_color('#222')
    ax.set_title(f'{name}\nMann-Whitney p={p:.3g}', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    print(f"{name}: steerable median={yes.median():.0f} mean={yes.mean():.2f} | "
          f"not median={no.median():.0f} mean={no.mean():.2f} | MW p={p:.4g}")
axes[0].set_ylabel('WordNet hyponymy depth (min_depth)')
fig.tight_layout()
fig.savefig('plots/hypodepth_boxplots_fv4.png', dpi=150)
print('saved plots/hypodepth_boxplots_fv4.png')
