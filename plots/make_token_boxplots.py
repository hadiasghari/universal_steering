"""Concept token-count vs steerability (v4 600 runs, all-layer benchmark).

Two panels: Llama-3.1-8B and Gemma-2-9B, each with its own tokenizer and its
own steerable set (V1 union V6, Lall_K1ev_def_fv4). Token count = length of
the bare concept string tokenized without special tokens. Run from repo root.
"""
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from transformers import AutoTokenizer

MODELS = [
    ('Llama-3.1-8B', 'meta-llama/Meta-Llama-3.1-8B-Instruct',
     'csvs/rfm_wordnet_gpt_oss_outputs_500_concepts_llama_3.1_8B_english_only{v}_Lall_K1ev_def_fv4.csv',
     '#3a6ea5'),
    ('Gemma-2-9B', 'google/gemma-2-9b-it',
     'csvs/rfm_wordnet_gpt_oss_outputs_500_concepts_gemma_2_9B_english_only{v}_Lall_K1ev_def_fv4.csv',
     '#c98a3d'),
]

fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
for ax, (name, tok_id, csv_tpl, color) in zip(axes, MODELS):
    tok = AutoTokenizer.from_pretrained(tok_id)
    steer = {}
    for v in ('', '_v6'):
        for r in list(csv.reader(open(csv_tpl.format(v=v))))[1:]:
            steer[r[0]] = steer.get(r[0], False) or float(r[1]) >= 0.5
    ntok = {c: len(tok.encode(c, add_special_tokens=False)) for c in steer}
    yes = [ntok[c] for c in steer if steer[c]]
    no = [ntok[c] for c in steer if not steer[c]]
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
    print(f"{name}: steerable median={sorted(yes)[len(yes)//2]} mean={sum(yes)/len(yes):.2f} | "
          f"not median={sorted(no)[len(no)//2]} mean={sum(no)/len(no):.2f} | MW p={p:.4g}")
axes[0].set_ylabel('concept length (tokens)')
fig.tight_layout()
fig.savefig('plots/token_count_boxplots_fv4.png', dpi=150)
print('saved plots/token_count_boxplots_fv4.png')
