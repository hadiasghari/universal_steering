"""Mediation probe for the DAG's parent-dominance path (RUNLOG 026).

Decomposes each extracted direction's shared-component loading into (a) a
GLOBAL axis (sign-aligned mean of all 600 unit directions at the model's
anchored read depth) and (b) a category-specific residual share (leave-one-out
supersense mean after projecting out the global axis). Adds both to the
lexical logistic regression to test whether concreteness's effect is mediated
by shared-axis loading. Run from repo root: ../.venv/bin/python nb_axis_mediation.py
"""
import io
import pickle

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)


DS = pd.read_csv('bbxdata/bbx_wordnet_ds.csv')
CSVS = {'llama': 'csvs/rfm_wordnet_gpt_oss_outputs_500_concepts_llama_3.1_8B_english_only{v}_Lall_K1ev_def_fv4.csv',
        'gemma': 'csvs/rfm_wordnet_gpt_oss_outputs_500_concepts_gemma_2_9B_english_only{v}_Lall_K1ev_def_fv4.csv'}
DIRS = {'llama': ('directions_wordnet_fv4', '_llama_3_8b_it_eng_only', -19),
        'gemma': ('directions_wordnet_gemma_fv4', '_gemma_2_9b_it', -17)}

for model, (ddir, suffix, key) in DIRS.items():
    v1 = pd.read_csv(CSVS[model].format(v=''), header=0, names=['concept', 's_v1'])
    v6 = pd.read_csv(CSVS[model].format(v='_v6'), header=0, names=['concept', 's_v6'])
    df = DS.merge(v1, on='concept').merge(v6, on='concept')
    df['steered'] = ((df.s_v1 >= 0.5) | (df.s_v6 >= 0.5)).astype(int)
    if model == 'gemma':
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained('google/gemma-2-9b-it')
        df['n_tokens'] = df.concept.map(
            lambda w: len(tok(' ' + w, add_special_tokens=False)['input_ids']))

    V = {}
    for c in df.concept:
        M = CPUUnpickler(open(f"{ddir}/rfm_{c.replace(' ', '_')}{suffix}.pkl", 'rb')).load()[key]
        v = (M[0] if M.shape[0] == 3 else M[:, 0]).float()
        V[c] = (v / v.norm()).numpy()
    A = np.stack([V[c] for c in df.concept])
    G = A @ A.T
    ev, evec = np.linalg.eigh(G)
    u = evec[:, -1]
    if u.sum() < 0:
        u = -u
    s = np.sign(u)
    s[s == 0] = 1
    As = s[:, None] * A
    gmean = As.mean(0)
    gmean /= np.linalg.norm(gmean)
    df['global_share'] = np.abs(As @ gmean)
    resid = As - np.outer(As @ gmean, gmean)
    resid /= np.linalg.norm(resid, axis=1, keepdims=True)
    cat = np.zeros(len(df))
    for ss in df.supersense.unique():
        idx = np.where(df.supersense.values == ss)[0]
        Msub = resid[idx]
        n = len(idx)
        mean_all = Msub.mean(0)
        for j, i in enumerate(idx):
            loo = (mean_all * n - Msub[j]) / (n - 1)
            loo /= np.linalg.norm(loo)
            cat[i] = abs(float(Msub[j] @ loo))
    df['cat_share'] = cat
    stats = {c: CPUUnpickler(open(f"{ddir}/rfm_{c.replace(' ', '_')}{suffix}_rfmstats.pkl", 'rb')).load()
             for c in df.concept}
    df['frac1'] = [stats[c][key]['frac1'] for c in df.concept]
    for v in ['conc', 'zipf', 'aoa', 'polysemy', 'n_tokens', 'global_share', 'cat_share', 'frac1']:
        df['z_' + v] = (df[v] - df[v].mean()) / df[v].std()
    print(f"\n===== {model}: global |cos| mean={df.global_share.mean():.3f}  "
          f"r(glob,conc)={df.global_share.corr(df.conc):+.2f}  r(cat,conc)={df.cat_share.corr(df.conc):+.2f}")
    base = 'z_conc + z_zipf + z_aoa + z_polysemy + z_n_tokens'
    for label, fml in [('lex (baseline)', base),
                       ('lex + glob + cat', f'{base} + z_global_share + z_cat_share'),
                       ('lex + frac1 + glob + cat', f'{base} + z_frac1 + z_global_share + z_cat_share')]:
        m = smf.logit('steered ~ ' + fml, data=df).fit(disp=0)
        co = m.summary2().tables[1]
        out = '  '.join(
            f"{i.replace('z_', '')} {r['Coef.']:+.2f}"
            f"{'**' if r['P>|z|'] < .01 else '*' if r['P>|z|'] < .05 else chr(8224) if r['P>|z|'] < .1 else ''}"
            for i, r in co.iterrows() if i != 'Intercept')
        print(f"{label}  AIC={m.aic:.1f}\n   {out}")
