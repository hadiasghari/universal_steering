"""Per-supersense outcome table (LaTeX rows) for the appendix.

Columns: supersense, n, % known (prompted ceiling) per model, % steerable
Llama (all-layer / single L15 / union), % steerable Gemma (all-layer),
2 examples steerable in both models, 2 examples steerable in neither.
Example rule: highest-zipf qualifying words (deterministic).
"both" = (Llama all-layer or L15) and Gemma all-layer; "neither" = complement
of each model's respective outcome. Run from repo root.
"""
import csv

import pandas as pd

DS = pd.read_csv('bbxdata/bbx_wordnet_ds.csv')


def load(model, tag):
    u = {}
    for vl in ('', '_v6'):
        f = f"csvs/rfm_wordnet_gpt_oss_outputs_500_concepts_{model}_english_only{vl}_{tag}.csv"
        for r in list(csv.reader(open(f)))[1:]:
            u[r[0]] = u.get(r[0], False) or float(r[1]) >= 0.5
    return u


def load_ceiling(model):
    return {r[0]: float(r[1]) >= 0.5
            for r in list(csv.reader(open(f"csvs/ceiling_{model}.csv")))[1:]}


la = load('llama_3.1_8B', 'Lall_K1ev_def_fv4')
l15 = load('llama_3.1_8B', 'Lm17_K1ev_magn_fv4')
ga = load('gemma_2_9B', 'Lall_K1ev_def_fv4')
kl = load_ceiling('llama_3.1_8B')
kg = load_ceiling('gemma_2_9B')

df = DS[['concept', 'supersense', 'zipf']].copy()
df['kl'] = df.concept.map(kl)
df['kg'] = df.concept.map(kg)
df['la'] = df.concept.map(la)
df['l15'] = df.concept.map(l15)
df['lu'] = df.la | df.l15
df['ga'] = df.concept.map(ga)
df['both'] = df.lu & df.ga
df['neither'] = ~df.lu & ~df.ga

rows = []
for ss, g in df.groupby('supersense'):
    n = len(g)
    pct = lambda col: round(100 * g[col].mean())
    ex_b = ', '.join(g[g.both].sort_values('zipf', ascending=False).concept.head(2))
    ex_n = ', '.join(g[g.neither].sort_values('zipf', ascending=False).concept.head(2))
    rows.append((ss.replace('noun.', ''), n, pct('kl'), pct('kg'),
                 pct('la'), pct('l15'), pct('lu'), pct('ga'),
                 ex_b or '---', ex_n or '---'))
rows.sort(key=lambda r: -r[6])

print(r"supersense & $n$ & \multicolumn{2}{c}{\% known} & "
      r"\multicolumn{3}{c}{\% steer Llama} & \% steer G & "
      r"steer both (ex.) & steer neither (ex.) \\")
print(r" & & L & G & all & L15 & $\cup$ & all & & \\ \hline")
for r in rows:
    print(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]} & {r[5]} & {r[6]} & {r[7]} & "
          f"\\textit{{{r[8]}}} & \\textit{{{r[9]}}} \\\\")
t = df
print(f"\\hline all & {len(t)} & {round(100*t.kl.mean())} & {round(100*t.kg.mean())} & "
      f"{round(100*t.la.mean())} & {round(100*t.l15.mean())} & {round(100*t.lu.mean())} & "
      f"{round(100*t.ga.mean())} & & \\\\")
