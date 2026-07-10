"""
R4 (averaging variant): build category-mean-subtracted steering directions (HA, jul7).

For each concept c in the selected supersenses, per layer:
    v_spec(c) = v(c) - s_c * mean_{c' != c, same supersense}( s_{c'} * v(c') )
where v(.) is the unit-normalized top (K=1) AGOP eigenvector and s are
per-layer sign-consistency flips derived from the top eigenvector of the
25x25 Gram matrix (saved eigenvector signs are arbitrary; without alignment
the category mean partially self-cancels -- 4-28% of within-category pairs
have negative cosine at mid layers). The leave-one-out mean is subtracted
in v(c)'s OWN orientation (s_c * mean), so whatever sign convention made
the original direction steer toward the concept is preserved in the residual.

v_spec is renormalized and written as row 0 of the (3, d) direction matrix
(rows 1-2 kept as-is; unused at K=1). rfmstats pickles are copied unchanged
(only 'evals' is read at K=1, and the vector is re-unit-normalized after
weighting, so stale stats cannot leak into the injected vector under
COEF_BEHAVIOR='default').

Reads:  directions_wordnet_fv2b/   Writes: directions_wordnet_fv2b_r4spec/
Hypothesis (paper R4): concepts whose direction is dominated by the shared
frame/category component are un-steerable as extracted but rescued by v_spec.
"""

import io
import pickle
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

SRC = Path('directions_wordnet_fv2b')
DST = Path('directions_wordnet_fv2b_r4spec')
MODEL_NAME = 'llama_3_8b_it_eng_only'
SUPERSENSES = ['noun.feeling', 'noun.food', 'noun.person',
               'noun.location', 'noun.artifact', 'noun.body']


class CPUUnpickler(pickle.Unpickler):
    """Directions saved on CUDA hosts fail to unpickle on Mac/MPS; map to CPU."""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)


def main():
    DST.mkdir(exist_ok=True)
    df = pd.read_csv('bbxdata/bbx_wordnet_ds.csv')

    for ss in SUPERSENSES:
        concepts = df[df.supersense == ss].concept.tolist()
        raw = {}   # concept -> {layer: (3, d) tensor}
        for c in concepts:
            with open(SRC / f'rfm_{c}_{MODEL_NAME}.pkl', 'rb') as f:
                raw[c] = CPUUnpickler(f).load()

        layers = sorted(raw[concepts[0]].keys())
        n = len(concepts)
        new_rows = {c: {} for c in concepts}
        report = []
        for lyr in layers:
            # unit K=1 vectors, (n, d)
            M = torch.stack([raw[c][lyr][0] / raw[c][lyr][0].norm() for c in concepts])
            # sign alignment: loadings of the Gram matrix's top eigenvector
            G = M @ M.T
            evals, evecs = torch.linalg.eigh(G)
            u = evecs[:, -1]
            if u.sum() < 0:                       # fix global sign to majority +1
                u = -u
            s = torch.sign(u)
            s[s == 0] = 1.0
            Ms = s[:, None] * M                    # consensus-oriented
            mean_all = Ms.mean(0)
            for i, c in enumerate(concepts):
                loo = (mean_all * n - Ms[i]) / (n - 1)
                v_spec = M[i] - s[i] * loo         # subtract in v(c)'s own orientation
                new_rows[c][lyr] = v_spec / v_spec.norm()
            # log how much was removed (residual norm before renormalization)
            resid = (M - s[:, None] * loo).norm(dim=1)  # approx (uses last loo; fine for logging)
            report.append((lyr, float((s < 0).float().mean()),
                           float((M @ (mean_all / mean_all.norm()).T).abs().mean())))

        flips = sum(r[1] for r in report) / len(report)
        shared = sum(r[2] for r in report) / len(report)
        print(f"{ss}: {n} concepts, {len(layers)} layers | "
              f"mean sign-flip rate {flips:.2f} | mean |cos| to category mean {shared:.2f}")

        for c in concepts:
            out = {}
            for lyr in layers:
                mat = raw[c][lyr].clone()
                mat[0] = new_rows[c][lyr]
                out[lyr] = mat
            with open(DST / f'rfm_{c}_{MODEL_NAME}.pkl', 'wb') as f:
                pickle.dump(out, f)
            shutil.copy(SRC / f'rfm_{c}_{MODEL_NAME}_rfmstats.pkl',
                        DST / f'rfm_{c}_{MODEL_NAME}_rfmstats.pkl')

    print(f"\nWrote {len(list(DST.glob('rfm_*.pkl')))} files to {DST}/")


if __name__ == '__main__':
    main()
