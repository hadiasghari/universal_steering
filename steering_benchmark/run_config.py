"""
Shared steering-run configuration + run tag (HA).

Single source of truth for the per-run steering knobs, imported by
eval_generations.py (generation), parse_results.py (judging), and
read_csv.py (aggregation). The RUN_TAG derived from these constants is
embedded in every artifact filename, so runs with different configs can
never collide with or silently reuse each other's caches.

NB: the coefficient VALUES are deliberately not in the tag -- they are
selected per (model, COEF_BEHAVIOR) in eval_generations and are stored
inside the output pickles as (coef, output) tuples, so their provenance
travels with the artifact.
"""

# --- per-run knobs (edit these; everything downstream follows) ---------------

N_COMPONENTS = 1           # number of top AGOP eigenvectors to combine (1..3)
COMPONENT_WEIGHTING = 'evals'  # 'evals' (eigenvalue-weighted) or 'equal'
COEF_BEHAVIOR = 'default'  # 'default' (unit vectors), 'magn' (per-layer gap scaling), 'clamp' (unimplemented)

# Extraction frame style (see utils.build_positive_prompts):
#   'orig' = each builder's class-specific template (benchmark-comparable)
#   'v2'   = universal dual frame ("fascinated by" / "deeply preoccupied with"), concept quoted
#   'v3'   = light-verb dual frame ("thinking about" / "on your mind"), concept quoted
FRAME_STYLE = 'orig'

# Layers to steer, as negative indices (depth = n_layers + key).
# Examples:  {-19}                        -> single layer, depth 13 on llama-8B
#            {-13, -15, -23}              -> depths 19+17+9
#            set(range(-8, -30, -1))      -> depths 3..24 band
#            set(range(-1, -80, -1))      -> all layers (benchmark default)
TARGET_KEYS = set(range(-1, -80, -1))

# --- run tag ------------------------------------------------------------------

def _layers_part(keys):
    ks = sorted(keys, reverse=True)          # e.g. [-8, -9, ..., -29]
    if len(ks) >= 60:
        return "Lall"
    lo, hi = ks[0], ks[-1]
    if len(ks) > 3 and ks == list(range(lo, hi - 1, -1)):
        return f"Lm{-lo}tom{-hi}"            # contiguous band
    return "L" + "-".join(f"m{-k}" for k in ks)


def make_run_tag():
    w = {'evals': 'ev', 'equal': 'eq'}[COMPONENT_WEIGHTING]
    b = {'default': 'def', 'magn': 'magn', 'clamp': 'clamp'}[COEF_BEHAVIOR]
    f = "" if FRAME_STYLE == 'orig' else f"_f{FRAME_STYLE}"
    return f"{_layers_part(TARGET_KEYS)}_K{N_COMPONENTS}{w}_{b}{f}"


RUN_TAG = make_run_tag()


def print_run_config(coefs=None):
    print("=" * 70)
    print(f"RUN CONFIG  tag={RUN_TAG}")
    print(f"  layers={_layers_part(TARGET_KEYS)} ({len(TARGET_KEYS)} keys)  "
          f"K={N_COMPONENTS}  weighting={COMPONENT_WEIGHTING}  coef_behavior={COEF_BEHAVIOR}")
    if coefs is not None:
        print(f"  coefs={coefs}")
    print("=" * 70)
