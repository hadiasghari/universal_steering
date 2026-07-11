"""Pipeline schematic (fig1_opt3), v4-era labels. Run from repo root."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 3.4))
ax.set_xlim(0, 12); ax.set_ylim(0, 3.4); ax.set_axis_off()

BOX = dict(boxstyle='round,pad=0.28', fc='#eef3f9', ec='#3a6ea5', lw=1.2)
BOX2 = dict(boxstyle='round,pad=0.28', fc='#fdf3e7', ec='#c98a3d', lw=1.2)

def box(x, y, text, style=BOX, fs=8.5):
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            bbox=style, linespacing=1.35)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                 mutation_scale=13, color='#555', lw=1.1))

box(1.15, 1.7, "600 WordNet concepts\n24 supersenses $\\times$ 25\n+ psycholinguistic norms\n(zipf, conc, AoA, polysemy)")
arrow(2.35, 1.7, 3.0, 1.7)
box(4.05, 1.7, "contrast prompts (v4)\npos: Imagine/assume: $\\{c\\}$. <stmt>\nneg: <stmt> only\n(100 statements / class)")
arrow(5.15, 1.7, 5.8, 1.7)
box(6.75, 1.7, "RFM extraction / layer\nkernel ridge + AGOP\n$\\to$ top eigenvector $v_c^{(\\ell)}$\n+ spectrum ($\\lambda_1$/tr, PR)")
arrow(7.8, 2.15, 8.45, 2.6)
arrow(7.8, 1.25, 8.45, 0.8)
box(9.35, 2.6, "steer: all layers\n(benchmark coefficients)", BOX2)
box(9.35, 0.8, "steer: one mid layer\n(magn-scaled dose; depth sweep)", BOX2)
arrow(10.35, 2.6, 10.8, 2.05)
arrow(10.35, 0.8, 10.8, 1.35)
box(11.05, 1.7, "judge (gpt-oss:20b):\nexpressed under\nV1 / V6 prompt?", BOX2, fs=8)
ax.text(6.0, 0.18, "steerability $\\sim$ frequency + concreteness + AoA + polysemy + spectrum concentration   (logistic regression, per model: Llama-3.1-8B, Gemma-2-9B)",
        ha='center', fontsize=8.5, style='italic', color='#333')

fig.tight_layout()
fig.savefig('plots/fig1_opt3.png', dpi=150, bbox_inches='tight')
print('saved plots/fig1_opt3.png')
