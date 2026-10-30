"""Generate IEEE-style figures from the QLoRA run (qwen-os-tutor-lora) and the dataset.

Usage (from project root):  python paper/figures/make_figures.py
Outputs PDF (vector, for LaTeX) + PNG (preview) into paper/figures/.
Inputs:  qwen-os-tutor-lora/checkpoint-150/trainer_state.json, dataset_v2_{train,val}.csv
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
OVERSAMPLE = {"comparative", "analytical", "negative_edge_case", "applied_conceptual"}
FACTOR = 3

# Okabe-Ito (colour-blind safe); markers/linestyles keep it readable in greyscale print
BLUE, ORANGE, GREEN, GREY, RED = "#0072B2", "#D55E00", "#009E73", "#6b6b6b", "#CC3311"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight",
})
COL_W = 3.5   # IEEE single column (inches)
FULL_W = 7.16  # IEEE double column


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", name)


state = json.loads((ROOT / "qwen-os-tutor-lora/checkpoint-150/trainer_state.json").read_text())
log = state["log_history"]
tr = [(h["step"], h["loss"]) for h in log if "loss" in h]
ev = [(h["step"], h["eval_loss"]) for h in log if "eval_loss" in h]
gn = [(h["step"], h["grad_norm"]) for h in log if "grad_norm" in h]
lr = [(h["step"], h["learning_rate"]) for h in log if "learning_rate" in h]
best = state["best_global_step"]

# ---- Fig 1: train vs validation loss -------------------------------------------
fig, ax = plt.subplots(figsize=(COL_W, 2.4))
ax.plot(*zip(*tr), color=BLUE, marker="o", ms=3, lw=1.2, label="Training loss")
ax.plot(*zip(*ev), color=ORANGE, marker="s", ms=4, lw=1.2, ls="--", label="Validation loss")
ax.axvline(best, color=GREY, lw=0.8, ls=":")
ax.annotate(f"best val. loss\n(step {best}: {dict(ev)[best]:.3f})", xy=(best, dict(ev)[best]),
            xytext=(best - 62, 1.15), fontsize=7, color=GREY,
            arrowprops=dict(arrowstyle="->", color=GREY, lw=0.6))
for e in (50, 100):
    ax.axvline(e, color="none")
ax.set_xlabel("Optimisation step (50 steps = 1 epoch)")
ax.set_ylabel("Cross-entropy loss")
ax.set_xlim(0, 155)
ax.legend(frameon=False, loc="upper right")
save(fig, "fig_loss_curves")

# ---- Fig 2: generalisation gap (zoom on the overfitting onset) ------------------
trd = dict(tr)
fig, ax = plt.subplots(figsize=(COL_W, 2.2))
steps = [s for s, _ in ev]
gap = [e - trd[s] for s, e in ev]
bars = ax.bar([str(s) for s in steps], gap, color=[BLUE, BLUE, ORANGE], width=0.55)
for b, g, (s, e) in zip(bars, gap, ev):
    ax.text(b.get_x() + b.get_width() / 2, g + 0.01, f"{g:.2f}\n(val {e:.3f})", ha="center", fontsize=7)
ax.set_ylim(0, max(gap) * 1.35)
ax.set_xlabel("Evaluation step")
ax.set_ylabel("Val. loss $-$ train loss")
ax.grid(axis="x", visible=False)
save(fig, "fig_generalisation_gap")

# ---- Fig 3: optimisation diagnostics (grad norm + LR schedule) ------------------
fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 2.1))
a.plot(*zip(*gn), color=GREEN, marker="^", ms=3, lw=1.1)
a.set_xlabel("Optimisation step"); a.set_ylabel("Gradient norm"); a.set_title("(a) Gradient norm", fontsize=8)
b.plot(*zip(*lr), color=RED, marker="o", ms=3, lw=1.1)
b.set_xlabel("Optimisation step"); b.set_ylabel("Learning rate"); b.set_title("(b) Cosine LR schedule", fontsize=8)
b.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
fig.tight_layout()
save(fig, "fig_optimisation")

# ---- Fig 4: dataset composition (train / val / train after oversampling) --------
trn = pd.read_csv(ROOT / "dataset_v2_train.csv")
val = pd.read_csv(ROOT / "dataset_v2_val.csv")
c_tr = trn["category"].value_counts()
c_va = val["category"].value_counts().reindex(c_tr.index, fill_value=0)
c_os = c_tr.copy()
for c in c_os.index:
    if c in OVERSAMPLE:
        c_os[c] = c_tr[c] * FACTOR
labels = [c.replace("_", " ") for c in c_tr.index]
x = range(len(labels)); w = 0.27
fig, ax = plt.subplots(figsize=(FULL_W, 2.5))
ax.bar([i - w for i in x], c_tr.values, w, color=BLUE, label=f"Train (n={len(trn)})")
ax.bar([i for i in x], c_os.values, w, color=ORANGE, hatch="//", edgecolor="white", label=f"Train, oversampled $\\times${FACTOR} (n={int(c_os.sum())})")
ax.bar([i + w for i in x], c_va.values, w, color=GREY, label=f"Validation (n={len(val)})")
for i, (a_, b_, c_) in enumerate(zip(c_tr.values, c_os.values, c_va.values)):
    ax.text(i - w, a_ * 1.08, a_, ha="center", fontsize=6.5)
    ax.text(i, b_ * 1.08, b_, ha="center", fontsize=6.5)
    ax.text(i + w, max(c_, 0.7) * 1.08, c_, ha="center", fontsize=6.5)
ax.set_yscale("log"); ax.set_ylim(0.7, 1200)
ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=15)
ax.set_ylabel("Examples (log scale)")
ax.grid(axis="x", visible=False)
ax.legend(frameon=False)
save(fig, "fig_dataset_composition")

# ---- Fig 5: token-length distribution (prompt vs completion) --------------------
fig, ax = plt.subplots(figsize=(COL_W, 2.3))
bins = range(0, 1600, 100)
ax.hist(trn["prompt"].str.len(), bins=bins, color=BLUE, alpha=0.75, label="Prompt (context + question)")
ax.hist(trn["completion"].str.len(), bins=bins, color=ORANGE, alpha=0.75, label="Completion (answer)")
ax.set_xlabel("Length (characters)"); ax.set_ylabel("Training examples")
ax.legend(frameon=False)
save(fig, "fig_length_distribution")
