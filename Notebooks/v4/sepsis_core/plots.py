"""Figures: publication style, colour-blind safe (Okabe-Ito palette), one look across all notebooks. Aggregates only."""
import numpy as np
import matplotlib.pyplot as plt

from . import config as C

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]


def style():
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False, "axes.titlesize": 10, "axes.labelsize": 10})


def _series(table, view, key):
    rows = [r for r in table if r["view"] == view]
    return [r["bucket"] for r in rows], np.array([r.get(key, np.nan) for r in rows], float), np.array([r["n_rows"] for r in rows])


def by_icu_hour(tables, horizon=6, metric="auroc", path=None, title=None):
    """tables: {model name: by_icu_hour table from runs.report}. Three panels, as the brief asks: all stays; without early-onset stays;
    early-onset vs late-onset septic stays (each against all non-septic stays). Panel 3 is the 'recognition vs prediction' view:
    hours 1-6 of early-onset stays are recognition of sepsis already under way."""
    style(); key = f"{metric}_t{horizon}"; fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=True)
    for i, (name, tab) in enumerate(tables.items()):
        col = OKABE_ITO[i % len(OKABE_ITO)]
        for ax, view in zip(axes[:2], ("all", "no_early_onset")):
            x, y, _ = _series(tab, view, key); ax.plot(range(len(x)), y, marker="o", ms=4, color=col, label=name)
        x, y, _ = _series(tab, "early_onset", key); axes[2].plot(range(len(x)), y, marker="o", ms=4, color=col, ls="--", label=f"{name}: early-onset (recognition)")
        x, y, _ = _series(tab, "late_onset", key); axes[2].plot(range(len(x)), y, marker="s", ms=4, color=col, ls="-", label=f"{name}: late-onset (prediction)")
    for ax, ttl in zip(axes, ("All stays", f"Without early-onset stays (onset < {C.CFG.EARLY_ONSET_HOURS} h)", "Early- vs late-onset septic stays")):
        ax.set_xticks(range(len(x))); ax.set_xticklabels(x, rotation=30, ha="right"); ax.set_xlabel("ICU hour of the prediction"); ax.set_title(ttl)
        if metric == "auroc": ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    axes[0].set_ylabel(f"{metric.upper()} for the label at t+{horizon} h"); axes[0].legend(fontsize=8); axes[2].legend(fontsize=7)
    fig.suptitle(title or f"How early can it tell? {metric.upper()} by ICU hour (VAL, t+{horizon} h)", y=1.02)
    fig.tight_layout()
    if path: fig.savefig(path, bbox_inches="tight")
    plt.show(); return fig


def seed_strip(groups, path=None, ylabel="VAL AUROC (mean of horizons)", title="Seed-to-seed spread", ref=None):
    """groups: {label: [value per seed]}. Dots = seeds, bar = mean. `ref`: optional {label: value} horizontal reference lines."""
    style(); fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(groups), 3.8))
    for i, (name, vals) in enumerate(groups.items()):
        v = np.asarray(vals, float); col = OKABE_ITO[i % len(OKABE_ITO)]
        ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color=col, s=28, zorder=3)
        ax.hlines(v.mean(), i - 0.25, i + 0.25, color=col, lw=2)
    for j, (name, val) in enumerate((ref or {}).items()): ax.axhline(val, color="grey", lw=1, ls=(":" if j else "--")); ax.text(len(groups) - 0.5, val, f" {name}", va="bottom", ha="right", fontsize=8, color="grey")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(list(groups), rotation=20, ha="right"); ax.set_ylabel(ylabel); ax.set_title(title)
    fig.tight_layout()
    if path: fig.savefig(path, bbox_inches="tight")
    plt.show(); return fig


def mass_curves(recs, path=None):
    """Mean total attention mass per epoch (PDF Part D): softmax is 1 by construction; sigmoid is free."""
    style(); fig, ax = plt.subplots(figsize=(5.5, 3.6)); arms = sorted({r["arm"] for r in recs})
    for r in recs:
        m = [x for x in r.get("mass_per_epoch", []) if x is not None]
        if m: ax.plot(range(len(m)), m, color=OKABE_ITO[arms.index(r["arm"]) % len(OKABE_ITO)], alpha=0.8, label=r["arm"])
    h, l = ax.get_legend_handles_labels(); uniq = dict(zip(l, h)); ax.legend(uniq.values(), uniq.keys())
    ax.set_xlabel("epoch"); ax.set_ylabel("mean total attention mass  $\\Sigma_j a_{tj}$"); ax.set_title("Total attention mass during training (one line per seed)")
    fig.tight_layout()
    if path: fig.savefig(path, bbox_inches="tight")
    plt.show(); return fig
