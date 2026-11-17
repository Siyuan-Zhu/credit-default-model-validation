"""Static validation figures written to reports/figures/."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .config import FIGURES_DIR, OUTPUT_DIR, TABLES_DIR  # noqa: E402
from .warehouse import connect  # noqa: E402

COLORS = {"champion": "#2a78d6", "challenger": "#eb6834"}
OUTCOME = {"non-default": "#1baf7a", "default": "#6250d6"}
INK, INK2, GRID, SURFACE, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#a3a19a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": NEUTRAL, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "legend.frameon": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "font.family": "DejaVu Sans", "figure.dpi": 150,
})


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, bbox_inches="tight")
    plt.close(fig)


def roc(test, labels, perf):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=NEUTRAL, lw=1, ls="--", label="Random (AUC 0.500)")
    for m in COLORS:
        fpr, tpr, _ = roc_curve(test.default_flag, test[f"pd_{m}"])
        auc = perf[(perf.model == m) & (perf.split == "test")].auc.iloc[0]
        ax.plot(fpr, tpr, color=COLORS[m], label=f"{labels[m]} (AUC {auc:.3f})")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1),
           title="ROC curve, test split")
    ax.legend(loc="lower right")
    _save(fig, "roc.png")


def ks(test, labels):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharey=True)
    for ax, m in zip(axes, COLORS):
        y = test.default_flag.to_numpy()
        s = test[f"pd_{m}"].to_numpy()
        grid = np.quantile(s, np.linspace(0, 1, 201))
        cb = [(s[y == 1] <= g).mean() for g in grid]
        cg = [(s[y == 0] <= g).mean() for g in grid]
        gap = np.subtract(cg, cb)
        i = int(np.argmax(gap))
        ax.plot(grid, cg, color=OUTCOME["non-default"], label="Non-defaults (cumulative)")
        ax.plot(grid, cb, color=OUTCOME["default"], label="Defaults (cumulative)")
        ax.vlines(grid[i], cb[i], cg[i], color=INK, lw=1.2)
        ax.annotate(f"KS = {gap[i]:.3f}", (grid[i], (cb[i] + cg[i]) / 2), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK,
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none"))
        ax.set(xlabel="Predicted PD", title=labels[m], xlim=(0, 1))
    axes[0].set_ylabel("Cumulative share of accounts")
    axes[0].legend(loc="lower right")
    _save(fig, "ks.png")


def calibration(cal, labels):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    lim = float(max(cal.mean_pd.max(), cal.observed_dr.max())) * 1.05
    ax.plot([0, lim], [0, lim], color=NEUTRAL, lw=1, ls="--", label="Perfect calibration")
    for m in COLORS:
        t = cal[cal.model == m]
        ax.plot(t.mean_pd, t.observed_dr, color=COLORS[m], marker="o", ms=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=labels[m])
    ax.set(xlabel="Mean predicted PD (decile)", ylabel="Observed default rate (decile)",
           xlim=(0, lim), ylim=(0, lim), title="Calibration by PD decile, test split")
    ax.legend(loc="upper left")
    _save(fig, "calibration.png")


def score_distribution(scores, labels):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4))
    bins = np.linspace(0, 1, 41)
    for row, m in enumerate(COLORS):
        col = f"pd_{m}"
        ax = axes[row, 0]
        tr, te = scores[scores.split == "train"][col], scores[scores.split == "test"][col]
        ax.hist(tr, bins=bins, density=True, color=NEUTRAL, alpha=0.6, label="Train", edgecolor=SURFACE)
        ax.hist(te, bins=bins, density=True, histtype="step", color=COLORS[m], lw=2, label="Test")
        ax.set(title=f"{labels[m]}: train vs test", ylabel="Density")
        ax.legend()
        ax = axes[row, 1]
        test = scores[scores.split == "test"]
        for outcome, flag in (("non-default", 0), ("default", 1)):
            ax.hist(test[test.default_flag == flag][col], bins=bins, density=True, histtype="step",
                    color=OUTCOME[outcome], lw=2, label=outcome.capitalize())
        ax.set(title=f"{labels[m]}: test by outcome")
        ax.legend()
    for ax in axes[1]:
        ax.set_xlabel("Predicted PD")
    _save(fig, "score_distribution.png")


def feature_importance(imp, labels):
    piv = imp.pivot(index="feature", columns="model", values="auc_drop").fillna(0)
    piv = piv.loc[piv.max(axis=1).sort_values().index].tail(14)
    fig, ax = plt.subplots(figsize=(7, 5.4))
    y = np.arange(len(piv))
    h = 0.38
    for k, m in enumerate(COLORS):
        ax.barh(y + (0.5 - k) * h, piv[m].clip(lower=0), height=h - 0.04, color=COLORS[m], label=labels[m])
    ax.set_yticks(y, piv.index)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Drop in test AUC when the feature is shuffled (mean of 5 repeats)",
           title="Permutation importance, test split")
    ax.legend(loc="lower right")
    _save(fig, "feature_importance.png")


def partial_dependence(pdp, labels):
    feats = list(dict.fromkeys(pdp.feature))
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4), sharey=True)
    for ax, f in zip(axes.ravel(), feats):
        for m in COLORS:
            t = pdp[(pdp.feature == f) & (pdp.model == m)]
            ax.plot(t.value, t.mean_pd, color=COLORS[m], marker="o", ms=3, label=labels[m])
        ax.set(title=f, xlabel="Feature value (2nd to 98th percentile of train)")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if abs(v) >= 1000 else f"{v:g}"))
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean predicted PD")
    axes[0, 0].legend()
    _save(fig, "partial_dependence.png")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    con = connect()
    scores = con.execute("select * from model.scores").df()
    con.close()
    labels = json.loads((OUTPUT_DIR / "validation.json").read_text())["labels"]
    perf = pd.read_csv(TABLES_DIR / "performance.csv")
    test = scores[scores.split == "test"]
    roc(test, labels, perf)
    ks(test, labels)
    calibration(pd.read_csv(TABLES_DIR / "calibration_deciles_test.csv"), labels)
    score_distribution(scores, labels)
    feature_importance(pd.read_csv(TABLES_DIR / "permutation_importance_test.csv"), labels)
    partial_dependence(pd.read_csv(TABLES_DIR / "partial_dependence.csv"), labels)
    print("figures:", ", ".join(sorted(p.name for p in FIGURES_DIR.glob("*.png"))))


if __name__ == "__main__":
    main()
