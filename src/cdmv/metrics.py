"""Discrimination, calibration and stability metrics used in the validation."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve


def ks_statistic(y, p):
    """Maximum distance between the cumulative score distributions of bads and goods."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def performance(y, p):
    auc = roc_auc_score(y, p)
    return {
        "n": int(len(y)),
        "default_rate": float(np.mean(y)),
        "mean_pd": float(np.mean(p)),
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y, p),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
    }


def bootstrap_auc(y, p_a, p_b, n_boot=1000, seed=42):
    """Paired bootstrap of AUC for two models and their difference (b - a)."""
    rng = np.random.default_rng(seed)
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    a, b = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        a.append(roc_auc_score(y[i], p_a[i]))
        b.append(roc_auc_score(y[i], p_b[i]))
    a, b = np.array(a), np.array(b)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"auc_a_ci": ci(a), "auc_b_ci": ci(b), "diff_mean": float(np.mean(b - a)),
            "diff_ci": ci(b - a), "n_boot": int(len(a))}


def calibration_table(y, p, n_bins=10):
    """Deciles of predicted PD (decile 1 = lowest risk)."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["decile"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False) + 1
    t = df.groupby("decile").agg(n=("y", "size"), defaults=("y", "sum"),
                                 mean_pd=("p", "mean"), observed_dr=("y", "mean"),
                                 min_pd=("p", "min"), max_pd=("p", "max")).reset_index()
    t["defaults"] = t["defaults"].astype(int)
    t["diff_pp"] = (t["observed_dr"] - t["mean_pd"]) * 100
    return t


def hosmer_lemeshow(table):
    """HL chi-square on a decile calibration table, df = bins - 2."""
    n, o, e = table["n"], table["defaults"], table["mean_pd"] * table["n"]
    chi2 = float((((o - e) ** 2) / (e * (1 - e / n))).sum())
    dof = len(table) - 2
    return {"chi2": chi2, "dof": dof, "p_value": float(stats.chi2.sf(chi2, dof))}


def psi(expected, actual, edges=None, n_bins=10, eps=1e-4):
    """Population stability index of ``actual`` against ``expected``.

    Bins are deciles of ``expected`` unless explicit inner ``edges`` are given.
    Returns (psi, per-bin table).
    """
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    if edges is None:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)[1:-1]))
    edges = np.asarray(edges, float)
    k = len(edges) + 1
    e = np.bincount(np.searchsorted(edges, expected, side="right"), minlength=k) / len(expected)
    a = np.bincount(np.searchsorted(edges, actual, side="right"), minlength=k) / len(actual)
    e_, a_ = np.clip(e, eps, None), np.clip(a, eps, None)
    parts = (a_ - e_) * np.log(a_ / e_)
    table = pd.DataFrame({"bin": range(k), "expected_share": e, "actual_share": a, "psi": parts})
    return float(parts.sum()), table


def psi_band(value):
    """Conventional reading of PSI."""
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "moderate shift"
    return "significant shift"
