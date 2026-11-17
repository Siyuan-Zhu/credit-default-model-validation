"""Weight-of-evidence binning for the scorecard-style champion model.

Bins are fit on the training split only. Each feature starts from up to
``max_bins`` quantile bins (or its distinct values when there are few), then
adjacent bins are merged until every bin holds at least ``min_bin_frac`` of the
rows and contains both outcomes. WoE = ln(%good / %bad), so a higher WoE means
lower risk.

Coarse classing then forces a monotone WoE trend. The binner builds two
candidates, one with WoE increasing in the feature and one decreasing, by
repeatedly merging the adjacent pair of bins that most violates the trend. It
keeps the candidate with the higher information value (ties go to the
direction of the Spearman correlation between feature and target). Everything
is computed from the rows passed to ``fit``, which is the training split only.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


class WoEBinner:
    def __init__(self, max_bins=10, min_bin_frac=0.05, smoothing=0.5, monotone=True):
        self.max_bins = max_bins
        self.min_bin_frac = min_bin_frac
        self.smoothing = smoothing
        self.monotone = monotone
        self.edges_ = {}
        self.woe_ = {}
        self.tables_ = {}
        self.direction_ = {}
        self.n_fit_rows_ = 0

    # ---- fitting -------------------------------------------------------
    def _initial_edges(self, x):
        uniq = np.unique(x)
        if len(uniq) <= self.max_bins:
            return list((uniq[:-1] + uniq[1:]) / 2)
        qs = np.quantile(x, np.linspace(0, 1, self.max_bins + 1)[1:-1])
        return sorted(set(np.round(qs, 10)))

    @staticmethod
    def _assign(x, edges):
        return np.searchsorted(np.asarray(edges), x, side="right")

    def _merge(self, x, y, edges):
        edges = list(edges)
        n = len(x)
        while edges:
            idx = self._assign(x, edges)
            k = len(edges) + 1
            counts = np.bincount(idx, minlength=k)
            bads = np.bincount(idx, weights=y, minlength=k)
            weak = (counts < self.min_bin_frac * n) | (bads == 0) | (bads == counts)
            if not weak.any():
                break
            # merge the smallest weak bin into its smaller neighbour
            b = int(np.argmin(np.where(weak, counts, np.inf)))
            if b == 0:
                drop = 0
            elif b == k - 1:
                drop = k - 2
            else:
                drop = b - 1 if counts[b - 1] <= counts[b + 1] else b
            edges.pop(drop)
        return edges

    def _woe(self, x, y, edges):
        """Counts, bads, WoE and IV parts for the bins defined by ``edges``."""
        idx = self._assign(x, edges)
        k = len(edges) + 1
        counts = np.bincount(idx, minlength=k)
        bads = np.bincount(idx, weights=y, minlength=k)
        goods = counts - bads
        n_bad, n_good = y.sum(), len(y) - y.sum()
        s = self.smoothing
        pct_good = (goods + s) / (n_good + s * k)
        pct_bad = (bads + s) / (n_bad + s * k)
        woe = np.log(pct_good / pct_bad)
        return counts, bads, woe, (pct_good - pct_bad) * woe

    def _monotone(self, x, y, edges, sign):
        """Merge adjacent bins until WoE moves in direction ``sign`` (+1 or -1)."""
        edges = list(edges)
        while edges:
            woe = self._woe(x, y, edges)[2]
            steps = sign * np.diff(woe)
            if (steps >= 0).all():
                break
            edges.pop(int(np.argmin(steps)))  # merge the worst violating pair
        return edges

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.n_fit_rows_ = len(y)
        self.fit_index_ = X.index.copy()
        for col in X.columns:
            x = X[col].to_numpy(dtype=float)
            edges = self._merge(x, y, self._initial_edges(x))
            direction = 0
            if self.monotone:
                rho = spearmanr(x, y).statistic if np.ptp(x) > 0 else 0.0
                rho = 0.0 if np.isnan(rho) else rho
                cands = []
                for sign in (1, -1):
                    e = self._monotone(x, y, edges, sign)
                    iv = float(self._woe(x, y, e)[3].sum())
                    # higher IV wins; on a tie prefer the Spearman direction
                    # (WoE rises with the feature when risk falls with it)
                    cands.append((round(iv, 10), -sign * rho, sign, e))
                _, _, direction, edges = max(cands, key=lambda c: (c[0], c[1]))
            counts, bads, woe, iv_parts = self._woe(x, y, edges)
            k = len(edges) + 1
            self.direction_[col] = direction
            self.edges_[col] = edges
            self.woe_[col] = woe
            lows = [-np.inf] + edges
            highs = edges + [np.inf]
            self.tables_[col] = pd.DataFrame({
                "feature": col,
                "bin": range(k),
                "lower": lows,
                "upper": highs,
                "label": [_label(lo, hi) for lo, hi in zip(lows, highs)],
                "n": counts,
                "share": counts / len(y),
                "bads": bads.astype(int),
                "bad_rate": bads / counts,
                "woe": woe,
                "iv": iv_parts,
            })
        return self

    # ---- outputs -------------------------------------------------------
    def bin_index(self, X, col):
        return self._assign(X[col].to_numpy(dtype=float), self.edges_[col])

    def transform(self, X):
        return pd.DataFrame(
            {col: self.woe_[col][self.bin_index(X, col)] for col in self.edges_},
            index=X.index,
        )

    def iv_table(self):
        rows = []
        for col, t in self.tables_.items():
            rows.append({"feature": col, "iv": t["iv"].sum(), "n_bins": len(t),
                         "monotone_woe": _is_monotone(t["woe"]),
                         "monotone_bad_rate": _is_monotone(t["bad_rate"])})
        return pd.DataFrame(rows).sort_values("iv", ascending=False, ignore_index=True)


def _is_monotone(values):
    d = np.diff(np.asarray(values, dtype=float))
    return bool(np.all(d >= 0) or np.all(d <= 0))


def _fmt(v):
    if float(v).is_integer():
        return f"{v:.0f}"
    return f"{v:.3g}" if abs(v) < 1000 else f"{v:,.0f}"


def _label(lo, hi):
    if np.isinf(lo):
        return f"< {_fmt(hi)}"
    if np.isinf(hi):
        return f">= {_fmt(lo)}"
    return f"[{_fmt(lo)}, {_fmt(hi)})"
