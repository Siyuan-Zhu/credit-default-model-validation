"""Weight-of-evidence binning for the scorecard-style champion model.

Bins are fit on the training split only. Each feature starts from up to
``max_bins`` quantile bins (or its distinct values when there are few), then
adjacent bins are merged until every bin holds at least ``min_bin_frac`` of the
rows and contains both outcomes. WoE = ln(%good / %bad), so a higher WoE means
lower risk. Bins are not forced to be monotone; the bin table reports whether
the bad-rate trend is monotone so a reviewer can judge it.
"""
import numpy as np
import pandas as pd


class WoEBinner:
    def __init__(self, max_bins=10, min_bin_frac=0.05, smoothing=0.5):
        self.max_bins = max_bins
        self.min_bin_frac = min_bin_frac
        self.smoothing = smoothing
        self.edges_ = {}
        self.woe_ = {}
        self.tables_ = {}

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

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        n_bad, n_good = y.sum(), len(y) - y.sum()
        s = self.smoothing
        for col in X.columns:
            x = X[col].to_numpy(dtype=float)
            edges = self._merge(x, y, self._initial_edges(x))
            idx = self._assign(x, edges)
            k = len(edges) + 1
            counts = np.bincount(idx, minlength=k)
            bads = np.bincount(idx, weights=y, minlength=k)
            goods = counts - bads
            pct_good = (goods + s) / (n_good + s * k)
            pct_bad = (bads + s) / (n_bad + s * k)
            woe = np.log(pct_good / pct_bad)
            iv_parts = (pct_good - pct_bad) * woe
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
            br = t["bad_rate"].to_numpy()
            diffs = np.diff(br)
            monotone = bool(np.all(diffs >= 0) or np.all(diffs <= 0))
            rows.append({"feature": col, "iv": t["iv"].sum(), "n_bins": len(t),
                         "monotone_bad_rate": monotone})
        return pd.DataFrame(rows).sort_values("iv", ascending=False, ignore_index=True)


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
