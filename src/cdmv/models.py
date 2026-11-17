"""Champion (WoE logistic regression scorecard) and challenger (gradient boosting)."""
import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .woe import WoEBinner

FEATURES = [
    "credit_limit", "util_m1", "avg_util_6m", "max_util_6m", "util_change_6m",
    "bill_growth_3m", "n_months_over_limit_6m", "months_delinquent_m1",
    "max_months_delinquent_6m", "n_months_delinquent_6m", "current_delinq_streak",
    "longest_delinq_streak_6m", "n_months_paid_duly_6m", "n_months_status_neg2_6m",
    "n_months_status_zero_6m", "pay_ratio_m1", "avg_pay_ratio_5m", "min_pay_ratio_5m",
    "n_months_zero_payment_6m", "pay_to_limit_6m",
]
TARGET = "default_flag"

IV_MIN = 0.02          # drop features with weak information value
CORR_MAX = 0.70        # drop the weaker of any pair of WoE features above this |r|
C_GRID = [0.01, 0.1, 1.0, 10.0]
GBM_GRID = {
    "learning_rate": [0.03, 0.1],
    "max_leaf_nodes": [8, 31],
    "max_iter": [200, 400],
}

# Scorecard scaling: 600 points at 50:1 good:bad odds, 20 points to double the odds.
BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)


def load_modeling_frame(con):
    return con.execute(
        "select f.*, s.split from features.account_features f "
        "join features.model_split s using (account_id) order by account_id"
    ).df()


# ---- champion ------------------------------------------------------------
def select_features(binner, X_woe):
    """IV filter, then greedy correlation filter in order of IV."""
    iv = binner.iv_table()
    log, keep = [], []
    corr = X_woe.corr().abs()
    for row in iv.itertuples():
        if row.iv < IV_MIN:
            log.append({"feature": row.feature, "iv": row.iv, "decision": "dropped",
                        "reason": f"IV < {IV_MIN}"})
            continue
        clash = [k for k in keep if corr.loc[row.feature, k] > CORR_MAX]
        if clash:
            log.append({"feature": row.feature, "iv": row.iv, "decision": "dropped",
                        "reason": f"WoE correlation {corr.loc[row.feature, clash[0]]:.2f} with {clash[0]}"})
            continue
        keep.append(row.feature)
        log.append({"feature": row.feature, "iv": row.iv, "decision": "kept", "reason": ""})
    return keep, pd.DataFrame(log)


def fit_champion(train, valid, features=FEATURES):
    binner = WoEBinner().fit(train[features], train[TARGET])
    Xtr, Xva = binner.transform(train[features]), binner.transform(valid[features])
    keep, selection = select_features(binner, Xtr)

    # Refit until every coefficient has the expected (negative) sign: a higher
    # WoE means lower risk, so a positive coefficient would invert the bin order.
    while True:
        grid = []
        for C in C_GRID:
            m = LogisticRegression(C=C, max_iter=1000).fit(Xtr[keep], train[TARGET])
            grid.append((roc_auc_score(valid[TARGET], m.predict_proba(Xva[keep])[:, 1]), C, m))
        best_auc, best_C, model = max(grid, key=lambda g: (round(g[0], 4), -g[1]))
        wrong = [f for f, c in zip(keep, model.coef_[0]) if c >= 0]
        if not wrong:
            break
        drop = wrong[0]
        keep.remove(drop)
        selection.loc[selection.feature == drop, ["decision", "reason"]] = [
            "dropped", "coefficient sign reversed in multivariate fit"]

    return {
        "binner": binner, "model": model, "features": keep, "C": best_C,
        "valid_auc": best_auc, "selection": selection,
        "c_grid": pd.DataFrame([{"C": C, "valid_auc": a} for a, C, _ in grid]),
    }


def predict_champion(art, df):
    X = art["binner"].transform(df[FEATURES])[art["features"]]
    return art["model"].predict_proba(X)[:, 1]


def pd_to_score(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return OFFSET + FACTOR * np.log((1 - p) / p)


def scorecard_points(art):
    """Points per bin; an account's score is the sum of its bin points."""
    m, feats = art["model"], art["features"]
    n = len(feats)
    rows = []
    for coef, f in zip(m.coef_[0], feats):
        t = art["binner"].tables_[f]
        for r in t.itertuples():
            # log-odds of good = -(intercept + sum coef * woe), spread evenly
            pts = FACTOR * (-(coef * r.woe) - m.intercept_[0] / n) + OFFSET / n
            rows.append({"feature": f, "bin": r.label, "n": r.n, "bad_rate": r.bad_rate,
                         "woe": r.woe, "points": pts})
    return pd.DataFrame(rows)


# ---- challenger ----------------------------------------------------------
def fit_challenger(train, valid, features=FEATURES, seed=42):
    results = []
    best = None
    keys = list(GBM_GRID)
    for values in itertools.product(*GBM_GRID.values()):
        params = dict(zip(keys, values))
        m = HistGradientBoostingClassifier(
            **params, min_samples_leaf=50, l2_regularization=1.0,
            early_stopping=False, random_state=seed,
        ).fit(train[features], train[TARGET])
        auc = roc_auc_score(valid[TARGET], m.predict_proba(valid[features])[:, 1])
        results.append({**params, "valid_auc": auc})
        if best is None or auc > best[0]:
            best = (auc, params, m)
    return {"model": best[2], "params": best[1], "valid_auc": best[0],
            "features": list(features), "grid": pd.DataFrame(results)}


def predict_challenger(art, df):
    return art["model"].predict_proba(df[art["features"]])[:, 1]
