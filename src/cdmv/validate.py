"""Compute every validation metric on the scored splits and save it for the report."""
import json
import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .config import MODEL_DIR, OUTPUT_DIR, SEED, TABLES_DIR
from .metrics import (bootstrap_auc, calibration_table, gains_table, hosmer_lemeshow,
                      performance, psi, psi_band)
from .models import (FEATURES, TARGET, load_modeling_frame, pd_to_score, predict_challenger,
                     predict_champion, reason_codes)
from .warehouse import connect

MODELS = {"champion": predict_champion, "challenger": predict_challenger}
LABELS = {"champion": "Champion (WoE logistic)", "challenger": "Challenger (gradient boosting)"}
N_REPEATS = 5                     # permutation-importance repeats
PDP_SAMPLE = 2000                 # accounts in the partial-dependence sample
PDP_RANGE = (0.02, 0.98)          # train quantiles spanned by the partial-dependence grid
REASON_QUANTILES = [0.50, 0.90, 0.95, 0.99]   # PD quantiles of the example accounts


def load_artifacts():
    arts = {}
    for name in MODELS:
        with open(MODEL_DIR / f"{name}.pkl", "rb") as f:
            arts[name] = pickle.load(f)
    return arts


def permutation_importance(art, predict, df, n_repeats=N_REPEATS, seed=SEED):
    rng = np.random.default_rng(seed)
    base = roc_auc_score(df[TARGET], predict(art, df))
    rows = []
    for f in FEATURES:
        drops = []
        for _ in range(n_repeats):
            d = df.copy()
            d[f] = rng.permutation(d[f].to_numpy())
            drops.append(base - roc_auc_score(d[TARGET], predict(art, d)))
        rows.append({"feature": f, "auc_drop": float(np.mean(drops)), "auc_drop_sd": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("auc_drop", ascending=False, ignore_index=True)


def partial_dependence(art, predict, df, feature, grid):
    out = []
    for v in grid:
        d = df.copy()
        d[feature] = v
        out.append(float(np.mean(predict(art, d))))
    return out


def main():
    con = connect()
    df = load_modeling_frame(con)
    con.close()
    arts = load_artifacts()
    for name, predict in MODELS.items():
        df[f"pd_{name}"] = predict(arts[name], df)
    splits = {s: df[df.split == s] for s in ("train", "validation", "test")}
    train, test = splits["train"], splits["test"]
    out = {"labels": LABELS}

    # performance by split
    perf = []
    for name in MODELS:
        for s, d in splits.items():
            perf.append({"model": name, "split": s, **performance(d[TARGET], d[f"pd_{name}"])})
    perf = pd.DataFrame(perf)
    perf.to_csv(TABLES_DIR / "performance.csv", index=False)
    out["performance"] = perf.to_dict(orient="records")

    # benchmark: paired bootstrap on test
    out["bootstrap_test"] = bootstrap_auc(test[TARGET], test.pd_champion, test.pd_challenger, seed=SEED)
    out["spearman_test"] = float(spearmanr(test.pd_champion, test.pd_challenger).statistic)

    # calibration on test
    out["calibration"] = {}
    cal_rows = []
    for name in MODELS:
        t = calibration_table(test[TARGET], test[f"pd_{name}"])
        hl = hosmer_lemeshow(t, fitted_on_sample=False)  # test is an independent holdout
        obs = t["observed_dr"].to_numpy()
        out["calibration"][name] = {
            "hosmer_lemeshow": hl,
            "max_abs_decile_gap_pp": float(t["diff_pp"].abs().max()),
            "observed_dr_monotone": bool(np.all(np.diff(obs) >= 0)),
            "top_bottom_decile_ratio": float(obs[-1] / obs[0]) if obs[0] > 0 else None,
        }
        cal_rows.append(t.assign(model=name))
    pd.concat(cal_rows).to_csv(TABLES_DIR / "calibration_deciles_test.csv", index=False)

    # gains / score bands for the champion on test (band 1 = riskiest)
    gains = gains_table(test[TARGET], test.pd_champion)
    gains["min_score"] = pd_to_score(gains["max_pd"])
    gains["max_score"] = pd_to_score(gains["min_pd"])
    gains.to_csv(TABLES_DIR / "gains_champion_test.csv", index=False)

    # adverse-action reason codes for example test accounts at chosen PD quantiles
    ranked = test.sort_values("pd_champion", ignore_index=True)
    picks = sorted({min(int(q * len(ranked)), len(ranked) - 1) for q in REASON_QUANTILES})
    ex = ranked.iloc[picks]
    rc = reason_codes(arts["champion"], ex)
    rc.insert(1, "pd_quantile", [i / len(ranked) for i in picks])
    rc.insert(2, "pd", ex.pd_champion.to_numpy())
    rc.insert(3, "score", pd_to_score(ex.pd_champion.to_numpy()))
    rc.to_csv(TABLES_DIR / "reason_codes_examples.csv", index=False)

    # score stability (PSI on train deciles of PD)
    score_psi = []
    for name in MODELS:
        for s in ("validation", "test"):
            v, _ = psi(train[f"pd_{name}"], splits[s][f"pd_{name}"])
            score_psi.append({"model": name, "comparison": f"train vs {s}", "psi": v, "band": psi_band(v)})
    score_psi = pd.DataFrame(score_psi)
    score_psi.to_csv(TABLES_DIR / "score_psi.csv", index=False)
    out["score_psi"] = score_psi.to_dict(orient="records")

    # feature stability (PSI on the champion's WoE bins, fit on train)
    binner = arts["champion"]["binner"]
    feat_psi = []
    for f in FEATURES:
        v, _ = psi(train[f], test[f], edges=binner.edges_[f])
        feat_psi.append({"feature": f, "psi": v, "band": psi_band(v),
                         "in_champion": f in arts["champion"]["features"]})
    feat_psi = pd.DataFrame(feat_psi).sort_values("psi", ascending=False, ignore_index=True)
    feat_psi.to_csv(TABLES_DIR / "feature_psi.csv", index=False)
    out["feature_psi_max"] = feat_psi.iloc[0].to_dict()

    # sensitivity: permutation importance on test
    imp = []
    for name, predict in MODELS.items():
        imp.append(permutation_importance(arts[name], predict, test).assign(model=name))
    imp = pd.concat(imp, ignore_index=True)
    imp.to_csv(TABLES_DIR / "permutation_importance_test.csv", index=False)

    # sensitivity: partial dependence for the challenger's top drivers (both models)
    top = imp[imp.model == "challenger"].head(4)["feature"].tolist()
    sample = test.sample(n=min(PDP_SAMPLE, len(test)), random_state=SEED)
    pdp = []
    for f in top:
        grid = np.unique(np.quantile(train[f], np.linspace(*PDP_RANGE, 25)))
        for name, predict in MODELS.items():
            for g, v in zip(grid, partial_dependence(arts[name], predict, sample, f, grid)):
                pdp.append({"feature": f, "model": name, "value": float(g), "mean_pd": v})
    pd.DataFrame(pdp).to_csv(TABLES_DIR / "partial_dependence.csv", index=False)
    out["pdp_features"] = top
    out["pdp_sample_n"] = int(len(sample))
    out["pdp_range"] = list(PDP_RANGE)
    out["n_repeats"] = N_REPEATS

    # segment performance on test by credit-limit tercile (cut points from train)
    edges = np.quantile(train["credit_limit"], [1 / 3, 2 / 3])
    seg_names = [f"< {edges[0]:,.0f}", f"[{edges[0]:,.0f}, {edges[1]:,.0f})", f">= {edges[1]:,.0f}"]
    seg = np.searchsorted(edges, test["credit_limit"], side="right")
    seg_rows = []
    for i, label in enumerate(seg_names):
        d = test[seg == i]
        row = {"segment": label, "n": int(len(d)), "default_rate": float(d[TARGET].mean())}
        for name in MODELS:
            row[f"auc_{name}"] = float(roc_auc_score(d[TARGET], d[f"pd_{name}"]))
            row[f"mean_pd_{name}"] = float(d[f"pd_{name}"].mean())
        seg_rows.append(row)
    pd.DataFrame(seg_rows).to_csv(TABLES_DIR / "segment_performance_test.csv", index=False)

    (OUTPUT_DIR / "validation.json").write_text(json.dumps(out, indent=2, default=float))
    print(perf.to_string(index=False))
    print(score_psi.to_string(index=False))
    print(feat_psi.head(5).to_string(index=False))
    print(json.dumps({k: out[k] for k in ("bootstrap_test", "spearman_test", "calibration")}, indent=2))


if __name__ == "__main__":
    main()
