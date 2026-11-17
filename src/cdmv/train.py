"""Fit the champion and challenger on train, select settings on validation, score all splits."""
import json
import pickle

import pandas as pd

from .config import MODEL_DIR, OUTPUT_DIR, SEED, TABLES_DIR
from .models import (C_GRID, FEATURES, GBM_FIXED, MIN_POINTS_SPREAD, fit_challenger, fit_champion, load_modeling_frame,
                     pd_to_score, predict_challenger, predict_champion, scorecard_points)
from .warehouse import connect


def main():
    con = connect()
    df = load_modeling_frame(con)
    train, valid = df[df.split == "train"], df[df.split == "validation"]

    champ = fit_champion(train, valid)
    chall = fit_challenger(train, valid, seed=SEED)

    scores = df[["account_id", "split", "default_flag"]].copy()
    scores["pd_champion"] = predict_champion(champ, df)
    scores["pd_challenger"] = predict_challenger(chall, df)
    scores["score_champion"] = pd_to_score(scores["pd_champion"])
    con.execute("create schema if not exists model")
    con.register("scores_df", scores)
    con.execute("create or replace table model.scores as select * from scores_df")
    con.close()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "champion.pkl", "wb") as f:
        pickle.dump(champ, f)
    with open(MODEL_DIR / "challenger.pkl", "wb") as f:
        pickle.dump(chall, f)

    # documentation tables
    bins = pd.concat(champ["binner"].tables_.values(), ignore_index=True)
    bins.to_csv(TABLES_DIR / "woe_bins.csv", index=False)
    iv = champ["binner"].iv_table().merge(
        champ["selection"][["feature", "decision", "reason"]], on="feature")
    iv.to_csv(TABLES_DIR / "iv_and_selection.csv", index=False)
    coefs = pd.DataFrame({"feature": ["intercept"] + champ["features"],
                          "coefficient": [champ["model"].intercept_[0]] + list(champ["model"].coef_[0])})
    coefs.to_csv(TABLES_DIR / "champion_coefficients.csv", index=False)
    scorecard_points(champ).to_csv(TABLES_DIR / "scorecard_points.csv", index=False)
    champ["c_grid"].to_csv(TABLES_DIR / "champion_c_grid.csv", index=False)
    chall["grid"].to_csv(TABLES_DIR / "challenger_grid.csv", index=False)

    summary = {
        "n_candidate_features": len(FEATURES),
        "champion": {"features": champ["features"], "C": champ["C"],
                     "valid_auc": champ["valid_auc"], "c_grid": C_GRID,
                     "min_points_spread": MIN_POINTS_SPREAD},
        "challenger": {"params": chall["params"], "fixed_params": GBM_FIXED,
                       "valid_auc": chall["valid_auc"]},
        "split_counts": df.split.value_counts().to_dict(),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(iv.to_string(index=False))


if __name__ == "__main__":
    main()
