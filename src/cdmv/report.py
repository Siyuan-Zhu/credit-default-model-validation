"""Write reports/model_validation.md and the README results block from pipeline outputs.

Every number in the report is read from the warehouse, reports/tables/*.csv or
output/*.json. Judgements (for example "stable" or a finding's severity) are
derived from those numbers with the thresholds defined below, so a rerun on
different data produces a consistent report.
"""
import json
import re

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR, REPORTS_DIR, ROOT, SQL_DIR, TABLES_DIR
from .metrics import psi_band
from .models import BASE_ODDS, BASE_SCORE, CORR_MAX, FEATURES, IV_MIN, PDO
from .warehouse import connect, query_file
from .woe import WoEBinner

# Thresholds used for the report's judgements (conventions, not regulation).
AUC_MIN, KS_MIN = 0.70, 0.30
HL_ALPHA = 0.05
OVERFIT_GAP = 0.015          # train AUC minus test AUC
NEGLIGIBLE_DROP = 0.0005     # permutation AUC drop below which a feature adds nothing


# ---- formatting helpers --------------------------------------------------------
def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def n0(x):
    return f"{int(x):,}"


def md_table(df, align=None):
    cols = list(df.columns)
    align = align or ["l"] + ["r"] * (len(cols) - 1)
    sep = ["---:" if a == "r" else ":---" for a in align]
    lines = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join(sep) + " |"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def fmt_range(lo, hi):
    return f"{lo:.3f} to {hi:.3f}"


# ---- load everything -----------------------------------------------------------
def load():
    con = connect()
    d = {}
    d["n_accounts"], d["default_rate"] = con.execute(
        "select count(*), avg(default_flag) from staging.stg_accounts").fetchone()
    d["status_share"] = con.execute(
        "select pay_status, count(*) * 1.0 / sum(count(*)) over () as share "
        "from staging.stg_account_months group by 1 order by 1").df()
    d["split"] = con.execute(
        "select split, count(*) as n, sum(default_flag) as defaults, avg(default_flag) as dr "
        "from features.model_split join features.account_features using (account_id) "
        "group by 1 order by case split when 'train' then 1 when 'validation' then 2 else 3 end").df()
    d["codes"] = query_file(con, SQL_DIR / "profiles" / "category_codes.sql")
    d["n_rows_months"] = con.execute("select count(*) from staging.stg_account_months").fetchone()[0]
    con.close()

    t = lambda name: pd.read_csv(TABLES_DIR / name)
    d["dq"] = t("dq_results.csv")
    d["perf"] = t("performance.csv")
    d["cal"] = t("calibration_deciles_test.csv")
    d["score_psi"] = t("score_psi.csv")
    d["feat_psi"] = t("feature_psi.csv")
    d["imp"] = t("permutation_importance_test.csv")
    d["iv"] = t("iv_and_selection.csv")
    d["coefs"] = t("champion_coefficients.csv")
    d["points"] = t("scorecard_points.csv")
    d["c_grid"] = t("champion_c_grid.csv")
    d["gbm_grid"] = t("challenger_grid.csv")
    d["segments"] = t("segment_performance_test.csv")
    d["val"] = json.loads((OUTPUT_DIR / "validation.json").read_text())
    d["train"] = json.loads((OUTPUT_DIR / "train_summary.json").read_text())
    return d


def perf_row(d, model, split):
    p = d["perf"]
    return p[(p.model == model) & (p.split == split)].iloc[0]


# ---- findings ------------------------------------------------------------------
def findings(d):
    out = []
    v = d["val"]
    champ_te, chall_te = perf_row(d, "champion", "test"), perf_row(d, "challenger", "test")
    champ_feats = d["train"]["champion"]["features"]

    out.append(("Stability", "High",
                "No out-of-time test is possible: the data covers one observation point "
                "(six monthly snapshots, April to September 2005) and one outcome month. The "
                "train/test PSI values reported below compare random samples of the same "
                "population, so they confirm the split, not stability over time.",
                "Before any use beyond demonstration, test on a later vintage and set PSI "
                "monitoring triggers (0.10 review, 0.25 redevelop)."))

    hl = v["calibration"]["champion"]["hosmer_lemeshow"]
    if hl["p_value"] < HL_ALPHA:
        cal = d["cal"][d["cal"].model == "champion"]
        over = cal[cal.diff_pp < 0].decile.tolist()
        under = cal[cal.diff_pp > 0].decile.tolist()
        out.append(("Calibration", "Medium",
                    f"Champion fails the Hosmer-Lemeshow test on test data "
                    f"(chi-square {hl['chi2']:.1f}, {hl['dof']} df, p = {hl['p_value']:.4f}). "
                    f"It over-predicts PD in deciles {_list(over)} and under-predicts in deciles "
                    f"{_list(under)}; the largest gap is "
                    f"{v['calibration']['champion']['max_abs_decile_gap_pp']:.1f} pp. Mean PD "
                    f"({pct(champ_te.mean_pd)}) matches the observed rate ({pct(champ_te.default_rate)}).",
                    "Recalibrate (for example isotonic or Platt scaling on a holdout) before "
                    "PD-level use such as loss forecasting; rank-ordering use is less affected."))

    b = v["bootstrap_test"]
    if b["diff_ci"][0] > 0:
        out.append(("Benchmark", "Medium",
                    f"The challenger's test AUC is higher by {b['diff_mean']:.3f} "
                    f"(95% paired bootstrap CI {fmt_range(*b['diff_ci'])}), so the champion's "
                    "simpler form gives up measurable discrimination.",
                    "Document the interpretability trade-off, or test whether interaction terms "
                    "or coarser, monotone bins close the gap."))
    elif b["diff_ci"][1] < 0:
        out.append(("Benchmark", "Low",
                    f"The champion outperforms the challenger on test AUC by {-b['diff_mean']:.3f}.",
                    "No action; retain the champion."))

    nonmono = d["iv"][(d["iv"].feature.isin(champ_feats)) & (~d["iv"].monotone_bad_rate)]
    if len(nonmono):
        out.append(("Methodology", "Medium",
                    f"{len(nonmono)} of {len(champ_feats)} champion features have bins whose bad "
                    f"rate is not monotone ({_list(nonmono.feature.tolist(), code=True)}). Some "
                    "reversals reflect real mass points (accounts with no bill or no payment); "
                    "others look like noise from fine quantile bins.",
                    "Coarse-class these features with documented business rationale and refit."))

    imp = d["imp"][(d["imp"].model == "champion") & (d["imp"].feature.isin(champ_feats))]
    weak = imp[imp.auc_drop < NEGLIGIBLE_DROP].feature.tolist()
    if weak:
        out.append(("Methodology", "Low",
                    f"{len(weak)} champion features change test AUC by less than {NEGLIGIBLE_DROP} "
                    f"when shuffled ({_list(weak, code=True)}).",
                    "Remove them for parsimony unless they are kept for business reasons."))

    gap = perf_row(d, "challenger", "train").auc - chall_te.auc
    if gap > OVERFIT_GAP:
        out.append(("Performance", "Low",
                    f"The challenger's train AUC exceeds its test AUC by {gap:.3f}; for the "
                    f"champion the train-minus-test gap is "
                    f"{perf_row(d, 'champion', 'train').auc - champ_te.auc:+.3f}. The challenger "
                    "shows some overfitting.",
                    "Tighten regularisation (fewer leaves, larger minimum leaf size) or use "
                    "early stopping."))

    if not v["calibration"]["champion"]["observed_dr_monotone"]:
        cal = d["cal"][d["cal"].model == "champion"].reset_index(drop=True)
        drops = [f"{int(cal.decile[i])} to {int(cal.decile[i + 1])} "
                 f"({pct(cal.observed_dr[i])} to {pct(cal.observed_dr[i + 1])})"
                 for i in range(len(cal) - 1) if cal.observed_dr[i + 1] < cal.observed_dr[i]]
        out.append(("Performance", "Low",
                    "Observed default rates across champion PD deciles are not strictly "
                    f"increasing: the rate falls from decile {_list(drops)}.",
                    "Recheck rank ordering in those deciles after coarse classing."))

    st = d["status_share"].set_index("pay_status")["share"]
    out.append(("Data", "Medium",
                f"Repayment status codes -2 and 0 are not defined in the source documentation but "
                f"make up {pct(st.get(-2, 0) + st.get(0, 0))} of account-months. The model treats "
                "them as not delinquent, which is the common reading but is not confirmed.",
                "Confirm code definitions with the data owner; rerun if the reading changes."))

    out.append(("Governance", "Medium",
                "Development and validation were done by the same person. This report follows "
                "the structure of an independent validation but is not one.",
                "Have an independent reviewer repeat the key tests."))
    return out


def _list(items, code=False):
    items = [f"`{i}`" if code else str(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# ---- sections ------------------------------------------------------------------
def headline_table(d):
    rows = []
    for m, label in d["val"]["labels"].items():
        r = perf_row(d, m, "test")
        rows.append([label, f3(r.auc), f3(r.gini), f3(r.ks), f4(r.brier), pct(r.mean_pd)])
    return md_table(pd.DataFrame(rows, columns=[
        "Model", "AUC", "Gini", "KS", "Brier", "Mean PD"]))


def report(d):
    v, tr = d["val"], d["train"]
    labels = v["labels"]
    champ_te, chall_te = perf_row(d, "champion", "test"), perf_row(d, "challenger", "test")
    b = v["bootstrap_test"]
    champ_feats = tr["champion"]["features"]
    find = findings(d)
    dq = d["dq"]
    n_fail = int((dq.status == "FAIL").sum())
    sev_counts = pd.Series([f[1] for f in find]).value_counts()

    disc_ok = champ_te.auc >= AUC_MIN and champ_te.ks >= KS_MIN
    s = []
    w = s.append

    w("# Model validation report: credit-card probability of default\n")
    w("> Generated by `make report` from the pipeline outputs. Every figure and number below is "
      "computed by the code in `src/cdmv/`; nothing is typed in by hand. This is an independent "
      "portfolio project on public data, written in the format of a model-risk validation.\n")
    w(md_table(pd.DataFrame([
        ["Model", "Probability of default (PD), next-month payment default, credit-card accounts"],
        ["Champion", f"{labels['champion']}, {len(champ_feats)} features, scorecard scaling "
                     f"{BASE_SCORE} points at {BASE_ODDS}:1 odds, {PDO} points to double the odds"],
        ["Challenger", f"{labels['challenger']}, {tr['n_candidate_features']} features"],
        ["Data", f"UCI Default of Credit Card Clients, {n0(d['n_accounts'])} accounts, Taiwan, 2005"],
        ["Validation scope", "Data quality, methodology, discrimination, calibration, stability, "
                             "sensitivity, benchmarking"],
    ], columns=["Item", "Description"]), align=["l", "l"]))
    w("")

    # summary
    w("## Summary\n")
    w(f"The champion separates defaulters from non-defaulters with a test AUC of {f3(champ_te.auc)} "
      f"and KS of {f3(champ_te.ks)}, "
      + ("above" if disc_ok else "below")
      + f" the thresholds this report uses (AUC {AUC_MIN:.2f}, KS {KS_MIN:.2f}). "
      f"Mean predicted PD on test is {pct(champ_te.mean_pd)} against an observed default rate of "
      f"{pct(champ_te.default_rate)}. "
      f"The gradient-boosting challenger reaches a test AUC of {f3(chall_te.auc)}; the paired "
      f"bootstrap difference is {b['diff_mean']:.3f} (95% CI {fmt_range(*b['diff_ci'])}). "
      f"{len(dq)} data-quality checks ran with {n_fail} failures and "
      f"{int((dq.status == 'WARN').sum())} warnings for known, documented issues.\n")
    w(f"The validation raises {len(find)} findings: "
      + ", ".join(f"{sev_counts.get(k, 0)} {k.lower()}" for k in ("High", "Medium", "Low"))
      + ". The high-severity finding is the absence of an out-of-time test, which the data "
      "cannot support. The model is suitable as a documented demonstration of PD development "
      "and validation; it is not approved for decisions on real accounts.\n")
    w("### Headline metrics (test split)\n")
    w(headline_table(d) + "\n")
    w("### Findings\n")
    w(md_table(pd.DataFrame(
        [[f"F{i}", a, sev, txt, rec] for i, (a, sev, txt, rec) in enumerate(find, 1)],
        columns=["ID", "Area", "Severity", "Finding", "Recommendation"]),
        align=["l", "l", "l", "l", "l"]) + "\n")

    # 1 purpose
    w("## 1. Purpose and intended use\n")
    w("The model estimates, for each credit-card account, the probability that the account "
      "defaults on its next monthly payment, using the previous six months of limit, billing, "
      "payment and repayment-status history. In a lending setting a PD model of this kind would "
      "feed account management (line decreases, collections prioritisation) and expected-loss "
      "estimates. Here the intended use is narrower: to show a reproducible development and "
      "validation process on public data. Out of scope: origination decisions, regulatory "
      "capital, and any decision about a real customer.\n")

    # 2 data
    w("## 2. Data\n")
    w(f"Source: UCI Machine Learning Repository, dataset 350 (Yeh, 2009, CC BY 4.0). "
      f"{n0(d['n_accounts'])} credit-card accounts from a bank in Taiwan; one row per account; "
      f"six monthly snapshots from April to September 2005 (unpivoted to "
      f"{n0(d['n_rows_months'])} account-months in staging). The target is "
      f"`default payment next month` (1 = default), with an overall default rate of "
      f"{pct(d['default_rate'], 2)}. The source does not define default more precisely "
      "(for example days past due), so the target definition is taken as given.\n")
    w("The split is stratified by target and done in SQL (`sql/features/020_model_split.sql`) "
      "by ordering accounts on an md5 hash of the account id and a fixed seed. Validation data "
      "is used only to choose hyperparameters; test data is used only for this report.\n")
    sp = d["split"].copy()
    sp["share"] = (sp.n / sp.n.sum()).map(pct)
    sp["n"] = sp.n.map(n0)
    sp["defaults"] = sp.defaults.map(n0)
    sp["dr"] = sp.dr.map(lambda x: pct(x, 2))
    w(md_table(sp.rename(columns={"split": "Split", "n": "Accounts", "defaults": "Defaults",
                                  "dr": "Default rate", "share": "Share"})) + "\n")
    w("There is no time dimension beyond the six snapshots and a single outcome month, so the "
      "test split is an out-of-sample holdout from the same period, not an out-of-time sample "
      "(finding F1).\n")

    # 3 data quality
    w("## 3. Data quality\n")
    w("Checks are SQL queries in `sql/dq/`, run by `src/cdmv/dq.py` after every build. A check "
      "with severity `error` and any failing rows stops the pipeline with a non-zero exit code; "
      "`warn` records a known issue that staging handles explicitly.\n")
    dqt = dq[["check_id", "layer", "description", "severity", "failing_rows", "total_rows", "status"]].copy()
    dqt["failing_rows"] = dqt.failing_rows.map(n0)
    dqt["total_rows"] = dqt.total_rows.map(n0)
    w(md_table(dqt.rename(columns={"check_id": "Check", "layer": "Layer", "description": "Rule",
                                   "severity": "Severity", "failing_rows": "Failing rows",
                                   "total_rows": "Rows", "status": "Status"}),
               align=["l", "l", "l", "l", "r", "r", "l"]) + "\n")
    w("Documented versus observed category codes (from `sql/profiles/category_codes.sql`; "
      "PAY counts are account-months across PAY_0 to PAY_6):\n")
    codes = d["codes"].copy()
    codes["n_values"] = codes.n_values.map(n0)
    codes["documented"] = codes.documented.map({True: "yes", False: "**no**"})
    w(md_table(codes.rename(columns={"field": "Field", "code": "Code", "n_values": "Count",
                                     "documented_meaning": "Documented meaning",
                                     "documented": "Documented"}),
               align=["l", "r", "r", "l", "l"]) + "\n")
    w("Treatment: EDUCATION codes 0, 5 and 6 are mapped to 4 (others) and MARRIAGE code 0 to 3 "
      "(others), with flags kept in staging; neither field is a model input. PAY codes -2 and 0 "
      "are kept as distinct values and counted separately (`n_months_status_neg2_6m`, "
      "`n_months_status_zero_6m`); both are treated as not delinquent "
      "(`months_delinquent = greatest(pay_status, 0)`). A common reading is -2 = no balance or "
      "no use and 0 = revolving credit paid at least the minimum, but the source does not "
      "confirm this (finding F" + str(_find_index(find, "Data")) + "). Negative bills are credit "
      "balances and bills above the limit are over-limit accounts; both are kept.\n")

    # 4 methodology
    w("## 4. Methodology\n")
    w("### 4.1 Features\n")
    w(f"{len(FEATURES)} candidate features are built in SQL (`sql/features/010_account_features.sql`) "
      "from limit, bill, payment and repayment-status fields: utilisation level, peak and trend; "
      "bill growth; over-limit months; delinquency level, counts and streaks (gaps-and-islands "
      "on the monthly table); counts of each repayment-status type; payment ratios against the "
      "previous statement; zero-payment months; and payments relative to the limit. Definitions "
      "and lineage are in `docs/data_dictionary.md`.\n")
    w("Demographic fields (SEX, MARRIAGE, AGE, EDUCATION) are staged and quality-checked but not "
      "used. Sex, marital status and age are prohibited bases under the US Equal Credit "
      "Opportunity Act (age is allowed only under specific conditions in Regulation B), and "
      "education can act as a proxy. The data is from Taiwan, so this follows a US convention "
      "rather than the source jurisdiction's rules.\n")
    w("### 4.2 Champion: weight-of-evidence logistic regression\n")
    binner = WoEBinner()
    w(f"Each feature is binned on the training split: up to {binner.max_bins} quantile bins (or "
      f"its distinct values when it has fewer), then adjacent bins are merged until each holds at "
      f"least {pct(binner.min_bin_frac, 0)} of accounts and contains both outcomes. Each bin is "
      "replaced by its weight of evidence, WoE = ln(share of non-defaults / share of defaults), "
      f"so higher WoE means lower risk. Features with information value (IV) below {IV_MIN} are "
      f"dropped, then, in order of IV, any feature whose WoE correlates above {CORR_MAX} with an "
      "already selected feature. Every coefficient must be negative (higher WoE, lower PD); a "
      "feature with a reversed sign is dropped and the model refit. The L2 penalty C is chosen on "
      "validation AUC.\n")
    iv = d["iv"].copy()
    iv["iv"] = iv.iv.map(f3)
    iv["monotone_bad_rate"] = iv.monotone_bad_rate.map({True: "yes", False: "no"})
    iv["reason"] = iv.reason.fillna("")
    w(md_table(iv.rename(columns={"feature": "Feature", "iv": "IV", "n_bins": "Bins",
                                  "monotone_bad_rate": "Monotone bad rate",
                                  "decision": "Decision", "reason": "Reason"}),
               align=["l", "r", "r", "l", "l", "l"]) + "\n")
    cg = d["c_grid"].copy()
    w("Validation AUC by C: " + ", ".join(f"C = {c:g}: {a:.4f}" for c, a in zip(cg.C, cg.valid_auc))
      + f". Selected C = {tr['champion']['C']:g}.\n")
    pts = d["points"].groupby("feature").points.agg(["min", "max"])
    co = d["coefs"].copy()
    co["points range"] = co.feature.map(lambda f: f"{pts.loc[f, 'min']:.0f} to {pts.loc[f, 'max']:.0f}"
                                        if f in pts.index else "")
    co["coefficient"] = co.coefficient.map(f4)
    w("Coefficients (on WoE inputs) and the range of scorecard points each feature can contribute "
      "(full points table: `reports/tables/scorecard_points.csv`):\n")
    w(md_table(co.rename(columns={"feature": "Feature", "coefficient": "Coefficient",
                                  "points range": "Points range"})) + "\n")
    w("### 4.3 Challenger: gradient boosting\n")
    g = d["gbm_grid"].sort_values("valid_auc", ascending=False)
    p = tr["challenger"]["params"]
    w(f"scikit-learn `HistGradientBoostingClassifier` on all {tr['n_candidate_features']} raw "
      "features (no binning), with minimum leaf size 50 and L2 regularisation 1.0. A grid of "
      f"{len(g)} settings was searched on validation AUC (range {fmt_range(g.valid_auc.min(), g.valid_auc.max())}); "
      f"selected: learning rate {p['learning_rate']}, {p['max_leaf_nodes']} leaves per tree, "
      f"{p['max_iter']} trees. Full grid: `reports/tables/challenger_grid.csv`.\n")

    # 5 performance
    w("## 5. Performance\n")
    pf = d["perf"].copy()
    pf["model"] = pf.model.map(labels)
    pf = pf[["model", "split", "n", "default_rate", "mean_pd", "auc", "gini", "ks", "brier", "log_loss"]]
    for c in ("auc", "gini", "ks"):
        pf[c] = pf[c].map(f3)
    for c in ("brier", "log_loss"):
        pf[c] = pf[c].map(f4)
    for c in ("default_rate", "mean_pd"):
        pf[c] = pf[c].map(pct)
    pf["n"] = pf.n.map(n0)
    w(md_table(pf.rename(columns={"model": "Model", "split": "Split", "n": "Accounts",
                                  "default_rate": "Default rate", "mean_pd": "Mean PD",
                                  "auc": "AUC", "gini": "Gini", "ks": "KS", "brier": "Brier",
                                  "log_loss": "Log loss"}), align=["l", "l"] + ["r"] * 8) + "\n")
    w(f"Bootstrap 95% CIs for test AUC ({b['n_boot']} paired resamples): champion "
      f"{fmt_range(*b['auc_a_ci'])}, challenger {fmt_range(*b['auc_b_ci'])}. Train-to-test AUC "
      f"change: champion {perf_row(d, 'champion', 'test').auc - perf_row(d, 'champion', 'train').auc:+.3f}, "
      f"challenger {chall_te.auc - perf_row(d, 'challenger', 'train').auc:+.3f}.\n")
    w("![ROC curve](figures/roc.png)\n")
    w("![KS](figures/ks.png)\n")

    # 6 calibration
    w("## 6. Calibration\n")
    for m in ("champion", "challenger"):
        c = d["cal"][d["cal"].model == m].copy()
        hl = v["calibration"][m]["hosmer_lemeshow"]
        w(f"**{labels[m]}**, test split. Hosmer-Lemeshow chi-square {hl['chi2']:.2f} on "
          f"{hl['dof']} df, p = {hl['p_value']:.4f} "
          + ("(calibration rejected at 5%)" if hl["p_value"] < HL_ALPHA else "(not rejected at 5%)")
          + f". Largest decile gap {v['calibration'][m]['max_abs_decile_gap_pp']:.1f} pp; top-to-bottom "
          f"decile default-rate ratio {v['calibration'][m]['top_bottom_decile_ratio']:.1f}.\n")
        c = c[["decile", "n", "defaults", "min_pd", "max_pd", "mean_pd", "observed_dr", "diff_pp"]]
        c["pd range"] = [fmt_range(a, z) for a, z in zip(c.min_pd, c.max_pd)]
        c["mean_pd"] = c.mean_pd.map(pct)
        c["observed_dr"] = c.observed_dr.map(pct)
        c["diff_pp"] = c.diff_pp.map(lambda x: f"{x:+.1f}")
        c = c[["decile", "n", "defaults", "pd range", "mean_pd", "observed_dr", "diff_pp"]]
        w(md_table(c.rename(columns={"decile": "Decile", "n": "Accounts", "defaults": "Defaults",
                                     "pd range": "PD range", "mean_pd": "Mean PD",
                                     "observed_dr": "Observed", "diff_pp": "Observed - predicted (pp)"}),
                   align=["l", "r", "r", "l", "r", "r", "r"]) + "\n")
    w("The Hosmer-Lemeshow test becomes very sensitive at this sample size (600 accounts per "
      "decile), so small absolute gaps can be statistically significant; the decile tables show "
      "the size and direction of each gap.\n")
    w("![Calibration](figures/calibration.png)\n")

    # 7 stability
    w("## 7. Stability\n")
    w("PSI compares the distribution of a score or feature in a comparison sample with the "
      "training sample: below 0.10 is read as stable, 0.10 to 0.25 as a moderate shift, above "
      "0.25 as a significant shift. Score PSI uses deciles of the training PD; feature PSI uses "
      "the champion's WoE bins, fit on train.\n")
    sp = d["score_psi"].copy()
    sp["model"] = sp.model.map(labels)
    sp["psi"] = sp.psi.map(f4)
    w(md_table(sp.rename(columns={"model": "Model", "comparison": "Comparison", "psi": "PSI",
                                  "band": "Reading"}), align=["l", "l", "r", "l"]) + "\n")
    fp = d["feat_psi"].copy()
    fp["psi"] = fp.psi.map(f4)
    fp["in_champion"] = fp.in_champion.map({True: "yes", False: "no"})
    w("Feature PSI, train vs test:\n")
    w(md_table(fp.rename(columns={"feature": "Feature", "psi": "PSI", "band": "Reading",
                                  "in_champion": "In champion"}), align=["l", "r", "l", "l"]) + "\n")
    mx = d["feat_psi"].iloc[0]
    w(f"All values are {psi_band(max(d['score_psi'].psi.max(), mx.psi))} (largest feature PSI "
      f"{f4(mx.psi)}, `{mx.feature}`). This is expected: the splits are random draws from one "
      "period, so these numbers check the split and the PSI code rather than stability over "
      "time (finding F1).\n")
    w("![Score distributions](figures/score_distribution.png)\n")

    # 8 sensitivity
    w("## 8. Sensitivity\n")
    w("Permutation importance: the drop in test AUC when one feature's values are shuffled "
      "(mean of 5 repeats). It measures how much each model relies on a feature, holding the "
      "fitted model fixed.\n")
    im = d["imp"].pivot(index="feature", columns="model", values="auc_drop")
    im = im.sort_values("challenger", ascending=False).head(12).reset_index()
    im["in champion"] = im.feature.map(lambda f: "yes" if f in champ_feats else "no")
    for m in ("champion", "challenger"):
        im[m] = im[m].map(f4)
    w(md_table(im[["feature", "in champion", "champion", "challenger"]].rename(columns={
        "feature": "Feature", "in champion": "In champion",
        "champion": "Champion AUC drop", "challenger": "Challenger AUC drop"}),
        align=["l", "l", "r", "r"]) + "\n")
    top_c = d["imp"][d["imp"].model == "champion"].iloc[0].feature
    top_g = d["imp"][d["imp"].model == "challenger"].iloc[0].feature
    lead = (f"Both models rely most on `{top_c}`. " if top_c == top_g else
            f"The champion relies most on `{top_c}` and the challenger on `{top_g}`. ")
    w(lead + "The partial-dependence plots below show the mean predicted PD on a "
      "2,000-account test sample as each of the challenger's top four features is varied across "
      "its training range, with all other features held at their observed values.\n")
    rng = d["points"].groupby("feature").points.agg(lambda x: x.max() - x.min()).sort_values(ascending=False)
    w(f"For the champion, the scorecard points range in section 4.2 gives the same view directly: "
      f"`{rng.index[0]}` moves an account's score by up to {rng.iloc[0]:.0f} points, against "
      f"{rng.iloc[1]:.0f} for `{rng.index[1]}` and {rng.iloc[2]:.0f} for `{rng.index[2]}` "
      f"({PDO} points double or halve the odds).\n")
    w("![Partial dependence](figures/partial_dependence.png)\n")
    w("![Feature importance](figures/feature_importance.png)\n")
    seg = d["segments"].copy()
    seg["n"] = seg.n.map(n0)
    seg["default_rate"] = seg.default_rate.map(pct)
    for m in ("champion", "challenger"):
        seg[f"auc_{m}"] = seg[f"auc_{m}"].map(f3)
        seg[f"mean_pd_{m}"] = seg[f"mean_pd_{m}"].map(pct)
    w("Performance by credit-limit band (NT dollars; band edges are terciles of the training split):\n")
    w(md_table(seg.rename(columns={"segment": "Credit limit", "n": "Accounts",
                                   "default_rate": "Default rate",
                                   "auc_champion": "Champion AUC", "mean_pd_champion": "Champion mean PD",
                                   "auc_challenger": "Challenger AUC",
                                   "mean_pd_challenger": "Challenger mean PD"})) + "\n")

    # 9 benchmark
    w("## 9. Benchmark: champion versus challenger\n")
    w(md_table(pd.DataFrame([
        ["Test AUC", f3(champ_te.auc), f3(chall_te.auc)],
        ["Test KS", f3(champ_te.ks), f3(chall_te.ks)],
        ["Test Brier", f4(champ_te.brier), f4(chall_te.brier)],
        ["Hosmer-Lemeshow p (test)",
         f4(v["calibration"]["champion"]["hosmer_lemeshow"]["p_value"]),
         f4(v["calibration"]["challenger"]["hosmer_lemeshow"]["p_value"])],
        ["Train-to-test AUC change",
         f"{champ_te.auc - perf_row(d, 'champion', 'train').auc:+.3f}",
         f"{chall_te.auc - perf_row(d, 'challenger', 'train').auc:+.3f}"],
        ["Features used", str(len(champ_feats)), str(tr["n_candidate_features"])],
        ["Explanation", "Additive points per bin", "Post-hoc (permutation, partial dependence)"],
    ], columns=["Measure", labels["champion"], labels["challenger"]]), align=["l", "r", "r"]) + "\n")
    w(f"Paired bootstrap AUC difference (challenger minus champion): {b['diff_mean']:.3f}, 95% CI "
      f"{fmt_range(*b['diff_ci'])}"
      + (", which excludes zero. " if b["diff_ci"][0] > 0 or b["diff_ci"][1] < 0 else ", which includes zero. ")
      + f"Spearman rank correlation between the two models' PDs on test: {v['spearman_test']:.3f}, "
      "so the two models largely agree on ordering and differ mainly at the margins. The champion "
      "is retained as the primary model because each prediction decomposes into points per "
      "feature, which supports adverse-action reasons and review; the challenger serves as the "
      "performance benchmark.\n")

    # 10 limitations
    w("## 10. Limitations and model risks\n")
    for item in [
        "**No out-of-time or out-of-population test.** One observation window and one outcome "
        "month; performance on later periods, other economic conditions or other portfolios is "
        "unknown.",
        "**Old, foreign, single-lender data.** Taiwan, 2005, during a period of high card "
        "delinquency in that market; behaviour and the credit environment differ from current "
        "US portfolios.",
        "**Target definition.** \"Default payment next month\" is not defined in days past due or "
        "charge-off terms, and the one-month horizon is far shorter than the 12-month horizon "
        "typical for PD models.",
        "**Undocumented codes.** PAY -2 and 0 dominate the repayment-status fields and their "
        "meaning is inferred.",
        "**Payment-timing assumption.** The payment in month m is assumed to settle the "
        "statement of month m+1 (the previous statement); payment-ratio features depend on it.",
        "**No macroeconomic or bureau data.** The model cannot respond to economic conditions "
        "and has no external credit information.",
        "**Fine-classed bins.** WoE bins come from quantiles, not from reviewed coarse classes; "
        "some trends are non-monotone.",
        "**Fairness not tested.** Protected attributes are excluded as inputs, but no disparate-"
        "impact analysis was done; excluding a field does not remove proxies for it.",
        "**Self-validation.** The developer and validator are the same person.",
    ]:
        w(f"- {item}")
    w("")

    # 11 conclusions
    w("## 11. Conclusions\n")
    hl_c = v["calibration"]["champion"]["hosmer_lemeshow"]
    w(f"1. Data quality: {len(dq)} checks, {n_fail} failures. The known undocumented codes are "
      "handled explicitly and recorded as warnings.")
    w(f"2. Discrimination: the champion's test AUC of {f3(champ_te.auc)} (Gini {f3(champ_te.gini)}, "
      f"KS {f3(champ_te.ks)}) is " + ("adequate" if disc_ok else "below threshold")
      + " for rank-ordering accounts by risk, and close to its train AUC.")
    w(f"3. Calibration: overall PD level is accurate ({pct(champ_te.mean_pd)} predicted vs "
      f"{pct(champ_te.default_rate)} observed), but decile-level calibration is "
      + ("rejected" if hl_c["p_value"] < HL_ALPHA else "not rejected")
      + f" (Hosmer-Lemeshow p = {hl_c['p_value']:.4f}).")
    w("4. Stability: PSI values are all "
      f"{psi_band(max(d['score_psi'].psi.max(), d['feat_psi'].psi.max()))}, but this only reflects "
      "a random split; stability over time is untested.")
    w(f"5. Benchmark: the challenger is more discriminating by {b['diff_mean']:.3f} AUC; the "
      "champion is kept for interpretability, and the gap is a documented finding.")
    w("6. Overall: fit for demonstration and for discussion of methodology. Use on real accounts "
      "would first require an out-of-time test, recalibration, coarse classing, confirmation of "
      "code definitions and an independent review.\n")

    w("## References\n")
    w("- Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning "
      "Repository. https://doi.org/10.24432/C55S3H (CC BY 4.0).")
    w("- Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the "
      "predictive accuracy of probability of default of credit card clients. *Expert Systems "
      "with Applications*, 36(2), 2473-2480.")
    w("- Board of Governors of the Federal Reserve System and OCC (2011). *Supervisory Guidance "
      "on Model Risk Management* (SR 11-7 / OCC 2011-12). Used as a structural reference only.")
    return "\n".join(s) + "\n"


def _find_index(find, area):
    for i, f in enumerate(find, 1):
        if f[0] == area:
            return i
    return "?"


def readme_results(d):
    v = d["val"]
    b = v["bootstrap_test"]
    champ_psi = d["score_psi"][(d["score_psi"].model == "champion") &
                               (d["score_psi"].comparison == "train vs test")].psi.iloc[0]
    hl = v["calibration"]["champion"]["hosmer_lemeshow"]
    lines = [
        "",
        "",
        headline_table(d),
        "",
        f"- Challenger minus champion test AUC: {b['diff_mean']:.3f} "
        f"(95% paired bootstrap CI {fmt_range(*b['diff_ci'])}).",
        f"- Champion calibration: mean PD {pct(perf_row(d, 'champion', 'test').mean_pd)} vs observed "
        f"{pct(perf_row(d, 'champion', 'test').default_rate)}; Hosmer-Lemeshow p = {hl['p_value']:.4f}.",
        f"- Champion score PSI, train vs test: {champ_psi:.4f} (random split, so this checks the "
        "split rather than stability over time).",
        f"- Data quality: {len(d['dq'])} SQL checks, {int((d['dq'].status == 'FAIL').sum())} failures, "
        f"{int((d['dq'].status == 'WARN').sum())} warnings for documented code and balance issues.",
    ]
    n_test = int(d["split"].set_index("split").loc["test", "n"])
    lines[0] = f"Test split ({n_test:,} accounts, not used for fitting or tuning):"
    return "\n".join(lines)


def main():
    d = load()
    (REPORTS_DIR / "model_validation.md").write_text(report(d))
    readme = ROOT / "README.md"
    if readme.exists():
        text = readme.read_text()
        new = re.sub(r"(<!-- results:start -->\n).*?(<!-- results:end -->)",
                     lambda m: m.group(1) + readme_results(d) + "\n" + m.group(2), text, flags=re.S)
        readme.write_text(new)
    print("wrote reports/model_validation.md" + (" and README results" if readme.exists() else ""))


if __name__ == "__main__":
    main()
