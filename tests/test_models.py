"""Smoke tests for the models and unit tests for the validation metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score

from cdmv.metrics import calibration_table, hosmer_lemeshow, ks_statistic, performance, psi
from cdmv.models import (fit_challenger, fit_champion, load_modeling_frame, pd_to_score,
                         predict_challenger, predict_champion, scorecard_points)
from cdmv.woe import WoEBinner

from conftest import build_warehouse, synthetic_raw


def test_models_train_and_predict():
    con = build_warehouse(synthetic_raw(n=3000, seed=1))
    df = load_modeling_frame(con)
    train, valid, test = (df[df.split == s] for s in ("train", "validation", "test"))

    champ = fit_champion(train, valid)
    assert champ["features"], "champion kept no features"
    assert (champ["model"].coef_[0] < 0).all()
    p = predict_champion(champ, test)
    assert p.shape == (len(test),) and np.all((p > 0) & (p < 1))
    assert roc_auc_score(test.default_flag, p) > 0.6

    chall = fit_challenger(train, valid)
    q = predict_challenger(chall, test)
    assert np.all((q > 0) & (q < 1))
    assert roc_auc_score(test.default_flag, q) > 0.6

    # scorecard points add up to the model's score
    pts = scorecard_points(champ)
    X = test.head(5)
    total = np.zeros(len(X))
    for f in champ["features"]:
        idx = champ["binner"].bin_index(X, f)
        total += pts[pts.feature == f].points.to_numpy()[idx]
    np.testing.assert_allclose(total, pd_to_score(predict_champion(champ, X)), atol=1e-6)


def test_woe_bins_respect_minimum_size():
    rng = np.random.default_rng(0)
    import pandas as pd
    X = pd.DataFrame({"a": rng.normal(size=5000), "b": rng.integers(0, 9, 5000)})
    y = (rng.random(5000) < 1 / (1 + np.exp(-X.a))).astype(int)
    b = WoEBinner(min_bin_frac=0.05).fit(X, y)
    for t in b.tables_.values():
        assert (t.share >= 0.05).all()
        assert ((t.bads > 0) & (t.bads < t.n)).all()
    assert b.tables_["a"].woe.is_monotonic_decreasing


def test_ks_and_auc_on_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert ks_statistic(y, p) == 1.0
    assert performance(y, p)["gini"] == 1.0


def test_psi():
    rng = np.random.default_rng(0)
    x = rng.normal(size=10_000)
    assert psi(x, x)[0] == 0
    assert psi(x, rng.normal(size=10_000))[0] < 0.01
    assert psi(x, rng.normal(loc=1.0, size=10_000))[0] > 0.25


def test_calibration_table_and_hl():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.6, 20_000)
    y = (rng.random(20_000) < p).astype(int)
    t = calibration_table(y, p)
    assert len(t) == 10 and t.n.sum() == 20_000
    assert hosmer_lemeshow(t)["p_value"] > 0.01  # well calibrated by construction
    assert hosmer_lemeshow(calibration_table(y, np.clip(p * 1.5, 0, 0.99)))["p_value"] < 0.001
