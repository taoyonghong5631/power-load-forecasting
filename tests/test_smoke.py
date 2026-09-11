# -*- coding: utf-8 -*-
"""轻量测试：不依赖 pytest，直接 ``python tests/test_smoke.py`` 也能跑。

重点覆盖两件容易出错、又最伤结果的事情：
1. 训练时的批量特征与递归预测时的单步特征必须完全一致（train-serving skew）；
2. 滞后/滚动特征不能用到"未来"的信息（数据泄漏）。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import metrics as M
from src.anomaly import (detection_metrics, inject_anomalies, sigma_scores,
                         top_rate_mask, threshold_free_metrics)
from src.config import get_config
from src.data import synthetic_dataset
from src.evaluate import make_origins, train_end_index
from src.features import build_features, build_supervised, feature_names, next_step_row


def test_next_step_row_matches_batch_features():
    """递归预测用的单步特征 == 批量构造的特征（同一时间点必须完全相等）。"""
    df = synthetic_dataset(45, seed=1)
    groups = ("base", "calendar", "temperature")
    cols = feature_names(groups)
    batch = build_features(df, groups)
    for pos in (200, 500, 800):
        single = next_step_row(df.iloc[:pos], groups, df.index[pos],
                               temp=float(df["temp"].iloc[pos]))
        assert np.allclose(single[cols].to_numpy(float),
                           batch.iloc[[pos]][cols].to_numpy(float),
                           equal_nan=True), "位置 %d 的单步特征与批量特征不一致" % pos


def test_no_future_leakage():
    """改变 t 时刻的负荷，不应该影响 t 时刻自己的特征。"""
    df = synthetic_dataset(20, seed=2)
    groups = ("base", "calendar", "temperature")
    cols = feature_names(groups)
    pos = 300
    before = build_features(df, groups).iloc[[pos]][cols].to_numpy(float)

    pert = df.copy()
    pert.iloc[pos, pert.columns.get_loc("load")] *= 5.0
    after = build_features(pert, groups).iloc[[pos]][cols].to_numpy(float)
    assert np.allclose(before, after, equal_nan=True), "t 的特征用到了 t 自己的负荷（泄漏）"

    # 但 t+1 的 lag_1 必须变化
    nxt = build_features(pert, groups).iloc[[pos + 1]][["lag_1"]].to_numpy(float)
    nxt0 = build_features(df, groups).iloc[[pos + 1]][["lag_1"]].to_numpy(float)
    assert not np.allclose(nxt, nxt0), "lag_1 没有反映上一时刻的变化"


def test_metrics_basics():
    y = np.array([1.0, 2.0, 3.0])
    m = M.forecast_metrics(y, y)
    assert abs(m["MAE"]) < 1e-9 and abs(m["RMSE"]) < 1e-9 and abs(m["MAPE"]) < 1e-9
    assert abs(M.improvement(10.0, 8.0) - 20.0) < 1e-9
    assert abs(M.r2(y, y) - 1.0) < 1e-9


def test_origins_respect_horizon():
    n, horizon = 1000, 24
    train_end = train_end_index(n, 0.8)
    origins = make_origins(n, train_end, horizon, stride=24, n_origins=10)
    assert origins and all(o >= train_end for o in origins)
    assert all(o + horizon <= n - 1 for o in origins), "预测窗口越界了"
    assert len(origins) <= 10


def test_anomaly_injection_and_scoring():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2013-01-01", periods=24 * 30, freq="h")
    base = 300 + 80 * np.sin(2 * np.pi * idx.hour / 24.0) + rng.normal(0, 6, len(idx))
    y = pd.Series(base, index=idx)

    corrupted, truth, kinds, events = inject_anomalies(y.to_numpy(), seed=3)
    assert truth.sum() > 0 and len(events) > 0
    assert not np.allclose(corrupted, y.to_numpy())

    scores = sigma_scores(corrupted, window=24)
    matched = top_rate_mask(scores, 0.05)
    assert abs(matched.sum() - round(len(scores) * 0.05)) <= 1

    mm = detection_metrics(truth, matched)
    assert set(mm) == {"TP", "FP", "FN", "Precision", "Recall", "F1"}
    auc = threshold_free_metrics(scores, truth)
    assert 0.0 <= auc["ROC_AUC"] <= 1.0


def test_supervised_shapes():
    df = synthetic_dataset(30, seed=4)
    groups = ("base", "calendar", "temperature")
    X, y, cols = build_supervised(df, groups)
    assert list(X.columns) == cols and len(X) == len(y)
    assert not X.isna().any().any() and not y.isna().any()
    # 前 168 小时因 lag_168 被丢弃
    assert len(X) <= len(df) - 168


def test_models_short_run():
    """XGBoost / ARIMA 在很小数据上也能 fit + predict（LSTM 单独测）。"""
    from src.models.xgb import XGBForecaster
    from src.models.arima import ARIMAForecaster

    df = synthetic_dataset(60, seed=5)
    cfg = get_config()
    cfg.task.look_back, cfg.task.horizon = 24, 12
    cfg.xgb.n_estimators = 50
    cfg.arima.train_window = 400
    cfg.arima.seasonal_order = (1, 1, 0, 24)

    for model in (XGBForecaster(cfg, ("base", "calendar", "temperature")),
                  ARIMAForecaster(cfg)):
        model.fit(df.iloc[:1200])
        pred = model.predict(df, 1199, 12)
        assert pred.shape == (12,), "%s 输出形状不对" % model.name
        assert np.isfinite(pred).all(), "%s 输出了 NaN/inf" % model.name


def test_lstm_short_run():
    """LSTM 递归多步预测的形状正确性（小数据、少轮数）。"""
    import torch
    from src.models.lstm import LSTMForecaster

    df = synthetic_dataset(45, seed=6)
    cfg = get_config()
    cfg.task.look_back, cfg.task.horizon = 24, 8
    cfg.lstm.epochs, cfg.lstm.batch_size, cfg.lstm.hidden_size = 2, 32, 8
    cfg.lstm.num_layers = 1

    model = LSTMForecaster(cfg, ("calendar", "temperature"))
    model.fit(df.iloc[:800], verbose=False)
    pred = model.predict(df, 799, 8)
    assert pred.shape == (8,) and np.isfinite(pred).all()

    plain = LSTMForecaster(cfg, ())
    plain.fit(df.iloc[:800], verbose=False)
    assert plain.predict(df, 799, 8).shape == (8,)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print("PASS  %s" % fn.__name__)
        except AssertionError as exc:
            failed += 1
            print("FAIL  %s: %s" % (fn.__name__, exc))
        except Exception as exc:
            failed += 1
            print("ERROR %s: %s: %s" % (fn.__name__, type(exc).__name__, exc))
    print("\n%d/%d 通过" % (len(tests) - failed, len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
