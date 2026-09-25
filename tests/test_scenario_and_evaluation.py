"""Portfolio, Prognosequellen, Auswertung und Kennzahlen; kein Bestellbestand darf etwas aus der Zukunft seines Ursprungs verwenden."""

import dataclasses

import numpy as np
import pytest

import stk_constants as C
import stk_evaluation as E
import stk_members as M
import stk_rules as R
import stk_scenario as S


def test_scenario_is_deterministic_seeded_and_swing_is_heterogeneous():
    a, b, c = S.generate(6, seed=1, swing=0.06), S.generate(6, seed=1, swing=0.06), S.generate(6, seed=2, swing=0.06)
    assert np.array_equal(a.y, b.y) and not np.array_equal(a.y, c.y) and a.y.shape == (6, C.N_DAYS)
    assert a.swing.min() > 0 and a.swing.max() / a.swing.min() > 1.2 and (S.generate(3, seed=1, swing=0.0).swing == 0).all()


def test_sources_by_hand_and_mean_of_four():
    port = S.generate(3, seed=5)
    m = 4
    fc = M.member_forecasts(port, m)
    org = M.origins(m)
    y = port.y[1]
    t, j = 700, 2
    i = t - org[0]
    assert fc["wm"][1, i, j] == pytest.approx(np.mean([y[t - 7 * (k + 1) + j % 7] for k in range(4)]))
    assert fc["wm"].shape == (3, len(org), m) and org[0] == C.FIRST_ORIGIN and org[-1] == C.N_DAYS - m and set(fc) == set(C.SOURCES)
    assert np.allclose(fc["mean4"], np.mean([fc[s] for s in C.MEMBERS], axis=0))


def _settings(**kw):
    return E.Settings(n_depots=10, **kw)


@pytest.fixture(scope="module")
def analysis():
    return E.analyse(_settings())


def test_analysis_shapes_and_keys(analysis):
    a = analysis
    T = C.N_DAYS - a.m + 1 - C.FIRST_TEST
    assert a.m == 4 and a.demand.shape == (10, T) and a.days.shape == (10, T) and len(a.keys) == 26 and a.keys[-1] == "oracle" and set(a.metrics) == set(a.keys)
    assert all(v.shape == (10, T) for v in a.levels.values()) and set(a.forecast) == set(C.SOURCES) and a.sim["cost"].shape == (26, 10)
    t = a.test_org[7]
    assert np.array_equal(a.demand[3, 7], a.port.y[3, t:t + a.m].sum()) and np.array_equal(a.days[3, 7], a.port.y[3, t])


def test_none_is_the_forecast_and_safety_stock_is_not_negative_for_the_parametric_rules(analysis):
    a = analysis
    for src in C.SOURCES:
        assert np.array_equal(a.levels[(src, "none")], a.forecast[src])
        for rule in ("rough", "lognormal"):
            assert (a.levels[(src, rule)] >= a.levels[(src, "none")] - 1e-9).all()
    assert (a.levels["oracle"] > 0).all()


def test_metrics_by_hand(analysis):
    a = analysis
    k = ("hw", "empirical")
    b = a.keys.index(k)
    dep = 2
    tau = a.settings.target
    p = tau / (1 - tau)
    cost = (a.sim["on_hand"][b, dep] + p * a.sim["backorder"][b, dep]).mean() / a.days[dep].mean()
    assert a.per_depot[k]["cost"][dep] == pytest.approx(cost) and a.metrics[k]["cost"] == pytest.approx(a.per_depot[k]["cost"].mean())
    assert a.per_depot[k]["alpha"][dep] == pytest.approx(np.mean(a.sim["backorder"][b, dep] == 0)) and a.per_depot[k]["beta"][dep] == pytest.approx(1 - a.sim["short"][b, dep].sum() / a.days[dep].sum())
    assert a.wape["hw"] == pytest.approx(np.mean(np.abs(a.forecast["hw"] - a.demand).mean(axis=1) / a.demand.mean(axis=1)))


def test_oracle_rule_is_the_cheapest_and_meets_the_target(analysis):
    a = analysis
    o = a.metrics["oracle"]
    assert o["alpha"] == pytest.approx(a.settings.target, abs=0.03) and o["beta"] > o["alpha"]
    assert all(a.metrics[k]["cost"] > o["cost"] for k in a.keys if k != "oracle")
    assert all(a.metrics[(s, "none")]["alpha"] < 0.8 for s in C.SOURCES)


def test_levels_ignore_the_future_of_the_origin():
    s = _settings()
    port = E._portfolio(s.portfolio_key)
    fc = E._members(s.portfolio_key, s.lead + 1)
    lv1, *_ = E.build_levels(port, fc, s)
    t = 850
    y2 = port.y.copy()
    y2[:, t:] = np.random.default_rng(0).integers(0, 999, size=y2[:, t:].shape)
    port2 = dataclasses.replace(port, y=y2)
    lv2, *_ = E.build_levels(port2, M.member_forecasts(port2, s.lead + 1), s)
    i = t - C.FIRST_TEST
    for k, v in lv1.items():
        if k == "oracle":
            assert np.array_equal(v, lv2[k])                                                    # das Orakel kennt den Erwartungswert, nicht die Ist-Werte
        else:
            assert np.allclose(v[:, :i + 1], lv2[k][:, :i + 1]), k


def test_lead_zero_and_high_target():
    a = E.analyse(E.Settings(n_depots=10, lead=0, target=0.99))
    assert a.m == 1 and np.allclose(a.demand, a.days) and a.metrics["oracle"]["alpha"] > 0.95


def test_members_are_computed_once_per_portfolio_and_horizon():
    s = _settings()
    assert E._members(s.portfolio_key, 4) is E._members(s.portfolio_key, 4)
