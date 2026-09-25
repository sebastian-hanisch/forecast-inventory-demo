"""Jede Zahl aus README und PRESET_HELP als Test. Die Verfahren sind deterministisch bei festem Seed; Bänder um die gerundeten Angaben, dazu Rangfolgen mit Abstand
(numpy-Versionen und Plattformen können den Zufallsstrom des HW-Fits und die Baumteilungen des Boostings um Rundung verschieben, feedback_ci_platform_robust_tests)."""

from functools import lru_cache

import pytest

import stk_constants as C
import stk_evaluation as E
import stk_presets as P

STD = "Standardfall: Lieferzeit 3 Tage, Ziel 95 %"


@lru_cache(maxsize=None)
def _preset(name):
    v = P.PRESETS[name]
    return E.analyse(E.Settings(v["n_depots"], v["noise"], v["trend"], v["swing"], v["lead"], v["target"], v["window"], v["seed"]))


def _reg(a, src, rule):
    return 100.0 * (a.metrics[(src, rule)]["cost"] / a.metrics["oracle"]["cost"] - 1.0)


def _al(a, src, rule):
    return a.metrics[(src, rule)]["alpha"]


def _band(x):
    return max(8.0, 0.15 * abs(x))


def test_standard_preset():
    a = _preset(STD)
    o = a.metrics["oracle"]
    assert o["alpha"] == pytest.approx(0.951, abs=0.02) and o["beta"] == pytest.approx(0.992, abs=0.006) and o["cost"] == pytest.approx(0.704, abs=0.04) and o["stock"] == pytest.approx(0.55, abs=0.05)
    for (src, rule, r, al) in (("mean4", "conformal", 42, 0.950), ("mean4", "none", 341, 0.547), ("mean4", "rough", 66, 0.885), ("mean4", "lognormal", 43, 0.937), ("mean4", "empirical", 44, 0.941), ("gbm", "empirical", 33, 0.945)):
        assert _reg(a, src, rule) == pytest.approx(r, abs=_band(r)) and _al(a, src, rule) == pytest.approx(al, abs=0.025), (src, rule)
    best = min(((s, r) for s in C.SOURCES for r in C.RULES), key=lambda k: a.metrics[k]["cost"])
    assert best[0] == "gbm" and best[1] in ("empirical", "lognormal") and _reg(a, *best) == pytest.approx(33, abs=8)
    assert all(_al(a, s, "none") < 0.65 for s in C.SOURCES) and all(_al(a, s, "rough") < 0.90 for s in C.SOURCES)
    assert _al(a, "hw", "lognormal") > _al(a, "hw", "rough") + 0.04 and _reg(a, "gbm", "empirical") < _reg(a, "gbm", "rough") < _reg(a, "gbm", "none")


def test_source_ranking_and_regression_failure_in_the_standard_case():
    a = _preset(STD)
    assert a.wape["gbm"] == pytest.approx(0.079, abs=0.015) and a.wape["hw"] == pytest.approx(0.086, abs=0.015) and a.wape["wm"] == pytest.approx(0.099, abs=0.015) and a.wape["reg"] == pytest.approx(0.139, abs=0.02) and a.wape["mean4"] == pytest.approx(0.083, abs=0.015)
    assert _al(a, "reg", "empirical") == pytest.approx(0.862, abs=0.03) and _al(a, "reg", "conformal") == pytest.approx(0.929, abs=0.025)
    assert _reg(a, "reg", "empirical") == pytest.approx(222, abs=40) and _reg(a, "reg", "conformal") == pytest.approx(53, abs=12) and _reg(a, "wm", "empirical") == pytest.approx(82, abs=14) and _reg(a, "hw", "empirical") == pytest.approx(55, abs=10)
    assert _reg(a, "reg", "empirical") > 2 * _reg(a, "reg", "conformal")


def test_lead_zero_preset():
    a = _preset("Lieferzeit 0 (der reine Newsvendor)")
    o = a.metrics["oracle"]
    assert o["alpha"] == pytest.approx(0.953, abs=0.02) and o["beta"] == pytest.approx(0.996, abs=0.005)
    for (rule, r, al) in (("none", 268, 0.545), ("rough", 36, 0.923), ("empirical", 18, 0.950), ("conformal", 17, 0.956)):
        assert _reg(a, "mean4", rule) == pytest.approx(r, abs=_band(r)) and _al(a, "mean4", rule) == pytest.approx(al, abs=0.025), rule
    assert _reg(a, "mean4", "rough") > _reg(a, "mean4", "empirical") + 10


def test_long_lead_preset():
    a = _preset("Lange Lieferzeit (13 Tage)")
    for (rule, r, al) in (("rough", 175, 0.809), ("empirical", 107, 0.929), ("conformal", 105, 0.908)):
        assert _reg(a, "mean4", rule) == pytest.approx(r, abs=_band(r)) and _al(a, "mean4", rule) == pytest.approx(al, abs=0.03), rule
    assert _al(a, "mean4", "conformal") < 0.95 - 0.015 and _reg(a, "mean4", "empirical") > _reg(_preset(STD), "mean4", "empirical") + 30


def test_high_target_preset():
    a = _preset("Hohes Ziel (99 %)")
    for (rule, r, al) in (("rough", 151, 0.939), ("lognormal", 51, 0.982), ("empirical", 52, 0.983), ("conformal", 59, 0.979), ("none", 1508, 0.547)):
        assert _reg(a, "mean4", rule) == pytest.approx(r, abs=_band(r)) and _al(a, "mean4", rule) == pytest.approx(al, abs=0.02), rule
    assert a.metrics["oracle"]["alpha"] == pytest.approx(0.989, abs=0.01) and _reg(a, "mean4", "none") > 1000


def test_low_target_preset():
    a = _preset("Niedriges Ziel (80 %)")
    o = a.metrics["oracle"]
    assert o["alpha"] == pytest.approx(0.806, abs=0.03) and o["beta"] == pytest.approx(0.963, abs=0.012)
    for (rule, r, al) in (("rough", 38, 0.761), ("empirical", 38, 0.807), ("conformal", 32, 0.799)):
        assert _reg(a, "mean4", rule) == pytest.approx(r, abs=_band(r)) and _al(a, "mean4", rule) == pytest.approx(al, abs=0.03), rule
    assert o["beta"] > o["alpha"] + 0.12


def test_regression_under_swing_preset():
    a = _preset("Regression bei starker Schwankung")
    assert _al(a, "reg", "empirical") == pytest.approx(0.816, abs=0.04) and _al(a, "reg", "conformal") == pytest.approx(0.903, abs=0.035) and _al(a, "gbm", "empirical") == pytest.approx(0.948, abs=0.025)
    assert _reg(a, "reg", "empirical") == pytest.approx(578, abs=90) and _reg(a, "reg", "conformal") == pytest.approx(137, abs=30) and _reg(a, "gbm", "empirical") == pytest.approx(53, abs=10)
    assert a.wape["reg"] == pytest.approx(0.240, abs=0.03) and a.wape["gbm"] == pytest.approx(0.090, abs=0.015) and a.wape["reg"] > 2 * a.wape["gbm"]


@lru_cache(maxsize=None)
def _target():
    return E.target_experiment(C.TARGET_LEVELS, C.EXP_SEEDS)


@lru_cache(maxsize=None)
def _lead():
    return E.lead_experiment(C.LEAD_LEVELS, C.EXP_SEEDS)


@lru_cache(maxsize=None)
def _swing():
    return E.swing_experiment(C.SWING_LEVELS, C.EXP_SEEDS)


def test_target_experiment():
    rows = {r["target"]: r["by"] for r in _target()}
    al = lambda t, rule: rows[t][("mean4", rule)]["alpha"][0]
    rg = lambda t, rule: rows[t][("mean4", rule)]["regret"][0]
    for t, vals in ((0.80, (0.760, 0.803, 0.811, 0.795)), (0.90, (0.842, 0.895, 0.901, 0.895)), (0.95, (0.892, 0.945, 0.948, 0.948)), (0.99, (0.947, 0.986, 0.985, 0.979))):
        for rule, v in zip(("rough", "lognormal", "empirical", "conformal"), vals):
            assert al(t, rule) == pytest.approx(v, abs=0.02), (t, rule)
        assert al(t, "rough") < t - 0.025 and abs(al(t, "empirical") - t) < 0.02
    assert rg(0.80, "none") == pytest.approx(79, abs=20) and rg(0.95, "none") == pytest.approx(357, abs=50) and rg(0.99, "none") == pytest.approx(1601, abs=250)
    assert rg(0.95, "rough") == pytest.approx(58, abs=12) and rg(0.99, "rough") == pytest.approx(124, abs=25) and rg(0.95, "lognormal") == pytest.approx(37, abs=8) and rg(0.99, "conformal") == pytest.approx(54, abs=12)
    for t, beta in ((0.80, 0.963), (0.90, 0.983), (0.95, 0.992), (0.99, 0.999)):
        assert rows[t]["oracle"]["beta"][0] == pytest.approx(beta, abs=0.01) and rows[t]["oracle"]["alpha"][0] == pytest.approx(t, abs=0.02)
    assert rows[0.80]["oracle"]["beta"][0] - rows[0.80]["oracle"]["alpha"][0] > 0.12


def test_lead_experiment():
    rows = {r["lead"]: r["by"] for r in _lead()}
    al = lambda L, rule: rows[L][("mean4", rule)]["alpha"][0]
    rg = lambda L, rule: rows[L][("mean4", rule)]["regret"][0]
    for L, (rough, logn, emp, conf) in ((0, (37, 15, 16, 17)), (3, (58, 37, 38, 39)), (7, (102, 62, 65, 69)), (13, (158, 90, 94, 99))):
        assert rg(L, "rough") == pytest.approx(rough, abs=_band(rough)) and rg(L, "lognormal") == pytest.approx(logn, abs=_band(logn)) and rg(L, "empirical") == pytest.approx(emp, abs=_band(emp)) and rg(L, "conformal") == pytest.approx(conf, abs=_band(conf)), L
    assert al(0, "rough") == pytest.approx(0.921, abs=0.025) and al(13, "rough") == pytest.approx(0.820, abs=0.03) and al(13, "empirical") == pytest.approx(0.936, abs=0.025) and al(13, "conformal") == pytest.approx(0.913, abs=0.03)
    assert rg(0, "none") == pytest.approx(278, abs=45) and rg(13, "none") == pytest.approx(504, abs=80)
    assert rg(0, "empirical") < rg(3, "empirical") < rg(7, "empirical") < rg(13, "empirical") and al(0, "rough") > al(13, "rough") + 0.06 and al(13, "conformal") < al(13, "empirical") - 0.01


def test_swing_experiment():
    rows = {r["swing"]: r for r in _swing()}
    g = lambda w, src, rule, q: rows[w]["by"][(src, rule)][q][0]
    assert rows[0.0]["wape"]["reg"][0] == pytest.approx(0.069, abs=0.012) and rows[0.06]["wape"]["reg"][0] == pytest.approx(0.125, abs=0.02) and rows[0.12]["wape"]["reg"][0] == pytest.approx(0.213, abs=0.03)
    assert rows[0.06]["wape"]["gbm"][0] == pytest.approx(0.083, abs=0.015) and rows[0.06]["wape"]["mean4"][0] == pytest.approx(0.083, abs=0.015)
    assert rows[0.12]["wape"]["gbm"][0] == pytest.approx(0.094, abs=0.015) and rows[0.12]["wape"]["mean4"][0] == pytest.approx(0.103, abs=0.015) and rows[0.0]["wape"]["wm"][0] == pytest.approx(0.094, abs=0.015)
    assert g(0.0, "reg", "empirical", "regret") == pytest.approx(11, abs=8) and g(0.0, "reg", "conformal", "regret") == pytest.approx(11, abs=8) and g(0.0, "reg", "empirical", "alpha") == pytest.approx(0.957, abs=0.02)
    assert g(0.06, "reg", "empirical", "alpha") == pytest.approx(0.875, abs=0.03) and g(0.06, "reg", "conformal", "alpha") == pytest.approx(0.930, abs=0.025) and g(0.06, "reg", "empirical", "regret") == pytest.approx(136, abs=30) and g(0.06, "reg", "conformal", "regret") == pytest.approx(51, abs=12)
    assert g(0.12, "reg", "empirical", "alpha") == pytest.approx(0.829, abs=0.04) and g(0.12, "reg", "conformal", "alpha") == pytest.approx(0.907, abs=0.035) and g(0.12, "reg", "empirical", "regret") == pytest.approx(383, abs=70) and g(0.12, "reg", "conformal", "regret") == pytest.approx(126, abs=28)
    assert g(0.12, "gbm", "empirical", "regret") == pytest.approx(54, abs=12) and g(0.12, "mean4", "empirical", "regret") == pytest.approx(74, abs=15) and g(0.0, "gbm", "empirical", "regret") == pytest.approx(25, abs=8) and g(0.0, "mean4", "empirical", "regret") == pytest.approx(22, abs=8)
    assert g(0.12, "reg", "empirical", "regret") > 2.5 * g(0.12, "reg", "conformal", "regret") and g(0.0, "reg", "empirical", "regret") < g(0.0, "gbm", "empirical", "regret")
