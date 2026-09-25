"""Die Bestandsregeln von Hand nachgerechnet, gegen unabhängige Schleifen und Monte-Carlo-Simulation, und ohne Blick in die Zukunft."""

import numpy as np
import pytest

import stk_constants as C
import stk_rules as R


def test_z_and_cumulative_and_demand_sums_by_hand():
    stats = pytest.importorskip("scipy.stats")
    for tau in (0.5, 0.8, 0.95, 0.99):
        assert R.z_of(tau) == pytest.approx(stats.norm.ppf(tau), abs=1e-9)
    F = np.arange(12, dtype=float).reshape(1, 3, 4)
    assert np.array_equal(R.cumulative(F), [[6.0, 22.0, 38.0]])
    y = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    assert np.array_equal(R.demand_sums(y, np.array([0, 2, 3]), 3), [[6.0, 12.0, 15.0]])


def test_scores_and_in_sample_index():
    assert R.scores(np.array([99.0]), np.array([49.0]))[0] == pytest.approx(np.log(100 / 50))
    org = np.arange(C.FIRST_ORIGIN, 700)
    idx = R.in_sample_index(org, 4)
    assert org[idx].max() + 4 <= C.FIT_END and org[idx].max() + 5 > C.FIT_END and idx[0] == 0


def test_none_rough_lognormal_empirical_by_hand():
    Fm = np.array([[100.0, 200.0]])
    e1 = np.array([[1.0, -1.0, 2.0, -2.0]])                                                     # Stichproben-Streuung (ddof = 1) = sqrt(10/3)
    assert np.array_equal(R.rule_none(Fm), Fm)
    tau = 0.95
    assert np.allclose(R.rule_rough(Fm, e1, 4, tau), Fm + R.z_of(tau) * np.sqrt(10 / 3) * 2.0)
    s = np.array([[0.1, -0.1, 0.2, -0.2]])
    sd = np.sqrt(0.1 / 3)
    assert np.allclose(R.rule_lognormal(Fm, s, tau), (Fm + 1) * np.exp(R.z_of(tau) * sd) - 1)
    q = np.quantile(s, tau, axis=1)[0]
    assert np.allclose(R.rule_empirical(Fm, s, tau), (Fm + 1) * np.exp(q) - 1)


def _naive_conformal(S_ext, e, W, tau):
    vals = np.sort([S_ext[j] for j in range(max(e - W + 1, 0), e + 1)])
    k = len(vals)
    return vals[int(min(max(np.ceil((k + 1) * tau), 1), k)) - 1]


def test_conformal_quantile_matches_a_naive_loop():
    rng = np.random.default_rng(0)
    S_ext = rng.normal(size=(2, 400))
    ends = np.array([100, 150, 399, 5])
    got = R.conformal_quantile(S_ext, ends, 45, 0.9)
    for dep in range(2):
        for a, e in enumerate(ends):
            assert got[dep, a] == pytest.approx(_naive_conformal(S_ext[dep], e, 45, 0.9))
    got95 = R.conformal_quantile(S_ext, ends, 30, 0.975)
    assert got95[0, 0] == pytest.approx(_naive_conformal(S_ext[0], 100, 30, 0.975)) and got95[0, 0] == np.sort(S_ext[0, 71:101])[-1]          # 31. von 30 Werten -> Rand


def test_conformal_uses_only_realised_scores():
    rng = np.random.default_rng(1)
    n_org = C.N_DAYS - 4 + 1 - C.FIRST_ORIGIN
    s_all = rng.normal(size=(2, n_org))
    org = np.arange(C.FIRST_ORIGIN, C.N_DAYS - 4 + 1)
    test_org = np.arange(C.FIRST_TEST, C.N_DAYS - 4 + 1)
    Ft = np.full((2, len(test_org)), 100.0)
    base = R.rule_conformal(Ft, s_all, org, test_org, 4, 60, 0.9)
    t = 800
    s2 = s_all.copy()
    s2[:, t - 4 + 1 - C.FIRST_ORIGIN:] = 9.0                                                    # Ursprünge, deren Zieltage nicht vor t liegen
    chg = R.rule_conformal(Ft, s2, org, test_org, 4, 60, 0.9)
    i = t - C.FIRST_TEST
    assert np.allclose(base[:, :i + 1], chg[:, :i + 1]) and not np.allclose(base, chg)


def test_oracle_is_exact_for_one_day_and_close_to_simulation_for_sums():
    mu = np.full((1, C.N_DAYS), 100.0)
    sigma = np.array([0.3])
    tau = 0.95
    one = R.rule_oracle(mu, sigma, np.array([800]), 1, tau)[0, 0]
    assert one == pytest.approx(100.0 * np.exp(0.3 * R.z_of(tau) - 0.5 * 0.09), rel=1e-9)                      # m = 1: Log-Normal exakt
    rng = np.random.default_rng(2)
    draws = (100.0 * np.exp(0.3 * rng.normal(size=(400000, 5)) - 0.045)).sum(axis=1)
    assert R.rule_oracle(mu, sigma, np.array([800]), 5, tau)[0, 0] == pytest.approx(np.quantile(draws, tau), rel=0.01)
    mu2 = np.tile(np.array([50.0, 80.0, 120.0, 60.0])[None, :], (1, 300))
    days = mu2[0, 800 % 4:800 % 4 + 4]
    d2 = (days[None, :] * np.exp(0.2 * rng.normal(size=(400000, 4)) - 0.02)).sum(axis=1)
    assert R.rule_oracle(mu2, np.array([0.2]), np.array([800]), 4, 0.9)[0, 0] == pytest.approx(np.quantile(d2, 0.9), rel=0.01)
