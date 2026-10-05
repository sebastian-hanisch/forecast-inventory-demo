"""Orakel-Test: Bestandsfortschreibung gegen eine Ereignisschleife (Ankunftsliste statt Ringpuffer) und den Newsvendor in geschlossener Form (normalverteilte Tage), konformes Quantil gegen die Rangdefinition,
Merkmale gegen eine Skalarrechnung (mit Leck-Prüfung) und der Histogramm-Baumkern gegen eine Zeilen-Brute-Force-Suche (blattweises Wachsen)."""

import dataclasses
import math

import numpy as np
import pytest

import stk_constants as C
import stk_features as F
import stk_gbm as G
import stk_inventory as I
import stk_rules as R
import stk_scenario as S
import stk_tree as T


def ref_sim(Sv, y, L, tau, h=1.0):
    p = h * tau / (1 - tau)
    NI, pend = Sv[0], []
    oh, bo, sh, od = (np.zeros(len(y)) for _ in range(4))
    for t in range(len(y)):
        NI += sum(q for (a, q) in pend if a == t)
        pend = [(a, q) for (a, q) in pend if a > t]
        q = max(Sv[t] - (NI + sum(q for (a, q) in pend)), 0.0)
        od[t] = q
        if L == 0:
            NI += q
        else:
            pend.append((t + L, q))
        sh[t] = max(y[t] - max(NI, 0.0), 0.0)
        NI -= y[t]
        oh[t], bo[t] = max(NI, 0.0), max(-NI, 0.0)
    return dict(cost=(h * oh + p * bo).mean(), alpha=(bo == 0).mean(), beta=1 - sh.sum() / y.sum(), stock=oh.mean(), orders=od, on_hand=oh, backorder=bo, short=sh)


def test_simulation_matches_an_event_loop():
    rng = np.random.default_rng(18)
    for it in range(120):
        T_, L, tau = int(rng.integers(1, 60)), int(rng.integers(0, 14)), float(rng.choice([0.5, 0.8, 0.9, 0.99]))
        Sv = rng.uniform(0, 80, size=T_)
        y = rng.integers(0, 40, size=T_).astype(float)
        y[0] += 1.0
        r, q = I.simulate(Sv[None, None, :], y[None, :], L, tau), ref_sim(Sv, y, L, tau)
        for k in ("cost", "alpha", "beta", "stock"):
            assert r[k][0, 0] == pytest.approx(q[k], rel=1e-9, abs=1e-12), (k, L)
        for k in ("orders", "on_hand", "backorder", "short"):
            assert np.allclose(r[k][0, 0], q[k]), k


def test_constant_level_matches_the_closed_form_newsvendor_for_normal_days():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    L, tau, mu, sd, n = 2, 0.95, 50.0, 10.0, 120000
    m = L + 1
    y = rng.normal(mu, sd, size=n)
    Sc = m * mu + stats.norm.ppf(tau) * sd * math.sqrt(m)
    r = I.simulate(np.full((1, 1, n), Sc), y[None, :], L, tau)
    s_, u, p = sd * math.sqrt(m), stats.norm.ppf(tau), tau / (1 - tau)
    cost = s_ * (u * stats.norm.cdf(u) + stats.norm.pdf(u)) + p * s_ * (stats.norm.pdf(u) - u * (1 - stats.norm.cdf(u)))
    assert r["alpha"][0, 0] == pytest.approx(tau, abs=0.006) and r["cost"][0, 0] == pytest.approx(cost, rel=0.02)


def test_conformal_quantile_follows_the_rank_definition():
    rng = np.random.default_rng(2)
    for _ in range(80):
        n_o, W, tau = int(rng.integers(20, 300)), int(rng.choice([30, 60, 120])), float(rng.choice([0.8, 0.9, 0.95, 0.99]))
        S_ext = rng.normal(size=(1, n_o))
        ends = rng.integers(0, n_o, size=4)
        got = R.conformal_quantile(S_ext, ends, W, tau)[0]
        for a, e in enumerate(ends):
            win = sorted(S_ext[0, max(0, e - W + 1):e + 1])
            k, r = len(win), 1
            while r < k and r < (k + 1) * tau - 1e-12:                                   # kleinster Rang r >= (k+1) tau, höchstens k
                r += 1
            assert got[a] == pytest.approx(win[r - 1])


def test_oracle_rule_close_to_monte_carlo_for_unequal_days():
    mus = np.array([60.0, 250.0, 90.0, 40.0, 120.0])
    mu = np.ones((1, C.N_DAYS))
    mu[0, 800:805] = mus
    rng = np.random.default_rng(3)
    for sigma, tau in ((0.3, 0.9), (0.14, 0.99)):
        draws = (mus[None, :] * np.exp(sigma * rng.standard_normal((400000, 5)) - 0.5 * sigma ** 2)).sum(axis=1)
        assert R.rule_oracle(mu, np.array([sigma]), np.array([800]), 5, tau)[0, 0] == pytest.approx(np.quantile(draws, tau), rel=0.01)


def test_features_match_a_scalar_computation_and_do_not_leak():
    port = S.generate(4, 0.14, 0.5, 10.0, 7, 0.06)
    rng = np.random.default_rng(1)
    for _ in range(60):
        i, t, hz = int(rng.integers(0, 4)), int(rng.integers(100, C.N_DAYS - 28)), int(rng.integers(1, 29))
        X, tgt, lvl = F.build(port, np.array([i]), np.array([t]), np.array([hz]))
        y, s = port.y[i], t + hz - 1
        lv28, lv91 = np.mean(y[t - 28:t]), np.mean(y[t - 91:t])
        wk = [s - 7 * q for q in range(1, 40) if s - 7 * q <= t - 1][:4]                # die vier jüngsten bekannten Tage mit dem Wochentag des Zieltags
        doy = s % 365
        row = [y[t - k] / (lv28 + 1) for k in range(1, 8)] + [y[d] / (lv28 + 1) for d in wk] + [np.mean(y[t - 7:t]) / (lv28 + 1), lv28 / (lv91 + 1), np.std(y[t - 28:t]) / (lv28 + 1)]
        row += [s % 7, math.sin(2 * math.pi * doy / 365), math.cos(2 * math.pi * doy / 365), port.holiday[s], port.after[s], hz, port.promo[i, s], np.mean(port.promo[i, t - 7:t])]
        assert np.allclose(X[0], row, atol=1e-9) and tgt[0] == pytest.approx(math.log((y[s] + 1) / (lv28 + 1)))
    y2 = port.y.copy()
    y2[:, 400:] = 12345.0
    X1, _, _ = F.build(port, np.arange(4), np.full(4, 400), np.full(4, 5))
    X2, _, _ = F.build(dataclasses.replace(port, y=y2), np.arange(4), np.full(4, 400), np.full(4, 5))
    assert np.allclose(X1, X2)


def _best_split_rows(X, g, h, idx, edges, lam, mcs):
    Gt, Ht = g[idx].sum(), h[idx].sum()
    best = None
    for f in range(X.shape[1]):
        for e in edges[f]:
            L, R_ = idx[X[idx, f] <= e], idx[X[idx, f] > e]
            if len(L) < mcs or len(R_) < mcs or h[L].sum() < 1.0 or h[R_].sum() < 1.0:
                continue
            gain = 0.5 * (g[L].sum() ** 2 / (h[L].sum() + lam) + g[R_].sum() ** 2 / (h[R_].sum() + lam) - Gt ** 2 / (Ht + lam))
            if gain > 0 and (best is None or gain > best[2] + 1e-12):
                best = (f, e, gain)
    return best


def _ref_leafwise(X, g, h, edges, num_leaves, lam, mcs):
    mk = lambda idx: dict(idx=idx, best=_best_split_rows(X, g, h, idx, edges, lam, mcs) if len(idx) >= 2 * mcs else None)
    active, n_leaves = [mk(np.arange(len(X)))], 1
    while n_leaves < num_leaves and any(nd["best"] for nd in active):
        nd = max((z for z in active if z["best"]), key=lambda z: z["best"][2])
        f, e, _ = nd["best"]
        active = [z for z in active if z is not nd] + [mk(nd["idx"][X[nd["idx"], f] <= e]), mk(nd["idx"][X[nd["idx"], f] > e])]
        n_leaves += 1
    return active


def test_histogram_tree_and_boosting_match_a_row_brute_force():
    rng = np.random.default_rng(11)
    for it in range(40):
        n, d = int(rng.integers(30, 120)), int(rng.integers(1, 4))
        X = rng.normal(size=(n, d))
        g, h = rng.normal(size=n), (rng.uniform(0.5, 2, size=n) if it % 3 else np.ones(n))
        nl, lam, mcs = int(rng.integers(2, 9)), float(rng.choice([0.0, 1.0, 5.0])), int(rng.choice([1, 3, 10]))
        edges = T.build_bin_edges(X, int(rng.choice([8, 16, 63])))
        with np.errstate(all="ignore"):
            tree = T.grow(X, g, h, edges, num_leaves=nl, lam=lam, min_child_weight=1.0, min_child_samples=mcs)
        leaves = _ref_leafwise(X, g, h, edges, nl, lam, mcs)
        ref = np.zeros(n)
        for lf in leaves:
            ref[lf["idx"]] = -g[lf["idx"]].sum() / (h[lf["idx"]].sum() + lam)
        assert tree.n_leaves == len(leaves) and np.allclose(T.predict_value(tree, X), ref, atol=1e-9)
    X = rng.normal(size=(150, 3))
    y = X[:, 0] ** 2 + np.sin(X[:, 1])
    ens = G.fit(X, y, num_leaves=5, max_bin=16, lam=1.0, min_child_samples=5, n_rounds=3, learning_rate=0.5)
    edges, Fv = T.build_bin_edges(X, 16), np.full(150, y.mean())
    for _ in range(3):
        step = np.zeros(150)
        for lf in _ref_leafwise(X, Fv - y, np.ones(150), edges, 5, 1.0, 5):
            step[lf["idx"]] = -(Fv - y)[lf["idx"]].sum() / (len(lf["idx"]) + 1.0)
        Fv = Fv + 0.5 * step
    assert np.allclose(G.predict(ens, X), Fv, atol=1e-8)
