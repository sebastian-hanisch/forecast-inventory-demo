"""Die Bestandsfortschreibung von Hand nachgerechnet und gegen den Newsvendor."""

import numpy as np
import pytest

import stk_inventory as I


def _run(S, y, L, tau=0.75):
    return I.simulate(np.asarray(S, dtype=float)[None, None, :], np.asarray(y, dtype=float)[None, :], L, tau)


def test_lead_time_zero_by_hand():
    # tau 0,75 -> p = 3, S = 10 konstant, Nachfrage 4, 12, 3.
    # Tag 0: NI 10, Bestellung 0, Nachfrage 4 -> NI 6. Tag 1: IP 6, Bestellung 4 -> NI 10, Nachfrage 12 -> NI -2 (Rückstand 2, Fehlmenge 2).
    # Tag 2: IP -2, Bestellung 12 -> NI 10, Nachfrage 3 -> NI 7.
    r = _run([10, 10, 10], [4, 12, 3], L=0)
    assert np.allclose(r["orders"][0, 0], [0, 4, 12]) and np.allclose(r["on_hand"][0, 0], [6, 0, 7]) and np.allclose(r["backorder"][0, 0], [0, 2, 0]) and np.allclose(r["short"][0, 0], [0, 2, 0])
    assert r["cost"][0, 0] == pytest.approx((6 + 3 * 2 + 7) / 3) and r["alpha"][0, 0] == pytest.approx(2 / 3) and r["beta"][0, 0] == pytest.approx(1 - 2 / 19) and r["stock"][0, 0] == pytest.approx(13 / 3)


def test_lead_time_two_by_hand():
    # L = 2, S = 10, Nachfrage 3 je Tag: die Bestellung von Tag t kommt zu Beginn von Tag t + 2 an.
    # Tag 0: q 0, NI 7. Tag 1: IP 7 -> q 3, NI 4. Tag 2: IP 4 + 3 = 7 -> q 3, NI 1. Tag 3: Eingang 3 -> NI 4, unterwegs 3 -> q 3, NI 1. Tag 4: dasselbe.
    r = _run([10] * 5, [3, 3, 3, 3, 3], L=2)
    assert np.allclose(r["orders"][0, 0], [0, 3, 3, 3, 3]) and np.allclose(r["on_hand"][0, 0], [7, 4, 1, 1, 1]) and (r["backorder"] == 0).all()


def test_backorders_are_delivered_and_the_stock_balance_holds():
    rng = np.random.default_rng(0)
    S = rng.uniform(20, 60, size=(3, 2, 200))
    y = rng.uniform(0, 30, size=(2, 200))
    for L in (0, 1, 4):
        r = I.simulate(S, y, L, 0.9)
        assert (r["orders"] >= 0).all() and (r["on_hand"] >= 0).all() and (r["backorder"] >= 0).all()
        assert ((r["on_hand"] > 0) & (r["backorder"] > 0)).sum() == 0
        b = 1
        net = r["on_hand"][b] - r["backorder"][b]
        arrivals = np.zeros((2, 200 + L + 1))
        arrivals[:, L:L + 200] = r["orders"][b]
        cum = np.cumsum(arrivals[:, :200] - y, axis=1)
        assert np.allclose(net, S[b][:, 0][:, None] + cum)                                    # Nettobestand = Anfangsbestand + Eingänge - Nachfrage


def test_constant_level_matches_the_newsvendor():
    # konstantes S, iid Nachfrage: Servicegrad P(D_m <= S), Kosten h E(S - D_m)+ + p E(D_m - S)+ mit D_m = Summe von m Tagen
    rng = np.random.default_rng(3)
    T, L, tau = 200000, 2, 0.9
    y = rng.gamma(20.0, 5.0, size=T)
    m = L + 1
    Dm = np.convolve(y, np.ones(m), mode="valid")
    S = float(np.quantile(Dm, 0.9))
    r = I.simulate(np.full((1, 1, T), S), y[None, :], L, tau)
    p = tau / (1 - tau)
    exact_cost = np.mean(np.maximum(S - Dm, 0) + p * np.maximum(Dm - S, 0))
    assert r["alpha"][0, 0] == pytest.approx(np.mean(Dm <= S), abs=0.004) and r["cost"][0, 0] == pytest.approx(exact_cost, rel=0.01)
    assert r["beta"][0, 0] > r["alpha"][0, 0]


def test_end_of_day_net_inventory_is_level_minus_protection_demand():
    rng = np.random.default_rng(4)
    T, L = 300, 3
    y = rng.uniform(5, 25, size=T)
    S = np.full(T, 150.0)
    r = _run(S, y, L)
    net = r["on_hand"][0, 0] - r["backorder"][0, 0]
    Dm = np.convolve(y, np.ones(L + 1), mode="valid")
    t = np.arange(20, T - L)
    assert np.allclose(net[t + L], S[t] - Dm[t])
