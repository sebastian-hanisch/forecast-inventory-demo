"""Die Mitglieder des Pools: vier Prognoseverfahren und ihr Mittelwert aus den Vorgängern, je Depot (das Boosting global über alle Depots), alle mit Parametern aus den Tagen vor FIT_END.

Ursprung t: bekannt sind die Tage 0..t-1, prognostiziert werden t..t+h-1. Die Prognosen werden für alle Ursprünge ab FIRST_ORIGIN berechnet (Training, Kalibrierfenster FIT_END..FIRST_TEST-1 und Testjahr)."""

import numpy as np

import stk_baselines as B
import stk_constants as C
import stk_features as F
import stk_gbm as G


def origins(horizon):
    """Alle Ursprünge ab FIRST_ORIGIN (Training, Kalibrierung und Test) bis zum letzten mit vollem Horizont."""
    return np.arange(C.FIRST_ORIGIN, C.N_DAYS - horizon + 1)


def gbm_forecasts(port, horizon):
    """Ein globales Boosting-Modell über alle Depots (Stück 6): Zieltage vor FIT_END im Training, Ziel = log-Verhältnis zum 28-Tage-Niveau; Rückgabe (n, n_org, h) für alle Ursprünge ab FIRST_ORIGIN."""
    depots = np.arange(port.n)
    d, o, h = F.training_rows(port, depots, horizon, stride=4, per_origin=2, last_target=C.FIT_END, seed=0)
    X, y, _ = F.build(port, d, o, h)
    ens = G.fit(X, y, num_leaves=15, min_child_samples=20, n_rounds=C.GBM_ROUNDS, learning_rate=0.1, seed=0)
    td, to, th, org = F.test_rows(port, depots, horizon, first=C.FIRST_ORIGIN)
    Xt, _, lvl = F.build(port, td, to, th)
    pred = F.to_orders(G.predict(ens, Xt), lvl)
    return pred.reshape(port.n, len(org), horizon)


def member_forecasts(port, horizon):
    """{Verfahren: (n, n_org, h)} für alle Ursprünge ab FIRST_ORIGIN; Parameter aus den Tagen vor FIT_END."""
    org = origins(horizon)
    out = {m: [] for m in ("wm", "hw", "reg")}
    for i in range(port.n):
        y = port.y[i]
        out["wm"].append(B.snaive_k(y, org, horizon, k=4))
        out["hw"].append(B.hw_forecast(y, org, horizon, first=C.FIT_END))
        X = B.regression_design(port.dow, port.holiday, port.after, port.promo[i])
        out["reg"].append(B.regression_forecast(y, X, org, horizon, first=C.FIT_END))
    res = {m: np.stack(v) for m, v in out.items()}
    res["gbm"] = gbm_forecasts(port, horizon)
    res["mean4"] = np.mean([res[m] for m in C.MEMBERS], axis=0)
    return {m: res[m] for m in C.SOURCES}
