"""Bestandsregeln: aus der Prognose der Nachfrage im **Schutzintervall** wird der Bestellbestand S (Order-up-to-Stufe).

Ein Ursprung t heißt: bekannt sind die Tage 0..t-1, zu Beginn von Tag t wird bestellt (Wiederbeschaffungszeit L Tage, täglich Überprüfung). Die Bestellung deckt die Nachfrage der m = L + 1 Tage t..t+L (das Schutzintervall). F_m sei die Prognose dieser Summe
(die Summe der Punktprognosen der Horizonte 1..m), D_m die tatsächliche Nachfrage; Fehler ("Score") ist s = log((D_m + 1) / (F_m + 1)). Zielservicegrad tau = kritisches Verhältnis p / (p + h) (Newsvendor).

  none        S = F_m                                              kein Sicherheitsbestand
  rough       S = F_m + z σ_1 √m                                   Faustregel: z aus tau, σ_1 = Standardabweichung der EIN-Tages-Fehler y - F auf den Trainingstagen
  lognormal   S = (F_m + 1) exp(z σ_m) - 1                         σ_m = Standardabweichung der m-Tage-Scores auf den Trainingstagen
  empirical   S = (F_m + 1) exp(q_tau) - 1                         q_tau = tau-Quantil der m-Tage-Scores auf den Trainingstagen
  conformal   dasselbe mit den Scores der zuletzt REALISIERTEN Ursprünge (o <= t - m, die letzten W ab FIT_END) und der Endlichkeitskorrektur ceil((k+1) tau)
  oracle      S = tau-Quantil der wahren Nachfrage im Schutzintervall (Summe unabhängiger log-normaler Tagesnachfragen mit bekanntem Erwartungswert und bekannter Streuung, nach Fenton/Wilkinson durch eine Log-Normalverteilung angenähert)"""

from statistics import NormalDist

import numpy as np

import stk_constants as C

_ND = NormalDist()


def z_of(tau):
    return _ND.inv_cdf(tau)


def cumulative(F):
    """Summe der Punktprognosen über das Schutzintervall: (n, O, m) -> (n, O)."""
    return F.sum(axis=2)


def demand_sums(y, org, m):
    """Tatsächliche Nachfrage der Tage o..o+m-1 für alle Ursprünge: (n, O)."""
    cs = np.concatenate([np.zeros((y.shape[0], 1)), np.cumsum(y, axis=1)], axis=1)
    return cs[:, org + m] - cs[:, org]


def scores(Dm, Fm):
    return np.log((Dm + 1.0) / (Fm + 1.0))


def in_sample_index(org, m):
    """Ursprünge, deren Zieltage alle vor FIT_END liegen: die Trainingsfehler der Regeln."""
    return np.nonzero(org + m <= C.FIT_END)[0]


def rule_none(Fm_test):
    return Fm_test


def rule_rough(Fm_test, e1_train, m, tau):
    """e1_train: (n, n_train) Ein-Tages-Fehler y - F auf den Trainingstagen."""
    return Fm_test + z_of(tau) * e1_train.std(axis=1, ddof=1)[:, None] * np.sqrt(m)


def rule_lognormal(Fm_test, s_train, tau):
    return np.maximum((Fm_test + 1.0) * np.exp(z_of(tau) * s_train.std(axis=1, ddof=1)[:, None]) - 1.0, 0.0)


def rule_empirical(Fm_test, s_train, tau):
    q = np.quantile(s_train, tau, axis=1)[:, None]
    return np.maximum((Fm_test + 1.0) * np.exp(q) - 1.0, 0.0)


def conformal_quantile(S_ext, ends, W, tau):
    """Konformes tau-Quantil (obere Stufe: ceil((k+1) tau), auf 1..k begrenzt) aus den letzten W Scores bis zum Ursprungsindex e: (n, len(ends)). S_ext: (n, O_ext) Scores der Ursprünge ab FIT_END."""
    n = S_ext.shape[0]
    P = np.concatenate([np.full((n, W), np.nan), S_ext], axis=1)
    win = np.lib.stride_tricks.sliding_window_view(P, W, axis=1)
    A = np.sort(win[:, np.asarray(ends) + 1, :], axis=-1)                                     # NaN am Ende
    k = np.sum(~np.isnan(A), axis=-1)
    rank = np.clip(np.ceil((k + 1) * tau), 1, np.maximum(k, 1)).astype(int)
    return np.take_along_axis(A, (rank - 1)[..., None], axis=-1)[..., 0]


def rule_conformal(Fm_test, s_all, org, test_org, m, W, tau):
    """s_all: (n, O) Scores aller Ursprünge; die Quelle der Kalibrierung sind die Ursprünge ab FIT_END."""
    S_ext = s_all[:, C.FIT_END - C.FIRST_ORIGIN:]
    ends = test_org - m - C.FIT_END                                                            # letzter Ursprung, dessen Zieltage vor t liegen
    q = conformal_quantile(S_ext, ends, W, tau)
    return np.maximum((Fm_test + 1.0) * np.exp(q) - 1.0, 0.0)


def rule_oracle(mu, sigma, test_org, m, tau):
    """Wahres tau-Quantil der Nachfrage im Schutzintervall (Fenton/Wilkinson): mu (n, T_all), sigma (n,) Streuung des Tagesrauschens."""
    days = test_org[:, None] + np.arange(m)[None, :]
    mm = mu[:, days]                                                                            # (n, T, m)
    M = mm.sum(axis=2)
    V = ((mm ** 2) * (np.exp(sigma[:, None, None] ** 2) - 1.0)).sum(axis=2)
    s2 = np.log1p(V / M ** 2)
    return np.exp(np.log(M) - 0.5 * s2 + np.sqrt(s2) * z_of(tau))
