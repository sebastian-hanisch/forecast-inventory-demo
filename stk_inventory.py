"""Bestandsfortschreibung: tägliche Überprüfung, Wiederbeschaffungszeit L Tage, Fehlmengen werden nachgeliefert (Backorders), Order-up-to-Politik mit zeitvariabler Stufe S_t.

Ablauf eines Tages t:
  1. Wareneingang: die Bestellung von Tag t - L kommt an (NI = Nettobestand = Lagerbestand - Rückstand)
  2. Bestellung: q_t = max(S_t - IP_t, 0) mit der Bestandsposition IP_t = NI + noch unterwegs (bei L = 0 kommt q_t sofort an)
  3. Nachfrage y_t: aus dem Lagerbestand bedient wird max(NI, 0); Rest ist Fehlmenge, der Rückstand bleibt bestehen
  4. Tagesende: Lagerbestand max(NI, 0) kostet h je Stück, Rückstand max(-NI, 0) kostet p je Stück (h = 1, p = tau / (1 - tau))

Ist S_t konstant, gilt am Ende von Tag t + L: NI = S - (Nachfrage der Tage t..t+L). Der Servicegrad je Tag ist dann P(D_m <= S), die Kosten sind die des Newsvendor mit der Schutzintervall-Nachfrage D_m (m = L + 1)."""

import numpy as np


def simulate(S, y, L, tau, h=1.0):
    """S: (B, n, T) Bestellbestand je Tag (B Regeln nebeneinander), y: (n, T) Nachfrage der Tage. Rückgabe: dict mit on_hand, backorder, short (je (B, n, T)), orders (B, n, T) und den Kennzahlen je (B, n)."""
    B, n, T = S.shape
    p = h * tau / (1.0 - tau)
    arr = np.zeros((B, n, T + L + 1))
    NI = S[:, :, 0].copy()
    on_hand = np.zeros((B, n, T))
    backorder = np.zeros((B, n, T))
    short = np.zeros((B, n, T))
    orders = np.zeros((B, n, T))
    for t in range(T):
        NI = NI + arr[:, :, t]
        pipeline = arr[:, :, t + 1:t + L + 1].sum(axis=2) if L > 0 else 0.0
        q = np.maximum(S[:, :, t] - (NI + pipeline), 0.0)
        orders[:, :, t] = q
        if L == 0:
            NI = NI + q
        else:
            arr[:, :, t + L] += q
        d = y[:, t][None, :]
        short[:, :, t] = np.maximum(d - np.maximum(NI, 0.0), 0.0)
        NI = NI - d
        on_hand[:, :, t] = np.maximum(NI, 0.0)
        backorder[:, :, t] = np.maximum(-NI, 0.0)
    cost = h * on_hand + p * backorder
    return {"on_hand": on_hand, "backorder": backorder, "short": short, "orders": orders,
            "cost": cost.mean(axis=2), "alpha": (backorder == 0).mean(axis=2), "beta": 1.0 - short.sum(axis=2) / y.sum(axis=1)[None, :], "stock": on_hand.mean(axis=2)}
