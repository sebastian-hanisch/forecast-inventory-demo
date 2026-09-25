"""Auswertung Prognose -> Bestand: fünf Prognosequellen (vier Verfahren und ihr Mittelwert) mal fünf Bestandsregeln, dazu die Regel mit der wahren Verteilung als Untergrenze, jede mit der Bestandsfortschreibung über das Testjahr bewertet; drei Experimente.

Kennzahlen (je Depot, dann über die Depots gemittelt): **Kosten** je Tag in Tagesbedarfen (Lagerbestand h = 1 je Stück und Tag, Rückstand p = tau / (1 - tau) je Stück und Tag; geteilt durch den mittleren Tagesbedarf des Depots), **Servicegrad alpha** (Anteil der Tage ohne Rückstand am Tagesende),
**Servicegrad beta** (mengenmäßiger Erfüllungsgrad: 1 - Fehlmenge / Nachfrage), **Lagerbestand** (mittlerer Lagerbestand in Tagesbedarfen). **Mehrkosten** sind gegen die Regel mit der wahren Verteilung (Orakel) gerechnet."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import stk_constants as C
import stk_inventory as I
import stk_members as M
import stk_rules as R
import stk_scenario as S


@dataclass(frozen=True)
class Settings:
    n_depots: int = C.DEFAULT_DEPOTS
    noise: float = C.DEFAULT_NOISE
    trend: int = C.DEFAULT_TREND
    swing: float = C.DEFAULT_SWING
    lead: int = C.DEFAULT_LEAD
    target: float = C.DEFAULT_TARGET
    window: int = C.DEFAULT_WINDOW
    seed: int = 3

    @property
    def portfolio_key(self):
        return (self.n_depots, self.noise, self.trend, self.swing, self.seed)


@dataclass
class Analysis:
    settings: Settings
    port: S.Portfolio
    m: int                          # Länge des Schutzintervalls (L + 1)
    test_org: np.ndarray
    forecast: dict                  # Quelle -> (n, T) Prognose der Nachfrage im Schutzintervall
    demand: np.ndarray              # (n, T) tatsächliche Nachfrage im Schutzintervall
    levels: dict                    # (Quelle, Regel) und "oracle" -> (n, T) Bestellbestand S
    keys: list                      # Reihenfolge der Kombinationen in sim
    sim: dict                       # Ergebnis von I.simulate über alle Kombinationen (Achse 0 = keys)
    metrics: dict                   # Schlüssel -> {"cost", "alpha", "beta", "stock"} (über die Depots gemittelt)
    per_depot: dict                 # Schlüssel -> {"cost": (n,), "alpha": (n,), ...}
    wape: dict                      # Quelle -> Fehler der Schutzintervall-Prognose (mittlerer absoluter Fehler durch mittlere Nachfrage)
    days: np.ndarray                # (n, T_days) Nachfrage der simulierten Tage
    scale: np.ndarray               # (n,) mittlerer Tagesbedarf im Testjahr


@lru_cache(maxsize=16)
def _portfolio(key):
    n, noise, trend, swing, seed = key
    return S.generate(n, noise, 0.5, float(trend), seed, swing)


@lru_cache(maxsize=8)
def _members(key, m):
    return M.member_forecasts(_portfolio(key), m)


def build_levels(port, member_fc, s):
    """Bestellbestände aller Quellen und Regeln für alle Testursprünge; Rückgabe (levels, Nachfrage im Schutzintervall (n, T), Prognosen (n, T), Ursprünge, Testursprünge)."""
    m = s.lead + 1
    org = M.origins(m)
    test_org = np.arange(C.FIRST_TEST, C.N_DAYS - m + 1)
    it = test_org - C.FIRST_ORIGIN
    Dm_all = R.demand_sums(port.y, org, m)
    train = R.in_sample_index(org, m)
    levels, fcs = {}, {}
    for src, F in member_fc.items():
        Fm = R.cumulative(F)
        sc = R.scores(Dm_all, Fm)
        e1 = port.y[:, org] - F[:, :, 0]
        e1_train = e1[:, org < C.FIT_END]
        Ft = Fm[:, it]
        fcs[src] = Ft
        levels[(src, "none")] = R.rule_none(Ft)
        levels[(src, "rough")] = np.maximum(R.rule_rough(Ft, e1_train, m, s.target), 0.0)
        levels[(src, "lognormal")] = R.rule_lognormal(Ft, sc[:, train], s.target)
        levels[(src, "empirical")] = R.rule_empirical(Ft, sc[:, train], s.target)
        levels[(src, "conformal")] = R.rule_conformal(Ft, sc, org, test_org, m, s.window, s.target)
    levels["oracle"] = R.rule_oracle(port.mu, port.noise, test_org, m, s.target)
    return levels, Dm_all[:, it], fcs, org, test_org


def assemble(port, s, member_fc):
    m = s.lead + 1
    levels, Dm, fcs, org, test_org = build_levels(port, member_fc, s)
    keys = [(src, rule) for src in C.SOURCES for rule in C.RULES] + ["oracle"]
    Sarr = np.stack([levels[k] for k in keys])                                                  # (B, n, T)
    days = port.y[:, test_org]                                                                  # Nachfrage von Tag t (Beginn des Schutzintervalls)
    sim = I.simulate(Sarr, days, s.lead, s.target)
    scale = days.mean(axis=1)
    cost = sim["cost"] / scale[None, :]
    stock = sim["stock"] / scale[None, :]
    metrics, per = {}, {}
    for b, k in enumerate(keys):
        per[k] = {"cost": cost[b], "alpha": sim["alpha"][b], "beta": sim["beta"][b], "stock": stock[b]}
        metrics[k] = {q: float(v.mean()) for q, v in per[k].items()}
    wape = {src: float(np.mean(np.abs(fcs[src] - Dm).mean(axis=1) / Dm.mean(axis=1))) for src in fcs}
    return Analysis(s, port, m, test_org, fcs, Dm, levels, keys, sim, metrics, per, wape, days, scale)


@lru_cache(maxsize=6)
def analyse(s):
    return assemble(_portfolio(s.portfolio_key), s, _members(s.portfolio_key, s.lead + 1))


# --- Experimente ---------------------------------------------------------------------------------------------------------------------------------------------

def _mean_se(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def _replace(base, **kw):
    d = {f: getattr(base, f) for f in base.__dataclass_fields__}
    d.update(kw)
    return Settings(**d)


def _collect(analyses, keys):
    """Mittel und Standardfehler über die Analysen: {Schlüssel: {Kennzahl: (Mittel, SE)}} und die Mehrkosten gegen das Orakel je Schlüssel."""
    out = {}
    for k in keys:
        out[k] = {q: _mean_se([a.metrics[k][q] for a in analyses]) for q in ("cost", "alpha", "beta", "stock")}
        out[k]["regret"] = _mean_se([100.0 * (a.metrics[k]["cost"] / a.metrics["oracle"]["cost"] - 1.0) for a in analyses])
    return out


def target_experiment(levels=None, seeds=None, source="mean4", base=None):
    """Zielservicegrad tau (Zykluswahrscheinlichkeit alpha): erreichter alpha und beta je Regel, dazu das Orakel."""
    levels = C.TARGET_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings(n_depots=C.EXP_DEPOTS) if base is None else base
    keys = [(source, r) for r in C.RULES] + ["oracle"]
    return [{"target": t, "n_seeds": len(seeds), "source": source, "by": _collect([analyse(_replace(base, target=t, seed=sd)) for sd in seeds], keys)} for t in levels]


def lead_experiment(levels=None, seeds=None, source="mean4", base=None):
    """Wiederbeschaffungszeit L: Mehrkosten und Servicegrad je Regel."""
    levels = C.LEAD_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings(n_depots=C.EXP_DEPOTS) if base is None else base
    keys = [(source, r) for r in C.RULES] + ["oracle"]
    return [{"lead": L, "n_seeds": len(seeds), "source": source, "by": _collect([analyse(_replace(base, lead=L, seed=sd)) for sd in seeds], keys)} for L in levels]


def swing_experiment(levels=None, seeds=None, base=None):
    """Niveauschwankung: alle Quellen mit der empirischen und der konformen Regel; dazu der Prognosefehler der Quelle (WAPE)."""
    levels = C.SWING_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings(n_depots=C.EXP_DEPOTS) if base is None else base
    keys = [(s, r) for s in C.SOURCES for r in ("empirical", "conformal")] + ["oracle"]
    rows = []
    for w in levels:
        an = [analyse(_replace(base, swing=w, seed=sd)) for sd in seeds]
        rows.append({"swing": w, "n_seeds": len(seeds), "by": _collect(an, keys), "wape": {s: _mean_se([a.wape[s] for a in an]) for s in C.SOURCES}})
    return rows
