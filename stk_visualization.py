"""Plotly-Abbildungen der Bestands-Demo. Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go

import stk_constants as C

SOURCE_COLORS = {"wm": "#17becf", "hw": "#e6550d", "reg": "#8c6bb1", "gbm": "#2e7d32", "mean4": "#1f77b4"}
RULE_COLORS = {"none": "#9e9e9e", "rough": "#d62728", "lognormal": "#e6550d", "empirical": "#8c6bb1", "conformal": "#00897b", "oracle": "#54a24b"}
ACTUAL = "#14233B"
WARN = "#f58518"
SHORT = {**C.SOURCE_SHORT, **C.RULE_SHORT, "oracle": "Orakel"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def build_stock(a, dep, key, start, span=90):
    """Bestandsverlauf eines Depots: Lagerbestand und Rückstand am Tagesende und die Tagesnachfrage."""
    T = len(a.test_org)
    i0 = int(np.clip(start - a.test_org[0], 0, max(T - span, 0)))
    sl = slice(i0, min(i0 + span, T))
    b = a.keys.index(key)
    x = a.test_org[sl]
    net = a.sim["on_hand"][b, dep, sl] - a.sim["backorder"][b, dep, sl]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=a.days[dep, sl], name="Tagesnachfrage", marker=dict(color="rgba(20,35,59,0.25)")))
    fig.add_trace(go.Scatter(x=x, y=np.maximum(net, 0), name="Lagerbestand", mode="lines", line=dict(color="#2e7d32", width=2), fill="tozeroy", fillcolor="rgba(46,125,50,0.15)"))
    fig.add_trace(go.Scatter(x=x, y=np.minimum(net, 0), name="Rückstand (Fehlmenge)", mode="lines", line=dict(color="#d62728", width=2), fill="tozeroy", fillcolor="rgba(214,39,40,0.25)"))
    fig.update_xaxes(title_text="Tag")
    fig.update_yaxes(title_text="Stück")
    fig.update_layout(barmode="overlay")
    return _base(fig, 360).update_layout(legend=dict(orientation="h", y=-0.3))


def build_protection(a, dep, source, rule, start, span=90):
    """Nachfrage im Schutzintervall: tatsächlich, Prognose der Quelle und Bestellbestand S der Regel (Abstand S - Prognose = Sicherheitsbestand)."""
    T = len(a.test_org)
    i0 = int(np.clip(start - a.test_org[0], 0, max(T - span, 0)))
    sl = slice(i0, min(i0 + span, T))
    x = a.test_org[sl]
    S = a.levels[(source, rule)][dep, sl]
    F = a.forecast[source][dep, sl]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=F, name="Prognose", mode="lines", line=dict(color=SOURCE_COLORS[source], width=2)))
    fig.add_trace(go.Scatter(x=x, y=S, name="Bestellbestand S", mode="lines", line=dict(color=WARN, width=2, dash="dash"), fill="tonexty", fillcolor="rgba(245,133,24,0.18)"))
    fig.add_trace(go.Scatter(x=x, y=a.demand[dep, sl], name="tatsächliche Nachfrage im Schutzintervall", mode="lines+markers", line=dict(color=ACTUAL, width=1.2), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=x, y=a.levels["oracle"][dep, sl], name="wahres Quantil (Orakel)", mode="lines", line=dict(color=RULE_COLORS["oracle"], width=1.6, dash="dot")))
    fig.update_xaxes(title_text="Ursprung (Beginn des Schutzintervalls)")
    fig.update_yaxes(title_text=f"Nachfrage über {a.m} Tage", rangemode="tozero")
    return _base(fig, 340).update_layout(legend=dict(orientation="h", y=-0.3))


def _heat(z, text, title, colorscale, zmid=None, zmin=None, zmax=None):
    fig = go.Figure(go.Heatmap(z=z, x=[SHORT[r] for r in C.RULES], y=[SHORT[s] for s in C.SOURCES], text=text, texttemplate="%{text}", colorscale=colorscale, zmid=zmid, zmin=zmin, zmax=zmax, showscale=False,
                               hovertemplate="%{y}, %{x}: %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")
    return _base(fig, 300)


def build_regret_heat(a):
    z = np.array([[100.0 * (a.metrics[(s, r)]["cost"] / a.metrics["oracle"]["cost"] - 1.0) for r in C.RULES] for s in C.SOURCES])
    txt = [[f"{v:+.0f} %".replace(".", ",") for v in row] for row in z]
    return _heat(np.log10(np.maximum(z, 1.0)), txt, "", "YlOrRd", zmin=0.5, zmax=2.7)


def build_alpha_heat(a):
    z = np.array([[a.metrics[(s, r)]["alpha"] for r in C.RULES] for s in C.SOURCES])
    txt = [[f"{100 * v:.1f} %".replace(".", ",") for v in row] for row in z]
    tgt = a.settings.target
    return _heat(z - tgt, txt, "", "RdBu", zmid=0.0, zmin=-0.15, zmax=0.15)


# --- Experimente ------------------------------------------------------------------------------------------------------------------------------


def _lines(fig, xs, series):
    for name, y, err, color, dash, width in series:
        fig.add_trace(go.Scatter(x=xs, y=y, error_y=dict(type="data", array=err) if err is not None else None, mode="lines+markers", name=name, line=dict(color=color, dash=dash, width=width)))


def build_target(rows):
    """Erreichter Servicegrad alpha je Regel gegen das Ziel; dazu der mengenmäßige Servicegrad beta der Orakel-Regel."""
    xs = [f"{100 * r['target']:g} %".replace(".", ",") for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[100 * r["target"] for r in rows], mode="lines", name="Ziel", line=dict(color="#7f7f7f", dash="dash", width=1.5)))
    for rule in ("rough", "lognormal", "empirical", "conformal"):
        k = rows[0]["source"], rule
        _lines(fig, xs, [(SHORT[rule], [100 * r["by"][k]["alpha"][0] for r in rows], [100 * r["by"][k]["alpha"][1] for r in rows], RULE_COLORS[rule], "solid", 2.4)])
    _lines(fig, xs, [("β der Orakel-Regel (mengenmäßig)", [100 * r["by"]["oracle"]["beta"][0] for r in rows], None, RULE_COLORS["oracle"], "dot", 2.4)])
    fig.update_xaxes(title_text="Zielservicegrad α (Anteil der Tage ohne Rückstand)", type="category")
    fig.update_yaxes(title_text="erreichter Servicegrad (%)")
    return _base(fig, 380).update_layout(legend=dict(orientation="h", y=-0.3))


def build_lead_regret(rows):
    xs = [str(r["lead"]) for r in rows]
    fig = go.Figure()
    for rule in ("rough", "lognormal", "empirical", "conformal"):
        k = rows[0]["source"], rule
        _lines(fig, xs, [(SHORT[rule], [r["by"][k]["regret"][0] for r in rows], [r["by"][k]["regret"][1] for r in rows], RULE_COLORS[rule], "solid", 2.4)])
    fig.update_xaxes(title_text="Wiederbeschaffungszeit L (Tage)", type="category")
    fig.update_yaxes(title_text="Mehrkosten gegen die Orakel-Regel (%)", rangemode="tozero")
    return _base(fig, 340)


def build_lead_alpha(rows):
    xs = [str(r["lead"]) for r in rows]
    fig = go.Figure()
    for rule in ("rough", "lognormal", "empirical", "conformal"):
        k = rows[0]["source"], rule
        _lines(fig, xs, [(SHORT[rule], [100 * r["by"][k]["alpha"][0] for r in rows], [100 * r["by"][k]["alpha"][1] for r in rows], RULE_COLORS[rule], "solid", 2.4)])
    fig.add_hline(y=100 * 0.95, line=dict(color="#7f7f7f", dash="dash"), annotation_text="Ziel", annotation_position="top left")
    fig.update_xaxes(title_text="Wiederbeschaffungszeit L (Tage)", type="category")
    fig.update_yaxes(title_text="erreichter Servicegrad α (%)")
    return _base(fig, 340)


def build_swing(rows):
    xs = [f"{r['swing']:g}".replace(".", ",") for r in rows]
    fig = go.Figure()
    for src in ("reg", "gbm", "mean4"):
        for rule, dash in (("empirical", "dot"), ("conformal", "solid")):
            k = (src, rule)
            _lines(fig, xs, [(f"{SHORT[src]}, {SHORT[rule].lower()}", [r["by"][k]["regret"][0] for r in rows], [r["by"][k]["regret"][1] for r in rows], SOURCE_COLORS[src], dash, 2.4)])
    fig.update_xaxes(title_text="Niveauschwankung der Depots", type="category")
    fig.update_yaxes(title_text="Mehrkosten gegen die Orakel-Regel (%)", rangemode="tozero")
    return _base(fig, 380).update_layout(legend=dict(orientation="h", y=-0.35))
