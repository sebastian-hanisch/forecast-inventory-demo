"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Depot- und Ausschnitts-Regler, Würfel-Knopf, Permalink-Grenzen, Extremwerte, drei Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import stk_constants as C
import stk_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=600)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]
    for el in list(at.caption) + list(at.markdown) + list(at.warning) + list(at.success) + list(at.info):
        assert "{de(" not in el.value and "{pct(" not in el.value and "{spct(" not in el.value, el.value[:120]


def test_default_run_shows_metrics_charts_and_a_verdict():
    at = _run()
    _ok(at)
    assert len(at.metric) == 5 and len(at.get("plotly_chart")) == 4 and len(at.info) + len(at.success) + len(at.warning) >= 1


@pytest.mark.parametrize("name", list(P.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = P.PRESETS[name]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == p[key]


def test_depot_and_origin_selections_survive_a_smaller_portfolio_and_longer_lead_time():
    at = _run(depot_slider=29, origin_slider=1094 - 20)
    _ok(at)
    at.slider(key="depots_slider").set_value(10).run()
    _ok(at)
    assert at.session_state["depot_slider"] <= 9
    at.slider(key="lead_slider").set_value(13).run()
    _ok(at)
    assert at.session_state["origin_slider"] <= C.N_DAYS - 14


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neues Portfolio generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_snapped_and_clamped():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["depots"] = "77"
    at.query_params["lead"] = "99"
    at.query_params["target"] = "0.5"
    at.query_params["window"] = "10"
    at.query_params["source"] = "arima"
    at.query_params["rule"] = "Empirical"
    at.query_params["swing"] = "abc"
    at.run()
    _ok(at)
    assert at.session_state["depots_slider"] == 80 and at.session_state["lead_slider"] == C.LEAD_MAX and at.session_state["target_slider"] == C.TARGET_MIN and at.session_state["window_slider"] == C.WINDOW_MIN
    assert at.session_state["source_select"] == "mean4" and at.session_state["rule_select"] == "empirical" and at.session_state["swing_slider"] == C.DEFAULT_SWING


@pytest.mark.parametrize("kw", [dict(depots_slider=C.DEPOTS_MIN, lead_slider=0, target_slider=C.TARGET_MAX), dict(lead_slider=C.LEAD_MAX, window_slider=C.WINDOW_MAX, target_slider=C.TARGET_MIN),
                                dict(swing_slider=C.SWING_MAX, noise_slider=C.NOISE_MAX, trend_slider=C.TREND_MIN), dict(source_select="wm", rule_select="none"),
                                dict(source_select="reg", rule_select="rough", swing_slider=0.0, noise_slider=C.NOISE_MIN)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def _small(monkeypatch):
    monkeypatch.setattr(C, "EXP_SEEDS", (0,))
    monkeypatch.setattr(C, "EXP_DEPOTS", 10)
    monkeypatch.setattr(C, "TARGET_LEVELS", (0.80, 0.95))
    monkeypatch.setattr(C, "LEAD_LEVELS", (0, 13))
    monkeypatch.setattr(C, "SWING_LEVELS", (0.0, 0.12))


def _click(at, key):
    next(b for b in at.button if b.key == key).click().run()
    _ok(at)


def test_target_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    at = _run()
    _click(at, "target_start")
    assert at.session_state["target_on"] and any("Ein Ziel ohne Definition ist kein Ziel" in w.value for w in at.warning)


def test_lead_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    at = _run()
    _click(at, "lead_start")
    assert at.session_state["lead_on"] and any("Die Mehrkosten wachsen mit der Wiederbeschaffungszeit" in w.value for w in at.warning)


def test_swing_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    at = _run()
    _click(at, "swing_start")
    assert at.session_state["swing_on"] and any("mitlaufende Kalibrierung" in w.value for w in at.warning)


def test_footer_and_grenzen_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
