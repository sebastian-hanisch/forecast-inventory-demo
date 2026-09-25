"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster des Portfolios, vgl. cmb_presets.py)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import stk_constants as C


def _source(value):
    v = str(value).strip().lower()
    if v not in C.SOURCES:
        raise ValueError(value)
    return v


def _rule(value):
    v = str(value).strip().lower()
    if v not in C.RULES:
        raise ValueError(value)
    return v


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "depots_slider": SettingSpec("depots", int, C.DEFAULT_DEPOTS, C.DEPOTS_MIN, C.DEPOTS_MAX),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "trend_slider": SettingSpec("trend", int, C.DEFAULT_TREND, C.TREND_MIN, C.TREND_MAX),
    "swing_slider": SettingSpec("swing", float, C.DEFAULT_SWING, C.SWING_MIN, C.SWING_MAX),
    "lead_slider": SettingSpec("lead", int, C.DEFAULT_LEAD, C.LEAD_MIN, C.LEAD_MAX),
    "target_slider": SettingSpec("target", float, C.DEFAULT_TARGET, C.TARGET_MIN, C.TARGET_MAX),
    "window_slider": SettingSpec("window", int, C.DEFAULT_WINDOW, C.WINDOW_MIN, C.WINDOW_MAX),
    "source_select": SettingSpec("source", _source, "mean4"),
    "rule_select": SettingSpec("rule", _rule, "conformal"),
    "seed_input": SettingSpec("seed", int, 3, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n_depots": "depots_slider", "noise": "noise_slider", "trend": "trend_slider", "swing": "swing_slider", "lead": "lead_slider", "target": "target_slider", "window": "window_slider",
               "source": "source_select", "rule": "rule_select", "seed": "seed_input"}
STEPS = {"depots_slider": C.DEPOTS_STEP, "noise_slider": C.NOISE_STEP, "trend_slider": C.TREND_STEP, "swing_slider": C.SWING_STEP, "target_slider": C.TARGET_STEP, "window_slider": C.WINDOW_STEP}


def _p(**kw):
    base = {"n_depots": C.DEFAULT_DEPOTS, "noise": C.DEFAULT_NOISE, "trend": C.DEFAULT_TREND, "swing": C.DEFAULT_SWING, "lead": C.DEFAULT_LEAD, "target": C.DEFAULT_TARGET, "window": C.DEFAULT_WINDOW,
            "source": "mean4", "rule": "conformal", "seed": 3}
    base.update(kw)
    return base


PRESETS = {
    "Standardfall: Lieferzeit 3 Tage, Ziel 95 %": _p(),
    "Lieferzeit 0 (der reine Newsvendor)": _p(lead=0),
    "Lange Lieferzeit (13 Tage)": _p(lead=13),
    "Hohes Ziel (99 %)": _p(target=0.99),
    "Niedriges Ziel (80 %)": _p(target=0.80),
    "Regression bei starker Schwankung": _p(swing=0.12, source="reg", rule="empirical"),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, min(spec.hi, value))
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            spec = SETTING_SPECS[key]
            snapped = spec.lo + round((st.session_state[key] - spec.lo) / step) * step
            snapped = min(spec.hi, max(spec.lo, snapped))
            st.session_state[key] = int(snapped) if isinstance(spec.default, int) else round(float(snapped), 3)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = ",".join(value) if isinstance(value, (list, tuple)) else str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        v = PRESETS[name][key]
        st.session_state[state_key] = list(v) if isinstance(v, list) else v


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


PRESET_HELP = {
    "Standardfall: Lieferzeit 3 Tage, Ziel 95 %": "Mittelwert der vier Prognosen, konforme Regel (Seed 3, L = 3, Ziel 95 %): α 95,0 %, +42 % Mehrkosten gegen die wahre Verteilung; ohne Sicherheitsbestand α 54,7 % (+341 %), Faustregel α 88,5 % (+66 %), log-normal 93,7 % (+43 %), empirisch 94,1 % (+44 %); die günstigste Kombination ist das Boosting mit der empirischen Regel (+33 %).",
    "Lieferzeit 0 (der reine Newsvendor)": "Lieferzeit 0: die wahre Verteilung erreicht α 95,3 % und β 99,6 %; Mittelwert der vier: Faustregel +36 % (α 92,3 %), empirisch +18 % (α 95,0 %), konform +17 % (α 95,6 %); ohne Sicherheitsbestand +268 %.",
    "Lange Lieferzeit (13 Tage)": "Lieferzeit 13 Tage, Mittelwert der vier: Faustregel α 80,9 % (+175 %), empirisch α 92,9 % (+107 %), konform α 90,8 % (+105 %); die Mehrkosten gegen die wahre Verteilung wachsen mit der Lieferzeit, die konforme Regel fällt unter das Ziel.",
    "Hohes Ziel (99 %)": "Ziel 99 %, Mittelwert der vier: Faustregel α 93,9 % (+151 %), log-normal α 98,2 % (+51 %), empirisch α 98,3 % (+52 %), konform α 97,9 % (+59 %); ohne Sicherheitsbestand +1508 %.",
    "Niedriges Ziel (80 %)": "Ziel 80 %: die wahre Verteilung erreicht α 80,6 % und β 96,3 %; Mittelwert der vier: Faustregel α 76,1 % (+38 %), empirisch α 80,7 % (+38 %), konform α 79,9 % (+32 %).",
    "Regression bei starker Schwankung": "Regression bei Schwankung 0,12: empirische Regel α 81,6 % (+578 %), konforme α 90,3 % (+137 %); das Boosting mit der empirischen Regel α 94,8 % (+53 %). Prognosefehler der Regression 24,0 %, des Boostings 9,0 %.",
}
