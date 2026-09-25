"""Presets und Permalink-Werte: Vollständigkeit, gültige Werte, Grenzen und Schrittweiten - reine Datenprüfungen ohne Streamlit-Session."""

import stk_constants as C
import stk_evaluation as E
import stk_presets as P


def _settings(p):
    return E.Settings(p["n_depots"], p["noise"], p["trend"], p["swing"], p["lead"], p["target"], p["window"], p["seed"])


def test_every_preset_has_help_and_all_keys():
    assert set(P.PRESETS) == set(P.PRESET_HELP)
    for name, p in P.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and P.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_slider_grid():
    for p in P.PRESETS.values():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            spec.caster(p[key])
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi
        for key, state_key in (("n_depots", "depots_slider"), ("noise", "noise_slider"), ("trend", "trend_slider"), ("swing", "swing_slider"), ("target", "target_slider"), ("window", "window_slider")):
            spec, step = P.SETTING_SPECS[state_key], P.STEPS[state_key]
            k = (p[key] - spec.lo) / step
            assert abs(k - round(k)) < 1e-6
        assert p["source"] in C.SOURCES and p["rule"] in C.RULES


def test_standard_preset_equals_the_default_settings():
    assert _settings(P.PRESETS["Standardfall: Lieferzeit 3 Tage, Ziel 95 %"]) == E.Settings()


def test_bounds_steps_and_unique_url_params():
    assert P.bounds("target_slider") == (C.TARGET_MIN, C.TARGET_MAX) and P.bounds("lead_slider") == (C.LEAD_MIN, C.LEAD_MAX)
    assert set(P.STEPS) == {"depots_slider", "noise_slider", "trend_slider", "swing_slider", "target_slider", "window_slider"}
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_casters_reject_bad_values():
    for caster, bad in ((P._source, "arima"), (P._rule, "quatsch")):
        try:
            caster(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    assert P._source(" HW ") == "hw" and P._rule(" Conformal ") == "conformal"
