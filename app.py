"""Von der Prognose zum Bestand - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zehntes Stück der Zeitreihen-Prognose-Linie der "Konzepte"-Reihe: der Zusammenfluss der Prognoseverfahren (Stücke 1 bis 9) mit der Bestandsplanung. Aus der Prognose der Nachfrage im Schutzintervall wird mit einer Regel
(Faustregel, log-normal, empirisch, konform) der Bestellbestand; eine Bestandsfortschreibung mit Wiederbeschaffungszeit bewertet Kosten und Servicegrad.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import stk_constants as C
from stk_evaluation import Settings, analyse, lead_experiment, swing_experiment, target_experiment
from stk_presets import PRESET_HELP, PRESETS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from stk_visualization import SHORT, build_alpha_heat, build_lead_alpha, build_lead_regret, build_protection, build_regret_heat, build_stock, build_swing, build_target

st.set_page_config(page_title="Von der Prognose zum Bestand – Sebastian Hanisch", layout="wide")


def de(x, digits=2):
    """Deutsche Zahlenschreibweise: Punkt als Tausendertrenner, Komma als Dezimalzeichen."""
    x = round(float(x), digits)
    if x == 0:
        x = 0.0
    return f"{x:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(x, digits=1):
    return f"{de(100 * x, digits)} %"


def spct(x, digits=0):
    """Prozent mit ausdrücklichem Vorzeichen (Minuszeichen U+2212); x in Prozent."""
    x = round(float(x), digits)
    if x == 0:
        x = 0.0
    return ("+" if x > 0 else "−" if x < 0 else "±") + f"{abs(x):.{digits}f}".replace(".", ",") + " %"


@st.cache_data(show_spinner=False)
def _target(levels, seeds):
    return target_experiment(levels, seeds)


@st.cache_data(show_spinner=False)
def _lead(levels, seeds):
    return lead_experiment(levels, seeds)


@st.cache_data(show_spinner=False)
def _swing(levels, seeds):
    return swing_experiment(levels, seeds)


st.title("📦 Von der Prognose zum Bestand")
st.markdown(
    """
Eine Prognose ist kein Bestand. Wer bestellt, muss die Nachfrage der **Wiederbeschaffungszeit** decken - und ein **Sicherheitsbestand** fängt ab, was die Prognose verfehlt. Wie groß der sein muss, hängt an der **Unsicherheit der Prognose**, nicht an ihrem Wert: aus den Intervallen (Stück 7) wird ein **Quantil**, aus dem Quantil der Bestellbestand.
Die Demo nimmt die Prognosen der Vorgänger (Wochenmittel, Holt-Winters, Regression, Boosting und ihren Mittelwert) auf einem **Portfolio von Depots** und bildet vier **Bestandsregeln** (Faustregel, log-normal, empirisch, konform) gegen eine Regel mit der **wahren Verteilung**. Eine **Bestandsfortschreibung** über das Testjahr
(tägliche Überprüfung, Wiederbeschaffungszeit, nachgelieferte Fehlmengen) misst, was das an **Kosten und Servicegrad** bringt - und was es kostet, nicht zu wissen, wie sicher die Prognose ist. Alle Daten sind erzeugt; die Rechnung ist in numpy geschrieben.
"""
)
st.caption(
    "Zehntes Stück der **Zeitreihen-Prognose-Linie** der \"Konzepte\"-Reihe: der **Zusammenfluss** mit der Bestandsplanung (Prognoseintervalle Stück 7, Kombination Stück 9). **Bezug zu OR:** die Bestandspolitik (Base-Stock, Newsvendor-Quantil) ist der Abnehmer der Prognose; "
    "die Vorab-Messreihe der Bestandsseite (Definitionen des Servicegrads, Schätzfehler kurzer Historien) ergänzt sie."
)

with st.expander("So wird aus der Prognose ein Bestand", expanded=True):
    st.markdown(
        """
1. **Das Schutzintervall.** Bestellt wird jeden Tag, die Ware kommt nach $L$ Tagen. Die Bestellung von Tag $t$ muss die Nachfrage der $m = L + 1$ Tage $t, \\dots, t+L$ decken. Die Prognose dieser Summe ist die Summe der Punktprognosen der Horizonte $1..m$.
2. **Der Bestellbestand $S$** (Order-up-to): jeden Tag wird bis zur Bestandsposition $S_t$ aufgefüllt. Beim **Newsvendor** ist der kostenoptimale Bestellbestand das **Quantil** der Nachfrage im Schutzintervall zum kritischen Verhältnis $\\tau = p/(p+h)$ (Fehlmenge $p$, Lagerung $h$ je Stück und Tag). Das Ziel $\\tau$ ist zugleich der Ziel-Servicegrad $\\alpha$.
3. **Vier Regeln** für dieses Quantil, jeweils mit der Prognose als Mitte: **ohne Sicherheitsbestand** ($S = F_m$), die **Faustregel** $F_m + z\\,\\sigma_1\\sqrt m$ (σ aus den Ein-Tages-Fehlern des Trainings), **log-normal** (Streuung der Summenfehler im Training), **empirisch** (Quantil der Summenfehler im Training), **konform** (Quantil der zuletzt realisierten Summenfehler, Stück 7).
   Als Untergrenze dient die Regel mit der **wahren Verteilung** (Orakel).
4. **Die Fortschreibung.** Ein Tag: Wareneingang, Bestellung $q_t = \\max(S_t - IP_t, 0)$, Nachfrage; Lagerbestand kostet $h = 1$ je Stück und Tag, Rückstand $p = \\tau/(1-\\tau)$ je Stück und Tag. Gemessen werden die Kosten (in Tagesbedarfen), der **Servicegrad $\\alpha$** (Anteil der Tage ohne Rückstand), der **Servicegrad $\\beta$** (mengenmäßig: 1 − Fehlmenge/Nachfrage) und der Lagerbestand.
        """
    )

st.caption("🎯 Schnellstart – ein Beispiel laden:")
preset_names = list(PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP.get(name), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Das Portfolio**")
    n_depots = st.slider("Zahl der Depots", *bounds("depots_slider"), key="depots_slider", step=C.DEPOTS_STEP, help="Wie viele Depots (Artikel) das Portfolio hat; das Boosting lernt aus allen.")
    noise = st.slider("Rauschen (Mittel der Depots)", *bounds("noise_slider"), key="noise_slider", step=C.NOISE_STEP, help="Mittlere Streuung des multiplikativen Tagesrauschens.")
    trend = st.slider("Trend (% je Jahr, Mittel)", *bounds("trend_slider"), key="trend_slider", step=C.TREND_STEP, help="Mittleres Wachstum der Depots in Prozent des Ausgangsniveaus je Jahr.")
    swing = st.slider("Niveauschwankung der Depots", *bounds("swing_slider"), key="swing_slider", step=C.SWING_STEP, help="Stärke der langsamen, depoteigenen Niveauschwankungen (im Log). Die statische Regression kann ihnen nicht folgen.")
    st.markdown("**Die Bestandspolitik**")
    lead = st.slider("Wiederbeschaffungszeit L (Tage)", *bounds("lead_slider"), key="lead_slider", help="Wie viele Tage eine Bestellung braucht; das Schutzintervall hat L + 1 Tage.")
    target = st.slider("Ziel-Servicegrad α", *bounds("target_slider"), key="target_slider", step=C.TARGET_STEP, format="%.2f", help="Zugleich das kritische Verhältnis p/(p+h): Ziel 0,95 heißt Fehlmenge 19-mal so teuer wie Lagerung.")
    window = st.slider("Kalibrierfenster der konformen Regel (Ursprünge)", *bounds("window_slider"), key="window_slider", step=C.WINDOW_STEP, help="Aus den letzten so vielen realisierten Ursprüngen wird das Fehlerquantil der konformen Regel bestimmt.")
    st.markdown("**Anzeige**")
    source = st.selectbox("Prognosequelle", list(C.SOURCES), key="source_select", format_func=lambda k: C.SOURCE_NAMES[k], help="Welche Prognose im Depot-Diagramm und in der Tabelle gezeigt wird.")
    rule = st.selectbox("Bestandsregel", list(C.RULES), key="rule_select", format_func=lambda k: C.RULE_NAMES[k], help="Welche Regel im Depot-Diagramm gezeigt wird.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Legt das ganze Portfolio fest.")
    st.button("🎲 Neues Portfolio generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"depots_slider": int(n_depots), "noise_slider": round(float(noise), 2), "trend_slider": int(trend), "swing_slider": round(float(swing), 2), "lead_slider": int(lead), "target_slider": round(float(target), 2),
                   "window_slider": int(window), "source_select": source, "rule_select": rule, "seed_input": int(seed)})

settings = Settings(int(n_depots), round(float(noise), 2), int(trend), round(float(swing), 2), int(lead), round(float(target), 2), int(window), int(seed))
with st.spinner("Die Prognosen werden berechnet und der Bestand über das Testjahr fortgeschrieben ..."):
    a = analyse(settings)
port = a.port
key = (source, rule)
mt, mo = a.metrics[key], a.metrics["oracle"]
ratio = settings.target / (1.0 - settings.target)

# --- Ein Depot -------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ein Depot: von der Prognose zum Bestellbestand")
st.session_state["depot_slider"] = min(port.n - 1, max(0, st.session_state.get("depot_slider", 0)))
dep = int(st.slider("Depot", 0, port.n - 1, key="depot_slider", help="Welches Depot des Portfolios gezeigt wird."))
lo_o, hi_o = int(a.test_org[0]), int(a.test_org[-1])
st.session_state["origin_slider"] = min(hi_o, max(lo_o, st.session_state.get("origin_slider", 800)))
origin = int(st.slider("Anfang des gezeigten Ausschnitts (Tag)", lo_o, hi_o, key="origin_slider", help="Ab diesem Tag werden 90 Tage gezeigt."))
st.markdown("##### Prognose, Sicherheitsbestand und tatsächliche Nachfrage im Schutzintervall")
st.plotly_chart(build_protection(a, dep, source, rule, origin), width="stretch", key="protection_chart")
st.caption(
    f"Depot {dep}: mittlere Tagesnachfrage {de(a.scale[dep], 0)} Stück, Schutzintervall {a.m} Tage. Blau die Prognose der Nachfrage über diese {a.m} Tage ({SHORT[source]}), gestrichelt der Bestellbestand $S$ der Regel **{SHORT[rule]}**: der schattierte Abstand ist der **Sicherheitsbestand**. "
    f"Schwarz die tatsächliche Nachfrage, grün gepunktet das wahre {int(round(100 * settings.target))} %-Quantil. Liegt schwarz über gestrichelt, entsteht ein Rückstand."
)
st.markdown("##### Der Bestand über die Zeit")
st.plotly_chart(build_stock(a, dep, key, origin), width="stretch", key="stock_chart")
pd_ = a.per_depot[key]
st.caption(
    f"In diesem Depot ({SHORT[source]}, {SHORT[rule]}): Kosten {de(pd_['cost'][dep], 2)} Tagesbedarfe je Tag, Servicegrad α {pct(pd_['alpha'][dep])}, β {pct(pd_['beta'][dep])}, mittlerer Lagerbestand {de(pd_['stock'][dep], 2)} Tagesbedarfe. "
    f"Grün der Lagerbestand am Tagesende, rot der Rückstand (Fehlmenge, die nachgeliefert wird), grau die Tagesnachfrage."
)

st.markdown("---")

# --- Auswertung -----------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was kostet es, die Unsicherheit nicht zu kennen?")
mcols = st.columns(5)
mcols[0].metric("Kosten (Tagesbedarfe/Tag)", de(mt["cost"], 3), delta=f"{spct(100 * (mt['cost'] / mo['cost'] - 1))} gegen Orakel", delta_color="inverse", help="Mittlere Kosten je Tag, in Tagesbedarfen des Depots; das Orakel ist die Regel mit der wahren Verteilung.")
mcols[1].metric("Servicegrad α", pct(mt["alpha"]), delta=f"Ziel {pct(settings.target, 0)}", delta_color="off", help="Anteil der Tage ohne Rückstand am Tagesende.")
mcols[2].metric("Servicegrad β", pct(mt["beta"]), delta=f"Orakel {pct(mo['beta'])}", delta_color="off", help="Mengenmäßiger Servicegrad: 1 − Fehlmenge / Nachfrage.")
mcols[3].metric("Lagerbestand", de(mt["stock"], 2), delta=f"Orakel {de(mo['stock'], 2)}", delta_color="off", help="Mittlerer Lagerbestand am Tagesende in Tagesbedarfen.")
mcols[4].metric("Prognosefehler", pct(a.wape[source]), help="Mittlerer absoluter Fehler der Schutzintervall-Prognose geteilt durch die mittlere Nachfrage (WAPE).")
h1, h2 = st.columns(2)
with h1:
    st.markdown("##### Mehrkosten gegen die Regel mit der wahren Verteilung")
    st.plotly_chart(build_regret_heat(a), width="stretch", key="regret_chart")
with h2:
    st.markdown("##### Erreichter Servicegrad α (Ziel: " + pct(settings.target, 0) + ")")
    st.plotly_chart(build_alpha_heat(a), width="stretch", key="alpha_chart")
regr = {k: 100 * (a.metrics[k]["cost"] / mo["cost"] - 1) for k in a.keys if k != "oracle"}
best_k = min(regr, key=regr.get)
none_k = (source, "none")
cells_ok = [k for k in regr if abs(a.metrics[k]["alpha"] - settings.target) <= 0.02 and k[1] != "none"]
if abs(mt["alpha"] - settings.target) <= 0.02:
    st.success(f"✅ Diese Kombination ({SHORT[source]}, {SHORT[rule]}) trifft den Zielservicegrad ({pct(mt['alpha'])} gegen {pct(settings.target, 0)}) und liegt bei {spct(regr[key])} Mehrkosten gegen die wahre Verteilung; die günstigste aller Kombinationen ist {SHORT[best_k[0]]} mit {SHORT[best_k[1]]} ({spct(regr[best_k])}).")
else:
    st.warning(f"⚠️ Diese Kombination ({SHORT[source]}, {SHORT[rule]}) verfehlt den Zielservicegrad: {pct(mt['alpha'])} statt {pct(settings.target, 0)}, bei {spct(regr[key])} Mehrkosten. Die günstigste Kombination ist {SHORT[best_k[0]]} mit {SHORT[best_k[1]]} ({spct(regr[best_k])}, α {pct(a.metrics[best_k]['alpha'])}).")
st.caption(
    f"Kosten in Tagesbedarfen je Tag und Depot; Lagerbestand h = 1, Rückstand p = {de(ratio, 1)} je Stück und Tag (kritisches Verhältnis {de(settings.target, 2)}). Ohne Sicherheitsbestand erreicht {SHORT[source]} nur α = {pct(a.metrics[none_k]['alpha'])} bei {spct(regr[none_k])} Mehrkosten. "
    f"**β ist bei diesem Ziel deutlich höher als α** ({pct(mo['beta'])} bei der wahren Verteilung): ein Ziel von {pct(settings.target, 0)} ohne Definition ist kein Ziel."
)
rows = [{"Regel": C.RULE_NAMES[r], "Kosten": de(a.metrics[(source, r)]["cost"], 3), "Mehrkosten gegen Orakel": spct(regr[(source, r)]), "α": pct(a.metrics[(source, r)]["alpha"]), "β": pct(a.metrics[(source, r)]["beta"]), "Lagerbestand": de(a.metrics[(source, r)]["stock"], 2)} for r in C.RULES]
rows.append({"Regel": "Orakel (wahre Verteilung)", "Kosten": de(mo["cost"], 3), "Mehrkosten gegen Orakel": "–", "α": pct(mo["alpha"]), "β": pct(mo["beta"]), "Lagerbestand": de(mo["stock"], 2)})
st.markdown(f"##### Alle Regeln für die Prognosequelle {SHORT[source]}")
st.dataframe(rows, hide_index=True)

st.markdown("---")

# --- Experimente ------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Was heißt \"95 % Servicegrad\"? Ziel und erreichter Servicegrad")
st.caption(f"{C.EXP_DEPOTS} Depots, Mittelwert der vier Prognosen, Wiederbeschaffungszeit 3 Tage; das Ziel α läuft von {C.TARGET_LEVELS[0]:.2f} bis {C.TARGET_LEVELS[-1]:.2f}. Gezeigt: der erreichte Servicegrad α je Regel und der mengenmäßige Servicegrad β der wahren Verteilung. "
           f"Mittel über {len(C.EXP_SEEDS)} feste Seeds (Fehlerbalken: Standardfehler). Dauer etwa eine halbe Minute.")
if st.button("Ziele durchrechnen", key="target_start"):
    st.session_state["target_on"] = True
if st.session_state.get("target_on"):
    rt = _target(C.TARGET_LEVELS, C.EXP_SEEDS)
    st.plotly_chart(build_target(rt), width="stretch", key="target_chart")
    src_k = rt[0]["source"]
    t95 = next((r for r in rt if abs(r["target"] - 0.95) < 1e-9), rt[len(rt) // 2])
    t0_ = rt[0]
    st.warning(
        f"**Befund:** Die **Faustregel** verfehlt das Ziel bei jeder Stufe: bei {pct(t95['target'], 0)} erreicht sie {pct(t95['by'][(src_k, 'rough')]['alpha'][0])}, bei {pct(t0_['target'], 0)} {pct(t0_['by'][(src_k, 'rough')]['alpha'][0])}. Log-normale ({pct(t95['by'][(src_k, 'lognormal')]['alpha'][0])}), empirische ({pct(t95['by'][(src_k, 'empirical')]['alpha'][0])}) "
        f"und konforme Regel ({pct(t95['by'][(src_k, 'conformal')]['alpha'][0])}) treffen es. **Und derselbe Bestand liefert einen mengenmäßigen Servicegrad β von {pct(t95['by']['oracle']['beta'][0])}** bei α {pct(t95['by']['oracle']['alpha'][0])}: „95 %“ heißt je nach Definition sehr Verschiedenes. Ein Ziel ohne Definition ist kein Ziel."
    )

st.markdown("---")

st.subheader("🔬 Wiederbeschaffungszeit: Wann wird die Unsicherheit teuer?")
st.caption(f"{C.EXP_DEPOTS} Depots, Mittelwert der vier Prognosen, Ziel 95 %; die Wiederbeschaffungszeit läuft über {', '.join(str(x) for x in C.LEAD_LEVELS)} Tage. Gezeigt: Mehrkosten gegen die wahre Verteilung und der erreichte Servicegrad. "
           f"Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer etwa eine Minute.")
if st.button("Wiederbeschaffungszeiten durchrechnen", key="lead_start"):
    st.session_state["lead_on"] = True
if st.session_state.get("lead_on"):
    rl = _lead(C.LEAD_LEVELS, C.EXP_SEEDS)
    l1, l2 = st.columns(2)
    with l1:
        st.plotly_chart(build_lead_regret(rl), width="stretch", key="lead_regret_chart")
    with l2:
        st.plotly_chart(build_lead_alpha(rl), width="stretch", key="lead_alpha_chart")
    src_k = rl[0]["source"]
    r0, r1 = rl[0], rl[-1]
    st.warning(
        f"**Befund:** Die Mehrkosten wachsen mit der Wiederbeschaffungszeit: bei L = {r0['lead']} liegt die empirische Regel {spct(r0['by'][(src_k, 'empirical')]['regret'][0])} über der wahren Verteilung, bei L = {r1['lead']} {spct(r1['by'][(src_k, 'empirical')]['regret'][0])}; die Faustregel {spct(r0['by'][(src_k, 'rough')]['regret'][0])} und {spct(r1['by'][(src_k, 'rough')]['regret'][0])}, ihr Servicegrad fällt von "
        f"{pct(r0['by'][(src_k, 'rough')]['alpha'][0])} auf {pct(r1['by'][(src_k, 'rough')]['alpha'][0])}. Die konforme Regel fällt bei langem Schutzintervall zurück ({pct(r1['by'][(src_k, 'conformal')]['alpha'][0])} bei L = {r1['lead']}); vermutlich, weil die Summenfehler überlappender Schutzintervalle stark verwandt sind und das Kalibrierfenster weniger unabhängige Werte hat, als es Ursprünge zählt (nicht isoliert)."
    )

st.markdown("---")

st.subheader("🔬 Prognosegüte und Anpassungsfähigkeit: Wer zahlt für eine schlechte Prognose?")
st.caption(f"{C.EXP_DEPOTS} Depots, Ziel 95 %, Wiederbeschaffungszeit 3 Tage; die Niveauschwankung wächst auf {', '.join(str(x).replace('.', ',') for x in C.SWING_LEVELS)}. Gezeigt: Mehrkosten gegen die wahre Verteilung für die Regression, das Boosting und den Mittelwert der vier, mit der empirischen und der konformen Regel. "
           f"Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer etwa zwei Minuten.")
if st.button("Niveauschwankung durchrechnen", key="swing_start"):
    st.session_state["swing_on"] = True
if st.session_state.get("swing_on"):
    rs = _swing(C.SWING_LEVELS, C.EXP_SEEDS)
    st.plotly_chart(build_swing(rs), width="stretch", key="swing_chart")
    s0, s1 = rs[0], rs[-1]
    st.warning(
        f"**Befund:** Ohne Schwankung ist die Regression die beste Quelle ({spct(s0['by'][('reg', 'empirical')]['regret'][0])} Mehrkosten mit der empirischen Regel, Prognosefehler {pct(s0['wape']['reg'][0])}). Bei Schwankung {de(s1['swing'], 2)} verfehlt sie mit der empirischen Regel, deren Fehlerquantil aus dem Training stammt, das Ziel "
        f"(α {pct(s1['by'][('reg', 'empirical')]['alpha'][0])}) und kostet {spct(s1['by'][('reg', 'empirical')]['regret'][0])}; die **konforme** Regel, die das Quantil aus den letzten Fehlern nachführt, hält α {pct(s1['by'][('reg', 'conformal')]['alpha'][0])} bei {spct(s1['by'][('reg', 'conformal')]['regret'][0])}. "
        f"Das Boosting und der Mittelwert der vier bleiben mit beiden Regeln nahe am Ziel ({spct(s1['by'][('gbm', 'empirical')]['regret'][0])} und {spct(s1['by'][('mean4', 'empirical')]['regret'][0])} mit der empirischen Regel). **Eine gute Prognose spart Bestand; eine mitlaufende Kalibrierung schützt vor der schlechten.**"
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ein Zielservicegrad steht für eine Definition** | α (Tage ohne Rückstand) und β (Mengenanteil) unterscheiden sich stark (95 % α ist rund 99 % β); ohne Definition ist das Ziel keines. | Ziel als Kostenverhältnis p/h oder als β angeben |
| **Der Bestellbestand ist das Quantil der Schutzintervall-Nachfrage** | Gilt für Order-up-to mit Backorders und täglicher Überprüfung. Mit Bestellfixkosten braucht man (s, S); mit verlorenen Verkäufen ändert sich die Kostenrechnung. | (s, S)-Politik, Lost-Sales-Modell |
| **Die Summenfehler sind austauschbar** | Die konforme Regel gewinnt Abdeckung nur, wenn die letzten Fehler den kommenden gleichen; überlappende Schutzintervalle machen die Fehler abhängig, das Fenster zählt weniger Werte, als es Ursprünge hat. | Größeres Fenster, adaptive Kalibrierung (Stück 7) |
| **Die Historie beschreibt die Zukunft** | Die empirische und die log-normale Regel schätzen die Streuung aus dem Training; ändert sie sich, verfehlen sie das Ziel. | Konforme Kalibrierung |
| **Die Prognose ist die Nachfrage** | Beobachtet wird hier die tatsächliche Nachfrage; bei Fehlmengen wäre sie nur zensiert sichtbar. | Zensierte Nachfrage schätzen |
| **Ein Artikel je Depot, ohne Kopplung** | Keine Mehrstufigkeit, kein Risk Pooling, keine Mindestbestellmenge, keine Lieferzeitunsicherheit. | Mehrstufige Bestandsmodelle |
| **Erzeugtes Portfolio, drei Seeds** | Das Vehikel erzeugt genau die Muster (multiplikativ, log-normal, AR(1)-Schwankungen); echte Portfolios sind unordentlicher. Die Zahlen gelten für diese Portfolios. | – |
"""
)
st.caption("Die Linie: Naive Prognose → Exponentielle Glättung → ARIMA → Dynamische Regression, dazu Croston, Boosting, Prognoseintervalle, Hierarchie, Kombination, **Bestand** und ein vortrainiertes Netz (alle Stücke gebaut).")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Schutzintervall und Fehler.** $m = L + 1$, Ursprung $t$ (bekannt: Tage $< t$). $F_m = \sum_{j=1}^m \hat y_{t+j-1}$, $D_m = \sum_{j=0}^{m-1} y_{t+j}$, Score $s = \log\frac{D_m + 1}{F_m + 1}$. Ziel $\tau = p/(p+h)$, $z = \Phi^{-1}(\tau)$.

**Regeln.** Ohne: $S = F_m$. **Faustregel:** $S = F_m + z\,\sigma_1 \sqrt m$ mit der Standardabweichung $\sigma_1$ der Ein-Tages-Fehler $y - \hat y_1$ auf den Trainingstagen. **Log-normal:** $S = (F_m + 1)\,e^{z\sigma_m} - 1$ mit der Standardabweichung $\sigma_m$ der Scores (Ursprünge, deren Zieltage vor Tag 610 liegen). **Empirisch:** $S = (F_m + 1)\,e^{q_\tau} - 1$ mit dem $\tau$-Quantil der Trainings-Scores.
**Konform:** dasselbe mit den Scores der Ursprünge $o \in (t - m - W,\ t - m]$, $o \ge 610$, und dem Rang $\lceil (k+1)\tau \rceil$ unter $k$ Scores. **Orakel:** $S$ ist das $\tau$-Quantil von $D_m = \sum y_{t+j}$ mit unabhängigen log-normalen Tagen (Erwartungswert $\mu_j$, Streuung $\sigma$); die Summe wird nach **Fenton/Wilkinson** durch eine Log-Normalverteilung mit gleichem Mittel $M = \sum \mu_j$ und gleicher Varianz $V = \sum \mu_j^2 (e^{\sigma^2} - 1)$ angenähert: $S = \exp\big(\ln M - s^2/2 + s z\big)$, $s^2 = \ln(1 + V/M^2)$.

**Fortschreibung.** Nettobestand $NI$, Bestandsposition $IP_t = NI_t + \text{unterwegs}$, Bestellung $q_t = \max(S_t - IP_t, 0)$, Eingang nach $L$ Tagen, Nachfrage $y_t$: $NI \leftarrow NI - y_t$. Tageskosten $h \max(NI, 0) + p \max(-NI, 0)$ mit $h = 1$, $p = \tau/(1-\tau)$. Bei konstantem $S$ ist $NI = S - D_m$ am Ende von Tag $t + L$: Servicegrad $P(D_m \le S)$ und Newsvendor-Kosten.
**Kennzahlen.** Kosten je Tag geteilt durch die mittlere Tagesnachfrage des Depots, über die Depots gemittelt. $\alpha$ = Anteil der Tage mit Rückstand 0 am Tagesende; $\beta = 1 - \sum \text{Fehlmenge} / \sum y$ mit Fehlmenge $= (y_t - \max(NI_t, 0))^+$; Lagerbestand in Tagesbedarfen. Mehrkosten $= 100\,(\text{Kosten}/\text{Kosten}_{\text{Orakel}} - 1)$.

Implementiert in `stk_rules.py` (die Regeln), `stk_inventory.py` (die Fortschreibung), `stk_members.py`, `stk_baselines.py`, `stk_ets.py`, `stk_tree.py`, `stk_gbm.py`, `stk_features.py` (die Prognosequellen aus den Vorgängern), `stk_scenario.py` (das Portfolio), `stk_evaluation.py` (Analyse, drei Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Zeitreihen-Prognose: von Naiv bis Vortraining](https://sebastianhanisch.net/konzepte-zeitreihen-prognose.html)."
)
