# 📦 Von der Prognose zum Bestand

**[→ Demo live ausprobieren](https://sebastianhanisch-forecast-inventory-demo.streamlit.app/)**

Zehntes Stück der **Zeitreihen-Prognose-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning. Der **Zusammenfluss** der Prognoseverfahren mit der Bestandsplanung: Abnehmer der Quantile aus den [Prognoseintervallen](https://github.com/sebastian-hanisch/forecast-interval-demo) und der Prognosen aus [Boosting](https://github.com/sebastian-hanisch/boosting-forecast-demo) und [Kombination](https://github.com/sebastian-hanisch/forecast-combination-demo).
Geplant ist ein weiteres Stück (ein vortrainiertes Netz; noch nicht gebaut).

Eine Prognose ist kein Bestand. Wer bestellt, muss die Nachfrage der **Wiederbeschaffungszeit** decken – und ein **Sicherheitsbestand** fängt ab, was die Prognose verfehlt. Wie groß der sein muss, hängt an der **Unsicherheit der Prognose**, nicht an ihrem Wert. Die Demo nimmt die Prognosen der Vorgänger (Wochenmittel, Holt-Winters, Regression, Boosting und ihren Mittelwert) auf einem **Portfolio von Depots**
und bildet vier **Bestandsregeln** (Faustregel, log-normal, empirisch, konform) gegen eine Regel mit der **wahren Verteilung**. Eine **Bestandsfortschreibung** über das Testjahr (tägliche Überprüfung, Wiederbeschaffungszeit, nachgelieferte Fehlmengen) misst **Kosten und Servicegrad**. Alle Daten sind erzeugt, die Rechnung ist in numpy geschrieben (`scipy` nur als Gegenprobe im Test).

**Bezug zu OR:** die Bestandspolitik (Base-Stock, Newsvendor-Quantil) ist der Abnehmer der Prognose; die Vorab-Messreihe der Bestandsseite (Definitionen des Servicegrads, Schätzfehler kurzer Historien) ergänzt sie – hier ist der Hook die **Unsicherheit der Prognose**, dort die **Historie eines einzelnen Artikels**.

## Warum dieses Problem – und was sich gegenüber dem Plan geändert hat

Der Plan der Linie erwartete: "Bessere Prognosen senken den Bestand." Das stimmt – aber die Messung zeigt, dass **die Regel für den Sicherheitsbestand mehr ausmacht als das Prognoseverfahren**, und dass die Mehrkosten selbst der besten Kombination hoch bleiben.

1. **Ohne Sicherheitsbestand gibt es kein Servicegrad-Ziel.** Der Bestellbestand gleich der Prognose erreicht α = 54,7 % (Mittelwert der vier, Ziel 95 %) bei +341 % Kosten gegen die wahre Verteilung; bei Ziel 99 % sind es +1508 %.
2. **Die Faustregel $F + z\sigma\sqrt{L+1}$ verfehlt das Ziel bei jeder Stufe.** Bei Ziel 95 % erreicht sie α = 88,5 %, bei 80 % 76,1 %, bei 99 % 93,9 %: sie nimmt die Streuung der Ein-Tages-Fehler und skaliert sie mit $\sqrt m$ (vermutlich unterschätzt das die Summenfehler, weil die Tagesfehler eines Schutzintervalls positiv korreliert und die Verteilung schief ist; nicht isoliert). Sie kostet im Standardfall +66 %.
3. **Log-normal und empirisch treffen das Ziel.** Beide messen die Streuung der **Summenfehler** über das Schutzintervall (α 93,7 und 94,1 % bei Ziel 95 %; +43 und +44 %). Die konforme Regel liegt bei α 95,0 % (+42 %).
4. **Selbst die beste Kombination kostet +33 % gegen die wahre Verteilung.** Das Boosting mit der empirischen Regel ist im Standardfall die günstigste Kombination (+33 %, α 94,5 %); die Mehrkosten sind der Preis des Prognosefehlers (WAPE der Schutzintervall-Prognose 7,9 %) und wachsen mit der Wiederbeschaffungszeit (+16 % bei L = 0, +94 % bei L = 13, empirische Regel).
5. **Eine mitlaufende Kalibrierung schützt vor der schlechten Prognose.** Die statische Regression verliert bei wachsender Niveauschwankung: mit der empirischen Regel (Quantil aus dem Training) sinkt α auf 86,2 % im Standardfall (+222 %) und auf 81,6 % bei Schwankung 0,12 (+578 %); die konforme Regel, die das Quantil aus den letzten Fehlern nachführt, hält 92,9 % (+53 %) beziehungsweise 90,3 % (+137 %). Das Boosting kommt mit beiden Regeln nahe an das Ziel.
6. **"95 % Servicegrad" ist ohne Definition kein Ziel.** Bei α = 95 % ist der mengenmäßige Servicegrad β der wahren Verteilung 99,2 %; bei α = 80 % ist β = 96,3 %. Die beiden Definitionen liegen so weit auseinander, dass ein Ziel ohne Angabe der Definition nichts festlegt (die Vorab-Messreihe der Bestandsseite zeigt dasselbe für einen einzelnen Artikel).

## Modell

- **Das Portfolio** (`stk_scenario.py`): 1 095 Tage Tagesnachfrage je Depot wie in den Vorgängern (multiplikativ: Niveau, Trend, Wochen- und Jahresmuster, Feiertage, Aktionen, log-normales Rauschen, eigene Parameter je Depot) mit depoteigener **Niveauschwankung** (mittelwertrückkehrender Zufallsgang im Log, $\varphi = 0{,}98$) wie im Kombinations-Stück.
- **Die Prognosequellen** (`stk_members.py`): Wochenmittel (Stück 1), Holt-Winters multiplikativ (Stück 2), Regression im Log auf Kalender und Aktionsplan (Stück 4), globales Boosting (Stück 6) und der Mittelwert der vier (Stück 9). Parameter aus den Tagen vor Tag 610; die Tage 610–729 kalibrieren die konforme Regel, das Testjahr beginnt bei Tag 730.
- **Das Schutzintervall** (`stk_rules.py`): Bestellt wird jeden Tag, die Ware kommt nach $L$ Tagen; die Bestellung deckt die $m = L + 1$ Tage $t, \dots, t + L$. $F_m$ ist die Summe der Punktprognosen der Horizonte $1..m$, $D_m$ die Nachfrage, der Fehler $s = \log\frac{D_m + 1}{F_m + 1}$. Ziel $\tau = p/(p+h)$ (Fehlmenge $p$, Lagerung $h = 1$ je Stück und Tag).
- **Fünf Regeln:** ohne Sicherheitsbestand ($S = F_m$); **Faustregel** $S = F_m + z\sigma_1\sqrt m$ ($\sigma_1$ = Streuung der Ein-Tages-Fehler im Training); **log-normal** $S = (F_m+1)e^{z\sigma_m} - 1$ ($\sigma_m$ = Streuung der Summen-Scores im Training); **empirisch** (τ-Quantil der Trainings-Scores); **konform** (Quantil der zuletzt realisierten Scores der letzten $W$ Ursprünge mit der Korrektur $\lceil (k+1)\tau\rceil$, Stück 7);
  **Orakel**: das wahre τ-Quantil der Nachfrage im Schutzintervall (Summe unabhängiger log-normaler Tage mit bekanntem Erwartungswert und bekannter Streuung, nach Fenton/Wilkinson durch eine Log-Normalverteilung angenähert; für $m = 1$ exakt).
- **Die Fortschreibung** (`stk_inventory.py`): tägliche Überprüfung, Order-up-to-Politik mit zeitvariablem $S_t$, Wiederbeschaffungszeit $L$, nachgelieferte Fehlmengen. Lagerbestand kostet $h = 1$, Rückstand $p = \tau/(1-\tau)$ je Stück und Tag. Kennzahlen je Depot (dann gemittelt): **Kosten** in Tagesbedarfen je Tag, **α** (Anteil der Tage ohne Rückstand), **β** (1 − Fehlmenge/Nachfrage), **Lagerbestand**; **Mehrkosten** gegen die Orakel-Regel.

## Methodik

- **Handrechnungen:** die Fortschreibung für $L = 0$ (Bestände 6, 0, 7, Bestellungen 0, 4, 12, Kosten, α = 2/3, β = 1 − 2/19) und für $L = 2$ (Bestellungen 0, 3, 3, 3, 3); Regeln (Faustregel, log-normal, empirisch) auf vier Zahlen; die Fenster- und Rangregel der konformen Regel.
- **Gegenprobe:** die Fortschreibung bei konstantem $S$ gegen den **Newsvendor** (Servicegrad $P(D_m \le S)$, Kosten $h E(S-D_m)^+ + pE(D_m-S)^+$ mit $D_m$ als Summe von $m$ Tagen; 200 000 Tage); die Bilanz Nettobestand = Anfangsbestand + Eingänge − Nachfrage; die Orakel-Regel gegen Monte-Carlo-Quantile (400 000 Ziehungen; 1 % Toleranz); `scipy` für die Normalquantile; die konforme Regel gegen eine **unabhängige Schleife**.
- **Eigenschaften:** am Ende von Tag $t + L$ ist der Nettobestand $S - D_m$; Rückstand und Lagerbestand gibt es nie gleichzeitig; Bestellungen sind nie negativ; **kein Bestellbestand verwendet etwas aus der Zukunft seines Ursprungs** (Ist-Werte ab dem Ursprung überschrieben, Bestellbestände bis dahin unverändert); die konforme Regel verwendet nur realisierte Fehler.
- **Statistik:** die Experimente mitteln über **drei feste Seeds** (Fehlerbalken = Standardfehler), die Preset-Zeilen sind **Einzelportfolios** (Seed 3).
- **Literatur** (nicht nachgebaut): Silver/Pyke/Peterson, *Inventory Management and Production Planning and Scheduling* (Servicegrad-Definitionen, Base-Stock); Snyder/Shen, *Fundamentals of Supply Chain Theory* (Newsvendor, (s, S)); Fenton 1960 / Wilkinson (Summe log-normaler Variablen); Vovk et al., Gibbs/Candès (konforme Vorhersage).

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| **Standardfall** (Preset, 30 Depots, L = 3, Ziel 95 %, Seed 3) | Orakel: α 95,1 %, β 99,2 %, Kosten 0,704 Tagesbedarfe je Tag, Lagerbestand 0,55. Mittelwert der vier: ohne SB α 54,7 % (+341 %), Faustregel 88,5 % (+66 %), log-normal 93,7 % (+43 %), empirisch 94,1 % (+44 %), **konform 95,0 % (+42 %)**. Günstigste Kombination: **Boosting mit empirischer Regel, α 94,5 %, +33 %**. | `test_standard_preset` |
| Prognosequellen im Standardfall | WAPE der Schutzintervall-Prognose: Boosting 7,9 %, Mittelwert 8,3 %, Holt-Winters 8,6 %, Wochenmittel 9,9 %, Regression 13,9 %. Mehrkosten mit der empirischen Regel: Regression **+222 %** (α 86,2 %), Wochenmittel +82 %, Holt-Winters +55 %; Regression konform +53 % (α 92,9 %). | `test_source_ranking_and_regression_failure_in_the_standard_case` |
| **Lieferzeit 0** (Preset, reiner Newsvendor) | Orakel α 95,3 %, β 99,6 %; Mittelwert der vier: ohne SB +268 %, Faustregel +36 % (α 92,3 %), empirisch +18 % (95,0 %), konform +17 % (95,6 %). | `test_lead_zero_preset` |
| **Lieferzeit 13** (Preset) | Mittelwert der vier: Faustregel +175 % (α 80,9 %), empirisch +107 % (92,9 %), **konform +105 % (90,8 %)**. | `test_long_lead_preset` |
| **Hohes Ziel 99 %** (Preset) | Faustregel +151 % (α 93,9 %), log-normal +51 % (98,2 %), empirisch +52 % (98,3 %), konform +59 % (97,9 %); ohne SB +1508 %. | `test_high_target_preset` |
| **Niedriges Ziel 80 %** (Preset) | Orakel α 80,6 %, β 96,3 %; Faustregel +38 % (α 76,1 %), empirisch +38 % (80,7 %), konform +32 % (79,9 %). | `test_low_target_preset` |
| **Regression bei Schwankung 0,12** (Preset) | Regression, empirisch: α 81,6 %, **+578 %**; konform α 90,3 %, +137 %; Boosting empirisch α 94,8 %, +53 %; WAPE Regression 24,0 %, Boosting 9,0 %. | `test_regression_under_swing_preset` |
| **Zielservicegrad** (Experiment, 20 Depots, 3 Seeds, Mittelwert der vier; Ziel 80 / 90 / 95 / 99 %) | Erreichtes α: Faustregel 76,0 / 84,2 / 89,2 / 94,7 %; log-normal 80,3 / 89,5 / 94,5 / 98,6 %; empirisch 81,1 / 90,1 / 94,8 / 98,5 %; konform 79,5 / 89,5 / 94,8 / 97,9 %. Orakel: β 96,3 / 98,3 / 99,2 / 99,9 %. Mehrkosten ohne SB: +79 / +176 / +357 / +1601 %. | `test_target_experiment` |
| **Wiederbeschaffungszeit** (Experiment; L = 0 / 3 / 7 / 13) | Mehrkosten gegen das Orakel: Faustregel +37 / +58 / +102 / +158 %; log-normal +15 / +37 / +62 / +90 %; empirisch +16 / +38 / +65 / +94 %; konform +17 / +39 / +69 / +99 %. α der Faustregel 92,1 / 89,2 / 85,6 / 82,0 %; α der konformen Regel bei L = 13: 91,3 %. Ohne SB: +278 … +504 %. | `test_lead_experiment` |
| **Niveauschwankung** (Experiment; 0 / 0,06 / 0,12) | WAPE Regression 6,9 / 12,5 / 21,3 %, Boosting … / 8,3 / 9,4 %, Mittelwert … / 8,3 / 10,3 %. Regression: empirisch α 95,7 / 87,5 / 82,9 %, +11 / +136 / +383 %; konform α 95,0 / 93,0 / 90,7 %, +11 / +51 / +126 %. Boosting empirisch +25 / +33 / +54 %; Mittelwert empirisch +22 / +38 / +74 %. | `test_swing_experiment` |

Die Preset-Zeilen sind **Einzelportfolios** (Seed 3); belastbar sind die Zeilen über drei Seeds.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ein Zielservicegrad steht für eine Definition** | α und β unterscheiden sich stark (95 % α ≈ 99 % β); ohne Definition ist das Ziel keines. | Ziel als Kostenverhältnis oder als β angeben |
| **Der Bestellbestand ist das Quantil der Schutzintervall-Nachfrage** | Gilt für Order-up-to mit Backorders und täglicher Überprüfung. Mit Bestellfixkosten braucht man (s, S); mit verlorenen Verkäufen ändert sich die Kostenrechnung. | (s, S)-Politik, Lost-Sales-Modell |
| **Die Summenfehler sind austauschbar** | Die konforme Regel gewinnt Abdeckung nur, wenn die letzten Fehler den kommenden gleichen; bei langem Schutzintervall fällt sie zurück (α 91,3 % bei L = 13), vermutlich, weil die Summenfehler überlappender Schutzintervalle stark verwandt sind (nicht isoliert). | Größeres Fenster, adaptive Kalibrierung |
| **Die Historie beschreibt die Zukunft** | Empirisch und log-normal schätzen die Streuung aus dem Training; ändert sie sich, verfehlen sie das Ziel. | Konforme Kalibrierung |
| **Die Prognose ist die Nachfrage** | Beobachtet wird hier die tatsächliche Nachfrage; bei Fehlmengen wäre sie nur zensiert sichtbar. | Zensierte Nachfrage schätzen |
| **Ein Artikel je Depot, ohne Kopplung** | Keine Mehrstufigkeit, kein Risk Pooling, keine Mindestbestellmenge, keine Lieferzeitunsicherheit, keine Preise. | Mehrstufige Bestandsmodelle |
| **Das Orakel ist eine Annäherung** | Die Summe der Tage wird nach Fenton/Wilkinson durch eine Log-Normalverteilung angenähert (auf 1 % gegen Monte Carlo geprüft); Rundung und Ausreißer sind nicht berücksichtigt. | – |
| **Erzeugtes Portfolio, drei Seeds** | Das Vehikel erzeugt genau die Muster (multiplikativ, log-normal, AR(1)-Schwankungen); echte Portfolios sind unordentlicher. Die Zahlen gelten für diese Portfolios. | – |

## Tests

Pytest-Suite (`pytest tests/ -v`, 54 Tests, rund vier Minuten wegen der Experimente): die Regeln von Hand und gegen unabhängige Schleifen und Monte Carlo (Orakel), die Fortschreibung von Hand und gegen den Newsvendor (Bilanz, Nettobestand = Stufe − Schutzintervall-Nachfrage), das Portfolio, die Prognosequellen, die Kennzahlen von Hand, kein Blick in die Zukunft, Preset- und Permalink-Klemmen,
AppTest-Rauchtests (Standard, jedes Preset, Depot- und Ausschnitts-Regler, Extremwerte, drei Experimente auf Abruf, keine unaufgelösten Platzhalter) und `test_claims.py` (jede Zahl aus diesem README und aus den Preset-Hinweisen mit Bändern und Rangfolgen).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `stk_constants.py` | Regler-Grenzen, Quellen, Regeln, Experiment-Seeds |
| `stk_presets.py` | Permalink/Presets-Mechanik, `PRESET_HELP` |
| `stk_scenario.py` | Das Portfolio (Depots, Niveauschwankungen) |
| `stk_members.py` | Die Prognosequellen über das Schutzintervall |
| `stk_baselines.py`, `stk_ets.py`, `stk_tree.py`, `stk_gbm.py`, `stk_features.py` | Wochenmittel, Holt-Winters, Regression, Boosting (aus den Vorgängern) |
| `stk_rules.py` | Die Bestandsregeln und das Orakel |
| `stk_inventory.py` | Die Bestandsfortschreibung |
| `stk_evaluation.py` | Analyse, Kennzahlen, drei Experimente |
| `stk_visualization.py` | Plotly-Abbildungen |

## Bewusst nicht umgesetzt

- (s, S)-Politik mit Bestellfixkosten, Lost Sales, Mindestbestellmengen, mehrere Stufen oder Lager (siehe Vorab-Messreihe der Bestandsseite).
- Prognose der Lieferzeit, Quantilprognosen aus dem Modell (Quantil-Boosting), abgestimmte hierarchische Prognosen als Quelle (Stück 8).
- Ein PDF-Export gehört nicht zur Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy (Gegenprobe im Test: scipy).
