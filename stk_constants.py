"""Konstanten der Bestands-Demo: Vehikel "Tagesbedarf mehrerer Depots" (Stück 10 der Zeitreihen-Prognose-Linie), Prognosequellen, Bestandsregeln, Regler, Experimente."""

EPS = 1e-9
SEED_MAX = 999999

N_DAYS = 1095
FIRST_TEST = 730
FIRST_ORIGIN = 91
FIT_END = 610                 # die Prognoseverfahren schätzen ihre Parameter auf den Tagen vor FIT_END; die Tage danach sind das Kalibrierfenster der konformen Regel
CAL_MAX = FIRST_TEST - FIT_END
LEVEL = 100.0
WEEKDAYS = ("Mo", "Di", "Mi", "Do", "Fr", "Sa", "So")
WEEKLY_PATTERN = (1.10, 1.05, 1.00, 1.05, 1.20, 0.55, 0.35)
HOLIDAY_DOY = (0, 89, 92, 120, 134, 143, 275, 358, 359, 360)
HOLIDAY_DROP = 0.5
HOLIDAY_REBOUND = 0.15
PROMO_LENGTH = 7
PROMO_PER_YEAR = 3
SEASON_PERIOD = 7
AR_PHI = 0.98
GBM_ROUNDS = 80

# --- Glättung aus Stück 2 ------------------------------------------------------------------------------------------------------------------------------
ETS_MODELS = {"hw_mult": ("add", "mul")}
INIT_DAYS = 28
INIT_WEEKS = 8
FIT_STAGE1 = 3000
FIT_TOP = 6
FIT_ROUNDS = 6
FIT_PER_START = 40
FIT_SEED = 20240924
PHI_MIN, PHI_MAX = 0.80, 0.98

# --- Prognosequellen (Reihenfolge = Anzeige) -------------------------------------------------------------------------------------------------------------
MEMBERS = ("wm", "hw", "reg", "gbm")
SOURCES = ("wm", "hw", "reg", "gbm", "mean4")
SOURCE_NAMES = {"wm": "Wochenmittel (Stück 1)", "hw": "Holt-Winters (Stück 2)", "reg": "Regression (Stück 4)", "gbm": "Boosting, global (Stück 6)", "mean4": "Mittelwert der vier (Stück 9)"}
SOURCE_SHORT = {"wm": "Wochenmittel", "hw": "Holt-Winters", "reg": "Regression", "gbm": "Boosting", "mean4": "Mittelwert der vier"}

# --- Bestandsregeln: aus der Prognose der Nachfrage im Schutzintervall wird der Bestellbestand S (Reihenfolge = Anzeige) -----------------------------------
RULES = ("none", "rough", "lognormal", "empirical", "conformal")
RULE_NAMES = {"none": "Ohne Sicherheitsbestand (S = Prognose)", "rough": "Faustregel: Prognose + z·σ·√(L+1)", "lognormal": "Log-normal (Streuung der Summenfehler)", "empirical": "Empirisch (Quantil der Trainingsfehler)",
              "conformal": "Konform (Quantil der zuletzt realisierten Fehler)"}
RULE_SHORT = {"none": "Ohne SB", "rough": "Faustregel", "lognormal": "Log-normal", "empirical": "Empirisch", "conformal": "Konform"}

# --- Regler und Voreinstellungen ------------------------------------------------------------------------------------------------------------------------
DEPOTS_MIN, DEPOTS_MAX, DEPOTS_STEP, DEFAULT_DEPOTS = 10, 100, 10, 30
NOISE_MIN, NOISE_MAX, NOISE_STEP, DEFAULT_NOISE = 0.04, 0.4, 0.02, 0.14
TREND_MIN, TREND_MAX, TREND_STEP, DEFAULT_TREND = -20, 40, 5, 10
SWING_MIN, SWING_MAX, SWING_STEP, DEFAULT_SWING = 0.0, 0.2, 0.02, 0.06
LEAD_MIN, LEAD_MAX, DEFAULT_LEAD = 0, 13, 3
TARGET_MIN, TARGET_MAX, TARGET_STEP, DEFAULT_TARGET = 0.80, 0.99, 0.01, 0.95
WINDOW_MIN, WINDOW_MAX, WINDOW_STEP, DEFAULT_WINDOW = 30, CAL_MAX, 15, 90

# --- Experimente (feste Seeds) ---------------------------------------------------------------------------------------------------------------------
EXP_SEEDS = tuple(range(3))
EXP_DEPOTS = 20
TARGET_LEVELS = (0.80, 0.90, 0.95, 0.99)
LEAD_LEVELS = (0, 3, 7, 13)
SWING_LEVELS = (0.0, 0.06, 0.12)
