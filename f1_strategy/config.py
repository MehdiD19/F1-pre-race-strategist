import os

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

# ---------------------------------------------------------------------------
# Fuel & car constants
# ---------------------------------------------------------------------------

FUEL_BURN_RATE      = 0.056   # seconds saved per lap as fuel burns off
FUEL_BURN_KG_PER_LAP = 1.6
FUEL_TIME_PER_KG    = 0.035

# Default fuel penalty (quali → race pace gap) for unknown circuits.
# Investigation showed Hungary 2023 actual gap was ~8.1 s; 4.0 was the old
# (wrong) value that only captured fuel weight and ignored race-mode delta.
FULL_FUEL_PENALTY   = 7.0

# Per circuit-type fuel penalty.  The gap between a driver's best qualifying
# lap and their early-race pace is NOT just fuel load — it also includes engine
# mapping conservation, cold-tyre lap 1 effects, and traffic.  Technical and
# street circuits have more structural traffic/conservation overhead.
CIRCUIT_TYPE_FUEL_PENALTY: dict[str, float] = {
    "street":    9.0,   # Monaco, Baku, Singapore, Melbourne, Las Vegas
    "technical": 8.0,   # Hungary, Barcelona, Abu Dhabi, Zandvoort, Mexico
    "mixed":     7.0,   # Silverstone, Suzuka, Spa, Monza, Austria (default)
    "power":     6.5,   # Bahrain, Saudi (clean air, open track)
}

# GP name substring → circuit type.  Matching is case-insensitive substring.
# If no key matches, the "mixed" penalty is used via get_fuel_penalty().
CIRCUIT_CLASSIFICATION: dict[str, str] = {
    # Street
    "Monaco":       "street",
    "Azerbaijan":   "street",
    "Singapore":    "street",
    "Australian":   "street",
    "Las Vegas":    "street",
    "Miami":        "street",
    # Technical
    "Hungarian":    "technical",
    "Spanish":      "technical",
    "Abu Dhabi":    "technical",
    "Dutch":        "technical",
    "Mexico":       "technical",
    "Brazilian":    "technical",
    "São Paulo":    "technical",
    # Power
    "Bahrain":      "power",
    "Saudi":        "power",
    # Mixed (explicit, also serves as documentation; catch-all in function)
    "British":      "mixed",
    "Japanese":     "mixed",
    "Italian":      "mixed",
    "Belgian":      "mixed",
    "Austrian":     "mixed",
    "Emilia":       "mixed",
    "Canadian":     "mixed",
    "United States": "mixed",
    "Qatar":        "mixed",
}


def get_fuel_penalty(gp_name: str) -> float:
    """Return the calibrated quali→race fuel penalty for a GP name.

    Does a case-insensitive substring match against CIRCUIT_CLASSIFICATION.
    Falls back to the "mixed" penalty if the circuit is unrecognised.
    """
    gp_lower = gp_name.lower()
    for key, circuit_type in CIRCUIT_CLASSIFICATION.items():
        if key.lower() in gp_lower:
            return CIRCUIT_TYPE_FUEL_PENALTY[circuit_type]
    return CIRCUIT_TYPE_FUEL_PENALTY["mixed"]


# ---------------------------------------------------------------------------
# Warmup & pit constants
# ---------------------------------------------------------------------------

# Compound-dependent warmup penalty profile (seconds lost per lap after a pit stop).
# Applied to the first N laps of each non-opening stint.
# Softer compounds heat their rubber quickly; Hards bleed warmup across 3 laps.
WARMUP_PROFILE: dict[str, list[float]] = {
    "SOFT":   [1.0],              # ~1.0 s total over 1 lap
    "MEDIUM": [1.5, 0.5],         # ~2.0 s total over 2 laps
    "HARD":   [2.0, 1.0, 0.5],    # ~3.5 s total over 3 laps
}

# ---------------------------------------------------------------------------
# Temperature correction
# ---------------------------------------------------------------------------

# Per-compound temperature sensitivity factor.
# Softer compounds are significantly more heat-sensitive than harder ones.
# Applied as:  correction_factor = 1 + TEMP_CORRECTION_FACTOR[compound] * delta_T
# where delta_T = race_track_temp - fp2_track_temp (positive = race is hotter).
TEMP_CORRECTION_FACTOR: dict[str, float] = {
    "SOFT":   0.005,   # 0.5 % more degradation per °C — most sensitive
    "MEDIUM": 0.003,   # 0.3 % — moderate sensitivity
    "HARD":   0.001,   # 0.1 % — least sensitive to temperature
}

# ---------------------------------------------------------------------------
# Degradation model fitting thresholds
# ---------------------------------------------------------------------------

# Minimum data points required to attempt any polynomial fit.
# Below this the compound uses COMPOUND_FALLBACK_DEG.
MIN_SAMPLES_FOR_LINEAR    = 10

# Minimum data points required to fit a degree-2 polynomial.
# Below this (but >= MIN_SAMPLES_FOR_LINEAR) a linear fit is used.
# A quadratic on too few points is unreliable and can invert (key bug fix).
MIN_SAMPLES_FOR_QUADRATIC = 20

# IQR multiplier for outlier rejection in practice long-run data.
# Laps beyond median ± OUTLIER_IQR_FACTOR × IQR are removed before fitting.
OUTLIER_IQR_FACTOR = 2.5

# Conservative fallback degradation curves when practice data is insufficient.
# Format: [a, b, c] for np.polyval → a*x² + b*x + c   (x = stint lap number)
# Values represent seconds of additional time lost vs. a fresh tyre.
# Intentionally conservative (low) — it is better to underestimate deg and
# risk a slightly wrong stop window than to generate phantom time savings.
COMPOUND_FALLBACK_DEG: dict[str, list[float]] = {
    "SOFT":   [0.000, 0.20, 0.0],   # 0.20 s/lap linear — ~3 s at lap 15
    "MEDIUM": [0.003, 0.08, 0.0],   # slightly accelerating — ~3.5 s at lap 30
    "HARD":   [0.000, 0.05, 0.0],   # 0.05 s/lap linear — ~2.5 s at lap 50
}

# ---------------------------------------------------------------------------
# Historical prior settings (Phase 2)
# ---------------------------------------------------------------------------

# Number of prior seasons to load for the historical degradation prior.
HISTORICAL_NUM_YEARS = 3

# Earliest season to include in the prior.  2022 marks the ground-effect
# regulation change; pre-2022 downforce levels produce different tyre loading
# so those seasons are excluded by default.
HISTORICAL_MIN_YEAR = 2022

# Minimum laps (after stripping pit-in/out) for a race stint to qualify for
# the historical prior.  Race stints are longer and cleaner than practice runs.
HISTORICAL_MIN_RACE_STINT_LAPS = 10

# Denominator for the practice data-confidence weight in blending.
# practice_weight = min(n / BLEND_N_SATURATION, 1.0) * min(r2 / BLEND_R2_SATURATION, 1.0)
# At n = BLEND_N_SATURATION and R² = BLEND_R2_SATURATION, practice data is trusted fully.
BLEND_N_SATURATION  = 40.0   # sample size at which practice gets full weight
BLEND_R2_SATURATION = 0.40   # R² at which practice gets full weight

# ---------------------------------------------------------------------------
# Stint generation & strategy constraints
# ---------------------------------------------------------------------------

MIN_STINT_LAPS   = 5
SOFT_MAX_STINT   = 25
MEDIUM_MAX_STINT = 38
HARD_MAX_STINT   = None        # no upper limit — capped by race distance

STINT_DISCRETIZATION = 2       # generate stint lengths in steps of 2 laps
MAX_STOPS            = 2       # default; set to 3 for extreme-deg circuits
SENSITIVITY_RANGE    = 0.15    # +/- 15 %

# ---------------------------------------------------------------------------
# Long-run extraction thresholds
# ---------------------------------------------------------------------------

MIN_LONG_RUN_SOFT     = 5
MIN_LONG_RUN_MED_HARD = 8

TRAFFIC_THRESHOLD = 1.5        # seconds gap to car ahead to count as "in traffic"

# ---------------------------------------------------------------------------
# Compound lists & lookups
# ---------------------------------------------------------------------------

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

COMPOUND_MAX_STINT = {
    "SOFT":   SOFT_MAX_STINT,
    "MEDIUM": MEDIUM_MAX_STINT,
    "HARD":   HARD_MAX_STINT,
}

COMPOUND_MIN_LONG_RUN = {
    "SOFT":   MIN_LONG_RUN_SOFT,
    "MEDIUM": MIN_LONG_RUN_MED_HARD,
    "HARD":   MIN_LONG_RUN_MED_HARD,
}
