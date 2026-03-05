import os

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

FUEL_BURN_RATE = 0.056            # seconds per lap lighter
FUEL_BURN_KG_PER_LAP = 1.6
FUEL_TIME_PER_KG = 0.035
FULL_FUEL_PENALTY = 4.0           # seconds added to quali pace for race-start fuel load
WARMUP_PENALTY = 1.2              # seconds penalty on first lap of a new stint (cold tires)
TEMP_CORRECTION_FACTOR = 0.003    # 0.3 % more degradation per degree C hotter

MIN_STINT_LAPS = 5
SOFT_MAX_STINT = 20
MEDIUM_MAX_STINT = 35
HARD_MAX_STINT = None             # no upper limit — capped by race distance

STINT_DISCRETIZATION = 2          # generate stint lengths in steps of 2 laps
MAX_STOPS = 2                     # default; set to 3 for extreme-deg circuits

SENSITIVITY_RANGE = 0.15          # +/- 15 %

MIN_LONG_RUN_SOFT = 5
MIN_LONG_RUN_MED_HARD = 8

TRAFFIC_THRESHOLD = 1.5           # seconds gap to car ahead to count as "in traffic"

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

COMPOUND_MAX_STINT = {
    "SOFT": SOFT_MAX_STINT,
    "MEDIUM": MEDIUM_MAX_STINT,
    "HARD": HARD_MAX_STINT,
}

COMPOUND_MIN_LONG_RUN = {
    "SOFT": MIN_LONG_RUN_SOFT,
    "MEDIUM": MIN_LONG_RUN_MED_HARD,
    "HARD": MIN_LONG_RUN_MED_HARD,
}
