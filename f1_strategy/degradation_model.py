"""Phase 2 — Tire Degradation Modeling from Practice Data.

Extracts long-run stints from FP2 (fallback FP3), isolates pure tire
degradation by removing the fuel-burn benefit, fits a polynomial per
compound, and applies a temperature correction for race day.

Phase-1 fixes applied (2025-03):
  1. IQR-based outlier filtering before fitting — removes blown laps that
     would otherwise corrupt the entire compound curve (e.g. a 140 s lap
     on an 85 s circuit pulling the Soft curve into 500 s territory).
  2. Polynomial degree selection by sample size — degree-2 only when n ≥ 20;
     degree-1 (linear) for 10 ≤ n < 20; conservative fallback for n < 10.
     This eliminates the downward-parabola overfitting on sparse Hard data.
  3. Downward-parabola rejection — if a degree-2 fit produces a leading
     coefficient a < 0 (tire gets faster with age), the fit is discarded
     and re-fitted as degree-1. Physically, tyres cannot improve with age
     once past the initial warm-up phase (which is already stripped).
  4. Per-compound temperature correction — Soft is ~5× more heat-sensitive
     than Hard; using one global factor was scaling up an already-broken
     Soft curve and making Hard worse.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    COMPOUND_FALLBACK_DEG,
    COMPOUND_MIN_LONG_RUN,
    COMPOUNDS,
    FUEL_BURN_RATE,
    MIN_SAMPLES_FOR_LINEAR,
    MIN_SAMPLES_FOR_QUADRATIC,
    OUTLIER_IQR_FACTOR,
    TEMP_CORRECTION_FACTOR,
)
from data_collection import CircuitProfile

logger = logging.getLogger(__name__)


@dataclass
class DegradationCurve:
    compound: str
    coefficients: np.ndarray   # polynomial [a, b, c] or [a, b] → np.polyval
    r_squared: float
    sample_size: int
    is_fallback: bool = False  # True when built from defaults, not fitted data
    raw_data: Optional[pd.DataFrame] = field(default=None, repr=False)

    def predict(self, tire_age: int | np.ndarray) -> float | np.ndarray:
        """Return predicted seconds lost relative to a fresh tyre."""
        return np.polyval(self.coefficients, tire_age)


# ---------------------------------------------------------------------------
# Outlier filtering
# ---------------------------------------------------------------------------

def _filter_outliers_iqr(
    values: np.ndarray, factor: float = OUTLIER_IQR_FACTOR
) -> np.ndarray:
    """Return a boolean mask: True for values within [Q1 - f*IQR, Q3 + f*IQR].

    IQR is used instead of std-dev because the distributions of practice
    lap deltas are non-normal and std is itself distorted by the outliers
    we are trying to remove.
    """
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return np.ones(len(values), dtype=bool)
    return (values >= q1 - factor * iqr) & (values <= q3 + factor * iqr)


# ---------------------------------------------------------------------------
# Long-run extraction
# ---------------------------------------------------------------------------

def _extract_long_runs(
    session, compounds: List[str]
) -> Dict[str, pd.DataFrame]:
    """Pull clean long-run data from a practice session, keyed by compound.

    Returns a dict mapping compound name to a DataFrame with columns
    ``stint_lap`` (0-based lap within stint) and ``delta_seconds``
    (fuel-corrected time delta from stint start).

    Base time is the median of the first 3 laps of each stripped stint
    rather than strictly lap 0.  This makes the reference more robust to
    a single anomalously fast or slow exit lap.
    """
    laps = session.laps

    clean = laps.pick_accurate().pick_wo_box()
    try:
        clean = clean.pick_track_status("1", how="equals")
    except Exception:
        pass

    result: Dict[str, List[pd.DataFrame]] = {c: [] for c in compounds}

    drivers = clean["Driver"].unique()
    for driver in drivers:
        driver_laps = clean.pick_drivers(driver)
        for stint_num in driver_laps["Stint"].unique():
            stint = driver_laps[driver_laps["Stint"] == stint_num].copy()
            stint = stint.sort_values("LapNumber")

            if len(stint) < 3:
                continue

            # Strip first and last lap (pit entry/exit distortion)
            stint = stint.iloc[1:-1]

            compound = stint["Compound"].iloc[0]
            if compound not in compounds:
                continue

            min_laps = COMPOUND_MIN_LONG_RUN.get(compound, 5)
            if len(stint) < min_laps:
                continue

            times = stint["LapTime"].dt.total_seconds().values
            if np.any(np.isnan(times)):
                continue

            # Stabilised base: median of first min(3, n) laps avoids a single
            # anomalous out-lap skewing every subsequent delta in this stint.
            base_time = float(np.median(times[:min(3, len(times))]))
            stint_laps = np.arange(len(times))

            raw_delta = times - base_time

            # Add back fuel-burn benefit to isolate pure tyre degradation
            corrected_delta = raw_delta + stint_laps * FUEL_BURN_RATE

            df = pd.DataFrame({
                "stint_lap":    stint_laps,
                "delta_seconds": corrected_delta,
            })
            result[compound].append(df)

    # Pool stints and apply cross-compound IQR outlier filtering
    merged: Dict[str, pd.DataFrame] = {}
    for compound in compounds:
        frames = result[compound]
        if not frames:
            continue
        pooled = pd.concat(frames, ignore_index=True)

        mask = _filter_outliers_iqr(pooled["delta_seconds"].values)
        n_removed = (~mask).sum()
        if n_removed:
            logger.info(
                "%s: removed %d outlier lap(s) before fitting (IQR factor %.1f)",
                compound, n_removed, OUTLIER_IQR_FACTOR,
            )
        merged[compound] = pooled[mask].reset_index(drop=True)

    return merged


# ---------------------------------------------------------------------------
# Polynomial fitting
# ---------------------------------------------------------------------------

def _fit_degradation(
    data: pd.DataFrame,
) -> Tuple[Optional[np.ndarray], float]:
    """Fit the best-supported polynomial to (stint_lap, delta_seconds) data.

    Degree selection:
      n < MIN_SAMPLES_FOR_LINEAR    → returns (None, 0.0) — use fallback
      n < MIN_SAMPLES_FOR_QUADRATIC → degree-1 (linear, no overfitting risk)
      n ≥ MIN_SAMPLES_FOR_QUADRATIC → degree-2, but if leading coeff a < 0
                                       (downward parabola = tyre gets faster),
                                       fall back to degree-1.

    Returns (coefficients_array, r_squared).  coefficients_array is None when
    there is insufficient data.
    """
    n = len(data)

    if n < MIN_SAMPLES_FOR_LINEAR:
        logger.warning(
            "Only %d data points — below minimum %d for any fit; using fallback",
            n, MIN_SAMPLES_FOR_LINEAR,
        )
        return None, 0.0

    x = data["stint_lap"].values.astype(float)
    y = data["delta_seconds"].values.astype(float)

    if n >= MIN_SAMPLES_FOR_QUADRATIC:
        coeffs = np.polyfit(x, y, deg=2)
        if coeffs[0] < 0:
            # Downward-opening parabola: the model would predict tyres getting
            # faster over long stints — physically wrong and the root cause of
            # the inverted Hard curve bug.  Reject and fall back to linear.
            logger.warning(
                "Deg-2 fit has a=%.5f < 0 (downward parabola) — "
                "falling back to linear fit",
                coeffs[0],
            )
            coeffs = np.polyfit(x, y, deg=1)
    else:
        coeffs = np.polyfit(x, y, deg=1)

    y_pred  = np.polyval(coeffs, x)
    ss_res  = np.sum((y - y_pred) ** 2)
    ss_tot  = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return coeffs, r_squared


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_degradation_curves(
    profile: CircuitProfile,
    compounds: Optional[List[str]] = None,
) -> Dict[str, DegradationCurve]:
    """Build per-compound degradation curves from practice data.

    Uses FP2 as primary source; falls back to FP3 for any compound that
    lacks sufficient long-run data.  When even combined practice data is
    insufficient, a conservative physical-default curve is substituted so
    that the strategy engine always has something to score against.
    """
    if compounds is None:
        compounds = list(COMPOUNDS)

    fp2_data = _extract_long_runs(profile.fp2_session, compounds)

    missing = [c for c in compounds if c not in fp2_data]
    fp3_data: Dict[str, pd.DataFrame] = {}
    if missing:
        logger.info("Falling back to FP3 for compounds: %s", missing)
        fp3_data = _extract_long_runs(profile.fp3_session, missing)

    # Merge: FP2 takes precedence; FP3 fills gaps
    combined = {**fp2_data, **fp3_data}

    temp_delta = profile.race_track_temp - profile.fp2_track_temp
    logger.info("Temperature delta FP2→Race: %+.1f °C", temp_delta)

    curves: Dict[str, DegradationCurve] = {}
    for compound in compounds:
        data = combined.get(compound)

        # Per-compound temperature correction factor (Fix 4)
        compound_tc = TEMP_CORRECTION_FACTOR.get(compound, 0.003)
        correction  = 1.0 + compound_tc * temp_delta

        if data is None or data.empty:
            # No practice data at all — use conservative physical default
            fallback_coeffs = np.array(COMPOUND_FALLBACK_DEG[compound])
            logger.warning(
                "%s: no long-run data → using fallback curve (%.3f s/lap linear)",
                compound, fallback_coeffs[-2],
            )
            curves[compound] = DegradationCurve(
                compound=compound,
                coefficients=fallback_coeffs,
                r_squared=0.0,
                sample_size=0,
                is_fallback=True,
            )
            continue

        coeffs, r_sq = _fit_degradation(data)

        if coeffs is None:
            # Insufficient data even after FP3 fallback — use physical default
            fallback_coeffs = np.array(COMPOUND_FALLBACK_DEG[compound])
            logger.warning(
                "%s: n=%d — below minimum %d for fit → using fallback curve",
                compound, len(data), MIN_SAMPLES_FOR_LINEAR,
            )
            curves[compound] = DegradationCurve(
                compound=compound,
                coefficients=fallback_coeffs,
                r_squared=0.0,
                sample_size=len(data),
                is_fallback=True,
                raw_data=data,
            )
            continue

        # Apply per-compound temperature correction
        coeffs_corrected = coeffs * correction

        curves[compound] = DegradationCurve(
            compound=compound,
            coefficients=coeffs_corrected,
            r_squared=r_sq,
            sample_size=len(data),
            is_fallback=False,
            raw_data=data,
        )
        logger.info(
            "%s: coeffs=%s  R²=%.3f  n=%d  temp_correction=×%.4f  fallback=%s",
            compound,
            np.array2string(coeffs_corrected, precision=5),
            r_sq,
            len(data),
            correction,
            False,
        )

    return curves
