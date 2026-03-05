"""Phase 2 — Tire Degradation Modeling from Practice Data.

Extracts long-run stints from FP2 (fallback FP3), isolates pure tire
degradation by removing the fuel-burn benefit, fits a polynomial per
compound, and applies a temperature correction for race day.

Phase-1 fixes applied (2025-03):
  1. IQR-based outlier filtering before fitting.
  2. Polynomial degree selection by sample size (degree-2 only when n ≥ 20).
  3. Downward-parabola rejection — if a degree-2 fit produces a < 0, the
     fit is rejected and re-done as degree-1.
  4. Per-compound temperature correction.

Phase-2 additions (2025-03):
  5. DegradationCurve gains ``source`` and ``prior_weight`` fields so callers
     can inspect how each curve was produced.
  6. ``_pad_to_length`` normalises coefficient arrays to a fixed length so
     linear (2-coeff) and quadratic (3-coeff) curves can be arithmetically
     blended without shape errors.
  7. ``blend_with_prior`` implements the Bayesian-style weighted combination
     of a practice-fitted curve and a historical race prior.  The practice
     weight grows with sample size and R² quality; a poor practice session
     (low n, low R²) almost entirely defers to the historical prior.
  8. ``build_degradation_curves`` now accepts an optional ``historical_prior``
     dict; when provided, each compound curve is blended before returning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    BLEND_N_SATURATION,
    BLEND_R2_SATURATION,
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

# Canonical internal coefficient length — all curves stored as [a, b, c]
# so that blending is always element-wise without shape mismatches.
_COEFF_LENGTH = 3


@dataclass
class DegradationCurve:
    compound: str
    coefficients: np.ndarray   # always length _COEFF_LENGTH after normalisation
    r_squared: float
    sample_size: int
    is_fallback: bool = False
    # "practice" | "historical" | "blend" | "fallback"
    source: str = "practice"
    # Fraction of the final curve that came from the historical prior.
    # 0.0 = pure practice, 1.0 = pure prior, 0.4 = 40% prior / 60% practice.
    prior_weight: float = 0.0
    raw_data: Optional[pd.DataFrame] = field(default=None, repr=False)

    def predict(self, tire_age: int | np.ndarray) -> float | np.ndarray:
        """Return predicted seconds lost relative to a fresh tyre."""
        return np.polyval(self.coefficients, tire_age)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_to_length(coeffs: np.ndarray, length: int = _COEFF_LENGTH) -> np.ndarray:
    """Left-pad a polynomial coefficient array with zeros to *length*.

    np.polyval([0, b, c], x) == np.polyval([b, c], x), so padding the
    leading term with zeros is algebraically transparent.
    """
    if len(coeffs) == length:
        return coeffs
    if len(coeffs) > length:
        return coeffs[-length:]
    return np.concatenate([np.zeros(length - len(coeffs)), coeffs])


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
# Long-run extraction from practice sessions
# ---------------------------------------------------------------------------

def _extract_long_runs(
    session, compounds: List[str]
) -> Dict[str, pd.DataFrame]:
    """Pull clean long-run data from a practice session, keyed by compound.

    Returns a dict mapping compound name to a DataFrame with columns
    ``stint_lap`` (0-based lap within stint) and ``delta_seconds``
    (fuel-corrected time delta from stint start).

    Base time is the median of the first 3 laps of each stripped stint
    rather than strictly lap 0, making it robust to a single anomalous
    exit lap skewing all subsequent deltas.
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

            stint = stint.iloc[1:-1]   # strip pit entry/exit distortion

            compound = stint["Compound"].iloc[0]
            if compound not in compounds:
                continue

            min_laps = COMPOUND_MIN_LONG_RUN.get(compound, 5)
            if len(stint) < min_laps:
                continue

            times = stint["LapTime"].dt.total_seconds().values
            if np.any(np.isnan(times)):
                continue

            base_time = float(np.median(times[:min(3, len(times))]))
            stint_laps = np.arange(len(times))
            raw_delta = times - base_time
            corrected_delta = raw_delta + stint_laps * FUEL_BURN_RATE

            result[compound].append(pd.DataFrame({
                "stint_lap":     stint_laps,
                "delta_seconds": corrected_delta,
            }))

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
                "%s: removed %d outlier lap(s) (IQR factor %.1f)",
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
      n < MIN_SAMPLES_FOR_LINEAR    → (None, 0.0) — caller should use fallback
      n < MIN_SAMPLES_FOR_QUADRATIC → degree-1 (linear)
      n ≥ MIN_SAMPLES_FOR_QUADRATIC → degree-2; but if leading coeff a < 0
                                       (downward parabola) fall back to deg-1.

    Returns (coefficients_padded_to_3, r_squared) or (None, 0.0).
    """
    n = len(data)

    if n < MIN_SAMPLES_FOR_LINEAR:
        logger.warning(
            "Only %d data points — below minimum %d; caller should use fallback",
            n, MIN_SAMPLES_FOR_LINEAR,
        )
        return None, 0.0

    x = data["stint_lap"].values.astype(float)
    y = data["delta_seconds"].values.astype(float)

    if n >= MIN_SAMPLES_FOR_QUADRATIC:
        coeffs = np.polyfit(x, y, deg=2)
        if coeffs[0] < 0:
            logger.warning(
                "Deg-2 fit has a=%.5f < 0 (downward parabola) — falling back to linear",
                coeffs[0],
            )
            coeffs = np.polyfit(x, y, deg=1)
    else:
        coeffs = np.polyfit(x, y, deg=1)

    # Normalise to canonical 3-coefficient form
    coeffs = _pad_to_length(coeffs)

    y_pred    = np.polyval(coeffs, x)
    ss_res    = np.sum((y - y_pred) ** 2)
    ss_tot    = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return coeffs, r_squared


# ---------------------------------------------------------------------------
# Bayesian blending
# ---------------------------------------------------------------------------

def blend_with_prior(
    practice_curve: Optional[DegradationCurve],
    prior_curve: Optional[DegradationCurve],
) -> DegradationCurve:
    """Return a Bayesian-weighted blend of a practice curve and a historical prior.

    The practice weight is determined by how much we trust the practice data:
      practice_weight = min(n / N_SAT, 1) × min(R² / R2_SAT, 1)

    When practice data is sparse (low n) or poor (low R²), the prior
    dominates — ensuring that e.g. 8 Hard laps from a single driver in FP2
    do not override multi-year race evidence.

    Edge cases
    ----------
    * Only practice available → return practice curve unchanged.
    * Only prior available    → return prior curve (source="historical").
    * Neither available       → should not happen; caller guards against it.
    """
    if prior_curve is None:
        return practice_curve  # nothing to blend with

    if practice_curve is None or practice_curve.is_fallback:
        # Config-fallback curves have no real evidence; defer entirely to prior
        result = DegradationCurve(
            compound=prior_curve.compound,
            coefficients=_pad_to_length(prior_curve.coefficients.copy()),
            r_squared=prior_curve.r_squared,
            sample_size=prior_curve.sample_size,
            is_fallback=False,
            source="historical",
            prior_weight=1.0,
        )
        logger.info(
            "%s: practice data absent/fallback — using historical prior 100%%  "
            "(n_prior=%d  R²_prior=%.3f)",
            prior_curve.compound, prior_curve.sample_size, prior_curve.r_squared,
        )
        return result

    # Compute trust weight for practice data
    n    = max(practice_curve.sample_size, 0)
    r2   = max(practice_curve.r_squared,  0.0)
    data_confidence = min(n  / BLEND_N_SATURATION,  1.0)
    fit_confidence  = min(r2 / BLEND_R2_SATURATION, 1.0)
    practice_weight = data_confidence * fit_confidence
    prior_weight    = 1.0 - practice_weight

    p_coeffs = _pad_to_length(practice_curve.coefficients)
    h_coeffs = _pad_to_length(prior_curve.coefficients)
    blended  = practice_weight * p_coeffs + prior_weight * h_coeffs

    # Blended R² is a weighted average (informational, not statistically exact)
    blended_r2 = (
        practice_weight * practice_curve.r_squared
        + prior_weight  * prior_curve.r_squared
    )
    blended_n = practice_curve.sample_size + prior_curve.sample_size

    logger.info(
        "%s blend: practice_weight=%.2f (n=%d  R²=%.3f) "
        "prior_weight=%.2f (n=%d  R²=%.3f)",
        practice_curve.compound,
        practice_weight, practice_curve.sample_size, practice_curve.r_squared,
        prior_weight,    prior_curve.sample_size,    prior_curve.r_squared,
    )

    return DegradationCurve(
        compound=practice_curve.compound,
        coefficients=blended,
        r_squared=blended_r2,
        sample_size=blended_n,
        is_fallback=False,
        source="blend",
        prior_weight=round(prior_weight, 3),
        raw_data=practice_curve.raw_data,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_degradation_curves(
    profile: CircuitProfile,
    compounds: Optional[List[str]] = None,
    historical_prior: Optional[Dict[str, "DegradationCurve"]] = None,
) -> Dict[str, DegradationCurve]:
    """Build per-compound degradation curves from practice data.

    1. Extracts long-run stints from FP2 (FP3 fallback).
    2. Fits a polynomial per compound with degree selection, outlier filtering,
       and per-compound temperature correction.
    3. If ``historical_prior`` is provided, blends each practice curve with the
       corresponding prior curve via ``blend_with_prior``.  A compound that had
       insufficient practice data defers almost entirely to the prior.

    Returns a dict mapping compound → DegradationCurve ready for scoring.
    """
    if compounds is None:
        compounds = list(COMPOUNDS)

    fp2_data = _extract_long_runs(profile.fp2_session, compounds)

    missing  = [c for c in compounds if c not in fp2_data]
    fp3_data: Dict[str, pd.DataFrame] = {}
    if missing:
        logger.info("Falling back to FP3 for compounds: %s", missing)
        fp3_data = _extract_long_runs(profile.fp3_session, missing)

    combined = {**fp2_data, **fp3_data}

    temp_delta = profile.race_track_temp - profile.fp2_track_temp
    logger.info("Temperature delta FP2→Race: %+.1f °C", temp_delta)

    practice_curves: Dict[str, DegradationCurve] = {}
    for compound in compounds:
        data = combined.get(compound)

        compound_tc = TEMP_CORRECTION_FACTOR.get(compound, 0.003)
        correction  = 1.0 + compound_tc * temp_delta

        if data is None or data.empty:
            fallback_coeffs = _pad_to_length(np.array(COMPOUND_FALLBACK_DEG[compound]))
            logger.warning(
                "%s: no long-run data → config fallback curve (%.3f s/lap)",
                compound, fallback_coeffs[-2],
            )
            practice_curves[compound] = DegradationCurve(
                compound=compound,
                coefficients=fallback_coeffs,
                r_squared=0.0,
                sample_size=0,
                is_fallback=True,
                source="fallback",
            )
            continue

        coeffs, r_sq = _fit_degradation(data)

        if coeffs is None:
            fallback_coeffs = _pad_to_length(np.array(COMPOUND_FALLBACK_DEG[compound]))
            logger.warning(
                "%s: n=%d below minimum → config fallback curve",
                compound, len(data),
            )
            practice_curves[compound] = DegradationCurve(
                compound=compound,
                coefficients=fallback_coeffs,
                r_squared=0.0,
                sample_size=len(data),
                is_fallback=True,
                source="fallback",
                raw_data=data,
            )
            continue

        coeffs_corrected = coeffs * correction
        practice_curves[compound] = DegradationCurve(
            compound=compound,
            coefficients=coeffs_corrected,
            r_squared=r_sq,
            sample_size=len(data),
            is_fallback=False,
            source="practice",
            raw_data=data,
        )
        logger.info(
            "%s practice: coeffs=%s  R²=%.3f  n=%d  temp×%.4f",
            compound,
            np.array2string(coeffs_corrected, precision=5),
            r_sq, len(data), correction,
        )

    # ── Phase 2: blend with historical prior ─────────────────────────────────
    if not historical_prior:
        return practice_curves

    final_curves: Dict[str, DegradationCurve] = {}
    for compound in compounds:
        practice = practice_curves.get(compound)
        prior    = historical_prior.get(compound)

        if practice is None and prior is None:
            logger.warning("%s: no practice or prior data — skipping", compound)
            continue

        if prior is None:
            final_curves[compound] = practice
        else:
            final_curves[compound] = blend_with_prior(practice, prior)

    return final_curves
