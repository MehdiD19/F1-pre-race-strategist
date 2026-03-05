"""Phase 2 — Tire Degradation Modeling from Practice Data.

Extracts long-run stints from FP2 (fallback FP3), isolates pure tire
degradation by removing the fuel-burn benefit, fits a degree-2 polynomial
per compound, and applies a temperature correction for race day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    COMPOUND_MIN_LONG_RUN,
    COMPOUNDS,
    FUEL_BURN_RATE,
    TEMP_CORRECTION_FACTOR,
)
from data_collection import CircuitProfile

logger = logging.getLogger(__name__)


@dataclass
class DegradationCurve:
    compound: str
    coefficients: np.ndarray   # degree-2 polynomial [a, b, c] → a*x² + b*x + c
    r_squared: float
    sample_size: int
    raw_data: Optional[pd.DataFrame] = field(default=None, repr=False)

    def predict(self, tire_age: int | np.ndarray) -> float | np.ndarray:
        """Return predicted seconds lost relative to a fresh tire."""
        return np.polyval(self.coefficients, tire_age)


def _extract_long_runs(
    session, compounds: List[str]
) -> Dict[str, pd.DataFrame]:
    """Pull clean long-run data from a practice session, keyed by compound.

    Returns a dict mapping compound name to a DataFrame with columns
    ``stint_lap`` (0-based lap within stint) and ``delta_seconds``
    (fuel-corrected time delta from stint start).
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

            # Strip first and last lap of stint (pit entry/exit distortion)
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

            base_time = times[0]
            stint_laps = np.arange(len(times))

            raw_delta = times - base_time

            # Add back fuel-burn benefit to isolate pure tire degradation
            corrected_delta = raw_delta + stint_laps * FUEL_BURN_RATE

            df = pd.DataFrame({
                "stint_lap": stint_laps,
                "delta_seconds": corrected_delta,
            })
            result[compound].append(df)

    merged: Dict[str, pd.DataFrame] = {}
    for compound in compounds:
        frames = result[compound]
        if frames:
            merged[compound] = pd.concat(frames, ignore_index=True)
    return merged


def _fit_degradation(data: pd.DataFrame) -> Tuple[np.ndarray, float]:
    """Fit degree-2 polynomial to (stint_lap, delta_seconds) data.

    Returns (coefficients, r_squared).
    """
    x = data["stint_lap"].values.astype(float)
    y = data["delta_seconds"].values.astype(float)

    coeffs = np.polyfit(x, y, deg=2)

    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return coeffs, r_squared


def build_degradation_curves(
    profile: CircuitProfile,
    compounds: Optional[List[str]] = None,
) -> Dict[str, DegradationCurve]:
    """Build per-compound degradation curves from practice data.

    Uses FP2 as primary source and falls back to FP3 for any compound
    that lacks sufficient long-run data.
    """
    if compounds is None:
        compounds = list(COMPOUNDS)

    fp2_data = _extract_long_runs(profile.fp2_session, compounds)

    missing = [c for c in compounds if c not in fp2_data]
    fp3_data: Dict[str, pd.DataFrame] = {}
    if missing:
        logger.info("Falling back to FP3 for compounds: %s", missing)
        fp3_data = _extract_long_runs(profile.fp3_session, missing)

    combined = {**fp2_data, **fp3_data}

    temp_delta = profile.race_track_temp - profile.fp2_track_temp
    correction = 1 + TEMP_CORRECTION_FACTOR * temp_delta
    logger.info(
        "Temperature correction: Δ%.1f °C → factor %.4f",
        temp_delta,
        correction,
    )

    curves: Dict[str, DegradationCurve] = {}
    for compound in compounds:
        data = combined.get(compound)
        if data is None or data.empty:
            logger.warning("No long-run data for %s — skipping", compound)
            continue

        coeffs, r_sq = _fit_degradation(data)

        # Apply temperature correction to all polynomial coefficients
        coeffs = coeffs * correction

        curve = DegradationCurve(
            compound=compound,
            coefficients=coeffs,
            r_squared=r_sq,
            sample_size=len(data),
            raw_data=data,
        )
        curves[compound] = curve
        logger.info(
            "%s curve: coeffs=%s  R²=%.3f  n=%d",
            compound,
            np.array2string(coeffs, precision=6),
            r_sq,
            len(data),
        )

    return curves
