"""Historical race data prior for tire degradation modeling.

Loads actual race sessions from previous seasons at the same Grand Prix and
builds per-compound degradation curves from the pooled stint data.

Why race data instead of practice data?
  - Race stints are 15–40 laps; practice long runs are 8–15 laps at best.
  - Race stints happen at full fuel load, race-pace commitment, and real
    tyre management — which is exactly what we want to model.
  - Practice sessions are dominated by setup runs, quali sims, and short
    tyre-comparison laps that add noise without adding degradation signal.

Era boundary (2022+):
  The 2022 regulation change introduced ground-effect aerodynamics, which
  fundamentally altered downforce levels and, consequently, the lateral and
  longitudinal loads applied to tyres in every corner.  Pre-2022 data would
  systematically understate degradation for circuits with high-speed corners
  and overstate it for low-speed circuits.  Only ground-effect era seasons
  are used as priors by default (configurable via HISTORICAL_MIN_YEAR).

Dependency note:
  This module imports ``_fit_degradation`` and ``_filter_outliers_iqr`` from
  ``degradation_model`` to keep fitting logic centralised.  The import is
  one-directional: ``degradation_model`` never imports from here.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import fastf1
import numpy as np
import pandas as pd

from config import (
    BLEND_N_SATURATION,
    COMPOUNDS,
    FUEL_BURN_RATE,
    HISTORICAL_MIN_RACE_STINT_LAPS,
    HISTORICAL_MIN_YEAR,
    HISTORICAL_NUM_YEARS,
    OUTLIER_IQR_FACTOR,
)
from degradation_model import DegradationCurve, _filter_outliers_iqr, _fit_degradation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Race stint extraction
# ---------------------------------------------------------------------------

def _extract_race_stints(
    race_session,
    compounds: List[str],
    min_laps: int = HISTORICAL_MIN_RACE_STINT_LAPS,
) -> Dict[str, pd.DataFrame]:
    """Extract per-compound degradation data from a completed race.

    Filtering steps applied:
    1. Remove pit-box laps (``pick_wo_box``).
    2. Keep only green-flag laps (TrackStatus == "1") to exclude safety car
       and VSC periods, which produce artificially slow laps that would
       flatten the fitted degradation curve.
    3. Strip the first 2 laps of each stint (pit-exit warm-up) and the last
       lap (often pushed hard or lifted for strategy).
    4. Require at least *min_laps* clean laps after stripping.
    5. Apply IQR outlier removal on the pooled delta_seconds per compound.

    Returns a dict compound → DataFrame(stint_lap, delta_seconds, year, driver).
    """
    laps = race_session.laps

    clean = laps.pick_wo_box()
    try:
        clean = clean.pick_track_status("1", how="equals")
    except Exception:
        pass

    result: Dict[str, List[pd.DataFrame]] = {c: [] for c in compounds}

    for driver in clean["Driver"].unique():
        d_laps = clean.pick_drivers(driver).sort_values("LapNumber")
        for stint_num in sorted(d_laps["Stint"].unique()):
            stint = d_laps[d_laps["Stint"] == stint_num].copy()

            # Need at least min_laps usable laps after stripping 2 front + 1 back
            if len(stint) < min_laps + 3:
                continue

            stint = stint.iloc[2:-1]   # strip pit-exit warm-up + last strategic lap

            if len(stint) < min_laps:
                continue

            compound = stint["Compound"].iloc[0]
            if not isinstance(compound, str) or compound not in compounds:
                continue

            times = stint["LapTime"].dt.total_seconds().values
            if np.any(np.isnan(times)) or np.any(times <= 0):
                continue

            # Sanity: median lap must be in a plausible race range
            median_t = float(np.median(times))
            if not (55.0 < median_t < 200.0):
                continue

            # Stabilised base: median of first 3 laps of the stripped stint
            base_time  = float(np.median(times[:3]))
            stint_laps = np.arange(len(times))
            raw_delta  = times - base_time
            # Add back fuel benefit to isolate tyre degradation
            corrected_delta = raw_delta + stint_laps * FUEL_BURN_RATE

            try:
                event_year = int(race_session.event["EventDate"].year)
            except Exception:
                event_year = -1

            result[compound].append(pd.DataFrame({
                "stint_lap":     stint_laps,
                "delta_seconds": corrected_delta,
                "year":          event_year,
                "driver":        driver,
            }))

    merged: Dict[str, pd.DataFrame] = {}
    for compound in compounds:
        frames = result[compound]
        if not frames:
            continue
        pooled = pd.concat(frames, ignore_index=True)
        mask = _filter_outliers_iqr(pooled["delta_seconds"].values, OUTLIER_IQR_FACTOR)
        n_removed = (~mask).sum()
        if n_removed:
            logger.debug(
                "Race prior %s: removed %d outlier laps (IQR filter)",
                compound, n_removed,
            )
        merged[compound] = pooled[mask].reset_index(drop=True)

    return merged


# ---------------------------------------------------------------------------
# Prior building
# ---------------------------------------------------------------------------

def build_historical_prior(
    gp: str,
    year: int,
    compounds: Optional[List[str]] = None,
    num_years: int = HISTORICAL_NUM_YEARS,
    min_year: int = HISTORICAL_MIN_YEAR,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, DegradationCurve]:
    """Build per-compound degradation priors from historical race data.

    Loads up to *num_years* prior seasons (going back to *min_year*) at the
    same GP, pools the per-compound stint data across years, and fits a
    single degradation curve per compound.

    The returned DegradationCurve objects carry ``source="historical"`` so
    downstream blending logic can distinguish them from practice-fitted curves.

    Returns an empty dict if no historical data can be loaded (network error,
    no prior years available, etc.).  Callers should handle an empty return
    gracefully and fall back to practice-only curves.

    Parameters
    ----------
    gp      : GP name string, same format as ``fastf1.get_session`` accepts.
    year    : The current analysis year (priors are taken from year-1, year-2, …).
    compounds : Compound list; defaults to config.COMPOUNDS.
    num_years : Maximum number of prior seasons to attempt loading.
    min_year  : Earliest allowed season (ground-effect era floor = 2022).
    progress  : Optional callable receiving progress strings for UI logging.
    """
    if compounds is None:
        compounds = list(COMPOUNDS)

    def _log(msg: str) -> None:
        logger.info(msg)
        if progress:
            progress(msg)

    # Build candidate list: (year-1), (year-2), … down to min_year
    candidate_years = [
        y for y in range(year - 1, min_year - 1, -1)
        if (year - y) <= num_years
    ]

    if not candidate_years:
        _log(f"  Historical prior: no eligible years for {year} {gp} (min_year={min_year})")
        return {}

    _log(f"  Historical prior: attempting years {candidate_years} for {gp}")

    # Accumulate stint data across successfully loaded years
    all_frames: Dict[str, List[pd.DataFrame]] = {c: [] for c in compounds}
    loaded_years: List[int] = []

    for prior_year in candidate_years:
        try:
            session = fastf1.get_session(prior_year, gp, "R")
            # Skip telemetry/weather/messages — we only need lap times
            session.load(telemetry=False, weather=False, messages=False)

            year_stints = _extract_race_stints(session, compounds)

            if not year_stints:
                _log(f"    {prior_year}: no usable race stints found — skipping")
                continue

            for compound, df in year_stints.items():
                all_frames[compound].append(df)

            counts = {c: len(all_frames[c][-1]) if all_frames[c] else 0
                      for c in compounds}
            _log(f"    {prior_year}: loaded — {counts} laps per compound")
            loaded_years.append(prior_year)

        except Exception as exc:
            _log(f"    {prior_year}: could not load ({exc}) — skipping")
            continue

    if not loaded_years:
        _log(f"  Historical prior: no prior years loaded for {gp}")
        return {}

    _log(f"  Historical prior: using data from {loaded_years}")

    # Fit one curve per compound from all pooled prior-year data
    prior_curves: Dict[str, DegradationCurve] = {}
    for compound in compounds:
        frames = all_frames[compound]
        if not frames:
            _log(f"  Prior {compound}: no data across any loaded year — skipping")
            continue

        pooled = pd.concat(frames, ignore_index=True)[["stint_lap", "delta_seconds"]]
        n_total = len(pooled)

        coeffs, r_sq = _fit_degradation(pooled)
        if coeffs is None:
            _log(
                f"  Prior {compound}: n={n_total} insufficient for any fit — skipping"
            )
            continue

        _log(
            f"  Prior {compound}: n={n_total}  R²={r_sq:.3f}  "
            f"coeffs={np.array2string(coeffs, precision=4)}"
        )

        prior_curves[compound] = DegradationCurve(
            compound=compound,
            coefficients=coeffs,
            r_squared=r_sq,
            sample_size=n_total,
            is_fallback=False,
            source="historical",
            prior_weight=1.0,
        )

    _log(
        f"  Historical prior complete: {list(prior_curves.keys())} fitted "
        f"({len(prior_curves)}/{len(compounds)} compounds)"
    )
    return prior_curves
