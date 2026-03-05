"""Phase 1 — Data Collection & Circuit Profiling.

Loads FP2, FP3, Qualifying, and Race sessions for a given GP + year,
then extracts circuit-level constants (total laps, pit-loss, temperatures).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import fastf1
import numpy as np
import pandas as pd

from config import CACHE_DIR

logger = logging.getLogger(__name__)

fastf1.Cache.enable_cache(CACHE_DIR)


@dataclass
class CircuitProfile:
    year: int
    gp: str
    total_laps: int
    pit_loss_seconds: float
    race_track_temp: float
    fp2_track_temp: float
    fp2_session: object = field(repr=False)
    fp3_session: object = field(repr=False)
    quali_session: object = field(repr=False)
    race_session: object = field(repr=False)


def _load_session(year: int, gp: str, identifier: str) -> fastf1.core.Session:
    session = fastf1.get_session(year, gp, identifier)
    session.load()
    return session


def _estimate_pit_loss(race_session: fastf1.core.Session) -> float:
    """Estimate pit-lane time loss by averaging per-stop deltas across all drivers.

    For each driver and each pit stop, computes:
        stop_loss = (in_lap_time + out_lap_time) − 2 × driver_median_clean_lap

    This captures the real time penalty paid per stop rather than a single
    global median, and naturally accounts for pit-lane length differences.
    """
    laps = race_session.laps
    clean = laps.pick_wo_box().pick_accurate()

    if clean.empty:
        logger.warning("No clean laps found for pit-loss estimation, using default 22 s")
        return 22.0

    pit_losses: list[float] = []

    for driver in laps["Driver"].unique():
        d_all   = laps.pick_drivers(driver).sort_values("LapNumber")
        d_clean = clean.pick_drivers(driver)

        if d_clean.empty or len(d_clean) < 3:
            continue

        driver_median = float(d_clean["LapTime"].dt.total_seconds().median())
        stints = sorted(d_all["Stint"].unique())

        for i in range(1, len(stints)):
            prev_stint = d_all[d_all["Stint"] == stints[i - 1]]
            curr_stint = d_all[d_all["Stint"] == stints[i]]

            if prev_stint.empty or curr_stint.empty:
                continue

            in_lap_t  = prev_stint.iloc[-1]["LapTime"]
            out_lap_t = curr_stint.iloc[0]["LapTime"]

            if pd.isna(in_lap_t) or pd.isna(out_lap_t):
                continue

            in_s  = float(in_lap_t.total_seconds())
            out_s = float(out_lap_t.total_seconds())

            if in_s <= 0 or out_s <= 0 or in_s > 300 or out_s > 300:
                continue

            stop_loss = (in_s + out_s) - 2.0 * driver_median
            if 5.0 < stop_loss < 60.0:
                pit_losses.append(stop_loss)

    if not pit_losses:
        logger.warning("No valid per-stop pit losses found, using default 22 s")
        return 22.0

    mean_loss = float(np.mean(pit_losses))
    logger.info(
        "Pit-loss: mean=%.1f s  std=%.1f s  n=%d stops  range=[%.1f, %.1f] s",
        mean_loss, float(np.std(pit_losses)),
        len(pit_losses), min(pit_losses), max(pit_losses),
    )

    if not (10.0 <= mean_loss <= 40.0):
        logger.warning("Pit-loss estimate %.1f s looks suspect, clamping to [15, 30]", mean_loss)
        mean_loss = float(np.clip(mean_loss, 15.0, 30.0))

    return mean_loss


def _mean_track_temp(session: fastf1.core.Session, first_half_only: bool = False) -> float:
    """Return mean TrackTemp from a session's weather data."""
    weather = session.weather_data
    if weather is None or weather.empty:
        logger.warning("No weather data for session, returning 30 °C default")
        return 30.0

    if first_half_only:
        total_time = weather["Time"].max()
        half_time = total_time / 2
        weather = weather[weather["Time"] <= half_time]

    return float(weather["TrackTemp"].mean())


def build_circuit_profile(
    year: int,
    gp: str,
    fp2_session: Optional[fastf1.core.Session] = None,
    fp3_session: Optional[fastf1.core.Session] = None,
    quali_session: Optional[fastf1.core.Session] = None,
    race_session: Optional[fastf1.core.Session] = None,
) -> CircuitProfile:
    """Load all sessions and compute the circuit profile.

    Pre-loaded session objects can be passed to skip re-downloading.
    """
    logger.info("Loading sessions for %d %s …", year, gp)

    if fp2_session is None:
        fp2_session = _load_session(year, gp, "FP2")
    if fp3_session is None:
        fp3_session = _load_session(year, gp, "FP3")
    if quali_session is None:
        quali_session = _load_session(year, gp, "Q")
    if race_session is None:
        race_session = _load_session(year, gp, "R")

    total_laps = int(race_session.laps["LapNumber"].max())
    pit_loss = _estimate_pit_loss(race_session)
    race_track_temp = _mean_track_temp(race_session, first_half_only=True)
    fp2_track_temp = _mean_track_temp(fp2_session, first_half_only=False)

    profile = CircuitProfile(
        year=year,
        gp=gp,
        total_laps=total_laps,
        pit_loss_seconds=pit_loss,
        race_track_temp=race_track_temp,
        fp2_track_temp=fp2_track_temp,
        fp2_session=fp2_session,
        fp3_session=fp3_session,
        quali_session=quali_session,
        race_session=race_session,
    )

    logger.info(
        "Circuit profile: %d laps, pit loss %.1f s, race temp %.1f °C, FP2 temp %.1f °C",
        total_laps,
        pit_loss,
        race_track_temp,
        fp2_track_temp,
    )
    return profile
