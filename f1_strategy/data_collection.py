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
    """Estimate pit-lane time loss by comparing box laps to clean laps."""
    laps = race_session.laps

    clean_laps = laps.pick_wo_box().pick_accurate()
    if clean_laps.empty:
        logger.warning("No clean laps found for pit-loss estimation, using default 22 s")
        return 22.0

    median_clean = clean_laps["LapTime"].dt.total_seconds().median()

    in_laps = laps.pick_box_laps(which="in")
    out_laps = laps.pick_box_laps(which="out")
    box_laps = pd.concat([in_laps, out_laps]).drop_duplicates()

    if box_laps.empty:
        logger.warning("No box laps found, using default 22 s")
        return 22.0

    median_box = box_laps["LapTime"].dt.total_seconds().median()
    pit_loss = median_box - median_clean

    if pit_loss < 10 or pit_loss > 40:
        logger.warning(
            "Pit-loss estimate %.1f s looks suspect, clamping to [15, 30]", pit_loss
        )
        pit_loss = np.clip(pit_loss, 15.0, 30.0)

    return float(pit_loss)


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
