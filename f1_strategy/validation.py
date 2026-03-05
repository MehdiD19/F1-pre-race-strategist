"""Phase 4 — Validation Against Reality.

Extracts each driver's actual race strategy, scores it through the model,
compares predicted vs. actual race time, and estimates traffic effects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import TRAFFIC_THRESHOLD
from data_collection import CircuitProfile
from degradation_model import DegradationCurve
from strategy_engine import Strategy, Stint, score_strategy

logger = logging.getLogger(__name__)


@dataclass
class DriverValidation:
    driver: str
    actual_strategy: Strategy
    actual_race_time: float            # real elapsed race time in seconds
    predicted_race_time: float         # model's prediction for their actual strategy
    model_error: float                 # predicted - actual
    predicted_rank: int                # rank of actual strategy among all candidates
    optimal_time: float                # best strategy time from the model
    delta_to_optimal: float            # actual strategy predicted time - optimal time
    avg_traffic_loss_per_lap: float    # seconds lost per lap when in traffic


def extract_actual_strategies(
    race_session, total_laps: int = 0
) -> Dict[str, Strategy]:
    """Extract each driver's actual stint sequence from race data.

    If *total_laps* is provided, drivers who completed fewer laps (DNFs)
    are excluded.
    """
    laps = race_session.laps
    drivers = laps["Driver"].unique()
    strategies: Dict[str, Strategy] = {}

    for driver in drivers:
        d_laps = laps.pick_drivers(driver).sort_values("LapNumber")
        if d_laps.empty:
            continue

        completed = int(d_laps["LapNumber"].max())
        if total_laps and completed < total_laps - 1:
            logger.info("Skipping %s — completed %d / %d laps (DNF)", driver, completed, total_laps)
            continue

        stints: List[Stint] = []
        for stint_num in sorted(d_laps["Stint"].unique()):
            stint_laps = d_laps[d_laps["Stint"] == stint_num]
            compound = stint_laps["Compound"].iloc[0]
            num_laps = len(stint_laps)
            if num_laps > 0 and isinstance(compound, str):
                stints.append(Stint(compound=compound, laps=num_laps))

        if stints:
            strategies[driver] = Strategy(stints=stints)

    return strategies


def _actual_race_time(race_session, driver: str) -> float:
    """Sum of all LapTime values for a driver (seconds)."""
    d_laps = race_session.laps.pick_drivers(driver)
    times = d_laps["LapTime"].dt.total_seconds().dropna()
    return float(times.sum())


def _estimate_traffic_loss(race_session, driver: str) -> float:
    """Estimate average time lost per lap when running in traffic.

    Traffic is defined as being within TRAFFIC_THRESHOLD seconds of the
    car immediately ahead on any given lap.  We compare the median lap
    time in traffic vs. free air.
    """
    laps = race_session.laps
    d_laps = laps.pick_drivers(driver).pick_accurate().pick_wo_box()
    if d_laps.empty:
        return 0.0

    d_laps = d_laps.sort_values("LapNumber").copy()
    d_times = d_laps.set_index("LapNumber")["LapTime"].dt.total_seconds()

    all_laps = laps.pick_accurate().pick_wo_box()
    in_traffic_laps = []
    free_air_laps = []

    for lap_num in d_times.index:
        driver_time_raw = d_laps[d_laps["LapNumber"] == lap_num]
        if driver_time_raw.empty:
            continue

        lap_time_s = d_times.get(lap_num)
        if lap_time_s is None or np.isnan(lap_time_s):
            continue

        lap_all = all_laps[all_laps["LapNumber"] == lap_num].copy()
        lap_all["_time_s"] = lap_all["LapTime"].dt.total_seconds()
        lap_all = lap_all.dropna(subset=["_time_s"]).sort_values("_time_s")

        driver_pos = lap_all.index[lap_all["Driver"] == driver]
        if len(driver_pos) == 0:
            free_air_laps.append(lap_time_s)
            continue

        idx = list(lap_all.index).index(driver_pos[0])
        if idx > 0:
            car_ahead_time = lap_all.iloc[idx - 1]["_time_s"]
            gap = lap_time_s - car_ahead_time
            if 0 < gap < TRAFFIC_THRESHOLD:
                in_traffic_laps.append(lap_time_s)
                continue

        free_air_laps.append(lap_time_s)

    if not in_traffic_laps or not free_air_laps:
        return 0.0

    return float(np.median(in_traffic_laps) - np.median(free_air_laps))


def validate_race(
    profile: CircuitProfile,
    deg_curves: Dict[str, DegradationCurve],
    ranked_strategies: List[Strategy],
    base_paces: Dict[str, float],
    drivers: Optional[List[str]] = None,
) -> List[DriverValidation]:
    """Compare model predictions against actual race outcomes."""
    race = profile.race_session
    actual_strats = extract_actual_strategies(race, total_laps=profile.total_laps)

    if drivers:
        actual_strats = {d: s for d, s in actual_strats.items() if d in drivers}

    optimal_time = ranked_strategies[0].total_time if ranked_strategies else 0.0

    results: List[DriverValidation] = []
    for driver, actual_strat in actual_strats.items():
        base_pace = base_paces.get(driver)
        if base_pace is None:
            logger.warning("No base pace for %s, skipping", driver)
            continue

        predicted_time = score_strategy(
            actual_strat, deg_curves, base_pace, profile.pit_loss_seconds, profile.total_laps
        )

        # Rank: count every strategy that scores strictly better than the driver's
        # actual strategy (both scored with the driver's own base pace so the
        # comparison is fair). No early break — the list is sorted by reference-
        # driver pace, but the driver's own pace shifts absolute times while
        # preserving relative order only approximately, so we iterate all.
        predicted_rank = 1 + sum(
            1 for strat in ranked_strategies
            if score_strategy(
                strat, deg_curves, base_pace, profile.pit_loss_seconds, profile.total_laps
            ) < predicted_time
        )

        actual_time = _actual_race_time(race, driver)
        traffic_loss = _estimate_traffic_loss(race, driver)

        driver_optimal = score_strategy(
            ranked_strategies[0], deg_curves, base_pace, profile.pit_loss_seconds, profile.total_laps
        ) if ranked_strategies else predicted_time

        results.append(DriverValidation(
            driver=driver,
            actual_strategy=actual_strat,
            actual_race_time=actual_time,
            predicted_race_time=predicted_time,
            model_error=predicted_time - actual_time,
            predicted_rank=predicted_rank,
            optimal_time=driver_optimal,
            delta_to_optimal=predicted_time - driver_optimal,
            avg_traffic_loss_per_lap=traffic_loss,
        ))

    results.sort(key=lambda v: v.actual_race_time)
    return results
