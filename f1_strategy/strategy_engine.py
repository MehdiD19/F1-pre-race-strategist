"""Phase 3 — Strategy Generation & Scoring.

Generates all valid race strategies under FIA and physical constraints,
scores each one via a lap-by-lap model, ranks them, and runs a
sensitivity analysis on the top candidates.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    COMPOUND_MAX_STINT,
    COMPOUNDS,
    FUEL_BURN_RATE,
    FULL_FUEL_PENALTY,
    MAX_STOPS,
    MIN_STINT_LAPS,
    SENSITIVITY_RANGE,
    STINT_DISCRETIZATION,
    WARMUP_PROFILE,
)

# FULL_FUEL_PENALTY is kept as import for callers that don't pass fuel_penalty
# explicitly (e.g. notebooks / tests).  The api_runner passes the circuit-
# calibrated value derived from get_fuel_penalty(gp_name).
from degradation_model import DegradationCurve

logger = logging.getLogger(__name__)


@dataclass
class Stint:
    compound: str
    laps: int

    def __repr__(self) -> str:
        return f"{self.laps}L {self.compound.capitalize()}"


@dataclass
class Strategy:
    stints: List[Stint]
    total_time: float = 0.0

    @property
    def num_stops(self) -> int:
        return len(self.stints) - 1

    @property
    def description(self) -> str:
        return " → ".join(str(s) for s in self.stints)

    def compounds_used(self) -> set:
        return {s.compound for s in self.stints}


# ---------------------------------------------------------------------------
# Strategy generation
# ---------------------------------------------------------------------------

def _stint_lengths(compound: str, total_laps: int) -> List[int]:
    """Return candidate stint lengths for a compound."""
    max_stint = COMPOUND_MAX_STINT.get(compound) or total_laps
    max_stint = min(max_stint, total_laps)

    lengths = list(range(MIN_STINT_LAPS, max_stint + 1, STINT_DISCRETIZATION))
    if MIN_STINT_LAPS not in lengths:
        lengths.insert(0, MIN_STINT_LAPS)
    return lengths


def generate_strategies(
    total_laps: int,
    compounds: Optional[List[str]] = None,
    max_stops: int = MAX_STOPS,
) -> List[Strategy]:
    """Enumerate all valid strategies up to *max_stops* pit stops."""
    if compounds is None:
        compounds = list(COMPOUNDS)

    strategies: List[Strategy] = []
    seen: set = set()

    for num_stops in range(1, max_stops + 1):
        num_stints = num_stops + 1
        compound_combos = product(compounds, repeat=num_stints)

        for combo in compound_combos:
            if len(set(combo)) < 2:
                continue

            length_options = [_stint_lengths(c, total_laps) for c in combo]
            for lengths in product(*length_options):
                if sum(lengths) != total_laps:
                    continue

                # Canonical key: sorted stints prevent permutation duplicates
                # (permutations score identically because degradation sums
                # are order-independent and fuel burn is a constant series).
                key = tuple(sorted(zip(combo, lengths)))
                if key in seen:
                    continue
                seen.add(key)

                stints = [Stint(compound=c, laps=l) for c, l in zip(combo, lengths)]
                strategies.append(Strategy(stints=stints))

    logger.info("Generated %d valid strategies", len(strategies))
    return strategies


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_strategy(
    strategy: Strategy,
    deg_curves: Dict[str, DegradationCurve],
    base_pace: float,
    pit_loss: float,
    total_laps: int,
) -> float:
    """Compute predicted total race time for a strategy.

    Parameters
    ----------
    strategy : Strategy
    deg_curves : per-compound degradation curves
    base_pace : driver base lap time (quali + fuel penalty), in seconds
    pit_loss : pit-lane time loss per stop, in seconds
    total_laps : total race distance in laps

    Returns
    -------
    Total predicted race time in seconds.
    """
    total_time = 0.0
    race_lap = 0

    for stint_idx, stint in enumerate(strategy.stints):
        curve = deg_curves.get(stint.compound)

        for lap_in_stint in range(stint.laps):
            tire_age = lap_in_stint
            race_lap += 1

            deg_penalty = curve.predict(tire_age) if curve else 0.0
            fuel_benefit = FUEL_BURN_RATE * race_lap

            lap_time = base_pace + deg_penalty - fuel_benefit

            # Compound-specific warmup: apply profile over first N laps of non-opening stints.
            # Softs heat in 1 lap; Mediums 2 laps; Hards 3 laps.
            if stint_idx > 0:
                warmup = WARMUP_PROFILE.get(stint.compound, [1.2])
                if lap_in_stint < len(warmup):
                    lap_time += warmup[lap_in_stint]

            total_time += lap_time

    # Add pit-stop time loss
    total_time += pit_loss * strategy.num_stops

    return total_time


def compute_base_pace(
    quali_session,
    drivers: Optional[List[str]] = None,
    fuel_penalty: Optional[float] = None,
) -> Dict[str, float]:
    """Derive per-driver base race pace from qualifying times.

    Tries Q3 first, then Q2, then Q1 so that every finisher gets a pace.

    base_pace = best qualifying time + fuel_penalty

    Parameters
    ----------
    quali_session : FastF1 session object for qualifying.
    drivers : optional list of driver codes to restrict results.
    fuel_penalty : quali→race pace gap in seconds.  If None, the global
        FULL_FUEL_PENALTY from config is used.  Callers should pass the
        circuit-calibrated value from ``config.get_fuel_penalty(gp_name)``
        to avoid the systematic base-pace underestimate identified in the
        2023 Hungarian GP investigation (4.0 s assumed vs 8.1 s observed).
    """
    penalty = fuel_penalty if fuel_penalty is not None else FULL_FUEL_PENALTY

    laps = quali_session.laps
    try:
        q1, q2, q3 = laps.split_qualifying_sessions()
    except Exception:
        q1 = q2 = q3 = None

    # Ordered preference: Q3 → Q2 → Q1 → all laps
    session_order = [s for s in [q3, q2, q1] if s is not None]
    if not session_order:
        session_order = [laps]

    all_drivers = laps["Driver"].unique()
    if drivers:
        all_drivers = [d for d in all_drivers if d in drivers]

    result: Dict[str, float] = {}
    for driver in all_drivers:
        for sess in session_order:
            d_laps = sess.pick_drivers(driver)
            if d_laps.empty:
                continue
            best = d_laps.pick_fastest()
            if best is None:
                continue
            try:
                best_time = best["LapTime"].total_seconds()
                result[driver] = best_time + penalty
                break  # stop at highest qualifying session with a time
            except Exception:
                continue

    logger.info(
        "Base paces computed for %d drivers (fuel penalty = %.2f s)",
        len(result), penalty,
    )
    return result


# ---------------------------------------------------------------------------
# Ranking & sensitivity
# ---------------------------------------------------------------------------

def rank_strategies(
    strategies: List[Strategy],
    deg_curves: Dict[str, DegradationCurve],
    base_pace: float,
    pit_loss: float,
    total_laps: int,
) -> List[Strategy]:
    """Score and sort strategies by predicted total time (ascending)."""
    for strat in strategies:
        strat.total_time = score_strategy(strat, deg_curves, base_pace, pit_loss, total_laps)
    strategies.sort(key=lambda s: s.total_time)
    return strategies


@dataclass
class SensitivityResult:
    strategy: Strategy
    nominal_time: float
    time_plus: float    # +15 % deg
    time_minus: float   # -15 % deg
    flips_type: bool    # True if optimal stop-count changes under perturbation


def _perturb_curves(
    deg_curves: Dict[str, DegradationCurve], factor: float
) -> Dict[str, DegradationCurve]:
    """Scale degradation polynomial coefficients by *factor*."""
    perturbed = {}
    for compound, curve in deg_curves.items():
        new_curve = DegradationCurve(
            compound=curve.compound,
            coefficients=curve.coefficients * factor,
            r_squared=curve.r_squared,
            sample_size=curve.sample_size,
        )
        perturbed[compound] = new_curve
    return perturbed


def sensitivity_analysis(
    strategies: List[Strategy],
    deg_curves: Dict[str, DegradationCurve],
    base_pace: float,
    pit_loss: float,
    total_laps: int,
    top_n: int = 10,
) -> List[SensitivityResult]:
    """Run +/- sensitivity on the top *top_n* strategies."""
    top = strategies[:top_n]

    curves_plus = _perturb_curves(deg_curves, 1 + SENSITIVITY_RANGE)
    curves_minus = _perturb_curves(deg_curves, 1 - SENSITIVITY_RANGE)

    # Find optimal stop-count under each perturbation for flip detection
    all_plus = [
        (s, score_strategy(s, curves_plus, base_pace, pit_loss, total_laps))
        for s in strategies
    ]
    all_minus = [
        (s, score_strategy(s, curves_minus, base_pace, pit_loss, total_laps))
        for s in strategies
    ]

    best_stops_nominal = strategies[0].num_stops if strategies else 1
    best_stops_plus = min(all_plus, key=lambda t: t[1])[0].num_stops if all_plus else best_stops_nominal
    best_stops_minus = min(all_minus, key=lambda t: t[1])[0].num_stops if all_minus else best_stops_nominal

    results: List[SensitivityResult] = []
    for strat in top:
        t_plus = score_strategy(strat, curves_plus, base_pace, pit_loss, total_laps)
        t_minus = score_strategy(strat, curves_minus, base_pace, pit_loss, total_laps)
        flips = best_stops_plus != best_stops_nominal or best_stops_minus != best_stops_nominal

        results.append(SensitivityResult(
            strategy=strat,
            nominal_time=strat.total_time,
            time_plus=t_plus,
            time_minus=t_minus,
            flips_type=flips,
        ))

    return results
