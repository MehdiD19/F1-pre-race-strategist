"""Serializable pipeline runner for the FastAPI backend.

Orchestrates all phases and returns a JSON-serializable dict of results.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from config import COMPOUNDS, HISTORICAL_MIN_YEAR, HISTORICAL_NUM_YEARS, SENSITIVITY_RANGE, get_fuel_penalty
from data_collection import build_circuit_profile
from degradation_model import DegradationCurve, build_degradation_curves
from historical_model import build_historical_prior
from strategy_engine import (
    compute_base_pace,
    compute_lap_timeline,
    generate_strategies,
    rank_strategies,
    strategy_delta_breakdown,
    sensitivity_analysis,
)
from validation import validate_race
from visualization import generate_narrative

logger = logging.getLogger(__name__)


def _serialize_degradation(curves: Dict[str, DegradationCurve]) -> Dict[str, Any]:
    out = {}
    for compound, curve in curves.items():
        max_laps = 42
        curve_x = list(range(0, max_laps + 1))
        curve_y = [float(curve.predict(x)) for x in curve_x]

        # Sensitivity bands — clamp to 0 so negatives don't flip the band
        curve_y_plus = [max(0.0, y * (1 + SENSITIVITY_RANGE)) for y in curve_y]
        curve_y_minus = [max(0.0, y * (1 - SENSITIVITY_RANGE)) for y in curve_y]

        raw_x: List[float] = []
        raw_y: List[float] = []
        raw_driver: List[str] = []
        raw_stint: List[str] = []
        if curve.raw_data is not None and not curve.raw_data.empty:
            raw_x = [float(v) for v in curve.raw_data["stint_lap"].tolist()]
            raw_y = [round(float(v), 4) for v in curve.raw_data["delta_seconds"].tolist()]
            raw_driver = curve.raw_data["driver"].tolist() if "driver" in curve.raw_data.columns else []
            raw_stint = curve.raw_data["stint_id"].tolist() if "stint_id" in curve.raw_data.columns else []

        out[compound] = {
            "r_squared":    round(float(curve.r_squared), 3),
            "sample_size":  int(curve.sample_size),
            "is_fallback":  bool(curve.is_fallback),
            "source":       curve.source,
            "prior_weight": round(float(curve.prior_weight), 3),
            "curve_x":      curve_x,
            "curve_y":      [round(v, 4) for v in curve_y],
            "curve_y_plus": [round(v, 4) for v in curve_y_plus],
            "curve_y_minus":[round(v, 4) for v in curve_y_minus],
            "raw_x": raw_x,
            "raw_y": raw_y,
            "raw_driver": raw_driver,
            "raw_stint": raw_stint,
        }
    return out


def run_analysis(
    year: int,
    gp: str,
    drivers: Optional[List[str]] = None,
    max_stops: int = 2,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run the full strategy analysis pipeline and return a serializable dict."""

    def log(msg: str) -> None:
        logger.info(msg)
        if progress:
            progress(msg)

    # ── Phase 1 ────────────────────────────────────────────────────────────
    log("Phase 1 — Loading FP2, FP3, Qualifying & Race sessions…")
    profile = build_circuit_profile(year, gp)
    log(
        f"Circuit loaded: {profile.total_laps} laps · pit loss {profile.pit_loss_seconds:.1f} s "
        f"· race temp {profile.race_track_temp:.1f} °C (FP2 {profile.fp2_track_temp:.1f} °C)"
    )

    # ── Phase 2a — Practice long-run extraction ────────────────────────────
    log("Phase 2 — Extracting practice long runs and building historical prior…")
    available = [
        c for c in COMPOUNDS
        if c in set(profile.fp2_session.laps["Compound"].unique())
        | set(profile.fp3_session.laps["Compound"].unique())
    ] or list(COMPOUNDS)

    # ── Phase 2b — Historical race prior ──────────────────────────────────
    historical_prior = {}
    try:
        historical_prior = build_historical_prior(
            gp=gp,
            year=year,
            compounds=available,
            num_years=HISTORICAL_NUM_YEARS,
            min_year=HISTORICAL_MIN_YEAR,
            progress=progress,
        )
        if historical_prior:
            log(
                f"  Historical prior loaded for: "
                f"{[c for c in historical_prior]} "
                f"(years back to {HISTORICAL_MIN_YEAR})"
            )
        else:
            log("  Historical prior: no data available — using practice only")
    except Exception as exc:
        log(f"  Historical prior failed ({exc}) — continuing with practice data only")
        historical_prior = {}

    # ── Phase 2c — Blend practice + prior ─────────────────────────────────
    deg_curves = build_degradation_curves(
        profile,
        compounds=available,
        historical_prior=historical_prior or None,
    )
    if not deg_curves:
        raise RuntimeError("Could not build any degradation curves for this race.")
    for c, curve in deg_curves.items():
        log(
            f"  {c}: source={curve.source}  R²={curve.r_squared:.3f}  "
            f"n={curve.sample_size}  prior_weight={curve.prior_weight:.2f}"
        )

    # ── Phase 3 ────────────────────────────────────────────────────────────
    log(f"Phase 3 — Generating all valid strategies (max {max_stops} stops)…")
    strategies = generate_strategies(
        profile.total_laps,
        compounds=list(deg_curves.keys()),
        max_stops=max_stops,
    )
    if not strategies:
        raise RuntimeError("No valid strategies generated.")
    log(f"  {len(strategies)} unique strategies to score")

    log("  Computing driver base paces from qualifying…")
    fuel_penalty = get_fuel_penalty(gp)
    log(f"  Circuit fuel penalty: {fuel_penalty:.1f} s (circuit type from '{gp}')")
    base_paces = compute_base_pace(
        profile.quali_session,
        drivers=drivers,
        fuel_penalty=fuel_penalty,
    )
    if not base_paces:
        raise RuntimeError("Could not derive any base paces from qualifying data.")

    reference_driver = min(base_paces, key=base_paces.get)
    reference_pace = base_paces[reference_driver]
    log(f"  Reference driver: {reference_driver} ({reference_pace:.3f} s/lap)")

    log("  Scoring and ranking…")
    ranked = rank_strategies(
        strategies, deg_curves, reference_pace, profile.pit_loss_seconds, profile.total_laps
    )
    sens = sensitivity_analysis(
        ranked, deg_curves, reference_pace, profile.pit_loss_seconds, profile.total_laps
    )

    # ── Phase 4 ────────────────────────────────────────────────────────────
    log("Phase 4 — Comparing model predictions against actual race results…")
    validations = validate_race(
        profile, deg_curves, ranked, base_paces, drivers=drivers
    )
    log(f"  {len(validations)} drivers validated")

    # ── Serialize ──────────────────────────────────────────────────────────
    log("Building output…")

    # Lap timeline for top 3 strategies (lap-by-lap predicted time)
    top_n_timeline = 3
    lap_timeline: Dict[str, Any] = {
        "laps": list(range(1, profile.total_laps + 1)),
        "strategies": [],
    }
    for i, s in enumerate(ranked[:top_n_timeline]):
        timeline = compute_lap_timeline(
            s, deg_curves, reference_pace, profile.pit_loss_seconds, profile.total_laps
        )
        lap_timeline["strategies"].append({
            "rank": i + 1,
            "description": s.description,
            "lap_times": [round(t, 2) for _, t in timeline],
        })

    # Pit window: strategies within +5s of optimal, extract stint-1 lap ranges per compound
    PIT_WINDOW_DELTA = 5.0
    optimal_time = ranked[0].total_time if ranked else 0.0
    near_optimal = [s for s in ranked if s.total_time <= optimal_time + PIT_WINDOW_DELTA]
    pit_window_laps: Dict[str, List[int]] = {}
    for s in near_optimal:
        if not s.stints:
            continue
        st1 = s.stints[0]
        c = st1.compound
        if c not in pit_window_laps:
            pit_window_laps[c] = []
        pit_window_laps[c].append(st1.laps)
    pit_window_ranges: List[Dict[str, Any]] = []
    for compound, laps_list in pit_window_laps.items():
        mn, mx = min(laps_list), max(laps_list)
        pit_window_ranges.append({
            "compound": compound,
            "min_laps": mn,
            "max_laps": mx,
            "pit_between_lap": f"{mn}–{mx}" if mn != mx else str(mn),
        })

    # Strategy delta breakdown: rank 1 vs rank 2
    strategy_delta: Optional[Dict[str, Any]] = None
    if len(ranked) >= 2:
        bd = strategy_delta_breakdown(
            ranked[0], ranked[1], deg_curves,
            reference_pace, profile.pit_loss_seconds, profile.total_laps,
        )
        strategy_delta = {
            "rank1": ranked[0].description,
            "rank2": ranked[1].description,
            "total_delta": round(bd["total_delta"], 2),
            "pit_delta": round(bd["pit_delta"], 2),
            "warmup_delta": round(bd["warmup_delta"], 2),
            "degradation_delta": round(bd["degradation_delta"], 2),
        }

    result: Dict[str, Any] = {
        "year": year,
        "gp": gp,
        "circuit": {
            "total_laps": profile.total_laps,
            "pit_loss": round(profile.pit_loss_seconds, 2),
            "race_temp": round(profile.race_track_temp, 1),
            "fp2_temp": round(profile.fp2_track_temp, 1),
            "temp_delta": round(profile.race_track_temp - profile.fp2_track_temp, 1),
            "fuel_penalty": round(fuel_penalty, 1),
        },
        "reference_driver": reference_driver,
        "reference_pace": round(reference_pace, 3),
        # Per-driver qualifying-derived base paces (quali best + circuit fuel penalty).
        # Every driver gets their own pace; the reference driver is simply the fastest.
        # Strategy RANKINGS are invariant to base_pace (it's an additive constant),
        # so the optimal strategy order is the same for all drivers.
        # What differs per driver is their absolute predicted time with any strategy.
        "driver_paces": {
            driver: round(pace, 3)
            for driver, pace in sorted(base_paces.items(), key=lambda x: x[1])
        },
        "total_strategies": len(ranked),
        "lap_timeline": lap_timeline,
        "pit_window": pit_window_ranges,
        "strategy_delta": strategy_delta,
        "degradation": _serialize_degradation(deg_curves),
        "strategies": [
            {
                "rank": i + 1,
                "stops": s.num_stops,
                "description": s.description,
                "total_time": round(s.total_time, 1),
                "delta": round(s.total_time - ranked[0].total_time, 1),
                "stints": [{"compound": st.compound, "laps": st.laps} for st in s.stints],
            }
            for i, s in enumerate(ranked[:20])
        ],
        "sensitivity": [
            {
                "description": r.strategy.description,
                "nominal": round(r.nominal_time, 1),
                "time_plus": round(r.time_plus, 1),
                "time_minus": round(r.time_minus, 1),
                "flips": r.flips_type,
            }
            for r in sens
        ],
        "validation": [
            {
                "driver": v.driver,
                "base_pace": round(base_paces.get(v.driver, 0.0), 3),
                "actual_strategy": v.actual_strategy.description,
                "actual_time": round(v.actual_race_time, 1),
                "predicted_time": round(v.predicted_race_time, 1),
                "optimal_time": round(v.optimal_time, 1),
                "error": round(v.model_error, 1),
                "rank": v.predicted_rank,
                "delta_to_optimal": round(v.delta_to_optimal, 1),
                "traffic_loss": round(v.avg_traffic_loss_per_lap, 2),
            }
            for v in validations
        ],
        "narrative": generate_narrative(year, gp, ranked, validations, sens),
    }

    log("Analysis complete!")
    return result
