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

from config import COMPOUNDS, SENSITIVITY_RANGE
from data_collection import build_circuit_profile
from degradation_model import DegradationCurve, build_degradation_curves
from strategy_engine import (
    compute_base_pace,
    generate_strategies,
    rank_strategies,
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
        if curve.raw_data is not None and not curve.raw_data.empty:
            raw_x = [float(v) for v in curve.raw_data["stint_lap"].tolist()]
            raw_y = [round(float(v), 4) for v in curve.raw_data["delta_seconds"].tolist()]

        out[compound] = {
            "r_squared": round(float(curve.r_squared), 3),
            "sample_size": int(curve.sample_size),
            "curve_x": curve_x,
            "curve_y": [round(v, 4) for v in curve_y],
            "curve_y_plus": [round(v, 4) for v in curve_y_plus],
            "curve_y_minus": [round(v, 4) for v in curve_y_minus],
            "raw_x": raw_x,
            "raw_y": raw_y,
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

    # ── Phase 2 ────────────────────────────────────────────────────────────
    log("Phase 2 — Extracting long runs and fitting degradation curves…")
    available = [
        c for c in COMPOUNDS
        if c in set(profile.fp2_session.laps["Compound"].unique())
        | set(profile.fp3_session.laps["Compound"].unique())
    ] or list(COMPOUNDS)

    deg_curves = build_degradation_curves(profile, compounds=available)
    if not deg_curves:
        raise RuntimeError("Could not build any degradation curves for this race.")
    for c, curve in deg_curves.items():
        log(f"  {c}: R²={curve.r_squared:.3f}  n={curve.sample_size}")

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
    base_paces = compute_base_pace(profile.quali_session, drivers=drivers)
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

    result: Dict[str, Any] = {
        "year": year,
        "gp": gp,
        "circuit": {
            "total_laps": profile.total_laps,
            "pit_loss": round(profile.pit_loss_seconds, 2),
            "race_temp": round(profile.race_track_temp, 1),
            "fp2_temp": round(profile.fp2_track_temp, 1),
            "temp_delta": round(profile.race_track_temp - profile.fp2_track_temp, 1),
        },
        "reference_driver": reference_driver,
        "reference_pace": round(reference_pace, 3),
        "total_strategies": len(ranked),
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
                "actual_strategy": v.actual_strategy.description,
                "actual_time": round(v.actual_race_time, 1),
                "predicted_time": round(v.predicted_race_time, 1),
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
