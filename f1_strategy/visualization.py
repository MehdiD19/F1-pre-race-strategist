"""Phase 5 — Presentation of Results.

Produces four deliverables per Grand Prix:
  1. Degradation curve charts with sensitivity bands
  2. Strategy ranking table (top 10)
  3. Model vs. Reality table
  4. Insight narrative
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

from config import SENSITIVITY_RANGE
from degradation_model import DegradationCurve
from strategy_engine import SensitivityResult, Strategy
from validation import DriverValidation

logger = logging.getLogger(__name__)

COMPOUND_COLORS = {
    "SOFT": "#FF3333",
    "MEDIUM": "#FFD700",
    "HARD": "#CCCCCC",
}


def plot_degradation_curves(
    curves: Dict[str, DegradationCurve],
    output_dir: str,
    max_stint_laps: int = 40,
) -> str:
    """Save a chart with one subplot per compound showing the fitted
    degradation curve and +/- sensitivity bands."""

    compounds = [c for c in ["SOFT", "MEDIUM", "HARD"] if c in curves]
    n = len(compounds)
    if n == 0:
        logger.warning("No degradation curves to plot")
        return ""

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    axes = axes[0]

    for ax, compound in zip(axes, compounds):
        curve = curves[compound]
        x = np.arange(0, max_stint_laps + 1)
        y = curve.predict(x)
        y_plus = y * (1 + SENSITIVITY_RANGE)
        y_minus = y * (1 - SENSITIVITY_RANGE)

        color = COMPOUND_COLORS.get(compound, "#888888")
        ax.plot(x, y, color=color, linewidth=2, label=compound.capitalize())
        ax.fill_between(x, y_minus, y_plus, alpha=0.2, color=color,
                        label=f"±{int(SENSITIVITY_RANGE*100)}%")
        ax.set_xlabel("Tire age (laps)")
        ax.set_ylabel("Degradation (s lost vs. fresh)")
        ax.set_title(f"{compound.capitalize()} Degradation")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "degradation_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved degradation chart → %s", path)
    return path


def format_strategy_table(
    strategies: List[Strategy],
    top_n: int = 10,
) -> str:
    """Return a formatted table of the top N strategies."""
    top = strategies[:top_n]
    if not top:
        return "No strategies to display."

    best_time = top[0].total_time
    rows = []
    for i, strat in enumerate(top, 1):
        rows.append([
            i,
            strat.num_stops,
            strat.description,
            f"{strat.total_time:.1f}",
            f"+{strat.total_time - best_time:.1f}",
        ])

    return tabulate(
        rows,
        headers=["Rank", "Stops", "Strategy", "Total (s)", "Δ to P1 (s)"],
        tablefmt="github",
    )


def format_validation_table(validations: List[DriverValidation]) -> str:
    """Return a formatted model-vs-reality comparison table."""
    if not validations:
        return "No validation data."

    rows = []
    for v in validations:
        rows.append([
            v.driver,
            v.actual_strategy.description,
            v.predicted_rank,
            f"{v.actual_race_time:.1f}",
            f"{v.predicted_race_time:.1f}",
            f"{v.model_error:+.1f}",
            f"{v.delta_to_optimal:+.1f}",
            f"{v.avg_traffic_loss_per_lap:+.2f}",
        ])

    return tabulate(
        rows,
        headers=[
            "Driver", "Actual Strategy", "Pred. Rank",
            "Actual Time (s)", "Pred. Time (s)", "Error (s)",
            "Δ Optimal (s)", "Traffic Loss/Lap (s)",
        ],
        tablefmt="github",
    )


def format_sensitivity_table(results: List[SensitivityResult]) -> str:
    """Return a formatted sensitivity analysis table."""
    if not results:
        return "No sensitivity data."

    rows = []
    for r in results:
        rows.append([
            r.strategy.description,
            f"{r.nominal_time:.1f}",
            f"{r.time_minus:.1f}",
            f"{r.time_plus:.1f}",
            "YES" if r.flips_type else "no",
        ])

    return tabulate(
        rows,
        headers=["Strategy", "Nominal (s)", "-15% Deg (s)", "+15% Deg (s)", "Type Flip?"],
        tablefmt="github",
    )


def generate_narrative(
    year: int,
    gp: str,
    strategies: List[Strategy],
    validations: List[DriverValidation],
    sensitivity: List[SensitivityResult],
) -> str:
    """Auto-generate a 2-3 sentence insight narrative for a race."""
    if not strategies:
        return "Insufficient data to generate insights."

    optimal = strategies[0]
    lines = []

    lines.append(
        f"For the {year} {gp} GP, the model's pre-race optimal strategy is a "
        f"{optimal.num_stops}-stop: {optimal.description} "
        f"(predicted total: {optimal.total_time:.1f} s)."
    )

    if sensitivity:
        flips = any(r.flips_type for r in sensitivity)
        if flips:
            lines.append(
                "Sensitivity analysis indicates this recommendation is FRAGILE: "
                "the optimal stop-count flips under a ±15% degradation shift, "
                "meaning Sunday conditions will be decisive."
            )
        else:
            lines.append(
                "The optimal strategy is ROBUST to ±15% degradation variation, "
                "maintaining its advantage across the sensitivity range."
            )

    if validations:
        aligned = sum(
            1 for v in validations
            if v.actual_strategy.num_stops == optimal.num_stops
        )
        total = len(validations)
        lines.append(
            f"{aligned} of {total} analyzed drivers matched the model's recommended "
            f"stop-count ({optimal.num_stops}-stop). "
            f"Median model error: {np.median([v.model_error for v in validations]):+.1f} s."
        )

    return " ".join(lines)


def save_outputs(
    output_dir: str,
    year: int,
    gp: str,
    curves: Dict[str, DegradationCurve],
    strategies: List[Strategy],
    validations: List[DriverValidation],
    sensitivity: List[SensitivityResult],
) -> None:
    """Write all deliverables to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    plot_degradation_curves(curves, output_dir)

    strat_table = format_strategy_table(strategies)
    val_table = format_validation_table(validations)
    sens_table = format_sensitivity_table(sensitivity)
    narrative = generate_narrative(year, gp, strategies, validations, sensitivity)

    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# {year} {gp} — Strategy Analysis Report\n\n")
        f.write("## Top 10 Strategies\n\n")
        f.write(strat_table + "\n\n")
        f.write("## Sensitivity Analysis\n\n")
        f.write(sens_table + "\n\n")
        f.write("## Model vs. Reality\n\n")
        f.write(val_table + "\n\n")
        f.write("## Insights\n\n")
        f.write(narrative + "\n")

    logger.info("Saved report → %s", report_path)
