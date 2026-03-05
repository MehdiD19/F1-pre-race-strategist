"""CLI entry point for the F1 Pre-Race Strategy Optimizer.

Usage examples:
    python main.py --year 2023 --gp Bahrain
    python main.py --year 2023 --gp Bahrain --drivers VER PER LEC
    python main.py --races 2023:Bahrain 2023:Spain 2023:Abu_Dhabi
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from config import COMPOUNDS
from data_collection import build_circuit_profile
from degradation_model import build_degradation_curves
from strategy_engine import (
    compute_base_pace,
    generate_strategies,
    rank_strategies,
    sensitivity_analysis,
)
from validation import validate_race
from visualization import save_outputs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("f1_strategy")


def run_analysis(
    year: int,
    gp: str,
    drivers: list[str] | None = None,
    output_root: str = "output",
) -> None:
    """Run the full pipeline for a single GP."""
    t0 = time.time()
    tag = f"{year}_{gp.replace(' ', '_')}"
    output_dir = os.path.join(output_root, tag)
    os.makedirs(output_dir, exist_ok=True)

    # Phase 1 — Data Collection
    logger.info("=" * 60)
    logger.info("Phase 1: Loading data for %d %s", year, gp)
    logger.info("=" * 60)
    profile = build_circuit_profile(year, gp)

    # Phase 2 — Degradation Modeling
    logger.info("=" * 60)
    logger.info("Phase 2: Building degradation curves")
    logger.info("=" * 60)
    available_compounds = [
        c for c in COMPOUNDS
        if c in {
            row
            for row in profile.fp2_session.laps["Compound"].unique()
        } | {
            row
            for row in profile.fp3_session.laps["Compound"].unique()
        }
    ]
    if not available_compounds:
        available_compounds = list(COMPOUNDS)

    deg_curves = build_degradation_curves(profile, compounds=available_compounds)
    if not deg_curves:
        logger.error("Could not build any degradation curves — aborting")
        return

    # Phase 3 — Strategy Generation & Scoring
    logger.info("=" * 60)
    logger.info("Phase 3: Generating and scoring strategies")
    logger.info("=" * 60)
    strategies = generate_strategies(
        profile.total_laps,
        compounds=list(deg_curves.keys()),
    )
    if not strategies:
        logger.error("No valid strategies generated — aborting")
        return

    base_paces = compute_base_pace(profile.quali_session, drivers=drivers)
    if not base_paces:
        logger.error("Could not compute base paces from qualifying — aborting")
        return

    reference_driver = min(base_paces, key=base_paces.get)
    reference_pace = base_paces[reference_driver]
    logger.info("Reference driver: %s (base pace %.3f s)", reference_driver, reference_pace)

    ranked = rank_strategies(
        strategies, deg_curves, reference_pace,
        profile.pit_loss_seconds, profile.total_laps,
    )

    sens = sensitivity_analysis(
        ranked, deg_curves, reference_pace,
        profile.pit_loss_seconds, profile.total_laps,
    )

    # Phase 4 — Validation
    logger.info("=" * 60)
    logger.info("Phase 4: Validating against actual race")
    logger.info("=" * 60)
    validations = validate_race(
        profile, deg_curves, ranked, base_paces, drivers=drivers,
    )

    # Phase 5 — Output
    logger.info("=" * 60)
    logger.info("Phase 5: Generating outputs")
    logger.info("=" * 60)
    save_outputs(
        output_dir, year, gp,
        deg_curves, ranked, validations, sens,
    )

    elapsed = time.time() - t0
    logger.info("Done — %d %s completed in %.0f s.  Output → %s", year, gp, elapsed, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="F1 Pre-Race Strategy Optimizer",
    )
    parser.add_argument("--year", type=int, help="Season year")
    parser.add_argument("--gp", type=str, help="Grand Prix name (e.g. Bahrain)")
    parser.add_argument(
        "--drivers", nargs="*", default=None,
        help="Driver abbreviations to analyze (default: all with quali times)",
    )
    parser.add_argument(
        "--races", nargs="*", default=None,
        help="Batch mode: space-separated year:gp pairs (e.g. 2023:Bahrain 2023:Spain)",
    )
    parser.add_argument(
        "--output", default="output",
        help="Root output directory (default: output/)",
    )
    args = parser.parse_args()

    if args.races:
        for entry in args.races:
            year_str, gp = entry.split(":", 1)
            gp = gp.replace("_", " ")
            run_analysis(int(year_str), gp, drivers=args.drivers, output_root=args.output)
    elif args.year and args.gp:
        run_analysis(args.year, args.gp, drivers=args.drivers, output_root=args.output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
