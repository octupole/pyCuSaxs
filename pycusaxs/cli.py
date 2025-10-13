"""
Command-line interface for pyCuSaxs.

This module provides the argument parser and CLI execution logic
for running SAXS calculations from the command line.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional

from .logger import setup_logging, get_logger
from .config import SaxsConfig, parse_grid_values
from .core import run_saxs_calculation
from .topology import Topology
from .saxs_defaults import SaxsDefaults
from .saxs_database import SaxsDatabase

logger = get_logger('cli')


def build_cli_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="CUDA-accelerated SAXS calculation from MD trajectories",
        epilog="""
Examples:
  %(prog)s -s protein.tpr -x traj.xtc -g 64 -b 0 -e 100
  %(prog)s -s system.tpr -x traj.xtc -g 128 --water tip3p --na 150
  %(prog)s -s system.tpr -x traj.xtc --info  # Print system information
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "-s", "--topology",
        required=True,
        help="Path to the topology file (.tpr, .pdb, .gro)"
    )
    parser.add_argument(
        "-x", "--trajectory",
        required=True,
        help="Path to the trajectory file (.xtc, .trr, .dcd)"
    )
    parser.add_argument(
        "-g", "--grid",
        type=int,
        nargs='+',
        metavar='N',
        default=[SaxsDefaults.GRID_SIZE],
        help="Grid size: single value (broadcast to 3D) or 3 values (nx ny nz)"
    )

    # Frame range
    parser.add_argument(
        "-b", "--begin",
        type=int,
        default=SaxsDefaults.INITIAL_FRAME,
        help=f"Initial frame index (default: {SaxsDefaults.INITIAL_FRAME})"
    )
    parser.add_argument(
        "-e", "--end",
        type=int,
        help="Last frame index (default: same as --begin)"
    )

    # Output
    parser.add_argument(
        "-o", "--out",
        default=SaxsDefaults.OUTPUT,
        help=f"Output file path (default: '{SaxsDefaults.OUTPUT}')"
    )

    # Calculation parameters
    parser.add_argument(
        "--dt",
        type=int,
        default=SaxsDefaults.DT,
        help=f"Frame stride (default: {SaxsDefaults.DT})"
    )
    parser.add_argument(
        "--order",
        type=int,
        default=SaxsDefaults.ORDER,
        help=f"B-spline interpolation order 1-8 (default: {SaxsDefaults.ORDER})"
    )
    parser.add_argument(
        "--gridS",
        dest="grid_scaled",
        type=int,
        default=SaxsDefaults.GRID_SCALED,
        help=f"Scaled grid size (default: {SaxsDefaults.GRID_SCALED})"
    )
    parser.add_argument(
        "--Scale",
        dest="scale_factor",
        type=float,
        default=SaxsDefaults.SCALE_FACTOR,
        help=f"Grid scale factor (default: {SaxsDefaults.SCALE_FACTOR})"
    )
    parser.add_argument(
        "--bin", "--Dq",
        dest="bin_size",
        type=float,
        default=SaxsDefaults.BIN_SIZE,
        help=f"Histogram bin size in Å⁻¹ (default: {SaxsDefaults.BIN_SIZE})"
    )
    parser.add_argument(
        "-q", "--qcut",
        dest="qcut",
        type=float,
        default=SaxsDefaults.QCUT,
        help=f"Reciprocal space cutoff in Å⁻¹ (default: {SaxsDefaults.QCUT})"
    )

    # Solvent parameters
    parser.add_argument(
        "--water",
        dest="water",
        default=SaxsDefaults.WATER_MODEL,
        help=f"Water model (tip3p, spc, etc.) (default: '{SaxsDefaults.WATER_MODEL}')"
    )
    parser.add_argument(
        "--na",
        dest="na",
        type=int,
        default=SaxsDefaults.SODIUM,
        help=f"Sodium ion count (default: {SaxsDefaults.SODIUM})"
    )
    parser.add_argument(
        "--cl",
        dest="cl",
        type=int,
        default=SaxsDefaults.CHLORINE,
        help=f"Chloride ion count (default: {SaxsDefaults.CHLORINE})"
    )

    # Mode switches
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the GUI instead of running in CLI mode"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print detailed system information and exit (no calculation)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging output"
    )

    # Database options
    default_db = str(SaxsDefaults.get_user_database_path())
    parser.add_argument(
        "--save-db",
        dest="save_db",
        type=str,
        default=None,
        nargs='?',
        const='',
        metavar="DB_FILE",
        help=f"Save SAXS profile to SQLite database (default: {default_db})"
    )
    parser.add_argument(
        "--save-reference",
        dest="save_reference",
        action="store_true",
        help="Save to reference solvent database (requires write permission)"
    )

    return parser


def handle_info_mode(topology_path: Path, trajectory_path: Path) -> int:
    """
    Handle --info flag: print system information and exit.

    Args:
        topology_path: Path to topology file
        trajectory_path: Path to trajectory file

    Returns:
        Exit code (0 for success, 1 for error)
    """
    if not topology_path.is_file():
        logger.error(f"Topology file not found: {topology_path}")
        return 1
    if not trajectory_path.is_file():
        logger.error(f"Trajectory file not found: {trajectory_path}")
        return 1

    try:
        topo = Topology(str(topology_path), str(trajectory_path))
        topo.print_system_summary(verbose=True)

        # Also print raw dictionary for database schema design
        print("\n" + "="*60)
        print("RAW DATA (for database design)")
        print("="*60)
        info = topo.get_system_info()
        print(json.dumps(info, indent=2, default=str))

        return 0
    except Exception as exc:
        logger.error(f"Error reading system information: {exc}", exc_info=True)
        return 1


def save_to_database(
    namespace: argparse.Namespace,
    required_params: dict,
    advanced_params: dict,
    grid_values: tuple
) -> int:
    """
    Save calculated SAXS profile to database.

    Args:
        namespace: Parsed command-line arguments
        required_params: Required calculation parameters
        advanced_params: Advanced calculation parameters
        grid_values: Grid dimensions tuple

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Determine which database to use
        if namespace.save_reference:
            db_path = str(SaxsDefaults.get_reference_database_path())
            logger.info(f"Saving to REFERENCE database: {db_path}")
        elif namespace.save_db:
            db_path = namespace.save_db
        else:
            db_path = str(SaxsDefaults.get_user_database_path())

        # Read the output SAXS profile
        output_file = advanced_params.get("out") or "saxs.dat"
        if not Path(output_file).exists():
            logger.warning(
                f"Output file {output_file} not found, cannot save to database"
            )
            return 0

        # Parse the SAXS profile from output file
        profile_data = []
        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        q = float(parts[0])
                        iq = float(parts[1])
                        profile_data.append((q, iq))

        # Get system information from topology
        topology_path = Path(required_params["topology"]).expanduser().resolve()
        trajectory_path = Path(required_params["trajectory"]).expanduser().resolve()
        topo = Topology(str(topology_path), str(trajectory_path))
        info = topo.get_system_info()

        # Calculate simulation time analyzed
        first_frame = required_params["initial_frame"]
        last_frame_requested = required_params["last_frame"]
        trajectory_dt = info.get('dt', 0.0)  # ps per frame in trajectory
        dt = advanced_params.get("dt", 1)  # frame stride

        # Calculate the actual last frame processed (accounting for stride)
        n_frames = last_frame_requested - first_frame + 1
        actual_frames = (n_frames + dt - 1) // dt
        last_frame_processed = first_frame + (actual_frames - 1) * dt

        # Total simulation time is from first to last PROCESSED frame
        sim_time_ps = (
            (last_frame_processed - first_frame) * trajectory_dt
            if trajectory_dt else 0.0
        )

        # Get supercell scale
        scale_raw = advanced_params.get(
            "grid_scaled",
            advanced_params.get("scale_factor", 0.0)
        )
        if isinstance(scale_raw, list) and len(scale_raw) > 0:
            supercell_scale = float(scale_raw[0]) / float(grid_values[0])
        elif isinstance(scale_raw, (int, float)) and scale_raw > 0:
            supercell_scale = float(scale_raw)
        else:
            supercell_scale = advanced_params.get("scale_factor", 1.0)

        # Calculate supercell volume
        supercell_volume = info['box_volume'] * (supercell_scale ** 3)

        # Save to database
        with SaxsDatabase(db_path) as db:
            profile_id = db.save_profile(
                profile_data=profile_data,
                water_model=info.get(
                    'detected_water_model',
                    advanced_params.get('water_model', '')
                ),
                n_water_molecules=info.get('n_water_molecules', 0),
                ion_counts=info.get('ion_counts', {}),
                box_x=info.get('box_x', 0.0),
                box_y=info.get('box_y', 0.0),
                box_z=info.get('box_z', 0.0),
                box_volume=info.get('box_volume', 0.0),
                supercell_scale=supercell_scale,
                supercell_volume=supercell_volume,
                simulation_time_ps=sim_time_ps,
                n_frames_analyzed=actual_frames,
                grid_size=grid_values,
                frame_stride=dt,
                bin_size=advanced_params.get('bin_size'),
                qcut=advanced_params.get('qcut'),
                order=advanced_params.get('order'),
                density_g_cm3=info.get('density_g_cm3'),
                n_atoms=info.get('n_atoms'),
                other_molecules=info.get('other_molecules', {}),
                notes=f"Generated from {topology_path.name}"
            )

            logger.info(f"SAXS profile saved to database: {db_path}")
            logger.info(f"Profile ID: {profile_id}")
            logger.info(f"Water Model: {info.get('detected_water_model', 'N/A')}")
            logger.info(f"Supercell Scale: {supercell_scale:.4f}")
            logger.info(f"Simulation Time: {sim_time_ps:.2f} ps")

        return 0

    except Exception as exc:
        logger.error(f"Error saving to database: {exc}", exc_info=True)
        return 1


def run_cli(args: List[str]) -> int:
    """
    Execute CLI mode with parsed arguments.

    Args:
        args: Command-line argument list

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = build_cli_parser()
    namespace = parser.parse_args(args)

    # Setup logging
    setup_logging(verbose=namespace.verbose)

    # Handle --info flag
    if namespace.info:
        topology_path = Path(namespace.topology).expanduser().resolve()
        trajectory_path = Path(namespace.trajectory).expanduser().resolve()
        return handle_info_mode(topology_path, trajectory_path)

    # Parse grid values
    try:
        grid_values = parse_grid_values(namespace.grid)
    except ValueError as exc:
        logger.error(f"Invalid grid size: {exc}")
        return 1

    # Validate grid_scaled and scale_factor
    if namespace.grid_scaled == 0 and namespace.scale_factor == 0.0:
        logger.error(
            "Both grid_scaled and scale_factor are zero. "
            "Either set --gridS > 0 or --Scale > 0."
        )
        return 1

    # Determine last frame
    last_frame = namespace.end if namespace.end is not None else namespace.begin

    # Build parameter dictionaries
    required_params = {
        "topology": namespace.topology,
        "trajectory": namespace.trajectory,
        "grid_size": grid_values,
        "initial_frame": namespace.begin,
        "last_frame": last_frame,
    }
    advanced_params = {
        "out": namespace.out,
        "dt": namespace.dt,
        "order": namespace.order,
        "grid_scaled": namespace.grid_scaled,
        "scale_factor": namespace.scale_factor,
        "bin_size": namespace.bin_size,
        "qcut": namespace.qcut,
        "water_model": namespace.water,
        "sodium": namespace.na,
        "chlorine": namespace.cl,
        "simulation": "",
    }

    # Run SAXS calculation
    try:
        results = list(run_saxs_calculation(required_params, advanced_params))
    except Exception as exc:
        logger.error(f"Calculation failed: {exc}", exc_info=namespace.verbose)
        return 1

    # Save to database if requested
    if namespace.save_db is not None or namespace.save_reference:
        return save_to_database(namespace, required_params, advanced_params, grid_values)

    return 0
