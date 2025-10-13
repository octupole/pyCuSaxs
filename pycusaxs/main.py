#!/usr/bin/env python3
"""GUI entry-point that ties the SAXS widget to topology/trajectory utilities."""

from __future__ import annotations

import argparse
import sys
import os
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import QObject, Signal, QTimer, QProcess

from .saxs_defaults import SaxsDefaults
from .saxs_widget import SaxsParametersWindow
from .topology import Topology


def build_output_paths(base: str, frame_range: range) -> List[Path]:
    """Derive per-frame PDB paths from a base filename or directory."""
    base_path = None
    if base:
        # Resolve path to absolute path
        try:
            base_path = Path(base).expanduser().resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid path '{base}': {e}")

    def resolve_for_frame(frame: int) -> Path:
        if base_path is None:
            return Path(f"trajectory_frame_{frame:05d}.pdb")

        if base_path.suffix.lower() == ".pdb":
            if len(frame_range) == 1:
                return base_path
            return base_path.with_name(f"{base_path.stem}_frame_{frame:05d}{base_path.suffix}")

        # Treat as directory (existing or to be created)
        return base_path / f"frame_{frame:05d}.pdb"

    return [resolve_for_frame(frame) for frame in frame_range]


def _invoke_cuda_backend(required_params: Dict[str, Any],
                         advanced_params: Dict[str, Any], topology: Topology) -> List[str]:
    try:
        import pycusaxs_cuda  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        return [f"cuda backend unavailable: {exc}"]

    grid = list(required_params["grid_size"])

    scaled_raw = advanced_params.get("grid_scaled")
    scaled_grid: List[int] = []
    if isinstance(scaled_raw, (list, tuple)):
        scaled_grid = [int(value) for value in scaled_raw]
    elif isinstance(scaled_raw, int) and scaled_raw > 0:
        scaled_grid = [int(scaled_raw)]

    begin = int(required_params["initial_frame"])
    end = int(required_params["last_frame"])
    stride = max(1, int(advanced_params.get("dt", SaxsDefaults.DT)))

    try:
        result = pycusaxs_cuda.run(
            obj_topology=topology,
            topology=required_params["topology"],
            trajectory=required_params["trajectory"],
            grid=grid,
            scaled_grid=scaled_grid,
            begin=begin,
            end=end,
            stride=stride,
            output=advanced_params.get("out", SaxsDefaults.OUTPUT),
            order=int(advanced_params.get("order", SaxsDefaults.ORDER)),
            scale_factor=float(advanced_params.get(
                "scale_factor", SaxsDefaults.SCALE_FACTOR)),
            bin_size=float(advanced_params.get(
                "bin_size", SaxsDefaults.BIN_SIZE)),
            qcut=float(advanced_params.get("qcut", SaxsDefaults.QCUT)),
            water_model=str(advanced_params.get(
                "water_model", SaxsDefaults.WATER_MODEL)),
            sodium=int(advanced_params.get("sodium", SaxsDefaults.SODIUM)),
            chlorine=int(advanced_params.get(
                "chlorine", SaxsDefaults.CHLORINE)),
            simulation=str(advanced_params.get("simulation", "")),
        )
    except Exception as exc:  # pragma: no cover - delegates to C++
        return [f"cuda backend error: {exc}"]

    summary = result.get("summary", "cuda backend returned no summary")
    return [summary]


def cuda_connect(required_params: Dict[str, Any], advanced_params: Dict[str, Any]) -> Iterable[str]:
    # Resolve file paths
    try:
        topology_path = Path(
            required_params["topology"]).expanduser().resolve()
        trajectory_path = Path(
            required_params["trajectory"]).expanduser().resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid file path: {e}")

    begin = int(required_params["initial_frame"])
    end = int(required_params["last_frame"])

    if not topology_path.is_file():
        raise FileNotFoundError(f"Topology file not found: {topology_path}")
    if not trajectory_path.is_file():
        raise FileNotFoundError(
            f"Trajectory file not found: {trajectory_path}")
    if end < begin:
        raise ValueError(
            "Last frame (-e) must be greater than or equal to initial frame (-b).")

    topo = Topology(str(topology_path), str(trajectory_path))
    total_frames = topo.universe.trajectory.n_frames
    if begin >= total_frames:
        raise ValueError(
            f"Initial frame {begin} exceeds available frames (0-{total_frames - 1})."
        )
    if end >= total_frames:
        raise ValueError(
            f"Last frame {end} exceeds available frames (0-{total_frames - 1})."
        )

    frame_range = range(begin, end + 1)
    output_base = advanced_params.get("out", "") or ""
    output_paths = build_output_paths(output_base, frame_range)

    lines = []
    counts = topo.count_molecules()
    lines.append(
        "Molecule counts — total: {0}, proteins: {1}, waters: {2}, ions: {3}, others: {4}".format(
            *counts)
    )

    lines.extend(_invoke_cuda_backend(required_params, advanced_params, topo))

    return lines


class SaxsMainWindow(SaxsParametersWindow):
    """Specialized window that runs trajectory export when Execute is pressed."""

    def __init__(self) -> None:
        super().__init__()
        self.process = None

    def execute(self) -> None:  # type: ignore[override]
        try:
            required_params = self.required_widget.parameters()
        except ValueError as error:
            QMessageBox.warning(self, "Invalid Input", str(error))
            return

        advanced_params = self.advanced_widget.parameters()

        # Validate grid_scaled and scale_factor
        if advanced_params.get('grid_scaled', 0) == 0 and advanced_params.get('scale_factor', 0.0) == 0.0:
            QMessageBox.warning(
                self,
                "Invalid Parameters",
                "Both grid_scaled and scale_factor are zero. Either set grid_scaled > 0 or scale_factor > 0."
            )
            return

        # Build CLI command
        cli_command = self.build_cli_command(required_params, advanced_params)
        self.cli_preview.setPlainText(cli_command)

        # Clear previous output
        self.output_view.setPlainText("Starting SAXS calculation...\n")
        self.output_view.repaint()

        # Build command arguments
        args = ["python", "-m", "pycusaxs.main"]

        # Add required arguments
        args.extend(["-s", str(required_params['topology'])])
        args.extend(["-x", str(required_params['trajectory'])])

        # Add grid
        grid = required_params['grid_size']
        if isinstance(grid, tuple) and len(grid) == 3:
            if grid[0] == grid[1] == grid[2]:
                args.extend(["-g", str(grid[0])])
            else:
                args.extend(["-g", f"{grid[0]},{grid[1]},{grid[2]}"])

        # Add frame range
        args.extend(["-b", str(required_params['initial_frame'])])
        args.extend(["-e", str(required_params['last_frame'])])

        # Add advanced parameters if not default
        if advanced_params.get('out'):
            args.extend(["-o", str(advanced_params['out'])])
        if advanced_params.get('dt', SaxsDefaults.DT) != SaxsDefaults.DT:
            args.extend(["--dt", str(advanced_params['dt'])])
        if advanced_params.get('order', SaxsDefaults.ORDER) != SaxsDefaults.ORDER:
            args.extend(["--order", str(advanced_params['order'])])
        if advanced_params.get('grid_scaled', SaxsDefaults.GRID_SCALED) != SaxsDefaults.GRID_SCALED:
            args.extend(["--gridS", str(advanced_params['grid_scaled'])])
        if advanced_params.get('scale_factor', SaxsDefaults.SCALE_FACTOR) != SaxsDefaults.SCALE_FACTOR:
            args.extend(["--Scale", str(advanced_params['scale_factor'])])
        if advanced_params.get('bin_size', SaxsDefaults.BIN_SIZE) != SaxsDefaults.BIN_SIZE:
            args.extend(["--bin", str(advanced_params['bin_size'])])
        if advanced_params.get('qcut', SaxsDefaults.QCUT) != SaxsDefaults.QCUT:
            args.extend(["-q", str(advanced_params['qcut'])])
        if advanced_params.get('water_model'):
            args.extend(["--water", str(advanced_params['water_model'])])
        if advanced_params.get('sodium', SaxsDefaults.SODIUM) != SaxsDefaults.SODIUM:
            args.extend(["--na", str(advanced_params['sodium'])])
        if advanced_params.get('chlorine', SaxsDefaults.CHLORINE) != SaxsDefaults.CHLORINE:
            args.extend(["--cl", str(advanced_params['chlorine'])])

        # Create and setup QProcess
        self.process = QProcess(self)
        self.process.setProcessChannelMode(
            QProcess.ProcessChannelMode.MergedChannels)

        # Connect signals for real-time output
        def append_output():
            data = self.process.readAllStandardOutput().data().decode('utf-8', errors='replace')
            self.output_view.appendPlainText(data.rstrip())
            self.output_view.verticalScrollBar().setValue(
                self.output_view.verticalScrollBar().maximum()
            )

        def handle_finished(exit_code, exit_status):
            if exit_code == 0:
                self.output_view.appendPlainText(
                    "\n=== Completed Successfully ===")
            else:
                self.output_view.appendPlainText(
                    f"\n=== Process exited with code {exit_code} ===")

        def handle_error(error):
            error_msg = f"Process error: {error}"
            self.output_view.appendPlainText(error_msg)
            QMessageBox.critical(self, "Execution Error", error_msg)

        self.process.readyReadStandardOutput.connect(append_output)
        self.process.finished.connect(handle_finished)
        self.process.errorOccurred.connect(handle_error)

        # Start the process
        self.process.start(args[0], args[1:])

        if not self.process.waitForStarted():
            QMessageBox.critical(self, "Error", "Failed to start process")


def _parse_grid_values(value: List[int]) -> Tuple[int, int, int]:

    if len(value) == 1:
        return tuple([int(value[0])] * 3)

    if len(value) == 3:
        return tuple([int(item) for item in value])

    raise ValueError("Grid size must contain either 1 or 3 integers.")


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate PDB snapshots from a trajectory without launching the GUI.",
    )
    parser.add_argument("-s", "--topology", required=True,
                        help="Path to the topology file (-s)")
    parser.add_argument("-x", "--trajectory", required=True,
                        help="Path to the trajectory file (-x)")
    parser.add_argument("-g", "--grid", type=int, nargs='+', metavar='N',
                        default=[SaxsDefaults.GRID_SIZE],
                        help="Give 1 value (broadcast to 3) or 3 values.")

    # parser.add_argument(
    #     "-g",
    #     "--grid",
    #     default=SaxsDefaults.GRID_SIZE,
    #     help=f"Grid size as nx[,ny,nz] (default: {SaxsDefaults.GRID_SIZE})",
    # )
    parser.add_argument(
        "-b", "--begin", type=int, default=SaxsDefaults.INITIAL_FRAME,
        help=f"Initial frame index (default: {SaxsDefaults.INITIAL_FRAME})",
    )
    parser.add_argument(
        "-e",
        "--end",
        type=int,
        help="Last frame index (default: same as --begin)",
    )
    parser.add_argument(
        "-o",
        "--out",
        default=SaxsDefaults.OUTPUT,
        help=f"Output file path (default: '{SaxsDefaults.OUTPUT}')",
    )
    parser.add_argument("--dt", type=int, default=SaxsDefaults.DT,
                        help=f"Frame interval (default: {SaxsDefaults.DT})")
    parser.add_argument("--order", type=int, default=SaxsDefaults.ORDER,
                        help=f"BSpline order (default: {SaxsDefaults.ORDER})")
    parser.add_argument(
        "--gridS",
        dest="grid_scaled",
        type=int,
        default=SaxsDefaults.GRID_SCALED,
        help=f"Scaled grid size (default: {SaxsDefaults.GRID_SCALED})",
    )
    parser.add_argument(
        "--Scale",
        dest="scale_factor",
        type=float,
        default=SaxsDefaults.SCALE_FACTOR,
        help=f"Grid scale factor (default: {SaxsDefaults.SCALE_FACTOR})",
    )
    parser.add_argument(
        "--bin",
        "--Dq",
        dest="bin_size",
        type=float,
        default=SaxsDefaults.BIN_SIZE,
        help=f"Histogram bin size (default: {SaxsDefaults.BIN_SIZE})",
    )
    parser.add_argument(
        "-q",
        "--qcut",
        dest="qcut",
        type=float,
        default=SaxsDefaults.QCUT,
        help=f"Reciprocal space cutoff (default: {SaxsDefaults.QCUT})",
    )
    parser.add_argument(
        "--water",
        dest="water",
        default=SaxsDefaults.WATER_MODEL,
        help=f"Model to use for the weighting function (default: '{SaxsDefaults.WATER_MODEL}')",
    )
    parser.add_argument(
        "--na",
        dest="na",
        type=float,
        default=SaxsDefaults.SODIUM,
        help=f"Sodium concentration (default: {SaxsDefaults.SODIUM})",
    )
    parser.add_argument(
        "--cl",
        dest="cl",
        type=float,
        default=SaxsDefaults.CHLORINE,
        help=f"Chlorine concentration (default: {SaxsDefaults.CHLORINE})",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the GUI instead of running in CLI mode.",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print detailed system information and exit (does not run SAXS calculation).",
    )
    default_db = str(SaxsDefaults.get_user_database_path())
    parser.add_argument(
        "--save-db",
        dest="save_db",
        type=str,
        default=None,
        nargs='?',
        const='',  # Empty string means use default
        metavar="DB_FILE",
        help=f"Save SAXS profile to SQLite database (default: {default_db})",
    )
    parser.add_argument(
        "--save-reference",
        dest="save_reference",
        action="store_true",
        help="Save to reference solvent database (requires write permission, for building reference library only)",
    )
    return parser


def _run_cli(namespace: argparse.Namespace) -> int:
    # Handle --info flag: print system information and exit
    if namespace.info:
        try:
            from pathlib import Path
            topology_path = Path(namespace.topology).expanduser().resolve()
            trajectory_path = Path(namespace.trajectory).expanduser().resolve()

            if not topology_path.is_file():
                print(f"Error: Topology file not found: {topology_path}", file=sys.stderr)
                return 1
            if not trajectory_path.is_file():
                print(f"Error: Trajectory file not found: {trajectory_path}", file=sys.stderr)
                return 1

            topo = Topology(str(topology_path), str(trajectory_path))
            topo.print_system_summary(verbose=True)

            # Also print the raw dictionary for database schema design
            print("\n" + "="*60)
            print("RAW DATA (for database design)")
            print("="*60)
            info = topo.get_system_info()
            import json
            print(json.dumps(info, indent=2, default=str))

            return 0
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1

    try:
        grid_values = _parse_grid_values(namespace.grid)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Validate grid_scaled and scale_factor
    if namespace.grid_scaled == 0 and namespace.scale_factor == 0.0:
        print(
            "Error: Both grid_scaled and scale_factor are zero. "
            "Either set --gridS > 0 or --Scale > 0.",
            file=sys.stderr
        )
        return 1

    last_frame = namespace.end if namespace.end is not None else namespace.begin

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
        "help": False,
        "water_model": namespace.water,
        "sodium": namespace.na,
        "chlorine": namespace.cl,
    }

    try:
        results = list(cuda_connect(required_params, advanced_params))
    except Exception as exc:  # pragma: no cover - CLI feedback
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # If --save-db or --save-reference was specified, save the profile to database
    if namespace.save_db is not None or namespace.save_reference:
        try:
            from .saxs_database import SaxsDatabase
            from pathlib import Path

            # Determine which database to use
            if namespace.save_reference:
                # Save to reference database (read-only, shipped with package)
                db_path = str(SaxsDefaults.get_reference_database_path())
                print(f"Saving to REFERENCE database: {db_path}")
            elif namespace.save_db:
                # Custom path specified
                db_path = namespace.save_db
            else:
                # Use default user database
                db_path = str(SaxsDefaults.get_user_database_path())

            # Read the output SAXS profile
            output_file = advanced_params.get("out") or "saxs.dat"
            if not Path(output_file).exists():
                print(f"Warning: Output file {output_file} not found, cannot save to database", file=sys.stderr)
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
            sim_time_ps = (last_frame_processed - first_frame) * trajectory_dt if trajectory_dt else 0.0

            # Get supercell scale
            scale_raw = advanced_params.get("grid_scaled", advanced_params.get("scale_factor", 0.0))
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
                    water_model=info.get('detected_water_model', advanced_params.get('water_model', '')),
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

                print(f"\n{'='*60}")
                print(f"SAXS profile saved to database: {db_path}")
                print(f"Profile ID: {profile_id}")
                print(f"Water Model: {info.get('detected_water_model', 'N/A')}")
                print(f"Supercell Scale: {supercell_scale:.4f}")
                print(f"Simulation Time: {sim_time_ps:.2f} ps")
                print(f"{'='*60}\n")

        except Exception as exc:
            print(f"Error saving to database: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1

    # Summary is now printed by the C++ backend with nice formatting
    # for line in results:
    #     print(line)
    return 0


def _run_gui() -> int:
    app = QApplication([sys.argv[0]])
    window = SaxsMainWindow()
    window.resize(640, 720)
    window.show()
    return app.exec()


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv:
        return _run_gui()

    if argv[0].lower() == "gui":
        return _run_gui()

    parser = _build_cli_parser()
    namespace = parser.parse_args(argv)

    if namespace.gui:
        return _run_gui()

    return _run_cli(namespace)


if __name__ == "__main__":
    sys.exit(main())
