#!/usr/bin/env python3
"""GUI entry-point that ties the SAXS widget to topology/trajectory utilities."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PySide6.QtWidgets import QApplication, QMessageBox

from .saxs_widget import SaxsParametersWindow
from .topology import Topology


def build_output_paths(base: str, frame_range: range) -> List[Path]:
    """Derive per-frame PDB paths from a base filename or directory."""
    base_path = Path(base).expanduser() if base else None

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


def format_parameters(required_params: Dict[str, Any], advanced_params: Dict[str, Any]) -> str:
    lines = ["Required Parameters:"]
    for key, value in required_params.items():
        lines.append(f"  {key}: {value}")

    lines.append("Advanced Parameters:")
    for key, value in advanced_params.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


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
    stride = max(1, int(advanced_params.get("dt", 1)))

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
            output=advanced_params.get("out", ""),
            order=int(advanced_params.get("order", 4)),
            scale_factor=float(advanced_params.get("scale_factor", 1.0)),
            bin_size=float(advanced_params.get("bin_size", 0.0)),
            qcut=float(advanced_params.get("qcut", 0.0)),
            water_model=str(advanced_params.get("water_model", "")),
            sodium=int(advanced_params.get("sodium", 0)),
            chlorine=int(advanced_params.get("chlorine", 0)),
            simulation=str(advanced_params.get("simulation", "")),
        )
    except Exception as exc:  # pragma: no cover - delegates to C++
        return [f"cuda backend error: {exc}"]

    summary = result.get("summary", "cuda backend returned no summary")
    return [summary]

def cuda_connect(required_params: Dict[str, Any], advanced_params: Dict[str, Any]) -> Iterable[str]:
    topology_path = Path(required_params["topology"]).expanduser()
    trajectory_path = Path(required_params["trajectory"]).expanduser()
    begin = int(required_params["initial_frame"])
    end = int(required_params["last_frame"])

    if not topology_path.is_file():
        raise FileNotFoundError(f"Topology file not found: {topology_path}")
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")
    if end < begin:
        raise ValueError("Last frame (-e) must be greater than or equal to initial frame (-b).")

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
        "Molecule counts — total: {0}, proteins: {1}, waters: {2}, ions: {3}, others: {4}".format(*counts)
    )

    lines.extend(_invoke_cuda_backend(required_params, advanced_params, topo))

    return lines


class SaxsMainWindow(SaxsParametersWindow):
    """Specialized window that runs trajectory export when Execute is pressed."""

    def execute(self) -> None:  # type: ignore[override]
        try:
            required_params = self.required_widget.parameters()
        except ValueError as error:
            QMessageBox.warning(self, "Invalid Input", str(error))
            return

        advanced_params = self.advanced_widget.parameters()
        summary = format_parameters(required_params, advanced_params)

        try:
            results = list(cuda_connect(required_params, advanced_params))
        except Exception as exc:  # pragma: no cover - GUI feedback
            QMessageBox.critical(self, "Execution failed", str(exc))
            return

        message = "\n".join([summary, "", *results])
        self.output_view.setPlainText(message)
        print(message)

def _parse_grid_values(value: str) -> tuple[int, int, int]:
    cleaned = value.replace(",", " ").split()
    if not cleaned:
        raise ValueError("Grid size must contain 1 or 3 integers.")
    try:
        numbers = [int(part) for part in cleaned]
    except ValueError as exc:
        raise ValueError("Grid size entries must be integers.") from exc
    if len(numbers) == 1:
        return tuple([numbers[0]] * 3)
    if len(numbers) == 3:
        return tuple(numbers)
    raise ValueError("Grid size must contain either 1 or 3 integers.")


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate PDB snapshots from a trajectory without launching the GUI.",
    )
    parser.add_argument("-s", "--topology", required=True, help="Path to the topology file (-s)")
    parser.add_argument("-x", "--trajectory", required=True, help="Path to the trajectory file (-x)")
    parser.add_argument(
        "-g",
        "--grid",
        default="128",
        help="Grid size as nx[,ny,nz] (default: 128)",
    )
    parser.add_argument(
        "-b", "--begin", type=int, default=0, help="Initial frame index (default: 0)",
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
        default="",
        help="Output PDB path or directory (default: trajectory_frame_XXXXX.pdb)",
    )
    parser.add_argument("--dt", type=float, default=0.0, help="Frame interval (--dt)")
    parser.add_argument("--order", type=int, default=3, help="BSpline order (--order)")
    parser.add_argument(
        "--gridS",
        dest="grid_scaled",
        type=int,
        default=0,
        help="Scaled grid size (--gridS)",
    )
    parser.add_argument(
        "--Scale",
        dest="scale_factor",
        type=float,
        default=1.0,
        help="Grid scale factor (--Scale)",
    )
    parser.add_argument(
        "--bin",
        "--Dq",
        dest="bin_size",
        type=float,
        default=0.0,
        help="Histogram bin size (--bin/--Dq)",
    )
    parser.add_argument(
        "-q",
        "--qcut",
        dest="qcut",
        type=float,
        default=0.0,
        help="Reciprocal space cutoff (-q/--qcut)",
    )
    parser.add_argument(
        "--water",
        dest="water",
        default="",
        help="Model to use for the weighting function (--water)",
    )
    parser.add_argument(
        "--na",
        dest="na",
        type=float,
        default=0.0,
        help="Sodium atoms (--na)",
    )
    parser.add_argument(
        "--cl",
        dest="cl",
        type=float,
        default=0.0,
        help="Chlorine atoms (--cl)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the GUI instead of running in CLI mode.",
    )
    return parser


def _run_cli(namespace: argparse.Namespace) -> int:
    try:
        grid_values = _parse_grid_values(namespace.grid)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
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

    summary = format_parameters(required_params, advanced_params)

    try:
        results = list(cuda_connect(required_params, advanced_params))
    except Exception as exc:  # pragma: no cover - CLI feedback
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(summary)
    print()
    # for line in results:
    #     print(line)
    return 0


def _run_gui() -> int:
    app = QApplication([sys.argv[0]])
    window = SaxsMainWindow()
    window.resize(640, 720)
    window.show()
    return app.exec()


def main(argv: List[str] | None = None) -> int:
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
