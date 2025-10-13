"""
Core business logic for SAXS calculations.

This module contains the main computation logic that coordinates between
Python trajectory processing and the CUDA backend.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List

from .logger import get_logger
from .topology import Topology
from .saxs_defaults import SaxsDefaults

logger = get_logger('core')


def build_output_paths(base: str, frame_range: range) -> List[Path]:
    """
    Derive per-frame PDB paths from a base filename or directory.

    Args:
        base: Base output path (file or directory)
        frame_range: Range of frame indices

    Returns:
        List of output paths for each frame

    Raises:
        ValueError: If base path is invalid
    """
    base_path = None
    if base:
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
            return base_path.with_name(
                f"{base_path.stem}_frame_{frame:05d}{base_path.suffix}"
            )

        # Treat as directory (existing or to be created)
        return base_path / f"frame_{frame:05d}.pdb"

    return [resolve_for_frame(frame) for frame in frame_range]


def invoke_cuda_backend(
    required_params: Dict[str, Any],
    advanced_params: Dict[str, Any],
    topology: Topology
) -> List[str]:
    """
    Invoke the CUDA backend for SAXS calculation.

    Args:
        required_params: Required calculation parameters
        advanced_params: Optional/advanced parameters
        topology: Topology object with molecular structure

    Returns:
        List of result summary lines

    Raises:
        ImportError: If CUDA backend is unavailable
        RuntimeError: If CUDA calculation fails
    """
    try:
        import pycusaxs_cuda  # type: ignore[import-not-found]
    except ImportError as exc:
        logger.error(f"CUDA backend unavailable: {exc}")
        return [f"cuda backend unavailable: {exc}"]

    grid = list(required_params["grid_size"])

    # Handle scaled grid
    scaled_raw = advanced_params.get("grid_scaled")
    scaled_grid: List[int] = []
    if isinstance(scaled_raw, (list, tuple)):
        scaled_grid = [int(value) for value in scaled_raw]
    elif isinstance(scaled_raw, int) and scaled_raw > 0:
        scaled_grid = [int(scaled_raw)]

    begin = int(required_params["initial_frame"])
    end = int(required_params["last_frame"])
    stride = max(1, int(advanced_params.get("dt", SaxsDefaults.DT)))

    logger.info(
        f"Starting SAXS calculation: frames {begin}-{end} "
        f"(stride={stride}), grid={grid}"
    )

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
            scale_factor=float(
                advanced_params.get("scale_factor", SaxsDefaults.SCALE_FACTOR)
            ),
            bin_size=float(advanced_params.get("bin_size", SaxsDefaults.BIN_SIZE)),
            qcut=float(advanced_params.get("qcut", SaxsDefaults.QCUT)),
            water_model=str(
                advanced_params.get("water_model", SaxsDefaults.WATER_MODEL)
            ),
            sodium=int(advanced_params.get("sodium", SaxsDefaults.SODIUM)),
            chlorine=int(advanced_params.get("chlorine", SaxsDefaults.CHLORINE)),
            simulation=str(advanced_params.get("simulation", "")),
        )
        logger.info("SAXS calculation completed successfully")
    except Exception as exc:
        logger.error(f"CUDA backend error: {exc}")
        return [f"cuda backend error: {exc}"]

    summary = result.get("summary", "cuda backend returned no summary")
    return [summary]


def run_saxs_calculation(
    required_params: Dict[str, Any],
    advanced_params: Dict[str, Any]
) -> Iterable[str]:
    """
    Main entry point for SAXS calculation.

    This function validates inputs, loads the topology, and invokes
    the CUDA backend for computation.

    Args:
        required_params: Required parameters (topology, trajectory, grid, frames)
        advanced_params: Advanced/optional parameters

    Returns:
        Iterator yielding result lines

    Raises:
        FileNotFoundError: If topology or trajectory files not found
        ValueError: If parameters are invalid
    """
    # Resolve file paths
    try:
        topology_path = Path(required_params["topology"]).expanduser().resolve()
        trajectory_path = Path(required_params["trajectory"]).expanduser().resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid file path: {e}")

    begin = int(required_params["initial_frame"])
    end = int(required_params["last_frame"])

    # Validate file existence
    if not topology_path.is_file():
        raise FileNotFoundError(f"Topology file not found: {topology_path}")
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

    # Validate frame range
    if end < begin:
        raise ValueError(
            f"Last frame ({end}) must be >= initial frame ({begin})"
        )

    logger.info(f"Loading topology: {topology_path.name}")
    logger.info(f"Loading trajectory: {trajectory_path.name}")

    # Load topology
    topo = Topology(str(topology_path), str(trajectory_path))
    total_frames = topo.universe.trajectory.n_frames

    # Validate frame indices
    if begin >= total_frames:
        raise ValueError(
            f"Initial frame {begin} exceeds available frames (0-{total_frames - 1})"
        )
    if end >= total_frames:
        raise ValueError(
            f"Last frame {end} exceeds available frames (0-{total_frames - 1})"
        )

    # Report molecular composition
    frame_range = range(begin, end + 1)
    output_base = advanced_params.get("out", "") or ""
    output_paths = build_output_paths(output_base, frame_range)

    lines = []
    counts = topo.count_molecules()
    molecule_summary = (
        f"Molecule counts — total: {counts[0]}, proteins: {counts[1]}, "
        f"waters: {counts[2]}, ions: {counts[3]}, others: {counts[4]}"
    )
    logger.info(molecule_summary)
    lines.append(molecule_summary)

    # Invoke CUDA backend
    lines.extend(invoke_cuda_backend(required_params, advanced_params, topo))

    return lines
