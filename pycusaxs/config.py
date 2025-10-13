"""
Configuration management for pyCuSaxs.

This module provides dataclasses and utilities for managing SAXS calculation
configuration parameters in a structured way.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from .saxs_defaults import SaxsDefaults


@dataclass
class SaxsConfig:
    """
    Centralized configuration for SAXS calculations.

    This dataclass consolidates all required and optional parameters
    for a SAXS calculation into a single, validated structure.

    Attributes:
        topology: Path to topology file (.tpr, .pdb, .gro)
        trajectory: Path to trajectory file (.xtc, .trr, .dcd)
        grid_size: Grid dimensions as (nx, ny, nz)
        initial_frame: Starting frame index
        last_frame: Ending frame index
        dt: Frame stride (default: 1)
        output: Output file path (default: saxs.dat)
        order: B-spline interpolation order 1-8 (default: 4)
        grid_scaled: Scaled grid dimensions (optional)
        scale_factor: Grid scaling factor (default: 1.0)
        bin_size: Histogram bin size in Å⁻¹ (default: 0.01)
        qcut: Q-space cutoff in Å⁻¹ (default: 0.5)
        water_model: Water model identifier (tip3p, spc, etc.)
        sodium: Sodium ion count (default: 0)
        chlorine: Chloride ion count (default: 0)
        simulation: Simulation ensemble type (nvt, npt)
    """

    # Required parameters
    topology: Path
    trajectory: Path
    grid_size: Tuple[int, int, int]
    initial_frame: int
    last_frame: int

    # Optional parameters with defaults
    dt: int = SaxsDefaults.DT
    output: str = SaxsDefaults.OUTPUT
    order: int = SaxsDefaults.ORDER
    grid_scaled: int = SaxsDefaults.GRID_SCALED
    scale_factor: float = SaxsDefaults.SCALE_FACTOR
    bin_size: float = SaxsDefaults.BIN_SIZE
    qcut: float = SaxsDefaults.QCUT
    water_model: str = SaxsDefaults.WATER_MODEL
    sodium: int = SaxsDefaults.SODIUM
    chlorine: int = SaxsDefaults.CHLORINE
    simulation: str = ""

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Convert paths to Path objects
        self.topology = Path(self.topology).expanduser().resolve()
        self.trajectory = Path(self.trajectory).expanduser().resolve()

        # Validate file existence
        if not self.topology.is_file():
            raise FileNotFoundError(f"Topology file not found: {self.topology}")
        if not self.trajectory.is_file():
            raise FileNotFoundError(f"Trajectory file not found: {self.trajectory}")

        # Validate frame range
        if self.last_frame < self.initial_frame:
            raise ValueError(
                f"Last frame ({self.last_frame}) must be >= initial frame ({self.initial_frame})"
            )

        # Validate grid size
        if any(dim <= 0 for dim in self.grid_size):
            raise ValueError(f"All grid dimensions must be positive: {self.grid_size}")

        # Validate other parameters
        if self.dt <= 0:
            raise ValueError(f"Frame stride (dt) must be positive: {self.dt}")
        if not (1 <= self.order <= 8):
            raise ValueError(f"B-spline order must be 1-8: {self.order}")
        if self.grid_scaled == 0 and self.scale_factor == 0.0:
            raise ValueError(
                "Either grid_scaled must be > 0 or scale_factor must be > 0"
            )

    def to_required_params(self) -> Dict[str, Any]:
        """
        Convert to required parameters dictionary for backward compatibility.

        Returns:
            Dictionary with required parameters
        """
        return {
            "topology": str(self.topology),
            "trajectory": str(self.trajectory),
            "grid_size": self.grid_size,
            "initial_frame": self.initial_frame,
            "last_frame": self.last_frame,
        }

    def to_advanced_params(self) -> Dict[str, Any]:
        """
        Convert to advanced parameters dictionary for backward compatibility.

        Returns:
            Dictionary with advanced/optional parameters
        """
        return {
            "out": self.output,
            "dt": self.dt,
            "order": self.order,
            "grid_scaled": self.grid_scaled,
            "scale_factor": self.scale_factor,
            "bin_size": self.bin_size,
            "qcut": self.qcut,
            "water_model": self.water_model,
            "sodium": self.sodium,
            "chlorine": self.chlorine,
            "simulation": self.simulation,
        }

    @classmethod
    def from_dicts(cls, required: Dict[str, Any], advanced: Dict[str, Any]) -> 'SaxsConfig':
        """
        Create SaxsConfig from separate required and advanced parameter dictionaries.

        Args:
            required: Dictionary with required parameters
            advanced: Dictionary with advanced parameters

        Returns:
            SaxsConfig instance
        """
        return cls(
            topology=required["topology"],
            trajectory=required["trajectory"],
            grid_size=required["grid_size"],
            initial_frame=required["initial_frame"],
            last_frame=required["last_frame"],
            dt=advanced.get("dt", SaxsDefaults.DT),
            output=advanced.get("out", SaxsDefaults.OUTPUT),
            order=advanced.get("order", SaxsDefaults.ORDER),
            grid_scaled=advanced.get("grid_scaled", SaxsDefaults.GRID_SCALED),
            scale_factor=advanced.get("scale_factor", SaxsDefaults.SCALE_FACTOR),
            bin_size=advanced.get("bin_size", SaxsDefaults.BIN_SIZE),
            qcut=advanced.get("qcut", SaxsDefaults.QCUT),
            water_model=advanced.get("water_model", SaxsDefaults.WATER_MODEL),
            sodium=advanced.get("sodium", SaxsDefaults.SODIUM),
            chlorine=advanced.get("chlorine", SaxsDefaults.CHLORINE),
            simulation=advanced.get("simulation", ""),
        )


def parse_grid_values(value: list) -> Tuple[int, int, int]:
    """
    Parse grid size values from command-line or GUI input.

    Args:
        value: List with 1 or 3 integer values

    Returns:
        Tuple of (nx, ny, nz)

    Raises:
        ValueError: If value doesn't contain 1 or 3 integers

    Examples:
        >>> parse_grid_values([128])
        (128, 128, 128)
        >>> parse_grid_values([64, 64, 128])
        (64, 64, 128)
    """
    if len(value) == 1:
        return (int(value[0]), int(value[0]), int(value[0]))

    if len(value) == 3:
        return (int(value[0]), int(value[1]), int(value[2]))

    raise ValueError("Grid size must contain either 1 or 3 integers.")
