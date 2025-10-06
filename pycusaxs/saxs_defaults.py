#!/usr/bin/env python3
"""Centralized default values for SAXS calculation parameters.

This module provides a single source of truth for all SAXS parameter defaults,
ensuring consistency between CLI and GUI interfaces.
"""

from __future__ import annotations
from typing import Dict, Any


class SaxsDefaults:
    """Default values for SAXS calculation parameters."""

    # Required parameters
    GRID_SIZE: str = "128"  # Can be parsed as single value or "nx,ny,nz"
    INITIAL_FRAME: int = 0
    LAST_FRAME: int = 0

    # Advanced parameters
    OUTPUT: str = ""
    DT: int = 1  # Frame interval
    ORDER: int = 4  # BSpline order
    GRID_SCALED: int = 0  # Scaled grid size (0 = use scale_factor instead)
    SCALE_FACTOR: float = 2.5  # Grid scale factor (ignored if grid_scaled > 0)
    BIN_SIZE: float = 0.01  # Histogram bin size
    QCUT: float = 1.5  # Reciprocal space cutoff
    WATER_MODEL: str = ""  # Water model identifier
    SODIUM: float = 0.0  # Sodium concentration
    CHLORINE: float = 0.0  # Chlorine concentration

    # UI-specific defaults
    GRID_SIZE_RANGE_MIN: int = 1
    GRID_SIZE_RANGE_MAX: int = 9_999_999
    DT_RANGE_MIN: int = 1
    DT_RANGE_MAX: int = 1_000_000
    ORDER_RANGE_MIN: int = 1
    ORDER_RANGE_MAX: int = 10
    GRID_SCALED_RANGE_MIN: int = 0  # 0 means use scale_factor instead
    GRID_SCALED_RANGE_MAX: int = 10_000
    SCALE_FACTOR_RANGE_MIN: float = 0.0
    SCALE_FACTOR_RANGE_MAX: float = 1_000.0
    SCALE_FACTOR_STEP: float = 0.1
    SCALE_FACTOR_DECIMALS: int = 3
    BIN_SIZE_RANGE_MIN: float = 0.0
    BIN_SIZE_RANGE_MAX: float = 1_000.0
    BIN_SIZE_STEP: float = 0.005
    BIN_SIZE_DECIMALS: int = 3
    QCUT_RANGE_MIN: float = 0.0
    QCUT_RANGE_MAX: float = 1_000.0
    QCUT_STEP: float = 0.1
    QCUT_DECIMALS: int = 3
    SODIUM_RANGE_MIN: float = 0.0
    SODIUM_RANGE_MAX: float = 1_000.0
    SODIUM_STEP: float = 1.0
    SODIUM_DECIMALS: int = 3
    CHLORINE_RANGE_MIN: float = 0.0
    CHLORINE_RANGE_MAX: float = 1_000.0
    CHLORINE_STEP: float = 1.0
    CHLORINE_DECIMALS: int = 3

    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        """Return all default values as a dictionary."""
        return {
            'grid_size': cls.GRID_SIZE,
            'initial_frame': cls.INITIAL_FRAME,
            'last_frame': cls.LAST_FRAME,
            'output': cls.OUTPUT,
            'dt': cls.DT,
            'order': cls.ORDER,
            'grid_scaled': cls.GRID_SCALED,
            'scale_factor': cls.SCALE_FACTOR,
            'bin_size': cls.BIN_SIZE,
            'qcut': cls.QCUT,
            'water_model': cls.WATER_MODEL,
            'sodium': cls.SODIUM,
            'chlorine': cls.CHLORINE,
        }
