#!/usr/bin/env python3
"""
Main entry point for pyCuSaxs.

This module provides the main() function that determines whether to
launch the GUI or execute CLI mode based on command-line arguments.
"""

from __future__ import annotations

import sys
from typing import List, Optional

from .logger import setup_logging, get_logger
from .cli import run_cli, build_cli_parser
from .gui import run_gui

logger = get_logger('main')


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for pyCuSaxs application.

    This function determines the execution mode (GUI or CLI) based on
    command-line arguments and dispatches to the appropriate handler.

    Args:
        argv: Command-line arguments (default: sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for error)

    Examples:
        # Launch GUI (no arguments)
        $ pycusaxs

        # Run CLI calculation
        $ pycusaxs -s system.tpr -x traj.xtc -g 64 -b 0 -e 100

        # Explicit GUI mode
        $ pycusaxs gui
        $ pycusaxs --gui
    """
    argv = list(sys.argv[1:] if argv is None else argv)

    # No arguments -> launch GUI
    if not argv:
        return run_gui()

    # Explicit GUI mode
    if argv[0].lower() == "gui":
        return run_gui()

    # Check for --gui flag
    parser = build_cli_parser()
    # Quick parse to check for --gui without validation
    namespace, _ = parser.parse_known_args(argv)
    if hasattr(namespace, 'gui') and namespace.gui:
        return run_gui()

    # Otherwise, run CLI mode
    return run_cli(argv)


if __name__ == "__main__":
    sys.exit(main())
