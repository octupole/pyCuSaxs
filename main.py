#!/usr/bin/env python3
"""Convenience wrapper to launch the pycusaxs CLI/GUI without installing."""

from pycusaxs.main import main as _main


if __name__ == "__main__":
    raise SystemExit(_main())
