"""
Progress tracking utilities for pyCuSaxs.

This module provides progress bar support for long-running SAXS calculations,
both for CLI and programmatic use.
"""

from typing import Optional, Iterator, Any
from contextlib import contextmanager

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from .logger import get_logger

logger = get_logger('progress')


class ProgressTracker:
    """
    Wrapper for progress tracking with tqdm support.

    This class provides a unified interface for progress tracking that
    gracefully degrades when tqdm is not available.

    Attributes:
        enabled: Whether progress tracking is enabled
        pbar: tqdm progress bar instance (if available)
    """

    def __init__(
        self,
        total: int,
        desc: str = "Processing",
        unit: str = "frame",
        enabled: bool = True,
        disable_tqdm: bool = False
    ):
        """
        Initialize progress tracker.

        Args:
            total: Total number of items to process
            desc: Description text for progress bar
            unit: Unit name for items (e.g., 'frame', 'step')
            enabled: Whether to enable progress tracking
            disable_tqdm: Force disable tqdm (use basic logging instead)
        """
        self.enabled = enabled and total > 0
        self.total = total
        self.desc = desc
        self.unit = unit
        self.current = 0
        self.pbar = None

        if self.enabled and TQDM_AVAILABLE and not disable_tqdm:
            self.pbar = tqdm(
                total=total,
                desc=desc,
                unit=unit,
                ncols=80,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
        elif self.enabled:
            logger.info(f"{desc}: 0/{total} {unit}s")

    def update(self, n: int = 1) -> None:
        """
        Update progress by n items.

        Args:
            n: Number of items completed (default: 1)
        """
        if not self.enabled:
            return

        self.current += n

        if self.pbar:
            self.pbar.update(n)
        else:
            # Log every 10% progress when tqdm is not available
            progress = (self.current * 100) // self.total
            if progress % 10 == 0:
                logger.info(
                    f"{self.desc}: {self.current}/{self.total} {self.unit}s "
                    f"({progress}%)"
                )

    def set_postfix(self, **kwargs) -> None:
        """
        Set additional information to display after the progress bar.

        Args:
            **kwargs: Key-value pairs to display (e.g., time=1.23)
        """
        if self.pbar:
            self.pbar.set_postfix(**kwargs)

    def close(self) -> None:
        """Close the progress bar."""
        if self.pbar:
            self.pbar.close()
        elif self.enabled:
            logger.info(f"{self.desc}: Complete ({self.current}/{self.total})")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


@contextmanager
def progress_bar(
    total: int,
    desc: str = "Processing",
    unit: str = "item",
    enabled: bool = True,
    disable_tqdm: bool = False
):
    """
    Context manager for progress tracking.

    Args:
        total: Total number of items
        desc: Description text
        unit: Unit name
        enabled: Whether to enable progress tracking
        disable_tqdm: Force disable tqdm

    Yields:
        ProgressTracker instance

    Examples:
        >>> with progress_bar(100, desc="Processing frames", unit="frame") as pbar:
        ...     for i in range(100):
        ...         # Do work
        ...         pbar.update(1)
    """
    tracker = ProgressTracker(total, desc, unit, enabled, disable_tqdm)
    try:
        yield tracker
    finally:
        tracker.close()


def iter_with_progress(
    iterable: Iterator[Any],
    total: Optional[int] = None,
    desc: str = "Processing",
    unit: str = "item",
    enabled: bool = True,
    disable_tqdm: bool = False
) -> Iterator[Any]:
    """
    Wrap an iterable with progress tracking.

    Args:
        iterable: Iterable to wrap
        total: Total number of items (required if iterable has no __len__)
        desc: Description text
        unit: Unit name
        enabled: Whether to enable progress tracking
        disable_tqdm: Force disable tqdm

    Yields:
        Items from the iterable

    Examples:
        >>> items = range(100)
        >>> for item in iter_with_progress(items, desc="Processing"):
        ...     # Process item
        ...     pass
    """
    # Try to get length from iterable
    if total is None:
        try:
            total = len(iterable)  # type: ignore
        except TypeError:
            logger.warning(
                "Cannot determine total length for progress tracking. "
                "Consider passing 'total' parameter."
            )
            total = 0

    if not enabled or total == 0:
        yield from iterable
        return

    if TQDM_AVAILABLE and not disable_tqdm:
        yield from tqdm(
            iterable,
            total=total,
            desc=desc,
            unit=unit,
            ncols=80
        )
    else:
        # Manual progress logging
        for i, item in enumerate(iterable, 1):
            yield item
            if i % max(1, total // 10) == 0:
                progress = (i * 100) // total
                logger.info(f"{desc}: {i}/{total} {unit}s ({progress}%)")


def format_time(seconds: float) -> str:
    """
    Format time duration in human-readable format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "1h 23m 45s")

    Examples:
        >>> format_time(3665)
        '1h 1m 5s'
        >>> format_time(125)
        '2m 5s'
        >>> format_time(45)
        '45s'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def estimate_remaining_time(
    current: int,
    total: int,
    elapsed: float
) -> str:
    """
    Estimate remaining time based on current progress.

    Args:
        current: Number of items completed
        total: Total number of items
        elapsed: Time elapsed in seconds

    Returns:
        Formatted remaining time string

    Examples:
        >>> estimate_remaining_time(25, 100, 60)
        '3m 0s'
    """
    if current == 0:
        return "unknown"

    time_per_item = elapsed / current
    remaining_items = total - current
    remaining_seconds = time_per_item * remaining_items

    return format_time(remaining_seconds)
