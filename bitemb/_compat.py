"""Compatibility patches for optional third-party dependencies."""

from __future__ import annotations


def patch_multiprocess_resource_tracker() -> None:
    """Suppress a Python 3.12 shutdown bug in multiprocess.resource_tracker."""
    try:
        import multiprocess.resource_tracker as resource_tracker
    except Exception:
        return

    resource_tracker.ResourceTracker.__del__ = lambda self: None


patch_multiprocess_resource_tracker()
