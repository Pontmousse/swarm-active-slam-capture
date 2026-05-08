"""
Backward-compatible entry point for plotting saved prototype runs.
Implementation lives in `telemetry.plotting`.
"""

from __future__ import annotations

from .telemetry.plotting import main

if __name__ == "__main__":
    main()
