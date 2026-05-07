"""Load `.env` from the repository tree (nearest file wins, walking upward from this package)."""

from __future__ import annotations

from pathlib import Path

_DONE = False


def maybe_load_dotenv() -> None:
    """Idempotent; no-op if `python-dotenv` is not installed."""
    global _DONE
    if _DONE:
        return
    _DONE = True
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    here = Path(__file__).resolve().parent
    for directory in [here, *here.parents]:
        candidate = directory / ".env"
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            break
