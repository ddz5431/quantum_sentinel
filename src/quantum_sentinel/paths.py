"""
Project Paths: Single Source of Truth

Shannon's principle: Centralize information to minimize redundancy and error propagation across the channel.
"""

from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """Find project root by searching for marker file."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root")


# Pre-computed paths (evaluated once, cached)
PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
TRACES_DIR = RESULTS_DIR / "traces"
ASSAYS_DIR = RESULTS_DIR / "assays"


def ensure_dirs() -> None:
    """Create directories if they don't exist."""
    for d in [DATA_DIR, RESULTS_DIR, TRACES_DIR, ASSAYS_DIR]:
        d.mkdir(parents=True, exist_ok=True)