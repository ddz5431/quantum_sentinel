from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """Find project root by looking for pyproject.toml."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent


PROJECT_ROOT = get_project_root()
RESULTS_DIR = PROJECT_ROOT / "results"
TRACES_DIR = RESULTS_DIR / "traces"


def ensure_dirs():
    """Create directories if needed."""
    for d in [RESULTS_DIR, TRACES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
