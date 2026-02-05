"""
Add the repo root to sys.path for devlog scripts run as files.
"""

from pathlib import Path
import sys


def ensure_project_root() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


ensure_project_root()
