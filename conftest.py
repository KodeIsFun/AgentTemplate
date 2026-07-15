"""Pytest config — makes `backend` importable when running tests directly.

The project layout uses a top-level `backend/` package; pytest's default
`rootdir` discovery handles this when there's a `pyproject.toml` / `setup.cfg`
/ `conftest.py` at the root. This file ensures `backend.*` resolves
regardless of where pytest is invoked from.
"""

import sys
from pathlib import Path

# Project root = parent of this conftest.py.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))