"""Put the repo root on sys.path so the digit-prefixed modules import.

Files like ``01_from_scratch.py`` are valid modules but not valid identifiers,
so they have to be loaded with importlib.import_module. That only works if the
repo root is importable, which this conftest guarantees regardless of where
pytest is invoked from.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
