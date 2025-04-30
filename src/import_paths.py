"""src/import_paths.py"""

import sys
from pathlib import Path

THIRD_PARTY_PATH = Path(__file__).resolve().parent / "third_party"

if str(THIRD_PARTY_PATH) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_PATH))
