"""Put the main thesis repo on sys.path so `recording`, `camera`, `pipeline` import."""
import os
import sys
from pathlib import Path

_default = Path(__file__).resolve().parent.parent / "Thesis"
_root = Path(os.environ.get("THESIS_REPO", _default))

if not _root.is_dir():
    raise RuntimeError(f"thesis repo not found at {_root} (set THESIS_REPO)")

if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
