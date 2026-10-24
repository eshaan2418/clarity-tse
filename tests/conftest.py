import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "data" / "fixtures" / "external"
if os.environ.get("CLARITY_USE_REAL_DATA") != "1":
    os.environ["CLARITY_EXTERNAL"] = str(FIX)
    os.environ["CLARITY_CACHE"] = str(REPO / "data" / "fixtures" / "cache")
    if not (FIX / "LibriSpeech").exists():
        subprocess.check_call([sys.executable, str(REPO / "scripts" / "make_fixtures.py"), "--out", str(FIX)])
