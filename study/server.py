"""Local-only listening-test server (binds 127.0.0.1; responses saved to study/data/, nothing uploaded).

Stimuli are read from data/cache/study_stimuli/<condition>/<item>.wav (made by study/make_stimuli.py),
passed through the hearing-safety limiter on every request, and served as 16-bit WAV.
"""

from __future__ import annotations

import argparse
import io
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import soundfile as sf

from clarity_tse.dsp.limiter import SafetyLimiter

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
STIM = REPO / "data" / "cache" / "study_stimuli"
DATA = ROOT / "data"
SAFE = re.compile(r"^[A-Za-z0-9_+\-]+$")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._send(200, (ROOT / "static" / "index.html").read_bytes(), "text/html")
        if path == "/api/trials":
            pid = self.path.split("pid=")[-1]
            f = ROOT / "trial_lists" / f"{pid}.json"
            if not SAFE.match(pid) or not f.exists():
                return self._send(404, b"unknown participant", "text/plain")
            return self._send(200, f.read_bytes(), "application/json")
        m = re.match(r"^/audio/([A-Za-z0-9_]+)/([A-Za-z0-9_+\-]+)\.wav$", path)
        if m:
            f = STIM / m.group(1) / f"{m.group(2)}.wav"
            if not f.exists():
                return self._send(404, b"missing stimulus", "text/plain")
            x, sr = sf.read(f, dtype="float32")
            y = SafetyLimiter(sr=sr).process(x.T if x.ndim > 1 else x)
            buf = io.BytesIO()
            sf.write(buf, y.T if y.ndim > 1 else y, sr, format="WAV", subtype="PCM_16")
            return self._send(200, buf.getvalue(), "audio/wav")
        return self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/response":
            return self._send(404, b"", "text/plain")
        n = int(self.headers.get("Content-Length", "0"))
        rec = json.loads(self.rfile.read(min(n, 100_000)))
        pid = str(rec.get("participant", ""))
        if not SAFE.match(pid):
            return self._send(400, b"bad pid", "text/plain")
        DATA.mkdir(exist_ok=True)
        with open(DATA / f"{pid}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        return self._send(200, b"ok", "text/plain")

    def log_message(self, *a) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"listening test on http://127.0.0.1:{a.port}  (local only; data -> {DATA})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
