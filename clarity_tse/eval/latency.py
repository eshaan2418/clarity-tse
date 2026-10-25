"""Latency harness: algorithmic latency from config, measured per-hop compute latency and RTF.

Three latency types are always reported separately (README "Latency accounting"):
  * algorithmic  = synthesis window (2*hop) + lookahead frames * hop       [from config]
  * compute      = wall-clock time to process one hop on the given hardware  [measured]
  * system       = mic -> ear including OS audio stack / Bluetooth          [measured or estimated]
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]


@dataclass
class ComputeStats:
    n_hops: int
    hop_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    p999_ms: float
    max_ms: float
    mean_ms: float
    rtf: float  # mean compute / hop duration
    deadline_misses: int  # hops whose compute exceeded the hop duration


def measure_per_hop(step: Callable[[], Any], hop_ms: float, n_hops: int = 2000, warmup: int = 100) -> ComputeStats:
    for _ in range(warmup):
        step()
    t = np.empty(n_hops)
    for i in range(n_hops):
        t0 = time.perf_counter_ns()
        step()
        t[i] = (time.perf_counter_ns() - t0) / 1e6
    return ComputeStats(n_hops=n_hops, hop_ms=hop_ms,
                        p50_ms=float(np.percentile(t, 50)),
                        p90_ms=float(np.percentile(t, 90)), p99_ms=float(np.percentile(t, 99)),
                        p999_ms=float(np.percentile(t, 99.9)), max_ms=float(t.max()), mean_ms=float(t.mean()),
                        rtf=float(t.mean() / hop_ms), deadline_misses=int((t > hop_ms).sum()))


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=REPO, stderr=subprocess.DEVNULL) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def hardware() -> dict[str, str]:
    info = dict(machine=platform.machine(), system=platform.system(), release=platform.release(),
                python=platform.python_version())
    try:
        info["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        info["cpu"] = platform.processor()
    return info


def provenance(config: dict[str, Any] | None = None, seed: int | None = None, ckpt_hash: str | None = None,
               stage: str = "desktop") -> dict[str, Any]:
    """Metadata block attached to every results file (non-negotiable rule 1)."""
    return dict(git_commit=git_commit(), hardware=hardware(), evidence_stage=stage, seed=seed,
                checkpoint_sha256=ckpt_hash, config=config or {}, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                torch_threads=os.environ.get("OMP_NUM_THREADS", "default"), load_avg=list(os.getloadavg()))


def write_result(path: str | Path, payload: dict[str, Any], **prov: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["provenance"] = provenance(**prov)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    raise TypeError(type(o))
