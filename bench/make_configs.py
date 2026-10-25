"""Generate M3 sweep configs (latency budgets x seeds + ablations) from configs/m3/base.yaml.

Design (DECISIONS D12, D15):
  * offline_s0      non-causal ceiling, also the distillation teacher
  * b{2,5,10,16,20,40}_s0   one model per algorithmic-latency budget (ms)
  * b16 = operating point (largest budget with algorithmic + p99 compute <= 20 ms): 3 seeds + all ablations
  * b10: 3 seeds (second reference point)
  * b2/b5 are initialised from b10_s0 ("latency-aware curriculum"); b2_scratch_s0 is the ablation
    without curriculum.

    STEPS=6000 TEACHER_STEPS=8000 uv run python -m bench.make_configs
"""
import copy
import os
from pathlib import Path

import yaml

BASE = yaml.safe_load(Path("configs/m3/base.yaml").read_text())
STEPS = int(os.environ.get("STEPS", 6000))
TEACHER_STEPS = int(os.environ.get("TEACHER_STEPS", 8000))
BUDGETS = {"b2": dict(hop=16), "b5": dict(hop=40), "b10": dict(hop=80), "b16": dict(hop=128),
           "b20": dict(hop=160), "b40": dict(n_fft=640, hop=320)}
TEACHER = "offline_s0"
OP = "b16"
CURRICULUM = {"b2", "b5"}
# short hops are expensive per second of audio (more frames): train them on shorter crops
CLIP = {"b2": 2.0, "b5": 2.5}


def cfg(name, seed=0, model=None, loss=None, init=None, train=None, data=None):
    c = copy.deepcopy(BASE)
    c["name"], c["seed"] = name, seed
    c["model"].update(model or {})
    c["loss"].update(loss or {})
    c["train"].update(dict(steps=STEPS, eval_every=max(500, STEPS // 8)))
    c["train"].update(train or {})
    c.setdefault("data", {}).update(data or {})
    c["init_from"] = init
    return c


def budget_cfg(b, seed, **kw):
    tr = dict(kw.pop("train", {}))
    if b in CLIP:
        tr["clip_s"] = CLIP[b]
    init = None
    if b in CURRICULUM and not kw.pop("scratch", False):
        init = "data/cache/ckpt/b10_s0.pt"
        tr.update(steps=STEPS // 2, warmup=100)
    name = kw.pop("name", f"{b}_s{seed}")
    return cfg(name, seed, model=dict(BUDGETS[b], **kw.pop("model", {})), init=init, train=tr, **kw)


def main() -> None:
    out = Path("configs/m3/gen")
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.yaml"):
        old.unlink()
    kd = dict(w_kd=0.5, teacher=f"data/cache/ckpt/{TEACHER}.pt")
    cs = [cfg(TEACHER, model=dict(causal=False, hop=128, bottleneck=160, hidden=320),
              train=dict(steps=TEACHER_STEPS, eval_every=TEACHER_STEPS // 8))]
    cs += [budget_cfg(b, 0, loss=kd) for b in ("b10", "b16", "b20", "b40", "b5", "b2")]
    cs += [budget_cfg(b, s, loss=kd) for b in (OP, "b10") for s in (1, 2)]
    cs += [budget_cfg(OP, 0, name=f"{OP}_nokd_s0"),
           budget_cfg(OP, 0, name=f"{OP}_noart_s0", loss=dict(kd, w_spec=0.0, w_over=0.0)),
           budget_cfg(OP, 0, name=f"{OP}_nodoa_s0", model=dict(use_doa=False), loss=kd),
           budget_cfg(OP, 0, name=f"{OP}_novoice_s0", model=dict(use_voice=False), loss=kd),
           budget_cfg(OP, 0, name=f"{OP}_1ch_s0", model=dict(channels=1, use_doa=False), loss=kd),
           budget_cfg(OP, 0, name=f"{OP}_noabsent_s0", loss=kd, data=dict(p_absent=0.0)),
           budget_cfg("b2", 0, name="b2_scratch_s0", loss=kd, scratch=True)]
    for i, c in enumerate(cs):
        c["order"] = i
        (out / f"{c['name']}.yaml").write_text(yaml.safe_dump(c))
    print(len(cs), "configs, steps", STEPS)


if __name__ == "__main__":
    main()
