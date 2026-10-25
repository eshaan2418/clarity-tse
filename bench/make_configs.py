"""Generate M3 sweep configs (latency budgets x seeds + ablations) from configs/m3/base.yaml."""
import copy
from pathlib import Path

import yaml

BASE = yaml.safe_load(Path("configs/m3/base.yaml").read_text())
BUDGETS = {"b2": dict(hop=16), "b5": dict(hop=40), "b10": dict(hop=80), "b16": dict(hop=128),
           "b20": dict(hop=160), "b40": dict(n_fft=640, hop=320)}
TEACHER = "offline_s0"


def cfg(name, seed=0, model=None, loss=None, init=None, train=None, data=None):
    c = copy.deepcopy(BASE)
    c["name"], c["seed"] = name, seed
    c["model"].update(model or {})
    c["loss"].update(loss or {})
    c["train"].update(train or {})
    c.setdefault("data", {}).update(data or {})
    c["init_from"] = init
    return c


def main() -> None:
    out = Path("configs/m3/gen")
    out.mkdir(parents=True, exist_ok=True)
    cs = [cfg(TEACHER, model=dict(causal=False, hop=128, bottleneck=160, hidden=320), train=dict(steps=30000))]
    kd = dict(w_kd=0.5, teacher=f"data/cache/ckpt/{TEACHER}.pt")
    for b, m in BUDGETS.items():
        for s in (0, 1, 2):
            cs.append(cfg(f"{b}_s{s}", s, model=m, loss=kd))
    # ablations at the 10 ms operating point (seed 0)
    cs += [cfg("b10_nokd_s0", model=BUDGETS["b10"]),
           cfg("b10_noart_s0", model=BUDGETS["b10"], loss=dict(kd, w_spec=0.0, w_over=0.0)),
           cfg("b10_nodoa_s0", model=dict(BUDGETS["b10"], use_doa=False), loss=kd),
           cfg("b10_novoice_s0", model=dict(BUDGETS["b10"], use_voice=False), loss=kd),
           cfg("b10_1ch_s0", model=dict(BUDGETS["b10"], channels=1, use_doa=False), loss=kd),
           cfg("b10_noabsent_s0", model=BUDGETS["b10"], loss=kd, data=dict(p_absent=0.0)),
           # latency-aware curriculum: 2 ms model initialised from the 10 ms model, half the steps
           cfg("b2_curr_s0", model=BUDGETS["b2"], loss=kd, init="data/cache/ckpt/b10_s0.pt",
               train=dict(steps=BASE["train"]["steps"] // 2, warmup=100))]
    for c in cs:
        (out / f"{c['name']}.yaml").write_text(yaml.safe_dump(c))
    print(len(cs), "configs")


if __name__ == "__main__":
    main()
