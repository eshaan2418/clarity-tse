"""Train a TSE model (streaming or offline) from a YAML config.

    uv run python -m bench.train --config configs/m3/b10_s0.yaml --out results/m3/runs

Writes <out>/<name>/{config.yaml, train_log.jsonl, summary.json} and the checkpoint under
data/cache/ckpt/<name>.pt (checkpoints are git-ignored; their SHA-256 is recorded).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from clarity_tse.eval import items as ev_items
from clarity_tse.eval import metrics as M
from clarity_tse.eval.latency import write_result
from clarity_tse.models.losses import tse_loss
from clarity_tse.models.tse import TSEConfig, TSENet
from clarity_tse.synth import corpus
from clarity_tse.synth.train_data import MixConfig, OnlineMixDataset, PoolDataset, pool_dir

CKPT_DIR = corpus.CACHE / "ckpt"
DEFAULTS: dict[str, Any] = dict(
    name="run", seed=0, init_from=None,
    model={},
    train=dict(steps=20000, batch=16, lr=1e-3, clip_s=4.0, workers=10, eval_every=2000, val_set="val_main",
               val_per_cell=2, device="auto", warmup=500, size_per_epoch=8000),
    loss=dict(w_spec=1.0, w_over=1.0, w_kd=0.0, teacher=None),
    data=dict(bank_size=None, p_absent=0.08, p_no_voice=0.15, p_no_doa=0.15, doa_noise_deg=5.0),
)


def deep_update(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(path: str | Path, device: str = "cpu") -> TSENet:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = TSENet(TSEConfig(**ck["model_cfg"]))
    net.load_state_dict(ck["state_dict"])
    return net.to(device).eval()


def shard_sampler(ds, seed: int) -> list[int]:
    """Shuffle shard order and items within shards (keeps each worker's shard cache hot)."""
    from clarity_tse.synth.train_data import POOL_SHARD

    rng = np.random.default_rng(seed)
    if not isinstance(ds, PoolDataset):
        return rng.permutation(len(ds)).tolist()
    order: list[int] = []
    shards = rng.permutation(len(ds.files))
    # interleave 4 shards at a time so batches mix scene geometries
    for g in range(0, len(shards), 4):
        group = [s * POOL_SHARD + rng.permutation(POOL_SHARD) for s in shards[g : g + 4]]
        order += np.stack(group, 1).reshape(-1).tolist() if len(group) == 4 else np.concatenate(group).tolist()
    return order


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def validate(net: TSENet, val_set: str, ids: list[str], device: str) -> dict[str, float]:
    net.eval()
    embs = ev_items.enrollment_embeddings(val_set)
    res = []
    for it, a in ev_items.iter_items(val_set, ids):
        mix = torch.tensor(a["mix"])[None].to(device)
        K = net.frames_for(mix.shape[-1])
        th = torch.tensor(ev_items.frame_theta(a["theta"], net.cfg.hop, K))[None].to(device)
        e = torch.tensor(embs[it["id"]])[None].to(device)
        y = net(mix, e, th)[0].cpu().numpy()
        res.append(M.si_sdr(y, a["target_early"]) - M.si_sdr(a["mix"][0], a["target_early"]))
    net.train()
    return dict(val_si_sdri=float(np.mean(res)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--override", default="{}", help="JSON dict deep-merged into config")
    args = ap.parse_args()
    cfg = deep_update(DEFAULTS, yaml.safe_load(Path(args.config).read_text()))
    cfg = deep_update(cfg, json.loads(args.override))
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])  # noqa: NPY002 (legacy global RNG used by some deps)
    tr = cfg["train"]
    device = pick_device(tr["device"])
    out = Path(args.out) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    mcfg = TSEConfig(**cfg["model"])
    net = TSENet(mcfg)
    if cfg["init_from"]:
        ck = torch.load(cfg["init_from"], map_location="cpu", weights_only=False)
        missing = net.load_state_dict(ck["state_dict"], strict=False)
        print("init_from", cfg["init_from"], missing)
    net = net.to(device).train()
    teacher = None
    if cfg["loss"]["teacher"] and cfg["loss"]["w_kd"] > 0:
        teacher = load_model(cfg["loss"]["teacher"], device)
        for p in teacher.parameters():
            p.requires_grad_(False)
    d = cfg["data"]
    mix_cfg = MixConfig(clip_s=tr["clip_s"], p_absent=d["p_absent"],
                        p_no_voice=d["p_no_voice"] if mcfg.use_voice else 1.0,
                        p_no_doa=d["p_no_doa"] if mcfg.use_doa else 1.0, doa_noise_deg=d["doa_noise_deg"])
    ds: torch.utils.data.Dataset
    if d.get("source", "pool") == "pool" and any(pool_dir("train").glob("shard_*/done")):
        ds = PoolDataset("train", mix_cfg, seed=cfg["seed"], n_shards=d.get("pool_shards"), crop_s=tr["clip_s"])
    else:
        ds = OnlineMixDataset("train", tr["size_per_epoch"], mix_cfg, seed=cfg["seed"], bank_size=d["bank_size"])
    print(f"[{cfg['name']}] dataset {type(ds).__name__} n={len(ds)} device={device} params={net.n_params()}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=tr["lr"], weight_decay=1e-4)
    steps = tr["steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / tr["warmup"]) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / steps))))
    val_ids = ev_items.stratified_ids(tr["val_set"], tr["val_per_cell"])
    log = open(out / "train_log.jsonl", "w")
    step, epoch, best = 0, 0, -1e9
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{cfg['name']}.pt"
    t0 = time.time()
    while step < steps:
        ds.set_epoch(epoch)
        dl = torch.utils.data.DataLoader(ds, batch_size=tr["batch"], num_workers=tr["workers"],
                                         sampler=shard_sampler(ds, cfg["seed"] * 1000 + epoch),
                                         drop_last=True, persistent_workers=False, prefetch_factor=4 if tr["workers"] else None)
        for b in dl:
            mix, ref = b["mix"].to(device), b["ref"].to(device)
            K = net.frames_for(mix.shape[-1])
            theta = b["theta"].to(device)[:, None].expand(-1, K).contiguous()
            emb = b["emb"].to(device)
            est = net(mix, emb, theta)
            t_out = None
            if teacher is not None:
                with torch.no_grad():
                    Kt = teacher.frames_for(mix.shape[-1])
                    t_out = teacher(mix, emb, b["theta"].to(device)[:, None].expand(-1, Kt).contiguous())
            loss, logs = tse_loss(est, ref, mix[:, 0], b["absent"].to(device), cfg["loss"]["w_spec"],
                                  cfg["loss"]["w_over"], t_out, cfg["loss"]["w_kd"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                rec = dict(step=step, loss=float(loss), grad_norm=float(gn), lr=sched.get_last_lr()[0],
                           elapsed_s=time.time() - t0, **logs)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if step % tr["eval_every"] == 0 or step == steps:
                v = validate(net, tr["val_set"], val_ids, device)
                log.write(json.dumps(dict(step=step, **v)) + "\n")
                log.flush()
                print(f"[{cfg['name']}] step {step} loss {float(loss):.2f} val_si_sdri {v['val_si_sdri']:.2f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                if v["val_si_sdri"] > best:
                    best = v["val_si_sdri"]
                    torch.save(dict(state_dict=net.state_dict(), model_cfg=mcfg.to_dict(), step=step, cfg=cfg), ckpt_path)
            if step >= steps:
                break
        epoch += 1
    log.close()
    summary = dict(name=cfg["name"], best_val_si_sdri=best, steps=steps, train_time_s=time.time() - t0,
                   n_params=net.n_params(), macs_per_frame=net.macs_per_frame(),
                   algorithmic_latency_ms=mcfg.latency.algorithmic_latency_ms, checkpoint=str(ckpt_path),
                   device=device)
    write_result(out / "summary.json", summary, config=cfg, seed=cfg["seed"], ckpt_hash=sha256_file(ckpt_path))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
